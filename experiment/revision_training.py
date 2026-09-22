#!/usr/bin/env python3
"""Versioned two-stage interfering-gain training with epoch-boundary resume."""
import argparse
import collections
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiment.benchmark_nn_overhead import digest
from experiment.benchmark_stateful_nn_overhead import DEFAULT_CHECKPOINT_ROOT, load_selected_models
from experiment.compare_stateful_prediction import StatefulPredictor
from experiment.prepare_stateful_trajectories import DEFAULT_SOURCE, build
from experiment import train_finite_window_vehicle_split as finite
from experiment import train_stateful_tbptt as stateful
from experiment.vehicle_split import trajectory_indices
from utils.directional_service import BEAM_AVERAGE_DB_CONVENTION

DEFAULT_SPLIT = ROOT/'experiment/results/stateful_tbptt_unified_split_20260913/vehicle_split_seed20.npz'
DEFAULT_TEST = ROOT/'data4sim/lbd1.00_800_830_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl'


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
    os.replace(temp, path)


def atomic_torch(path, obj):
    path = Path(path)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        torch.save(obj, stream)
    os.replace(temp, path)


def require_current_gain_convention(metadata):
    if metadata.get('interference_db_convention') != BEAM_AVERAGE_DB_CONVENTION:
        raise ValueError('Interfering-gain dB convention mismatch: use data/checkpoints with the original 1e-9 amplitude epsilon (-180 dB for zero channels); do not relabel existing -300 dB artifacts')


def train(args):
    if min(args.stage1_epochs,args.stage2_epochs,args.batch_size,args.cpu_threads) < 1:
        raise ValueError('Epochs, batch size and CPU threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable; no silent CPU fallback')
    source_meta = json.loads(args.data.with_suffix('.json').read_text())
    if source_meta.get('interference_label') != 'beam-average':
        raise ValueError('Training requires explicitly audited beam-average labels')
    require_current_gain_convention(source_meta)
    config = dict(data_sha256=digest(args.data), split_sha256=digest(args.split_file),
        stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size, seed=args.seed, chunk_length=10, backend=device.type,
        stage1_lr=1e-3, stage2_lr=1e-4, weight_decay=1e-4,
        interference_db_convention=BEAM_AVERAGE_DB_CONVENTION,
        max_samples=args.max_samples, max_trajectories=args.max_trajectories,
        smoke=bool(args.max_samples or args.max_trajectories or source_meta.get('smoke')),
        source_code={str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__), ROOT/'experiment/train_finite_window_vehicle_split.py',
             ROOT/'experiment/train_stateful_tbptt.py', ROOT/'utils/NN_utils.py',
             ROOT/'utils/directional_service.py')})
    if source_meta['output_sha256'] != config['data_sha256']:
        raise ValueError('Training dataset hash changed')
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output/'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_file = args.output/'training_config.json'
        if config_file.exists():
            if not args.resume:
                raise FileExistsError('Use --resume for this existing training directory')
            if json.loads(config_file.read_text()) != config:
                raise ValueError('Training configuration changed; use a new output directory')
        else:
            if any((args.output/s).exists() for s in ('stage1','stage2')):
                raise FileExistsError('Unversioned training outputs exist')
            atomic_json(config_file, config)
        data = stateful.load_data(args.data)
        epochs_this_run = 0
        train_ids, val_ids, _, _ = trajectory_indices(data, args.split_file)
        task = 'interfering_gain'
        for stage, epochs in ((1,args.stage1_epochs), (2,args.stage2_epochs)):
            dest = args.output/f'stage{stage}'/task
            dest.mkdir(parents=True, exist_ok=True)
            if (dest/'metadata.json').exists():
                completed = json.loads((dest/'metadata.json').read_text())
                if completed['epochs_completed'] == epochs and digest(dest/'best.pth') == completed['best_checkpoint_sha256']:
                    print(f'STAGE {stage} ALREADY COMPLETE', flush=True)
                    continue
            torch.manual_seed(args.seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(args.seed)
            model = finite.make_model(task, device)
            if stage == 2:
                model.load_state_dict(torch.load(args.output/'stage1'/task/'best.pth', map_location=device, weights_only=True))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if stage == 1 else 1e-4, weight_decay=1e-4)
            scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2, min_lr=1e-6)
                         if stage == 2 else None)
            ti, vi = train_ids, val_ids
            if args.max_trajectories:
                ti, vi = ti[:args.max_trajectories], vi[:max(2, args.max_trajectories//3)]
            if stage == 1:
                tt, tl = finite.finite_window_samples(data, ti, seed=args.seed)
                vt, vl = finite.finite_window_samples(data, vi, seed=args.seed+1)
                if args.max_samples:
                    tt, tl = tt[:args.max_samples], tl[:args.max_samples]
                    vt, vl = vt[:max(2,args.max_samples//3)], vl[:max(2,args.max_samples//3)]
            def epoch_run(training, epoch):
                if stage == 1:
                    return finite.run_epoch(model,task,data,tt if training else vt,tl if training else vl,
                        device,10,args.batch_size,optimizer if training else None,training,
                        args.seed+epoch if training else args.seed)
                return stateful.run_epoch(model,task,data,ti if training else vi,device,10,args.batch_size,
                    optimizer if training else None,training,args.seed+epoch if training else args.seed,
                    1.0,True)
            resume_file = dest/'resume.pt'
            start, best, best_epoch, history, elapsed = 0, float('inf'), 0, [], 0.
            best_weights = None
            if resume_file.exists():
                saved = torch.load(resume_file,map_location=device,weights_only=False)
                model.load_state_dict(saved['model'])
                optimizer.load_state_dict(saved['optimizer'])
                if scheduler:
                    scheduler.load_state_dict(saved['scheduler'])
                torch.set_rng_state(saved['torch_rng'].cpu())
                if device.type == 'cuda':
                    torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
                start,best,best_epoch,history,elapsed = (saved[k] for k in ('epoch','best','best_epoch','history','elapsed'))
                best_weights = {k:v.detach().cpu().clone() for k,v in saved['best_weights'].items()}
            elif stage == 2:
                validation = epoch_run(False, 0)
                best = validation['mae_db']
                best_weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                history = [dict(epoch=0, **{f'val_{k}':v for k,v in validation.items()})]
            started = time.monotonic()
            for epoch in range(start+1, epochs+1):
                train_metrics = epoch_run(True, epoch)
                val = epoch_run(False, epoch)
                lr = optimizer.param_groups[0]['lr']
                if val['mae_db'] < best:
                    best,best_epoch = val['mae_db'],epoch
                    best_weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if scheduler:
                    scheduler.step(val['mae_db'])
                history.append(dict(epoch=epoch,learning_rate=lr,
                    **{f'train_{k}':v for k,v in train_metrics.items()}, **{f'val_{k}':v for k,v in val.items()}))
                atomic_torch(resume_file,dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict() if scheduler else None,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                    best=best,best_epoch=best_epoch,best_weights=best_weights,history=history,
                    elapsed=elapsed+time.monotonic()-started))
                atomic_torch(dest/'best.pth',best_weights)
                atomic_json(dest/'history.json',history)
                atomic_json(args.output/'training_status.json',dict(stage=stage,epoch=epoch,total_epochs=epochs,best_mae_db=best))
                print(json.dumps(dict(stage=stage,**history[-1])),flush=True)
                epochs_this_run += 1
                if args.stop_after_epochs and epochs_this_run >= args.stop_after_epochs:
                    print('STOPPED AT EPOCH BOUNDARY; resume with the same configuration',flush=True)
                    return
            atomic_torch(dest/'best.pth',best_weights)
            atomic_torch(dest/'last.pth',{k:v.detach().cpu() for k,v in model.state_dict().items()})
            atomic_json(dest/'history.json',history)
            atomic_json(dest/'metadata.json',dict(task=task,best_epoch=best_epoch,best_validation_metric=best,
                epochs_completed=epochs,best_checkpoint_sha256=digest(dest/'best.pth'),
                input_normalization='paper',interference_label='beam-average',smoke=config['smoke'],
                interference_db_convention=BEAM_AVERAGE_DB_CONVENTION,
                data_sha256=config['data_sha256'],split_sha256=config['split_sha256'],
                chunk_length=10 if stage==2 else None,stage=stage,selection_metric='minimum validation MAE',
                elapsed_seconds=elapsed+time.monotonic()-started))
        atomic_json(args.output/'training_status.json',dict(status='completed',stage=2,epoch=args.stage2_epochs,
            smoke=config['smoke'],best_checkpoint=str((args.output/'stage2/interfering_gain/best.pth').resolve())))


def assemble(args):
    if json.loads((args.training/'training_status.json').read_text()).get('status') != 'completed':
        raise ValueError('Complete both training stages before assembling the model bundle')
    paths = {task: args.base_models/task for task in ('beam','desired_gain')}
    paths['interfering_gain'] = args.training/'stage2/interfering_gain'
    metadata = {task:json.loads((path/'metadata.json').read_text()) for task,path in paths.items()}
    if metadata['interfering_gain'].get('interference_label') != 'beam-average':
        raise ValueError('Wrong interference label')
    require_current_gain_convention(metadata['interfering_gain'])
    if not metadata['interfering_gain'].get('smoke') and len({m['split_sha256'] for m in metadata.values()}) != 1:
        raise ValueError('The three formal models must use the same vehicle split')
    manifest = {task:dict(path=str(path.resolve()),sha256=digest(path/'best.pth'),metadata=metadata[task]) for task,path in paths.items()}
    for task,row in manifest.items():
        if row['sha256'] != metadata[task]['best_checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
    dest_manifest = args.output/'bundle.json'
    if dest_manifest.exists() and json.loads(dest_manifest.read_text()) != manifest:
        raise FileExistsError('Model bundle changed; choose a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    for task,path in paths.items():
        dest = args.output/task
        dest.mkdir(exist_ok=True)
        for name in ('best.pth','metadata.json'):
            if (dest/name).exists() and digest(dest/name) != digest(path/name):
                raise FileExistsError(dest/name)
            shutil.copy2(path/name,dest/name)
    atomic_json(dest_manifest,manifest)
    print('MODEL BUNDLE',args.output,flush=True)


def cache(args):
    import fcntl
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _cache(args)


def _cache(args):
    torch.set_num_threads(args.cpu_threads)
    models,inventory = load_selected_models(args.models)
    meta = json.loads((args.models/'interfering_gain/metadata.json').read_text())
    if meta.get('interference_label') != 'beam-average':
        raise ValueError('Refusing legacy interfering-gain checkpoint')
    require_current_gain_convention(meta)
    expected = dict(source=str(args.source.resolve()),source_sha256=digest(args.source),models=inventory,
                    start=args.start,end=args.end,top_k=5,device=args.device,
                    interference_label='beam-average',smoke=bool(meta.get('smoke')),
                    interference_db_convention=BEAM_AVERAGE_DB_CONVENTION,
                    source_code={str(p.relative_to(ROOT)):digest(p) for p in
                        (Path(__file__),ROOT/'experiment/compare_stateful_prediction.py',ROOT/'utils/NN_utils.py')},
                    alignment='source x predicts target x+0.1; consumers use matching target')
    manifest_path = args.output.with_suffix('.json')
    if args.output.exists() and not manifest_path.exists():
        incomplete = args.output.with_suffix(f'.incomplete.{time.time_ns()}.pkl')
        os.replace(args.output,incomplete)
        print('PRESERVED UNCOMMITTED CACHE',incomplete,flush=True)
    if args.output.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(k)!=v for k,v in expected.items()) or digest(args.output)!=old['cache_sha256']:
            raise FileExistsError('Existing cache has a different protocol/hash')
        print('CACHE ALREADY COMPLETE',args.output,flush=True)
        return
    with args.source.open('rb') as f:
        timeline = pickle.load(f)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    for model in models.values():
        model.to(device).eval()
    streams = {task:StatefulPredictor(model) for task,model in models.items()}
    prepared = collections.OrderedDict()
    with torch.inference_mode():
        # Process earlier available frames to carry causal history into the interval.
        previous = None
        for frame in sorted(timeline):
            if frame > args.end+1e-6:
                break
            if previous is not None and not np.isclose(frame-previous,.1):
                for stream in streams.values(): stream.reset()
            records = timeline[frame]
            ids = sorted(records,key=str)
            x = torch.as_tensor(np.stack([records[v]['CSI_preprocessed'][-1] for v in ids]),
                                dtype=torch.float32,device=device)[:,None,:]
            predictions = {}
            for task,stream in streams.items():
                value = stream.step(ids,x)
                if task == 'beam':
                    value = value.topk(5,dim=-1,sorted=True).indices
                else:
                    scale,offset = models[task].params_norm
                    value = scale*(value-offset)
                predictions[task] = value.cpu().numpy()
            if frame >= args.start-1e-6:
                prepared[frame] = {}
                for j,v in enumerate(ids):
                    r = dict(records[v])
                    r['CSI_preprocessed'] = x[j].cpu().numpy()
                    r['shared_prediction'] = dict(source_frame=float(frame),target_frame=round(float(frame)+.1,7),
                        gain=predictions['desired_gain'][j],beam=predictions['beam'][j],
                        interference=predictions['interfering_gain'][j])
                    prepared[frame][v] = r
            previous = frame
    if len(prepared) != round((args.end-args.start)/.1)+1:
        raise ValueError('Requested cache interval is not fully present')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp = args.output.with_suffix(f'.{os.getpid()}.tmp')
    with temp.open('wb') as f: pickle.dump(prepared,f,protocol=4)
    os.replace(temp,args.output)
    atomic_json(manifest_path,dict(**expected,frames=len(prepared),cache_sha256=digest(args.output)))
    print('CACHE COMPLETE',args.output,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='phase',required=True)
    d=sub.add_parser('data'); d.add_argument('--source',type=Path,default=DEFAULT_SOURCE); d.add_argument('--output',type=Path,required=True)
    t=sub.add_parser('train'); t.add_argument('--data',type=Path,required=True); t.add_argument('--split-file',type=Path,default=DEFAULT_SPLIT)
    t.add_argument('--output',type=Path,required=True); t.add_argument('--device',default='cuda:0')
    t.add_argument('--stage1-epochs',type=int,default=100); t.add_argument('--stage2-epochs',type=int,default=100)
    t.add_argument('--batch-size',type=int,default=128); t.add_argument('--seed',type=int,default=20)
    t.add_argument('--max-samples',type=int,default=0); t.add_argument('--max-trajectories',type=int,default=0)
    t.add_argument('--cpu-threads',type=int,default=1); t.add_argument('--resume',action='store_true')
    t.add_argument('--stop-after-epochs',type=int,default=0,help='Operational limit for this invocation; resume preserves the full training schedule')
    a=sub.add_parser('assemble'); a.add_argument('--base-models',type=Path,default=DEFAULT_CHECKPOINT_ROOT)
    a.add_argument('--training',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('cache'); c.add_argument('--models',type=Path,required=True); c.add_argument('--source',type=Path,default=DEFAULT_TEST)
    c.add_argument('--output',type=Path,required=True); c.add_argument('--device',default='cuda:0'); c.add_argument('--cpu-threads',type=int,default=1)
    c.add_argument('--start',type=float,default=800.); c.add_argument('--end',type=float,default=830.)
    args=p.parse_args()
    if args.phase=='data': build(args.source,args.output,interference_label='beam-average')
    elif args.phase=='train':
        if min(args.stage1_epochs,args.stage2_epochs,args.batch_size)<1: p.error('Positive epoch/batch counts required')
        train(args)
    elif args.phase=='assemble': assemble(args)
    else: cache(args)


if __name__=='__main__': main()
