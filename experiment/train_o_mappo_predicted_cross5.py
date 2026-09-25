#!/usr/bin/env python3
"""Restartable prediction-only cross5 preparation, PPO, exact grid and report."""
import argparse
import collections
import concurrent.futures
import dataclasses
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiment import o_mappo_predicted_cross5 as env
from experiment import train_o_mappo_hierarchical32 as old
from experiment.revision_pipeline import atomic_json, digest, extract, native
from experiment.benchmark_stateful_nn_overhead import load_selected_models
from experiment.compare_stateful_prediction import StatefulPredictor
from experiment.prepare_stateful_trajectories import frame_values
from utils.data_utils import preprocess_input_np
from utils import o_mappo as om

DEFAULT = ROOT/'experiment/results/o_mappo_predicted_cross5_20260925'
DATA = ROOT/'experiment/results/o_mappo_h32_retrained_20260924_v4'
MODELS = ROOT/'experiment/results/revision_directional_20260922/models'
TEST = ROOT/'experiment/results/revision_directional_20260922/test_predictions.pkl'
INITIAL = ROOT/'experiment/results/o_mappo_eall_training_20260924/training/E_all/seed11/best_positive.pt'
RATES = list(range(1, 36, 2))
SEEDS = [11, 22, 33]
WORKER = None


def sources():
    return [Path(__file__), ROOT/'experiment/o_mappo_predicted_cross5.py',
        ROOT/'experiment/o_mappo_slot_tracking.py', ROOT/'experiment/o_mappo_target_check.py',
        ROOT/'experiment/o_mappo_optimizer_information.py',
        ROOT/'experiment/train_o_mappo_hierarchical32.py', ROOT/'experiment/prepare_stateful_trajectories.py',
        ROOT/'experiment/compare_stateful_prediction.py', ROOT/'experiment/revision_pipeline.py',
        *sorted((ROOT/'utils').glob('*.py'))]


def prepare(root, device):
    root.mkdir(parents=True, exist_ok=True)
    folder = root/'data'; folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    old.initialize_worker(str(DATA))
    channels, idx = old.DATA
    models, inventory = load_selected_models(MODELS)
    source = dict(models=inventory, arrays=old.read(DATA/'arrays_complete.json'),
        prediction_noise_seed=20260917, intervals=dict(train=[200,650], validation=[700,710],
        selection=[710,720], test=[800,830]), data_root=str(DATA),
        alignment='report at x predicts x+0.1; previous report supplies current RA',
        source_code={str(p.relative_to(ROOT)):digest(p) for p in [
            ROOT/'experiment/prepare_stateful_trajectories.py',
            ROOT/'experiment/compare_stateful_prediction.py', ROOT/'utils/NN_utils.py',
            ROOT/'utils/data_utils.py']})
    if (folder/'complete.json').exists():
        p = old.read(folder/'complete.json')
        if p['inputs'] != native(source): raise ValueError('Preparation inputs changed')
        assert p['report_sha256'] == digest(folder/'reports.npy')
        print('PREPARATION ALREADY COMPLETE', flush=True); return
    for model in models.values(): model.to(device).eval()
    streams = {name: StatefulPredictor(model) for name, model in models.items()}
    reports = np.lib.format.open_memmap(folder/'reports.tmp.npy', mode='w+', dtype=np.float32,
                                        shape=(len(channels), 2, 4))
    rng = np.random.default_rng(20260917)
    started = time.monotonic()
    with torch.inference_mode():
        for fi, frame in enumerate(idx['frames']):
            lo, hi = idx['offsets'][fi:fi+2]
            ids = list(idx['names'][lo:hi])
            order = sorted(range(len(ids)), key=lambda j: str(ids[j]))
            sorted_ids = [ids[j] for j in order]
            clean, *_ = frame_values([dict(h=channels[lo+j]) for j in order], interference_label='beam-average')
            noise = (rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))*np.sqrt(1e-14/2)
            x = preprocess_input_np((clean+noise).astype(np.complex64)).astype(np.float32)
            x = torch.as_tensor(x[:, None], device=device)
            for name, column in [('desired_gain',0), ('interfering_gain',1)]:
                value = streams[name].step(sorted_ids, x)
                scale, offset = models[name].params_norm
                reports[lo+np.asarray(order), column] = (scale*(value-offset)).cpu().numpy()
            if fi % 500 == 0:
                print('PREPARE', fi, len(idx['frames']), round(time.monotonic()-started,1), flush=True)
    reports.flush(); del reports
    os.replace(folder/'reports.tmp.npy', folder/'reports.npy')
    atomic_json(folder/'complete.json', dict(inputs=source, report_sha256=digest(folder/'reports.npy')))
    print('PREPARED', time.monotonic()-started, flush=True)


def initialize(root, device):
    global WORKER
    torch.set_num_threads(1); old.single_thread_solvers()
    old.initialize_worker(str(DATA))
    reports = np.load(Path(root)/'data/reports.npy', mmap_mode='r')
    WORKER = (Path(root), device, reports)


def timeline_slice(start, seconds):
    _, _, reports = WORKER
    channels, idx = old.DATA
    selected = np.flatnonzero((idx['frames'] >= start-1e-7) & (idx['frames'] <= start+seconds+1e-7))
    if len(selected) != round(seconds*10)+1: raise ValueError('Non-contiguous training segment')
    out = collections.OrderedDict()
    for i in selected:
        frame = float(idx['frames'][i]); lo, hi = idx['offsets'][i:i+2]
        out[frame] = {idx['names'][j]: dict(h=channels[j], pos=idx['positions'][j],
            angle=idx['angles'][j], v=idx['speeds'][j], shared_prediction=dict(
                gain=reports[j,0], interference=reports[j,1], source_frame=frame,
                target_frame=round(frame+.1,7))) for j in range(lo,hi)}
    return out


def rollout(job):
    checkpoint, rate, start, seconds, seed, learn = job
    policy = om.OMAPPPolicy.load(str(checkpoint), seed=seed)
    timeline = timeline_slice(start, seconds)
    result, traffic, memory, diagnostic = env.simulate(timeline, policy, rate, seed,
                                                       WORKER[1], learn=learn)
    metrics, _ = extract(old.paper_args(rate*1e6), result, traffic, min(2,len(timeline)-2))
    metrics.update(data_rate_mbps=rate, mean_event_reward=diagnostic['mean_event_reward'],
        average_system_power_w=metrics['power_w'], queue_violation_percent=metrics['violation_percent'],
        optimizer_failures=int(result.optimizer_failure_record.sum()),
        trigger_ratio=float(result.trigger_record.sum()/max(result.decision_record.sum(),1)))
    for transition in memory.transitions:
        transition.vehicle = (rate, start, seed, transition.vehicle)
    return metrics, memory.transitions if learn else diagnostic['probes']


def init_policy(seed):
    reference = om.OMAPPPolicy.load(str(INITIAL))
    cfg = env.configuration(reference.config)
    cfg.actor_learning_rate = cfg.critic_learning_rate = 1e-4
    policy = om.OMAPPPolicy(cfg, seed=seed)
    policy.actor.load_state_dict(reference.actor.state_dict())
    policy.critic.load_state_dict(reference.critic.state_dict())
    assert cfg.batch_size == 256 and cfg.ppo_epochs == 4 and cfg.actor_hidden_sizes == (64,64)
    return policy


def freeze(root, rounds, seconds):
    p = dict(training_seeds=SEEDS, rates=RATES, rounds=rounds, rollout_seconds=seconds,
        initial_checkpoint=str(INITIAL), initial_sha256=digest(INITIAL),
        initialization='warm-start identical actor and critic; fresh optimizer per training seed',
        data_sha256=digest(root/'data/complete.json'), test_sha256=digest(TEST),
        train_interval=[200,650], validation_interval=[700,710], selection_interval=[710,720],
        test_interval=[800,830], test_seeds=[1,2,3], warmup_frames=2,
        reward=dataclasses.asdict(om.o_mappo_reward_presets()['qos_energy020_load1']),
        batch_size=256, ppo_epochs=4, learning_rate='cosine 1e-4 to 1e-5 by round120',
        training='exact 100-slot directional service, same environment as evaluation',
        selection='positive-round best balanced all-load validation cost per seed, then separate exact all-load selection',
        actor='31->64->64->2; forecast SINR/INR/load; same state scaling',
        no_hidden_csi='actor, critic and optimizer accept whitelisted prediction/public records only',
        bf='HO32 then slot cross5; no free nominal search; predictions not used to pick beams',
        ra='measured serving gain, previous causal interfering-gain report, bounded report-derived occupancy',
        code={str(p.relative_to(ROOT)):digest(p) for p in sources()})
    if (root/'protocol.json').exists() and old.read(root/'protocol.json') != native(p):
        raise ValueError('Frozen experiment changed; choose a new root')
    atomic_json(root/'protocol.json', p)
    return p


def train(root, devices, rounds, seconds):
    protocol = freeze(root, rounds, seconds)
    policies, histories, best, probes, last_probs = {}, {}, {}, {}, {}
    progress = root/'training_progress.json'
    resume = old.read(progress)['round'] if progress.exists() else -1
    for seed in SEEDS:
        folder = root/'training'/f'seed{seed}'; folder.mkdir(parents=True,exist_ok=True)
        if resume >= 0:
            policies[seed] = om.OMAPPPolicy.load(str(folder/f'round{resume:04d}.pt'), seed=seed, load_optimizers=True)
            histories[seed] = [h for h in old.read(folder/'validation_history.json') if h['round'] <= resume]
            positive = [h['score'] for h in histories[seed] if h['round'] > 0]
            best[seed] = min(positive, default=float('inf'))
            probes[seed] = np.load(folder/'probes.npy')
            with torch.no_grad(): last_probs[seed] = policies[seed].actor(torch.from_numpy(probes[seed])).softmax(-1).numpy()
        else:
            policies[seed] = init_policy(seed); histories[seed] = []; best[seed] = float('inf')
            old.save_policy(policies[seed], folder/'round0000.pt')
    pools = [concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
        initializer=initialize, initargs=(str(root),d)) for d in devices]
    def dispatch(jobs):
        pending = [pools[j % len(pools)].submit(rollout, job) for j,job in enumerate(jobs)]
        return [f.result() for f in pending]
    started = time.monotonic()
    try:
        reference_path = root/'validation_reference.json'
        if reference_path.exists(): reference = np.array(old.read(reference_path)['costs'])
        else:
            path = root/'training/seed11/round0000.pt'
            outputs = dispatch([(path,r,700,10,9900+r,False) for r in RATES])
            reference = np.maximum([x[0]['violation_percent']+.02*x[0]['power_w'] for x in outputs],1e-3)
            atomic_json(reference_path,dict(costs=reference.tolist(),rows=[x[0] for x in outputs]))
        for rn in range(resume+1, rounds+1):
            if rn:
                jobs = []
                lr = float(1e-5+(1e-4-1e-5)*.5*(1+np.cos(np.pi*min(rn/120,1))))
                for seed in SEEDS:
                    folder = root/'training'/f'seed{seed}'; policy = policies[seed]
                    for optimizer in (policy.actor_optimizer, policy.critic_optimizer):
                        for group in optimizer.param_groups: group['lr'] = lr
                    old.save_policy(policy, folder/'rollout.pt')
                    for rate in RATES:
                        rng = np.random.default_rng(np.random.SeedSequence([seed,rn,rate]))
                        start = float(rng.integers(2000, int((650-seconds)*10)))/10
                        jobs.append((folder/'rollout.pt',rate,start,seconds,seed*100000+rn*100+rate,True))
                outputs = dispatch(jobs)
                for j, seed in enumerate(SEEDS):
                    group = outputs[j*len(RATES):(j+1)*len(RATES)]
                    memory = om.OMAPPOMemory()
                    for _, transitions in group: memory.transitions.extend(transitions)
                    update = policies[seed].update(memory)
                    row = dict(round=rn,seed=seed,learning_rate=lr,elapsed_s=time.monotonic()-started,
                        update=update,loads=[x[0] for x in group])
                    atomic_json(root/'training'/f'seed{seed}'/'updates'/f'{rn:04d}.json',row)
                    print('TRAIN', seed,rn,'samples',len(memory),'reward',np.mean([x[0]['mean_event_reward'] for x in group]),flush=True)
            if rn % 10 == 0 or rn == rounds:
                jobs=[]
                for seed in SEEDS:
                    path=root/'training'/f'seed{seed}'/f'round{rn:04d}.pt'
                    old.save_policy(policies[seed],path)
                    jobs.extend((path,r,700,10,9900+r,False) for r in RATES)
                outputs=dispatch(jobs)
                for j,seed in enumerate(SEEDS):
                    folder=root/'training'/f'seed{seed}'; group=outputs[j*len(RATES):(j+1)*len(RATES)]
                    rows=[x[0] for x in group]; value,costs=old.score(rows,reference)
                    if seed not in probes:
                        probes[seed]=np.concatenate([x[1] for x in group]).astype(np.float32)
                        np.save(folder/'probes.npy',probes[seed])
                    with torch.no_grad(): probs=policies[seed].actor(torch.from_numpy(probes[seed])).softmax(-1).numpy()
                    prior=last_probs.get(seed,probs); last_probs[seed]=probs
                    h=dict(round=rn,score=value,costs=costs.tolist(),rows=rows,
                        probe_flip_fraction=float(np.mean(probs.argmax(-1)!=prior.argmax(-1))),
                        probe_probability_change=float(np.abs(probs-prior).mean()))
                    histories[seed].append(h)
                    if rn>0 and value<best[seed]:
                        best[seed]=value; old.save_policy(policies[seed],folder/'best_positive.pt')
                        atomic_json(folder/'best_positive.json',h)
                    atomic_json(folder/'validation_history.json',histories[seed])
                    atomic_json(folder/'convergence.json',old.convergence(histories[seed]))
                    print('VALIDATE',seed,rn,'score',value,'stable',old.convergence(histories[seed])['passed'],flush=True)
                atomic_json(progress,dict(round=rn,elapsed_s=time.monotonic()-started))
        atomic_json(root/'training_complete.json',dict(rounds=rounds,seeds={str(s):old.convergence(histories[s]) for s in SEEDS}))
    finally:
        for pool in pools: pool.shutdown()


def exact_case(root, label, device):
    protocol=old.read(root/'protocol.json')
    for name,h in protocol['code'].items():
        if digest(ROOT/name)!=h:raise ValueError('Frozen evaluation code changed: '+name)
    for name,h in protocol['policies'].items():
        if digest(name)!=h:raise ValueError('Frozen policy changed: '+name)
    spec=protocol['jobs'][label]
    if spec['source']=='test':
        with TEST.open('rb') as stream: timeline=pickle.load(stream)
    else:
        initialize(root.parent,device); timeline=timeline_slice(710,10)
    policy=om.OMAPPPolicy.load(spec['policy'],seed=spec['seed'])
    torch.set_num_threads(1); old.single_thread_solvers()
    started=time.monotonic()
    result,traffic,_,diagnostic=env.simulate(timeline,policy,spec['rate'],spec['seed'],device,
        progress=lambda n,t: atomic_json(root/'progress'/f'{label}.json',dict(frame=n,total=t,pid=os.getpid())))
    metrics,raw=extract(old.paper_args(spec['rate']*1e6),result,traffic,2)
    metrics.update(handovers=int(result.handover_record[2:].sum()),
        optimizer_failures=int(result.optimizer_failure_record.sum()),
        trigger_ratio=float(result.trigger_record.sum()/max(result.decision_record.sum(),1)))
    folder=root/'runs';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{label}.npz',**raw)
    diagnostic.pop('probes')
    atomic_json(folder/f'{label}_diagnostics.json',diagnostic)
    atomic_json(folder/f'{label}.json',dict(**spec,metrics=metrics,frames=len(result.energy_record),
        elapsed_s=time.monotonic()-started,traffic_sha256=traffic['sha256'],
        policy_sha256=digest(spec['policy']),protocol_sha256=digest(root/'protocol.json'),
        raw_sha256=digest(folder/f'{label}.npz')))


def grid(root, policies, rates, seeds, source, devices):
    root.mkdir(parents=True,exist_ok=True)
    jobs={f'{label}_rate{r}_seed{s}':dict(label=label,policy=str(path),rate=r,seed=s,source=source)
          for label,path in policies.items() for r in rates for s in seeds}
    p=dict(jobs=jobs,code={str(p.relative_to(ROOT)):digest(p) for p in sources()},
           policies={str(p):digest(p) for p in policies.values()})
    if (root/'protocol.json').exists() and old.read(root/'protocol.json')!=p: raise ValueError('Grid changed')
    atomic_json(root/'protocol.json',p)
    def worker(device, labels):
        for label in labels:
            output=root/'runs'/f'{label}.json'
            if output.exists():
                r=old.read(output)
                assert r['protocol_sha256']==digest(root/'protocol.json')
                assert r['raw_sha256']==digest(output.with_suffix('.npz'))
                continue
            (root/'logs').mkdir(exist_ok=True)
            with (root/'logs'/f'{label}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'case','--root',str(root),
                    '--label',label,'--device',device],check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            print('COMPLETE',source,label,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(worker,d,list(jobs)[j::len(devices)]) for j,d in enumerate(devices)]
        for future in futures: future.result()
    rows=[old.read(root/'runs'/f'{label}.json') for label in jobs]
    atomic_json(root/'summary.json',dict(cases=len(rows),runs=rows))
    return rows


def run(args):
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    devices=args.devices.split(',')
    prepare(root,devices[0])
    train(root,devices,args.rounds,args.rollout_seconds)
    candidates={f'seed{s}':root/'training'/f'seed{s}'/'best_positive.pt' for s in SEEDS}
    rows=grid(root/'selection',candidates,RATES,[101],'selection',devices)
    ranking=[]
    # Normalize using the no-training prediction-input validation control.
    reference=np.asarray(old.read(root/'validation_reference.json')['costs'])
    for label,path in candidates.items():
        selected=sorted([r for r in rows if r['label']==label],key=lambda r:r['rate'])
        costs=np.array([r['metrics']['violation_percent']+.02*r['metrics']['power_w'] for r in selected])
        ratios=costs/reference
        ranking.append(dict(label=label,policy=str(path),score=float(.5*(ratios.mean()+ratios.max()))))
    ranking.sort(key=lambda r:(r['score'],r['label']))
    atomic_json(root/'selection.json',dict(ranking=ranking,selected=ranking[0]))
    grid(root/'test',{'predicted_cross5':Path(ranking[0]['policy'])},RATES,[1,2,3],'test',devices)
    atomic_json(root/'complete.json',dict(selection=ranking[0],test_cases=54,finished_unix=time.time()))
    print('FULL EXPERIMENT COMPLETE',root,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','run','case','smoke','launch'])
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--devices',default=','.join(['cuda:'+str(j) for j in range(7)]*2))
    p.add_argument('--rounds',type=int,default=160)
    p.add_argument('--rollout-seconds',type=float,default=5.)
    p.add_argument('--label')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root,a.device)
    elif a.command=='train':train(a.root,a.devices.split(','),a.rounds,a.rollout_seconds)
    elif a.command=='case':exact_case(a.root,a.label,a.device)
    elif a.command=='run':run(a)
    elif a.command=='launch':
        a.root.mkdir(parents=True,exist_ok=True)
        argv=[sys.executable,'-u',str(Path(__file__)),'run','--root',str(a.root),
            '--devices',a.devices,'--rounds',str(a.rounds),'--rollout-seconds',str(a.rollout_seconds)]
        with (a.root/'pipeline.log').open('a') as log:
            proc=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,PYTHONHASHSEED='0'))
        atomic_json(a.root/'launch.json',dict(pid=proc.pid,argv=argv,started_unix=time.time()))
        print('LAUNCHED',proc.pid,a.root,flush=True)
    else:
        initialize(a.root,a.device); policy=init_policy(11)
        start=time.monotonic()
        result,traffic,memory,d=env.simulate(timeline_slice(710,a.rollout_seconds),policy,15,101,a.device,learn=True)
        print('SMOKE',extract(old.paper_args(15e6),result,traffic,2)[0],len(memory),time.monotonic()-start,flush=True)
        print('PPO',policy.update(memory),flush=True)


if __name__=='__main__':main()
