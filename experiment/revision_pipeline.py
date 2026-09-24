#!/usr/bin/env python3
"""Audited causal-prediction/directional-service grid; no historical reuse.

prepare freezes inputs; run is restartable at case boundaries; status is read
only. Oracle-CR-LB is exported from Oracle-MC's frozen P2 instances, power only.
"""
import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiment.revision_training import atomic_json, require_current_gain_convention

METHODS=('meet_cobra','oracle_mc','reactive_obra','wo_gap_ho','wo_pet_bf','wo_otr_ra','o_mappo','mts_report')
LEGACY_POLICY=ROOT/'experiment/results/o_mappo/final_load1/final_policy.pt'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def native(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,dict): return {str(k):native(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [native(v) for v in obj]
    return obj


def read(path): return json.loads(Path(path).read_text())
def items(text,kind=str): return [kind(v.strip()) for v in text.split(',') if v.strip()]
def key(method,rate,seed): return f'{method}_rate{rate}_seed{seed}'


def protocol(args):
    cache_meta=read(args.cache.with_suffix('.json'))
    if cache_meta.get('interference_label')!='beam-average' or digest(args.cache)!=cache_meta['cache_sha256']:
        raise ValueError('Unvalidated prediction cache')
    require_current_gain_convention(cache_meta)
    if cache_meta.get('smoke') and not args.allow_smoke:
        raise ValueError('Smoke-trained model is not allowed in a formal grid')
    methods=items(args.methods)
    if not methods or not set(methods)<=set(METHODS): raise ValueError('Unknown methods')
    if 'reactive_obra' in methods and args.reactive_input!='current':
        raise ValueError('Explicitly select --reactive-input current or exclude reactive_obra')
    if cache_meta['frames']<4: raise ValueError('Insufficient context/service frames')
    code=[Path(__file__),ROOT/'experiment/revision_training.py',ROOT/'experiment/pql_ba_experiment.py',
          ROOT/'experiment/o_mappo_shared_frontend.py',ROOT/'experiment/compare_stateful_prediction.py']
    code+=sorted((ROOT/'utils').glob('*.py'))
    return dict(version=1,cache=str(args.cache.resolve()),cache_sha256=digest(args.cache),
        cache_manifest=cache_meta,methods=methods,rates=items(args.rates,int),seeds=items(args.seeds,int),
        backend=args.backend,warmup_frames=2,frame_s=.1,ho_ms=10,top_k=5,
        environment={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','numba')},
        gap_iterations=2,gap_capacity_correction=True,gap_rb_usage_capped=True,
        o_mappo_beam_search=args.o_mappo_beam_search,
        service='actual directional beams, explicit orthogonal RB indices, per-RB service feeds queues',
        prediction_timing='record x predicts x+1; current BF/RA uses report from x-1',
        information=dict(meet_cobra='causal prediction reports and observed serving gains only',
            mts_report='causal shared reports; no private H in matching or RA interference',
            o_mappo='frozen original privileged-information actor; target optimizer uses the configured acquisition overhead; candidate gains retain original full-CSI maxima',
            reactive_obra='current perfect gain measurements; no NN; privileged measurement reference'),
        baseline_handover='Existing HO_EE_Greedy_offload (energy-cost ordering), retained for Reactive-OBRA and w/o GAP-HO; not an RSS-first implementation',
        oracle_cr_lb='continuous relaxation of the final P2 instance along Oracle-MC; conditional power reference, not global dynamic optimum; infeasible ceiling flagged',
        policy=str(LEGACY_POLICY),policy_sha256=digest(LEGACY_POLICY),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code})


def prepare(args):
    p=protocol(args)
    if not p['rates'] or min(p['rates'])<1 or max(p['rates'])>35 or not p['seeds']:
        raise ValueError('Invalid grid')
    dest=args.root/'protocol.json'
    if dest.exists():
        if read(dest)!=p: raise FileExistsError('Protocol changed; use a new root')
    else:
        if args.root.exists() and any(args.root.iterdir()):
            raise FileExistsError('Refusing a nonempty unversioned root')
        atomic_json(dest,p)
    print(json.dumps(dict(root=str(args.root),cases=len(p['methods'])*len(p['rates'])*len(p['seeds']),
                         smoke=p['cache_manifest']['smoke'],protocol_sha256=digest(dest)),indent=2))


def validate(root,device=None):
    p=read(root/'protocol.json')
    if device and device.split(':')[0]!=p['backend']: raise ValueError('Physical backend changed')
    for name,sha in p['code_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError(f'Frozen code changed: {name}')
    if digest(p['cache'])!=p['cache_sha256'] or digest(p['policy'])!=p['policy_sha256']:
        raise ValueError('Frozen input changed')
    return p,digest(root/'protocol.json')


def completed(root,name,sha):
    path=root/'runs'/f'{name}.json'
    if not path.exists(): return False
    row=read(path)
    if row['protocol_sha256']!=sha: raise ValueError(f'Wrong protocol: {name}')
    for directory,field,suffix in (('raw','raw_sha256','.npz'),('diagnostics','diagnostics_sha256','.json')):
        if digest(root/directory/f'{name}{suffix}')!=row[field]: raise ValueError(f'Corrupt result: {name}')
    return True


def extract(args,result,traffic,warmup):
    n=len(result.energy_record)
    frames,vehicles,queues=[],[],[]
    for fi,rows in result.queue_per_vehicle_record.items():
        for v in sorted(rows,key=str):
            frames.append(fi); vehicles.append(str(v)); queues.append(rows[v])
    q=np.asarray(queues); frames=np.asarray(frames); vehicles=np.asarray(vehicles)
    rates={str(v):r for v,r in traffic['rates'].items()}
    per_rate=np.array([rates[v] for v in vehicles])
    expected_u=np.array([np.mean(q[frames==i] > per_rate[frames==i,None]*args.lat_slot_ub*args.slot_len) for i in range(n)])
    expected_q=np.array([q[frames==i].mean() for i in range(n)])
    np.testing.assert_allclose(expected_u,result.violation_probability_record,atol=1e-12)
    np.testing.assert_allclose(expected_q,result.average_queue_record,rtol=1e-12,atol=1e-8)
    counts=np.array([np.bincount(list(result.association_record[i].values()),minlength=5) for i in range(n)])
    rb=result.rb_allocated_record
    caps=np.array([args.num_RB_macro]+[args.num_RB_micro]*4)
    powers=np.array([args.p_macro]+[args.p_micro]*4)
    if not np.isfinite(q).all() or (q<0).any() or (rb<0).any() or (rb>caps+1e-8).any():
        raise AssertionError('Invalid queue or RB count')
    np.testing.assert_allclose(rb@powers*.1,result.energy_record,atol=1e-8,rtol=1e-10)
    selected=slice(warmup,None)
    delay=q[frames>=warmup]/per_rate[frames>=warmup,None]*1000
    metrics=dict(power_w=float(np.mean(result.energy_record[selected])/.1),
        violation_percent=float(100*expected_u[selected].mean()),mean_proxy_ms=float(delay.mean()),
        p90_proxy_ms=float(np.percentile(delay,90)),p99_proxy_ms=float(np.percentile(delay,99)),
        macro_association_percent=float(100*counts[selected,0].sum()/counts[selected].sum()),
        pilots_per_vehicle_slot=float(result.pilot_record[selected].mean()))
    raw=dict(energy_j=result.energy_record,rb_per_bs=rb,queue_bits=q,queue_frame=frames,queue_vehicle=vehicles,
        violation_probability=expected_u,mean_queue_bits=expected_q,association_counts=counts,
        handover_count=result.handover_record,pilots=result.pilot_record)
    return metrics,raw


def case(args):
    import torch
    from experiment import o_mappo_shared_frontend as shared
    from utils.ho_utils import make_paired_traffic
    from utils.revision_meet_sim import run_revised_meet
    from utils import mts_report_sim as mts
    from utils.mts_report_bounded import estimate_bounded_report_load
    from utils.mts_gs_hbf import candidate_configs
    from utils.o_mappo import OMAPPPolicy
    from utils.o_mappo_sim import run_sim_o_mappo
    from utils import alg_utils as alg
    p,sha=validate(args.root,args.device)
    if args.method not in p['methods'] or args.rate not in p['rates'] or args.seed not in p['seeds']:
        raise ValueError('Case outside frozen grid')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    name=key(args.method,args.rate,args.seed)
    (args.root/'locks').mkdir(exist_ok=True)
    with (args.root/'locks'/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if completed(args.root,name,sha): print('SKIP',name,flush=True); return
        if shutil.disk_usage(args.root).free<2*2**30: raise RuntimeError('Less than 2 GiB free')
        torch.set_num_threads(1); shared.single_thread_solvers()
        from utils.compiled_matching import km_algorithm_compiled
        alg.km_algorithm=km_algorithm_compiled
        km_algorithm_compiled(np.zeros((2,2)))
        with Path(p['cache']).open('rb') as f: timeline=pickle.load(f)
        params=shared.paper_args(args.rate*1e6); params.device=torch.device('cpu')
        traffic=make_paired_traffic(params,timeline,args.seed)
        np.random.seed(args.seed)
        diagnostic=[]; service=[]; gap=[]; beam_search=[]
        def progress(n,total):
            atomic_json(args.root/'progress'/f'{name}.json',dict(frame=n,total=total,pid=os.getpid(),device=args.device))
            if n%25==0 or n==total: print('FRAME',n,total,flush=True)
        started=time.monotonic()
        if args.method=='mts_report':
            mts.estimate_report_load=estimate_bounded_report_load
            result=mts.run_sim_mts_report(params,shared.MICRO_BS_LOCATIONS,timeline,candidate_configs()['pressure_early'],
                traffic,seed=args.seed,physics_device=args.device,ho_interruption_ms=10,k=5,diagnostics=diagnostic,
                progress_callback=progress,directional_service=True,predicted_ra_interference=True,service_diagnostics=service)
        elif args.method=='o_mappo':
            result=run_sim_o_mappo(params,shared.MICRO_BS_LOCATIONS,timeline,OMAPPPolicy.load(p['policy']),
                seed=args.seed,prt=False,optimizer_solver='milp',traffic_trace=traffic,ho_interruption_ms=10,
                paired_fading_seed=args.seed,physics_device=args.device,progress_callback=progress,
                directional_service=True,service_diagnostics=service,
                beam_search_variant=p.get('o_mappo_beam_search','exhaustive'),
                beam_search_diagnostics=beam_search)
        else:
            result=run_revised_meet(params,shared.MICRO_BS_LOCATIONS,timeline,args.method,traffic,args.seed,args.device,
                diagnostics=diagnostic,service_diagnostics=service,gap_diagnostics=gap,progress_callback=progress)
        metrics,raw=extract(params,result,traffic,p['warmup_frames'])
        if args.method=='o_mappo':
            raw['acquisition_count']=result.full_sweep_record
            raw['local_search_count']=result.local_sweep_record
            raw['decision_count']=result.decision_record
            metrics['acquisition_count']=int(result.full_sweep_record[p['warmup_frames']:].sum())
            metrics['local_search_count']=int(result.local_sweep_record[p['warmup_frames']:].sum())
        if len(service)!=len(result.energy_record) or any(d['slots']!=params.slots_per_frame for d in service):
            raise AssertionError('Not every queue update used directional service')
        if args.method=='oracle_mc':
            raw['p2_relaxed_power_w']=np.r_[np.nan,[d['relaxed_p2_power_w'] for d in diagnostic[:-1]]]
            raw['p2_relaxation_feasible']=np.r_[False,[d['relaxation_feasible'] for d in diagnostic[:-1]]]
            metrics['oracle_cr_lb_power_reference_w']=float(raw['p2_relaxed_power_w'][2:].mean())
            metrics['oracle_cr_lb_feasible_fraction']=float(raw['p2_relaxation_feasible'][2:].mean())
        sums={field:sum(d[field] for d in service[2:]) for field in service[0] if field not in ('frame','slots')}
        comparison=dict(interference_ratio=sums['approximate_interference_sum']/sums['directional_interference_sum']
            if sums['directional_interference_sum'] else None,
            service_capacity_ratio=sums['approximate_service_bits']/sums['directional_service_bits']
            if sums['directional_service_bits'] else None)
        for folder in ('raw','diagnostics','runs'): (args.root/folder).mkdir(exist_ok=True)
        temp=args.root/'raw'/f'{name}.{os.getpid()}.tmp.npz'; output=args.root/'raw'/f'{name}.npz'
        np.savez_compressed(temp,**raw); os.replace(temp,output)
        dpath=args.root/'diagnostics'/f'{name}.json'
        atomic_json(dpath,native(dict(ho=diagnostic,gap=gap,service=service,comparison=comparison,
                                    beam_search=beam_search)))
        atomic_json(args.root/'runs'/f'{name}.json',dict(method=args.method,rate_mbps=args.rate,seed=args.seed,
            protocol_sha256=sha,traffic_sha256=traffic['sha256'],raw_sha256=digest(output),
            diagnostics_sha256=digest(dpath),metrics=metrics,comparison=comparison,device=args.device,
            frames=len(result.energy_record),elapsed_s=time.monotonic()-started))
        print('COMPLETE',name,json.dumps(metrics),flush=True)


def status(root):
    p=read(root/'protocol.json'); sha=digest(root/'protocol.json')
    expected=[key(m,r,s) for m in p['methods'] for r in p['rates'] for s in p['seeds']]
    done=[name for name in expected if (root/'runs'/f'{name}.json').exists() and read(root/'runs'/f'{name}.json')['protocol_sha256']==sha]
    # Cheap status is not a raw-data audit; run/summarize verify content hashes.
    missing=[name for name in expected if name not in done]
    progress={name:read(root/'progress'/f'{name}.json') for name in missing if (root/'progress'/f'{name}.json').exists()}
    return dict(root=str(root),completed=len(done),expected=len(expected),
        missing_count=len(missing),missing_preview=missing[:20],unfinished_progress=progress,
        launcher=read(root/'launch.json') if (root/'launch.json').exists() else None,
        queue=read(root/'queue_status.json') if (root/'queue_status.json').exists() else None)


def run(args):
    p,sha=validate(args.root)
    devices=items(args.devices)
    if not devices or any(d.split(':')[0]!=p['backend'] for d in devices): raise ValueError('Wrong devices')
    if args.detach:
        with (args.root/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(args.root.resolve()),
                 '--devices',args.devices]
            for option in ('methods','rates','seeds'):
                if getattr(args,option): cmd += ['--'+option,getattr(args,option)]
            with (args.root/'queue.log').open('a') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    start_new_session=True,env=dict(os.environ,PYTHONHASHSEED='0'))
            atomic_json(args.root/'launch.json',dict(pid=process.pid,command=cmd,launched_at=time.time()))
        print(json.dumps(read(args.root/'launch.json'),indent=2)); return
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Launcher received termination signal')
    signal.signal(signal.SIGTERM,interrupted)
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        methods=items(args.methods) if args.methods else p['methods']
        rates=items(args.rates,int) if args.rates else p['rates']
        seeds=items(args.seeds,int) if args.seeds else p['seeds']
        if not set(methods)<=set(p['methods']) or not set(rates)<=set(p['rates']) or not set(seeds)<=set(p['seeds']):
            raise ValueError('Selection outside protocol')
        pending=[(m,r,s) for m in methods for r in rates for s in seeds if not completed(args.root,key(m,r,s),sha)]
        atomic_json(args.root/'queue_status.json',dict(phase='running',pending=len(pending),pid=os.getpid()))
        (args.root/'logs').mkdir(exist_ok=True)
        buckets=[pending[i::len(devices)] for i in range(len(devices))]
        failures=[]; stop=threading.Event(); children=[]
        def worker(device,jobs):
            for m,r,s in jobs:
                if stop.is_set(): break
                name=key(m,r,s)
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'case','--root',str(args.root.resolve()),
                    '--method',m,'--rate',str(r),'--seed',str(s),'--device',device]
                with (args.root/'logs'/f'{name}.log').open('a') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,PYTHONHASHSEED='0'))
                    children.append(child)
                    try: code=child.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait(); code=-1
                if code:
                    failures.append(name); stop.set(); break
                print('DONE',name,flush=True)
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(devices))
        try:
            futures=[pool.submit(worker,d,b) for d,b in zip(devices,buckets)]
            for f in futures: f.result()
        except BaseException:
            stop.set()
            for child in children:
                if child.poll() is None: child.terminate()
            atomic_json(args.root/'queue_status.json',dict(phase='interrupted',pid=os.getpid()))
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        atomic_json(args.root/'queue_status.json',dict(phase='failed' if failures else 'selection_completed',failures=failures))
        if failures: raise RuntimeError('Failed cases: '+','.join(failures))
        print(json.dumps(status(args.root),indent=2))


def summarize(args):
    import csv
    from scipy.stats import t
    p,sha=validate(args.root)
    rows=[]; missing=[]
    for m in p['methods']:
        for r in p['rates']:
            for s in p['seeds']:
                name=key(m,r,s)
                if completed(args.root,name,sha): rows.append(read(args.root/'runs'/f'{name}.json'))
                else: missing.append(name)
    if missing and not args.allow_partial: raise RuntimeError(f'{len(missing)} cases missing; use --allow-partial only for diagnostics')
    summaries=[]
    for m in p['methods']:
        for r in p['rates']:
            group=[x for x in rows if x['method']==m and x['rate_mbps']==r]
            if not group: continue
            stats={}
            for name in group[0]['metrics']:
                a=np.array([x['metrics'][name] for x in group])
                sd=float(a.std(ddof=1)) if len(a)>1 else None
                stats[name]=dict(mean=float(a.mean()),sd=sd,ci95_halfwidth=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))) if sd is not None else None)
            comparisons={name:[x['comparison'][name] for x in group] for name in group[0]['comparison']}
            summaries.append(dict(method=m,rate_mbps=r,seeds=[x['seed'] for x in group],metrics=stats,comparison=comparisons))
    dest=args.root/'aggregate'; dest.mkdir(exist_ok=True)
    atomic_json(dest/'summary.json',dict(protocol_sha256=sha,groups=summaries,missing=missing,
        smoke=p['cache_manifest']['smoke'],confidence_scope='Per-seed repetitions in a fixed scenario, not independent deployments'))
    with (dest/'curves.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['method','rate_mbps','seeds','power_w','violation_percent','p90_proxy_ms','p99_proxy_ms','macro_association_percent'])
        for g in summaries:
            w.writerow([g['method'],g['rate_mbps'],len(g['seeds']),*[g['metrics'][k]['mean'] for k in
                ('power_w','violation_percent','p90_proxy_ms','p99_proxy_ms','macro_association_percent')]])
    references=[dict(rate_mbps=g['rate_mbps'],seeds=g['seeds'],
        power=g['metrics']['oracle_cr_lb_power_reference_w'],feasible_fraction=g['metrics']['oracle_cr_lb_feasible_fraction'])
        for g in summaries if g['method']=='oracle_mc']
    atomic_json(dest/'oracle_cr_lb_power_only.json',dict(definition=p['oracle_cr_lb'],groups=references))
    print('SUMMARY',dest,'missing',len(missing),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='phase',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--root',type=Path,required=True); a.add_argument('--cache',type=Path,required=True)
    a.add_argument('--methods',default=','.join(METHODS)); a.add_argument('--rates',default=','.join(map(str,range(1,36,2))))
    a.add_argument('--seeds',default='1,2,3,4,5'); a.add_argument('--reactive-input',choices=('pending','current'),default='pending')
    a.add_argument('--backend',choices=('cpu','cuda'),default='cuda'); a.add_argument('--allow-smoke',action='store_true')
    a.add_argument('--o-mappo-beam-search',choices=('exhaustive','hierarchical32'),default='exhaustive')
    a=sub.add_parser('case'); a.add_argument('--root',type=Path,required=True); a.add_argument('--method',choices=METHODS,required=True)
    a.add_argument('--rate',type=int,required=True); a.add_argument('--seed',type=int,required=True); a.add_argument('--device',default='cuda:0')
    a=sub.add_parser('run'); a.add_argument('--root',type=Path,required=True); a.add_argument('--devices',default='cuda:0')
    a.add_argument('--methods'); a.add_argument('--rates'); a.add_argument('--seeds'); a.add_argument('--detach',action='store_true')
    a=sub.add_parser('status'); a.add_argument('--root',type=Path,required=True)
    a=sub.add_parser('summarize'); a.add_argument('--root',type=Path,required=True); a.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    if args.phase=='prepare': prepare(args)
    elif args.phase=='case': case(args)
    elif args.phase=='run': run(args)
    elif args.phase=='status': print(json.dumps(status(args.root),indent=2))
    else: summarize(args)


if __name__=='__main__': main()
