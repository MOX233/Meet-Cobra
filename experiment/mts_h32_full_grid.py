#!/usr/bin/env python3
"""Frozen H32-cross5 MTS: all paper loads and three paired traffic seeds."""
import argparse
import concurrent.futures
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiment import mts_hierarchical_tracking as pilot
from experiment import revision_pipeline as grid

DEFAULT=ROOT/'experiment/results/mts_h32_full_grid_20260925'
RATES=list(range(1,36,2))
SEEDS=[1,2,3]
VARIANT='hier32_cross5'


def key(rate,seed): return f'mts_h32_cross5_rate{rate}_seed{seed}'


def verify(root):
    p=grid.read(root/'protocol.json')
    for name,checksum in p['code'].items(): assert grid.digest(ROOT/name)==checksum,name
    assert grid.digest(p['cache'])==p['cache_sha256']
    return p


def completed(root,name):
    path=root/'runs'/f'{name}.json'
    if not path.exists(): return False
    r=grid.read(path)
    assert r['protocol_sha256']==grid.digest(root/'protocol.json')
    for folder,field,suffix in [('raw','raw_sha256','.npz'),('diagnostics','diagnostics_sha256','.json')]:
        assert grid.digest(root/folder/f'{name}{suffix}')==r[field]
    return True


def prepare(root):
    source=grid.read(pilot.DEFAULT/'protocol.json')
    assert grid.read(pilot.DEFAULT/'audit.json')['passed']
    for name,checksum in source['code'].items(): assert grid.digest(ROOT/name)==checksum,name
    reference=grid.read(pilot.REFERENCE/'protocol.json')
    assert reference['rates']==RATES and reference['seeds']==SEEDS
    p=dict(source,source_protocol_sha256=grid.digest(pilot.DEFAULT/'protocol.json'),
           variant=VARIANT,variants=[VARIANT],rates=RATES,seeds=SEEDS,seconds=30)
    p['code']=dict(source['code'],**{str(Path(__file__).relative_to(ROOT)):grid.digest(Path(__file__))})
    dest=root/'protocol.json'
    if dest.exists(): assert grid.read(dest)==p,'Frozen protocol changed'
    else: grid.atomic_json(dest,p)
    reused=[]
    for rate in pilot.RATES:
        source_name=f'{VARIANT}_rate{rate}_seed1'; name=key(rate,1)
        source_path=pilot.DEFAULT/'runs'/f'{source_name}.json'
        row=grid.read(source_path)
        assert row['protocol_sha256']==p['source_protocol_sha256'] and row['frames']==300
        assert row['variant']==VARIANT
        ref=grid.read(pilot.REFERENCE/'runs'/f'mts_report_rate{rate}_seed1.json')
        assert row['traffic_sha256']==ref['traffic_sha256']
        for folder,field,suffix in [('raw','raw_sha256','.npz'),('diagnostics','diagnostics_sha256','.json')]:
            src=pilot.DEFAULT/folder/f'{source_name}{suffix}'; target=root/folder/f'{name}{suffix}'
            assert grid.digest(src)==row[field]
            target.parent.mkdir(exist_ok=True)
            if target.exists(): assert grid.digest(target)==row[field]
            else:
                temp=target.with_suffix(target.suffix+'.copying')
                shutil.copyfile(src,temp); os.replace(temp,target)
        row=dict(row,method='mts_report',protocol_sha256=grid.digest(dest),
                 reused_from=str(source_path),source_row_sha256=grid.digest(source_path))
        target=root/'runs'/f'{name}.json'
        if target.exists(): assert grid.read(target)==row
        else: grid.atomic_json(target,row)
        reused.append(dict(rate=rate,seed=1,source=str(source_path),sha256=grid.digest(source_path)))
    grid.atomic_json(root/'reuse.json',dict(cases=reused))
    print('Prepared 54 cases; reused 6 validated pilot cases',flush=True)


def run_case(root,rate,seed,device):
    p=verify(root)
    assert rate in RATES and seed in SEEDS
    name=key(rate,seed)
    (root/'locks').mkdir(exist_ok=True)
    with (root/'locks'/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if completed(root,name): print('SKIP',name,flush=True); return
        torch.set_num_threads(1); pilot.shared.single_thread_solvers()
        from utils.compiled_matching import km_algorithm_compiled
        pilot.alg.km_algorithm=km_algorithm_compiled
        km_algorithm_compiled(np.zeros((2,2)))
        with Path(p['cache']).open('rb') as f: timeline=pickle.load(f)
        args=pilot.shared.paper_args(rate*1e6); args.device=torch.device('cpu')
        traffic=pilot.make_paired_traffic(args,timeline,seed)
        ref=grid.read(pilot.REFERENCE/'runs'/f'mts_report_rate{rate}_seed{seed}.json')
        assert traffic['sha256']==ref['traffic_sha256']
        np.random.seed(seed)
        adapter=pilot.BeamAdapter(VARIANT); service=[]
        def progress(n,total):
            grid.atomic_json(root/'progress'/f'{name}.json',dict(frame=n,total=total,pid=os.getpid(),device=device))
            if n%50==0 or n==total: print('FRAME',n,total,flush=True)
        start=time.monotonic()
        with ExitStack() as stack:
            adapter.activate(stack)
            result=pilot.sim.run_sim_mts_report(args,pilot.shared.MICRO_BS_LOCATIONS,timeline,
                pilot.candidate_configs()['pressure_early'],traffic,seed=seed,physics_device=device,
                ho_interruption_ms=10,k=5,progress_callback=progress,directional_service=True,
                predicted_ra_interference=True,service_diagnostics=service)
        assert len(service)==len(adapter.frames)==300
        assert all(d['slots']==100 for d in service)
        metrics,raw=grid.extract(args,result,traffic,2)
        totals={k:sum(d[k] for d in adapter.frames[2:]) for k in adapter.frames[0] if k!='frame_probing_fraction'}
        assert totals['one_probe_slots']==0
        assert totals['probe_count']==32*totals['acquisition_count']+5*totals['five_probe_slots']
        metrics.update(micro_probes_per_active_slot=totals['probe_count']/totals['micro_active_slots'],
            mean_serving_gain_db=totals['serving_gain_db_sum']/totals['micro_active_slots'],
            within_frame_beam_changes=totals['beam_changes'])
        for folder in ('raw','diagnostics','runs'): (root/folder).mkdir(exist_ok=True)
        rawpath=root/'raw'/f'{name}.npz'; temp=rawpath.with_suffix('.tmp.npz')
        np.savez_compressed(temp,**raw); os.replace(temp,rawpath)
        dp=root/'diagnostics'/f'{name}.json'
        grid.atomic_json(dp,dict(beam=adapter.frames,service=service))
        grid.atomic_json(root/'runs'/f'{name}.json',dict(method='mts_report',variant=VARIANT,
            rate_mbps=rate,seed=seed,frames=300,device=device,elapsed_s=time.monotonic()-start,
            protocol_sha256=grid.digest(root/'protocol.json'),traffic_sha256=traffic['sha256'],
            raw_sha256=grid.digest(rawpath),diagnostics_sha256=grid.digest(dp),metrics=metrics))
        print('COMPLETE',name,metrics,flush=True)


def audit(root):
    p=verify(root); rows=[]; provenance=[]
    for rate in RATES:
        for seed in SEEDS:
            name=key(rate,seed); path=root/'runs'/f'{name}.json'
            assert completed(root,name),name
            row=grid.read(path)
            assert row['frames']==300 and row['variant']==VARIANT
            ref=grid.read(pilot.REFERENCE/'runs'/f'mts_report_rate{rate}_seed{seed}.json')
            assert row['traffic_sha256']==ref['traffic_sha256']
            with np.load(root/'raw'/f'{name}.npz') as raw:
                q,f,rb=raw['queue_bits'],raw['queue_frame'],raw['rb_per_bs']
                assert np.isfinite(q).all() and (q>=0).all()
                assert ((rb>=0)&(rb<=np.array([133,66,66,66,66])+1e-8)).all()
                np.testing.assert_allclose(rb@np.array([1,.2,.2,.2,.2])*.1,raw['energy_j'],rtol=1e-10,atol=1e-8)
                u=np.array([np.mean(q[f==i]>rate*1e6*.02) for i in range(300)])
                np.testing.assert_allclose(u,raw['violation_probability'],atol=1e-12)
                delay=q[f>=2]/(rate*1e6)*1000; counts=raw['association_counts'][2:]
                actual=dict(power_w=raw['energy_j'][2:].mean()/.1,violation_percent=100*u[2:].mean(),
                    mean_proxy_ms=delay.mean(),p90_proxy_ms=np.percentile(delay,90),p99_proxy_ms=np.percentile(delay,99),
                    macro_association_percent=100*counts[:,0].sum()/counts.sum(),pilots_per_vehicle_slot=raw['pilots'][2:].mean())
                for k,v in actual.items(): np.testing.assert_allclose(v,row['metrics'][k],rtol=1e-10,atol=1e-8)
            d=grid.read(root/'diagnostics'/f'{name}.json')
            assert len(d['service'])==len(d['beam'])==300 and all(x['slots']==100 for x in d['service'])
            for x in d['beam']:
                assert x['one_probe_slots']==0
                assert x['micro_active_slots']==x['acquisition_count']+x['five_probe_slots']
                assert x['probe_count']==32*x['acquisition_count']+5*x['five_probe_slots']
            rows.append(row)
            provenance.append(dict(file=str(path),sha256=grid.digest(path),raw_sha256=row['raw_sha256'],
                                   diagnostics_sha256=row['diagnostics_sha256']))
            if len(rows)%9==0: print('AUDIT',len(rows),'/54',flush=True)
    groups=[]
    for rate in RATES:
        cases=[r for r in rows if r['rate_mbps']==rate]
        metrics={}
        for k in cases[0]['metrics']:
            values=[r['metrics'][k] for r in cases]
            metrics[k]=dict(mean=float(np.mean(values)),minimum=min(values),maximum=max(values),per_seed=values)
        groups.append(dict(rate_mbps=rate,metrics=metrics))
    summary=dict(cases=54,rates=RATES,seeds=SEEDS,seconds=30,warmup_frames=2,variant=VARIANT,
        cache_sha256=p['cache_sha256'],protocol_sha256=grid.digest(root/'protocol.json'),
        raw_metrics_recomputed=True,paired_traffic_verified=True,groups=groups,provenance=provenance,runs=rows)
    grid.atomic_json(root/'summary.json',summary)
    return summary


def status(root):
    done=[]; running=[]
    for r in RATES:
        for s in SEEDS:
            name=key(r,s)
            if (root/'runs'/f'{name}.json').exists(): done.append(name)
            elif (root/'progress'/f'{name}.json').exists(): running.append(dict(name=name,**grid.read(root/'progress'/f'{name}.json')))
    print(dict(completed=len(done),expected=54,unfinished_progress=running),flush=True)


def run(root,devices):
    prepare(root)
    (root/'logs').mkdir(exist_ok=True)
    jobs=[(r,s) for r in RATES for s in SEEDS if not completed(root,key(r,s))]
    buckets=[jobs[i::len(devices)] for i in range(len(devices))]
    def worker(device,jobs):
        for rate,seed in jobs:
            name=key(rate,seed)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'case','--root',str(root),
                 '--rate',str(rate),'--seed',str(seed),'--device',device]
            with (root/'logs'/f'{name}.log').open('a') as f:
                code=subprocess.call(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            print('JOB',name,'EXIT',code,flush=True)
            if code: raise RuntimeError(name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker,devices,buckets))
    audit(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','case','run','audit','status'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--rate',type=int,choices=RATES)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    args=parser.parse_args()
    if args.command=='case': run_case(args.root,args.rate,args.seed,args.device)
    elif args.command=='audit': audit(args.root)
    elif args.command=='prepare': prepare(args.root)
    elif args.command=='status': status(args.root)
    else:
        args.root.mkdir(parents=True,exist_ok=True)
        with (args.root/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.root,args.devices.split(','))


if __name__=='__main__': main()
