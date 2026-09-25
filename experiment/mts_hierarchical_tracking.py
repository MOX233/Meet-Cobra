#!/usr/bin/env python3
"""Isolated three-arm MTS BF pilot, paired with the formal directional grid."""
import argparse
import concurrent.futures
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from unittest.mock import patch

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiment import revision_pipeline as grid
from experiment import o_mappo_shared_frontend as shared
from utils import mts_report_sim as sim, mts_report as report, alg_utils as alg
from utils import directional_service as directional
from utils.mts_report_bounded import estimate_bounded_report_load
from utils.mts_gs_hbf import candidate_configs
from utils.hierarchical_tracking import search_frame
from utils.ho_utils import make_paired_traffic

REFERENCE=ROOT/'experiment/results/revision_directional_20260922/grid'
DEFAULT=ROOT/'experiment/results/mts_hierarchical_tracking_20260925_v2'
RATES=[1,5,15,21,25,29]
VARIANTS=['report_hold','hier32_hold','hier32_cross5']


class BeamAdapter:
    def __init__(self, variant):
        self.variant=variant
        self.original_probe=sim.probe_and_hold
        self.original_pairs=directional.state_pairs
        self.original_candidates=report.build_report_candidates
        self.frames=[]
        self.pairs=None

    def probes(self,physical,connection,states,reports,switched,ho_slots,k=5,tracking_pilots=1):
        if self.variant=='report_hold':
            gains,pilots,chosen=self.original_probe(physical,connection,states,reports,switched,ho_slots,k,tracking_pilots)
            pairs=self.original_pairs(physical,connection,states)
        else:
            gains,pilots,pairs,chosen=search_frame(physical,connection,states,switched,ho_slots,
                                                 self.variant=='hier32_cross5')
        self.pairs=pairs
        active=pilots>0
        own=np.zeros_like(pilots)
        for j,v in enumerate(physical.ids):
            if connection[v]>0: own[:,j]=pairs[:,j,connection[v]-1]
        changes=(own[1:]!=own[:-1]) & active[1:] & active[:-1]
        self.frames.append(dict(micro_active_slots=int(active.sum()),probe_count=int(pilots.sum()),
            serving_gain_db_sum=float(gains[active].sum()),beam_changes=int(changes.sum()),
            acquisition_count=int((pilots==32).sum()),five_probe_slots=int((pilots==5).sum()),
            one_probe_slots=int((pilots==1).sum()),
            frame_probing_fraction=float(pilots.sum()*physical.args.pilot_overhead_factor/max(1,active.sum()))))
        return gains,pilots,chosen

    def slot_pairs(self,physical,connection,states):
        assert self.pairs.shape==(physical.args.slots_per_frame,len(physical.ids),4)
        return self.pairs

    def candidates(self,args,*a,**kw):
        follow=5 if self.variant=='hier32_cross5' else 1
        kw['pilot_average_override']=(32+(args.slots_per_frame-1)*follow)/args.slots_per_frame
        return self.original_candidates(args,*a,**kw)

    def load(self,args,reports,connection,rates,config,locations,last_measurements,*a,**kw):
        follow=5 if self.variant=='hier32_cross5' else 1
        average=(32+(args.slots_per_frame-1)*follow)/args.slots_per_frame
        fraction=1-args.pilot_overhead_factor*average
        adjusted={v:r/(fraction if connection.get(v,0)>0 else 1) for v,r in rates.items()}
        return estimate_bounded_report_load(args,reports,connection,adjusted,config,locations,last_measurements,*a,**kw)

    def activate(self, stack):
        stack.enter_context(patch.object(sim,'probe_and_hold',self.probes))
        stack.enter_context(patch.object(directional,'state_pairs',self.slot_pairs))
        stack.enter_context(patch.object(sim,'estimate_report_load',
            estimate_bounded_report_load if self.variant=='report_hold' else self.load))
        if self.variant!='report_hold':
            stack.enter_context(patch.object(report,'build_report_candidates',self.candidates))


def prepare(root):
    parent=grid.read(REFERENCE/'protocol.json')
    names=['experiment/mts_hierarchical_tracking.py','utils/hierarchical_tracking.py',
        'experiment/revision_pipeline.py','experiment/o_mappo_shared_frontend.py',
        'experiment/pql_ba_experiment.py','utils/mts_report_sim.py','utils/mts_report.py',
        'utils/mts_report_bounded.py','utils/mts_gs_hbf.py','utils/mts_gs_hbf_sim.py',
        'utils/directional_service.py','utils/gpu_phy.py','utils/hierarchical_beam.py',
        'utils/alg_utils.py','utils/ho_utils.py','utils/queue_utils.py','utils/compiled_matching.py']
    for name in names:
        if name in parent['code_sha256'] and name!='experiment/revision_pipeline.py':
            assert grid.digest(ROOT/name)==parent['code_sha256'][name],name
    p=dict(cache=parent['cache'],cache_sha256=parent['cache_sha256'],reference=str(REFERENCE),
        reference_protocol_sha256=grid.digest(REFERENCE/'protocol.json'),
        variants=VARIANTS,rates=RATES,seeds=[1],frames=300,warmup_frames=2,ho_ms=10,
        code={name:grid.digest(ROOT/name) for name in names},
        design='Same GS preferences/5-frame association, prediction reports and OTR-RA; BF method and its mean probing cost change. Newly estimated occupancy accounts for BF cost for the two new arms; formal control is unchanged.',
        acquisition='32 paid serving-link probes in first available slot of every frame; no extra local search in that slot',
        tracking='Current plus four axial neighbours, refreshed every slot; boundaries wrap. No free current-beam probe.',
        physics='Actual per-slot beams and explicit RB assignments determine directional service and queue updates',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    path=root/'protocol.json'
    if path.exists():
        old=grid.read(path)
        p['git_head']=old['git_head']
        assert old==p,'Protocol changed; use a new output root'
    else: grid.atomic_json(path,p)
    return p


def run_case(root,variant,rate,device,limit=None):
    p=grid.read(root/'protocol.json')
    for name,sha in p['code'].items(): assert grid.digest(ROOT/name)==sha,name
    assert grid.digest(p['cache'])==p['cache_sha256']
    assert variant in VARIANTS and rate in RATES
    name=f'{variant}_rate{rate}_seed1'+(f'_smoke{limit}' if limit else '')
    dest=root/'smoke' if limit else root
    (dest/'locks').mkdir(parents=True,exist_ok=True)
    with (dest/'locks'/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (dest/'runs'/f'{name}.json').exists():
            row=grid.read(dest/'runs'/f'{name}.json')
            assert row['protocol_sha256']==grid.digest(root/'protocol.json')
            assert row['raw_sha256']==grid.digest(dest/'raw'/f'{name}.npz')
            print('SKIP',name,flush=True); return
        torch.set_num_threads(1); shared.single_thread_solvers()
        from utils.compiled_matching import km_algorithm_compiled
        alg.km_algorithm=km_algorithm_compiled
        km_algorithm_compiled(np.zeros((2,2)))
        with Path(p['cache']).open('rb') as f: timeline=pickle.load(f)
        if limit: timeline=dict(list(timeline.items())[:limit+1])
        args=shared.paper_args(rate*1e6); args.device=torch.device('cpu')
        traffic=make_paired_traffic(args,timeline,1)
        if not limit:
            reference=grid.read(REFERENCE/'runs'/f'mts_report_rate{rate}_seed1.json')
            assert traffic['sha256']==reference['traffic_sha256']
        np.random.seed(1)
        adapter=BeamAdapter(variant); diagnostics=[]; service=[]
        def progress(n,total):
            grid.atomic_json(dest/'progress'/f'{name}.json',dict(frame=n,total=total,pid=os.getpid(),device=device))
            if n%25==0 or n==total: print('FRAME',n,total,flush=True)
        start=time.monotonic()
        with ExitStack() as stack:
            adapter.activate(stack)
            result=sim.run_sim_mts_report(args,shared.MICRO_BS_LOCATIONS,timeline,candidate_configs()['pressure_early'],
                traffic,seed=1,physics_device=device,ho_interruption_ms=10,k=5,diagnostics=diagnostics,
                progress_callback=progress,directional_service=True,predicted_ra_interference=True,service_diagnostics=service)
        assert len(service)==len(adapter.frames)==len(result.energy_record)
        assert all(d['slots']==args.slots_per_frame for d in service)
        metrics,raw=grid.extract(args,result,traffic,2)
        totals={k:sum(d[k] for d in adapter.frames[2:]) for k in adapter.frames[0] if k!='frame_probing_fraction'}
        metrics.update(micro_probes_per_active_slot=totals['probe_count']/max(1,totals['micro_active_slots']),
            mean_serving_gain_db=totals['serving_gain_db_sum']/max(1,totals['micro_active_slots']),
            within_frame_beam_changes=totals['beam_changes'])
        if variant=='report_hold' and not limit:
            for k,v in reference['metrics'].items():
                np.testing.assert_allclose(metrics[k],v,rtol=1e-10,atol=1e-8,err_msg=k)
        for folder in ('runs','raw','diagnostics'): (dest/folder).mkdir(exist_ok=True)
        rawpath=dest/'raw'/f'{name}.npz'
        temporary=rawpath.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**raw); os.replace(temporary,rawpath)
        dpath=dest/'diagnostics'/f'{name}.json'
        # Original probe metadata describes report_hold only. New arms use
        # the exact per-frame counters below, not the legacy five-probe count.
        grid.atomic_json(dpath,dict(beam=adapter.frames,service=service))
        grid.atomic_json(dest/'runs'/f'{name}.json',dict(variant=variant,rate_mbps=rate,seed=1,device=device,
            protocol_sha256=grid.digest(root/'protocol.json'),traffic_sha256=traffic['sha256'],
            raw_sha256=grid.digest(rawpath),diagnostics_sha256=grid.digest(dpath),
            frames=len(result.energy_record),elapsed_s=time.monotonic()-start,metrics=metrics))
        print('COMPLETE',name,json.dumps(metrics),flush=True)


def launch(root,devices):
    prepare(root)
    (root/'logs').mkdir(exist_ok=True)
    jobs=[(v,r) for r in RATES for v in VARIANTS]
    buckets=[jobs[i::len(devices)] for i in range(len(devices))]
    def worker(device,jobs):
        for variant,rate in jobs:
            name=f'{variant}_rate{rate}_seed1'
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'case','--root',str(root),
                 '--variant',variant,'--rate',str(rate),'--device',device]
            with (root/'logs'/f'{name}.log').open('a') as log:
                code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print('JOB',name,'EXIT',code,flush=True)
            if code: raise RuntimeError(name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        list(executor.map(worker,devices,buckets))
    summarize(root)


def summarize(root):
    rows=[]
    for rate in RATES:
        for variant in VARIANTS:
            path=root/'runs'/f'{variant}_rate{rate}_seed1.json'
            if path.exists(): rows.append(grid.read(path))
    grid.atomic_json(root/'summary.json',dict(completed=len(rows),expected=18,runs=rows))
    for r in rows:
        m=r['metrics']
        print(r['rate_mbps'],r['variant'],*[round(m[k],5) for k in
              ('power_w','violation_percent','p99_proxy_ms','micro_probes_per_active_slot')],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','case','run','summarize'])
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--rate',type=int,choices=RATES)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5')
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.command=='prepare': prepare(a.root)
    elif a.command=='case': run_case(a.root,a.variant,a.rate,a.device,a.limit)
    elif a.command=='run': launch(a.root,a.devices.split(','))
    else: summarize(a.root)


if __name__=='__main__': main()
