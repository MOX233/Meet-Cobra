#!/usr/bin/env python3
"""Frozen E_all actor: HO-only H32 acquisition and per-slot cross-five BF.

Experiment-local adapters preserve all formal simulator sources/defaults.
Current-frame ideal planning CSI is retained, as in the approved O-MAPPO.
Actual serving beams use only paid probes of that slot's physical channel.
"""
import argparse
import concurrent.futures
from contextlib import ExitStack
import dataclasses
import fcntl
import os
from pathlib import Path
import pickle
import subprocess
import sys
from unittest.mock import patch

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiment import o_mappo_stage1_consistency as stage
from experiment import o_mappo_eall_full_grid as formal
from experiment.revision_pipeline import atomic_json,digest
from utils.gpu_phy import GPUFramePHY
from utils import directional_service as directional
from utils.hierarchical_tracking import acquire32,cross_neighbors,probe_candidates

DEFAULT=ROOT/'experiment/results/o_mappo_slot_cross5_20260925'
RATES=[1,5,15,21,25,29]
read=stage.read


@torch.inference_mode()
def track_frame(physical,connection,states,ho_slots=10):
    """Do not mutate learner states while precomputing private physical arrays."""
    ids,args,device=physical.ids,physical.args,physical.device
    slots=len(physical.h)
    gains=np.full((slots,len(ids)),-180.)
    pilots=np.zeros((slots,len(ids)),dtype=int)
    pairs=np.zeros((slots,len(ids),4),dtype=int)
    micro=np.array([j for j,v in enumerate(ids) if connection[v]>0],dtype=int)
    final={}
    if not len(micro): return gains,pilots,pairs,final
    ix=torch.as_tensor(micro,device=device)
    bs=torch.tensor([connection[ids[j]]-1 for j in micro],device=device)
    h=physical.h.permute(0,1,3,2,4)[:,ix,bs]
    acquisition=np.array([states[ids[j]].current_sweep_pilots==32 for j in micro])
    blocked=np.array([ho_slots if states[ids[j]].last_handover else 0 for j in micro])
    current=torch.tensor([states[ids[j]].tx_beam*args.M_r+states[ids[j]].rx_beam
                          for j in micro],device=device)
    chosen=torch.zeros((slots,len(micro)),dtype=torch.long,device=device)
    measured=torch.full((slots,len(micro)),-180.,device=device,dtype=torch.float64)
    for slot in range(slots):
        first=np.flatnonzero(acquisition & (blocked==slot))
        tracking=np.flatnonzero((blocked<=slot) & ~(acquisition & (blocked==slot)))
        if len(first):
            j=torch.as_tensor(first,device=device)
            beam,value=acquire32(h[slot,j],physical.tx,physical.rx)
            current[j]=beam; measured[slot,j]=physical.db(value)
            pilots[slot,micro[first]]=32
        if len(tracking):
            j=torch.as_tensor(tracking,device=device)
            candidates=cross_neighbors(current[j])
            response=probe_candidates(h[slot,j],candidates,physical.tx,physical.rx)
            best=response.argmax(1)[:,None]
            current[j]=candidates.gather(1,best)[:,0]
            measured[slot,j]=physical.db(response.gather(1,best)[:,0])
            pilots[slot,micro[tracking]]=5
        chosen[slot]=current
    gains[:,micro]=measured.cpu().numpy()
    pairs[:,micro,bs.cpu().numpy()]=chosen.cpu().numpy()
    final={ids[j]:int(p) for j,p in zip(micro,current.cpu().tolist())}
    return gains,pilots,pairs,final


class SlotTracking:
    def __init__(self):
        self.original_apply=stage.sim.apply_o_mappo_command
        self.original_load=stage.om.OMAPPPolicy.load
        self.original_sim=stage.sim.run_sim_o_mappo
        self.original_service=directional.DirectionalService.update
        self.frames=[]; self.service=[]; self.pending=False

    def load(self,*a,**kw):
        policy=self.original_load(*a,**kw)
        assert policy.config.tracking_pilots==1
        policy.config=dataclasses.replace(policy.config,tracking_pilots=5)
        return policy

    def apply(self,learner,command,record,config,tx,rx):
        # Suppress event-driven nine-pair tracking; slot tracking supersedes it.
        # Retain the original current-frame CSI convention for the nominal
        # post-HO planning beam. Actual service reacquires after interruption.
        if (command is not None and not command.trigger and learner.action>0
                and learner.tx_beam is not None and learner.rx_beam is not None):
            learner.current_sweep_pilots=0
            learner.last_handover=False; learner.last_trigger=0
            return stage.om.OMAPPOActionOutcome(False,False,0)
        return self.original_apply(learner,command,record,config,tx,rx)

    def physical(self,physical,connection,states):
        assert not self.pending
        self.gains,self.pilots,self.pairs,self.final=track_frame(physical,connection,states)
        self.states=states; self.ids=physical.ids; self.connection=connection
        self.before={v:(s.tx_beam,s.rx_beam) for v,s in states.items()}
        self.pending=True; self.checked=0
        active=self.pilots>0
        own=np.zeros_like(self.pilots)
        for j,v in enumerate(self.ids):
            if connection[v]>0: own[:,j]=self.pairs[:,j,connection[v]-1]
        changes=(own[1:]!=own[:-1]) & active[1:] & active[:-1]
        self.frames.append(dict(acquisitions=int((self.pilots==32).sum()),
            tracking_slots=int((self.pilots==5).sum()),probe_count=int(self.pilots.sum()),
            micro_active_slots=int(active.sum()),beam_changes=int(changes.sum()),
            gain_db_sum=float(self.gains[active].sum()),
            pilot_frame_mean=float(self.pilots.mean())))
        return self.gains

    def state_pairs(self,physical,connection,states):
        assert self.pending and states is self.states and physical.ids==self.ids
        assert all((s.tx_beam,s.rx_beam)==self.before[v] for v,s in states.items())
        return self.pairs

    def service_update(self,evaluator,args,**kw):
        slot=kw['slot_idx']; assert slot==self.checked
        for j,v in enumerate(self.ids):
            bs=self.connection[v]
            if bs:
                assert kw['num_pilot_dict'][v][bs-1]==self.pilots[slot,j]
                if self.pilots[slot,j]:
                    np.testing.assert_allclose(kw['g_dict'][v][bs],self.gains[slot,j],atol=1e-10)
                else: assert kw['RA_dict'][v]==0
        self.checked+=1
        return self.original_service(evaluator,args,**kw)

    def finish(self):
        assert self.pending and self.checked==100
        for v,p in self.final.items():
            self.states[v].tx_beam,self.states[v].rx_beam=divmod(p,8)
        self.pending=False

    def simulate(self,*a,**kw):
        previous=kw['progress_callback']
        def progress(n,total):
            self.finish()
            if previous: previous(n,total)
        return self.original_sim(*a,**dict(kw,progress_callback=progress,service_diagnostics=self.service))

    def activate(self,stack):
        stack.enter_context(patch.object(stage.om.OMAPPPolicy,'load',self.load))
        stack.enter_context(patch.object(stage.sim,'apply_o_mappo_command',self.apply))
        stack.enter_context(patch.object(GPUFramePHY,'fixed_pairs',lambda physical,c,s:self.physical(physical,c,s)))
        stack.enter_context(patch.object(directional,'state_pairs',self.state_pairs))
        stack.enter_context(patch.object(directional.DirectionalService,'update',
            lambda evaluator,args,**kw:self.service_update(evaluator,args,**kw)))
        stack.enter_context(patch.object(stage.sim,'run_sim_o_mappo',self.simulate))


def prepare(root):
    ref=read(formal.DEFAULT/'protocol.json')
    source=ref['jobs']['E_all_trained_rate15_seed1']
    for name,h in ref['code'].items(): assert digest(ROOT/name)==h,name
    assert digest(source['policy'])==formal.APPROVED_SHA
    p=dict(rates=RATES,seeds=[1],seconds=30,warmup_frames=2,ho_ms=10,
        policy=source['policy'],policy_sha256=source['policy_sha256'],
        timeline=ref['timeline'],timeline_sha256=ref['timeline_sha256'],
        formal_reference=str(formal.DEFAULT),reference_protocol_sha256=digest(formal.DEFAULT/'protocol.json'),
        code=dict(ref['code'],**{str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                              'utils/hierarchical_tracking.py':digest(ROOT/'utils/hierarchical_tracking.py')}),
        actor='frozen approved E_all two-layer actor; no retraining',
        changes='HO acquisition: 32 actual-slot probes after interruption; continuous five-point tracking with beam state carried across frames. Remove event-driven nine-pair search. Set tracking_pilots=5 for physical costs and existing decision cost formulas.',
        retained='E_all actor features and optimizer rules, ideal current-frame planning CSI, candidate search gain calculation, OTR-RA rule, traffic, fading, RB mapping, 10 ms interruption.',
        temporal='End-of-frame beam state committed only after service; no future physical channel is exposed to actor or optimizer.',
        planning='Nominal post-HO planning beam uses current-frame provided CSI as in the formal baseline; physical service reacquires using 32 paid probes. Candidate CSI acquisition costs remain uncharged as in the formal baseline.')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists(): assert read(root/'protocol.json')==p
    else: atomic_json(root/'protocol.json',p)
    return p


def verify(root):
    p=read(root/'protocol.json')
    for name,h in p['code'].items(): assert digest(ROOT/name)==h,name
    assert digest(p['policy'])==p['policy_sha256']
    assert digest(p['timeline'])==p['timeline_sha256']
    return p


def label(variant,rate,limit=None):
    return f'{variant}_rate{rate}_seed1'+(f'_smoke{limit}' if limit else '')


def case(root,variant,rate,device,limit=None):
    p=verify(root); name=label(variant,rate,limit)
    folder=root/('smoke' if limit else 'runs'); folder.mkdir(exist_ok=True)
    path=folder/f'{name}.json'
    if path.exists():
        row=read(path)
        assert row['protocol_sha256']==digest(root/'protocol.json')
        assert row['raw_sha256']==digest(path.with_suffix('.npz'))
        print('SKIP',name,flush=True); return
    with Path(p['timeline']).open('rb') as f: timeline=pickle.load(f)
    if limit: timeline=dict(list(timeline.items())[:limit+1])
    adapter=SlotTracking()
    with ExitStack() as stack:
        stack.enter_context(patch.object(stage,'POLICY',Path(p['policy'])))
        if variant=='slot_cross5': adapter.activate(stack)
        row,raw,diag,result=stage.simulate(timeline,'E_all',rate,1,device)
    if variant=='slot_cross5':
        assert len(adapter.frames)==len(result.energy_record) and not adapter.pending
        np.testing.assert_allclose([f['pilot_frame_mean'] for f in adapter.frames],result.pilot_record,atol=1e-12)
        assert all(f['probe_count']==32*f['acquisitions']+5*f['tracking_slots'] for f in adapter.frames)
        assert np.all(result.local_sweep_record==0)
        assert sum(f['acquisitions'] for f in adapter.frames)==result.full_sweep_record.sum()
        totals={k:sum(f[k] for f in adapter.frames[2:]) for k in adapter.frames[0]}
        row['metrics'].update(micro_probes_per_active_slot=totals['probe_count']/max(1,totals['micro_active_slots']),
            mean_measured_gain_db=totals['gain_db_sum']/max(1,totals['micro_active_slots']),
            within_frame_beam_changes=totals['beam_changes'])
        diag.update(beam=adapter.frames,service=adapter.service)
    if not limit:
        reference=formal.DEFAULT/'runs'/f'E_all_trained_rate{rate}_seed1.json'
        old=read(reference); assert row['traffic_sha256']==old['traffic_sha256']
        if variant=='control':
            with np.load(reference.with_suffix('.npz')) as z:
                for k,v in raw.items(): np.testing.assert_array_equal(v,z[k],err_msg=k)
            row['exact_reference_parity']=True
    np.savez_compressed(path.with_suffix('.npz'),**raw)
    dp=path.with_name(name+'_diagnostics.json'); atomic_json(dp,diag)
    row.update(variant=variant,protocol_sha256=digest(root/'protocol.json'),
        raw_sha256=digest(path.with_suffix('.npz')),diagnostics_sha256=digest(dp))
    atomic_json(path,row)
    print('COMPLETE',name,row['metrics'],flush=True)


def audit(root):
    p=verify(root); rows=[]
    for rate in RATES:
        path=root/'runs'/f'{label("slot_cross5",rate)}.json'; r=read(path)
        assert r['protocol_sha256']==digest(root/'protocol.json') and r['frames']==300
        assert r['actor_sha256']==p['policy_sha256']
        assert digest(path.with_suffix('.npz'))==r['raw_sha256']
        assert digest(path.with_name(path.stem+'_diagnostics.json'))==r['diagnostics_sha256']
        with np.load(path.with_suffix('.npz')) as z:
            q,f,rb=z['queue_bits'],z['queue_frame'],z['rb_per_bs']
            assert np.isfinite(q).all() and (q>=0).all()
            assert ((rb>=0)&(rb<=np.array([133,66,66,66,66])+1e-8)).all()
            np.testing.assert_allclose(rb@np.array([1,.2,.2,.2,.2])*.1,z['energy_j'],atol=1e-8)
            delay=q[f>=2]/(rate*1e6)*1000
            u=np.array([np.mean(q[f==i]>rate*1e6*.02) for i in range(300)])
            np.testing.assert_allclose(u,z['violation_probability'],atol=1e-12)
            actual=dict(power_w=z['energy_j'][2:].mean()/.1,violation_percent=100*u[2:].mean(),
                p90_proxy_ms=np.percentile(delay,90),p99_proxy_ms=np.percentile(delay,99))
            for k,v in actual.items(): np.testing.assert_allclose(v,r['metrics'][k],rtol=1e-10,atol=1e-8)
        old=read(formal.DEFAULT/'runs'/f'E_all_trained_rate{rate}_seed1.json')
        meet=read(formal.GRID/'runs'/f'meet_cobra_rate{rate}_seed1.json')
        assert r['traffic_sha256']==old['traffic_sha256']==meet['traffic_sha256']
        rows.append(dict(rate_mbps=rate,slot_cross5=r['metrics'],current=old['metrics'],meet=meet['metrics'],
            file=str(path),sha256=digest(path)))
    control=read(root/'runs'/f'{label("control",15)}.json')
    assert control['exact_reference_parity'] and control['protocol_sha256']==digest(root/'protocol.json')
    summary=dict(passed=True,paired_seed=1,comparison_loads=RATES,rows=rows,
        policy_sha256=p['policy_sha256'],raw_metrics_recomputed=True,
        control_raw_parity=True,protocol_sha256=digest(root/'protocol.json'))
    atomic_json(root/'summary.json',summary)
    for r in rows: print(r['rate_mbps'],{m:{k:round(r[m][k],5) for k in ['power_w','violation_percent','p99_proxy_ms']} for m in ['current','slot_cross5','meet']},flush=True)


def run(root,devices):
    prepare(root); (root/'logs').mkdir(exist_ok=True)
    jobs=[('control',15)]+[('slot_cross5',r) for r in RATES]
    buckets=[jobs[i::len(devices)] for i in range(len(devices))]
    def worker(device,jobs):
        for variant,rate in jobs:
            name=label(variant,rate)
            with (root/'logs'/f'{name}.log').open('a') as log:
                result=subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'case','--root',str(root),
                    '--variant',variant,'--rate',str(rate),'--device',device],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            print('JOB',name,'EXIT',result,flush=True)
            if result: raise RuntimeError(name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker,devices,buckets))
    audit(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','case','run','audit'])
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--variant',choices=['control','slot_cross5'],default='slot_cross5')
    p.add_argument('--rate',type=int,choices=RATES,default=15)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--limit',type=int)
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    a=p.parse_args()
    if a.command=='prepare': prepare(a.root)
    elif a.command=='case': case(a.root,a.variant,a.rate,a.device,a.limit)
    elif a.command=='audit': audit(a.root)
    else:
        a.root.mkdir(parents=True,exist_ok=True)
        with (a.root/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(a.root,a.devices.split(','))


if __name__=='__main__': main()
