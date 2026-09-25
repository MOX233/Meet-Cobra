#!/usr/bin/env python3
"""Extend the user-approved E_all checkpoint to the paper's paired full grid."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import o_mappo_eall_training as training
from experiment.revision_pipeline import atomic_json, digest
import numpy as np

SOURCE=training.OUTPUT
GRID=ROOT/'experiment/results/revision_directional_20260922/grid'
DEFAULT=ROOT/'experiment/results/o_mappo_eall_full_grid_20260925'
APPROVED_SHA='e1be46476dc17cbd2300a47b8930dceb173e469b9e8cf1dc18c35a31411e5b43'
LABEL='E_all_trained'
RATES=list(range(1,36,2))
SEEDS=[1,2,3]
read=training.old.read


def prepare(root):
    training.verify(SOURCE)
    selected=read(SOURCE/'selection.json')['selected_trained']['E_all']
    policy=Path(selected['policy'])
    assert selected['label']=='E_all_seed11' and digest(policy)==APPROVED_SHA
    assert read(policy.with_suffix('.json'))['round']==20
    source=SOURCE/'paired_test'
    previous=read(source/'protocol.json')
    paper=read(GRID/'protocol.json')
    assert paper['rates']==RATES and paper['seeds']==SEEDS
    assert paper['warmup_frames']==2 and paper['ho_ms']==10
    assert previous['seconds']==30 and previous['timeline_sha256']==paper['cache_sha256']
    code={str(p.relative_to(ROOT)):digest(p) for p in training.sources()}
    assert previous['code']==code
    allowed={'utils/o_mappo.py','utils/o_mappo_sim.py','experiment/revision_pipeline.py'}
    for name,h in paper['code_sha256'].items():
        if name not in allowed: assert digest(ROOT/name)==h,name
    jobs={training.case_label(LABEL,r,s):dict(label=LABEL,route='E_all',policy=str(policy),
        policy_sha256=APPROVED_SHA,rate=r,seed=s) for r in RATES for s in SEEDS}
    protocol=dict(timeline=previous['timeline'],timeline_sha256=previous['timeline_sha256'],
        seconds=30,jobs=jobs,code=code)
    extension=dict(source_protocol=str(source/'protocol.json'),source_sha256=digest(source/'protocol.json'),
        paper_protocol=str(GRID/'protocol.json'),paper_sha256=digest(GRID/'protocol.json'),
        selection_sha256=digest(SOURCE/'selection.json'),script_sha256=digest(Path(__file__)),
        checkpoint_sha256=APPROVED_SHA,route='E_all',rates=RATES,seeds=SEEDS,warmup_frames=2)
    for name,value in [('protocol.json',protocol),('extension_protocol.json',extension)]:
        if (root/name).exists(): assert read(root/name)==value,'Frozen protocol changed'
        else: atomic_json(root/name,value)
    destination=root/'runs'
    destination.mkdir(exist_ok=True)
    reused=[]
    for label,spec in previous['jobs'].items():
        if spec['label']!=LABEL: continue
        assert spec==jobs[label]
        assert training.verified_case(source,label)
        row=read(source/'runs'/f'{label}.json')
        ref=read(GRID/'runs'/f'meet_cobra_rate{spec["rate"]}_seed{spec["seed"]}.json')
        assert row['frames']==ref['frames']==300 and row['traffic_sha256']==ref['traffic_sha256']
        for suffix in ('.npz','_diagnostics.json'):
            src=source/'runs'/f'{label}{suffix}'
            target=destination/src.name
            if target.exists(): assert digest(target)==digest(src)
            else:
                temp=target.with_suffix(target.suffix+'.copying')
                shutil.copyfile(src,temp); os.replace(temp,target)
        row['protocol_sha256']=digest(root/'protocol.json')
        row['reused_from']=str(source/'runs'/f'{label}.json')
        target=destination/f'{label}.json'
        if target.exists(): assert read(target)==row
        else: atomic_json(target,row)
        reused.append(dict(source=row['reused_from'],sha256=digest(Path(row['reused_from'])),
            raw_sha256=row['raw_sha256'],diagnostics_sha256=row['diagnostics_sha256']))
    assert len(reused)==9
    atomic_json(root/'reuse.json',dict(cases=reused))
    return policy,Path(previous['timeline'])


def audit(root):
    protocol=read(root/'protocol.json')
    extension=read(root/'extension_protocol.json')
    assert extension['script_sha256']==digest(Path(__file__))
    assert digest(Path(protocol['timeline']))==protocol['timeline_sha256']
    for name,h in protocol['code'].items(): assert digest(ROOT/name)==h,name
    rows,provenance=[],[]
    for rate in RATES:
        for seed in SEEDS:
            label=training.case_label(LABEL,rate,seed)
            spec=protocol['jobs'][label]
            assert digest(Path(spec['policy']))==spec['policy_sha256']==APPROVED_SHA
            assert training.verified_case(root,label)
            path=root/'runs'/f'{label}.json'; row=read(path)
            ref=read(GRID/'runs'/f'meet_cobra_rate{rate}_seed{seed}.json')
            assert row['frames']==ref['frames']==300
            assert row['traffic_sha256']==ref['traffic_sha256']
            assert row['actor_sha256']==APPROVED_SHA and row['variant']=='E_all'
            assert (row['rate_mbps'],row['seed'])==(rate,seed)
            assert np.isfinite(list(row['metrics'].values())).all()
            with np.load(path.with_suffix('.npz')) as z:
                rb,q,f=z['rb_per_bs'],z['queue_bits'],z['queue_frame']
                assert np.isfinite(q).all() and np.all(q>=0)
                assert np.all((rb>=0)&(rb<=np.array([133,66,66,66,66])+1e-8))
                np.testing.assert_allclose(rb@np.array([1,.2,.2,.2,.2])*.1,z['energy_j'],rtol=1e-10,atol=1e-8)
                u=np.array([np.mean(q[f==i]>rate*1e6*.02) for i in range(300)])
                np.testing.assert_allclose(u,z['violation_probability'],atol=1e-12)
                delay=q[f>=2]/(rate*1e6)*1000; counts=z['association_counts'][2:]
                actual=dict(power_w=z['energy_j'][2:].mean()/.1,violation_percent=100*u[2:].mean(),
                    p90_proxy_ms=np.percentile(delay,90),p99_proxy_ms=np.percentile(delay,99),
                    macro_association_percent=100*counts[:,0].sum()/counts.sum())
                for k,v in actual.items(): np.testing.assert_allclose(v,row['metrics'][k],atol=1e-9,rtol=1e-10)
            rows.append(row)
            provenance.append(dict(file=str(path),sha256=digest(path),raw_sha256=row['raw_sha256']))
    groups=[]
    for rate in RATES:
        cases=[r for r in rows if r['rate_mbps']==rate]
        metrics={}
        for key in cases[0]['metrics']:
            values=[r['metrics'][key] for r in cases]
            metrics[key]=dict(mean=float(np.mean(values)),minimum=min(values),maximum=max(values),per_seed=values)
        groups.append(dict(rate_mbps=rate,metrics=metrics))
    summary=dict(cases=len(rows),rates=RATES,seeds=SEEDS,seconds=30,warmup_frames=2,
        policy_sha256=APPROVED_SHA,protocol_sha256=digest(root/'protocol.json'),
        paired_with=str(GRID),raw_metrics_recomputed=True,groups=groups,provenance=provenance,
        runs=rows,route='E_all',training_seed=11,selected_round=20,
        optimizer_failures=sum(r['metrics']['optimizer_failures'] for r in rows))
    atomic_json(root/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args(); args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'pipeline.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.audit_only:
        print('AUDITED',audit(args.root)['cases'],flush=True); return
    policy,timeline=prepare(args.root)
    print('REUSED 9 paired cases; 45 additional cases required',flush=True)
    if args.prepare_only: return
    training.exact_grid(args.root,{LABEL:dict(route='E_all',path=policy)},timeline,
        30,RATES,SEEDS,args.devices.split(','))
    print('COMPLETE AND AUDITED',audit(args.root)['cases'],flush=True)


if __name__=='__main__': main()
