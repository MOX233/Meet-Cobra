#!/usr/bin/env python3
"""Complete paired CSI-privileged HO32/cross5 controls without retraining."""
import argparse
import collections
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
from unittest.mock import patch

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiment import o_mappo_slot_tracking as pilot
from experiment.train_o_mappo_predicted_cross5 import DEFAULT, INITIAL, TEST, RATES, old
from experiment.revision_pipeline import atomic_json,digest


def prepare(root):
    reference=pilot.verify(pilot.DEFAULT)
    assert reference['policy_sha256']==digest(INITIAL)
    assert reference['timeline_sha256']==digest(TEST)
    p=dict(rates=RATES,seeds=[1,2,3],policy=str(INITIAL),policy_sha256=digest(INITIAL),
        timeline=str(TEST),timeline_sha256=digest(TEST),seconds=30,warmup_frames=2,
        code=dict(reference['code'],**{str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}),
        source_protocol_sha256=digest(pilot.DEFAULT/'protocol.json'))
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists() and old.read(root/'protocol.json')!=p:
        raise ValueError('Control protocol changed')
    atomic_json(root/'protocol.json',p)
    (root/'runs').mkdir(exist_ok=True)
    for rate in pilot.RATES:
        name=f'true_cross5_rate{rate}_seed1'
        destination=root/'runs'/f'{name}.json'
        if destination.exists():continue
        source=pilot.DEFAULT/'runs'/f'slot_cross5_rate{rate}_seed1.json'
        row=old.read(source)
        assert row['raw_sha256']==digest(source.with_suffix('.npz'))
        for suffix in ('.npz','_diagnostics.json'):
            src=source.with_name(source.stem+suffix)
            shutil.copyfile(src,root/'runs'/f'{name}{suffix}')
        row.update(label='true_cross5',rate=rate,policy=str(INITIAL),policy_sha256=digest(INITIAL),
            protocol_sha256=digest(root/'protocol.json'),reused_from=str(source))
        atomic_json(destination,row)


def case(root,rate,seed,device):
    p=old.read(root/'protocol.json')
    for name,h in p['code'].items():assert digest(ROOT/name)==h,name
    assert digest(INITIAL)==p['policy_sha256'] and digest(TEST)==p['timeline_sha256']
    with TEST.open('rb') as stream:timeline=pickle.load(stream)
    adapter=pilot.SlotTracking()
    with ExitStack() as stack:
        stack.enter_context(patch.object(pilot.stage,'POLICY',INITIAL))
        adapter.activate(stack)
        row,raw,diagnostic,_=pilot.stage.simulate(timeline,'E_all',rate,seed,device)
    name=f'true_cross5_rate{rate}_seed{seed}'
    path=root/'runs'/name
    np.savez_compressed(path.with_suffix('.npz'),**raw)
    diagnostic['tracking']=adapter.frames
    diagnostic['service']=adapter.service
    atomic_json(path.with_name(name+'_diagnostics.json'),diagnostic)
    row.update(label='true_cross5',rate=rate,policy=str(INITIAL),policy_sha256=digest(INITIAL),
        protocol_sha256=digest(root/'protocol.json'),raw_sha256=digest(path.with_suffix('.npz')),
        diagnostics_sha256=digest(path.with_name(name+'_diagnostics.json')))
    atomic_json(path.with_suffix('.json'),row)


def run(root,devices):
    prepare(root)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[(r,s) for r in RATES for s in (1,2,3)]
    def worker(device,jobs):
        for rate,seed in jobs:
            label=f'true_cross5_rate{rate}_seed{seed}'
            path=root/'runs'/f'{label}.json'
            if path.exists():
                row=old.read(path)
                assert row['protocol_sha256']==digest(root/'protocol.json')
                assert row['raw_sha256']==digest(path.with_suffix('.npz'))
                continue
            (root/'logs').mkdir(exist_ok=True)
            with (root/'logs'/f'{label}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'case','--root',str(root),
                    '--rate',str(rate),'--seed',str(seed),'--device',device],check=True,cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT)
            print('CONTROL COMPLETE',rate,seed,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(worker,d,jobs[j::len(devices)]) for j,d in enumerate(devices)]
        for f in futures:f.result()
    rows=[old.read(root/'runs'/f'true_cross5_rate{r}_seed{s}.json') for r,s in jobs]
    atomic_json(root/'summary.json',dict(cases=len(rows),runs=rows))
    print('CONTROL GRID COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','case','launch'])
    p.add_argument('--root',type=Path,default=DEFAULT/'true_control')
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    p.add_argument('--device',default='cuda:0');p.add_argument('--rate',type=int);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.command=='run':run(a.root,a.devices.split(','))
    elif a.command=='case':case(a.root,a.rate,a.seed,a.device)
    else:
        a.root.mkdir(parents=True,exist_ok=True)
        argv=[sys.executable,'-u',str(Path(__file__)),'run','--root',str(a.root),'--devices',a.devices]
        with (a.root/'pipeline.log').open('a') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONHASHSEED='0'))
        atomic_json(a.root/'launch.json',dict(pid=child.pid,argv=argv,started_unix=time.time()))
        print('LAUNCHED CONTROL',child.pid,flush=True)


if __name__=='__main__':main()
