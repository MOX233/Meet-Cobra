#!/usr/bin/env python3
"""Small, explicitly non-production integration check; never a formal result."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiment.revision_training import atomic_json
from experiment.benchmark_nn_overhead import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--second-device',help='Optional additional GPU for concurrent launcher check')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('Use a fresh smoke directory')
    args.output.mkdir(parents=True)
    def run(script,*arguments):
        cmd=[sys.executable,'-u',str(ROOT/'experiment'/script),*map(str,arguments)]
        print('CHECK',cmd,flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True,env=dict(os.environ,PYTHONHASHSEED='0'))
    rng=np.random.default_rng(91)
    timeline={}
    for t in range(16):
        timeline[round(200+t*.1,1)]={str(v):dict(h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-6)
                                  for v in range(12)}
    source=args.output/'synthetic_training.pkl'
    with source.open('wb') as f: pickle.dump(timeline,f)
    data=args.output/'data.npz'
    run('revision_training.py','data','--source',source,'--output',data)
    meta=json.loads(data.with_suffix('.json').read_text()); meta['smoke']=True
    atomic_json(data.with_suffix('.json'),meta)
    split=args.output/'split.npz'
    np.savez(split,train_vehicle_ids=np.array([str(v) for v in range(8)]),validation_vehicle_ids=np.array([str(v) for v in range(8,12)]))
    train=args.output/'training'
    command=('train','--data',data,'--split-file',split,'--output',train,'--device',args.device,
             '--stage1-epochs',2,'--stage2-epochs',2,'--max-samples',64,'--max-trajectories',4,'--batch-size',16)
    run('revision_training.py',*command,'--stop-after-epochs',1)
    run('revision_training.py',*command,'--resume','--stop-after-epochs',2)
    run('revision_training.py',*command,'--resume')
    before=digest(train/'stage2/interfering_gain/best.pth')
    run('revision_training.py',*command,'--resume')
    assert before==digest(train/'stage2/interfering_gain/best.pth')
    # Compare resumed and uninterrupted training on this deterministic fixture.
    reference=args.output/'uninterrupted_training'
    ref_command=tuple(reference if value==train else value for value in command)
    run('revision_training.py',*ref_command)
    import torch
    a=torch.load(train/'stage2/interfering_gain/best.pth',map_location='cpu',weights_only=True)
    b=torch.load(reference/'stage2/interfering_gain/best.pth',map_location='cpu',weights_only=True)
    for name in a: torch.testing.assert_close(a[name],b[name],rtol=1e-6,atol=1e-7)
    models=args.output/'models'
    run('revision_training.py','assemble','--training',train,'--output',models)
    cache=args.output/'test_cache.pkl'
    run('revision_training.py','cache','--models',models,'--output',cache,'--device',args.device,'--end',800.8)
    before=digest(cache)
    run('revision_training.py','cache','--models',models,'--output',cache,'--device',args.device,'--end',800.8)
    assert before==digest(cache)
    grid=args.output/'grid'
    run('revision_pipeline.py','prepare','--root',grid,'--cache',cache,'--rates',13,'--seeds',1,
        '--backend',args.device.split(':')[0],'--allow-smoke','--reactive-input','current')
    run('revision_pipeline.py','run','--root',grid,'--devices',args.device,'--methods','meet_cobra')
    devices=args.device+(','+args.second_device if args.second_device else '')
    run('revision_pipeline.py','run','--root',grid,'--devices',devices,'--detach')
    launched=json.loads((grid/'launch.json').read_text())['launched_at']
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        state=json.loads((grid/'queue_status.json').read_text())
        if (grid/'queue_status.json').stat().st_mtime>=launched and state['phase']!='running':
            assert state['phase']=='selection_completed',state
            break
        time.sleep(1)
    else: raise TimeoutError('Detached smoke launcher did not finish')
    hashes={f.name:digest(f) for f in (grid/'raw').glob('*.npz')}
    run('revision_pipeline.py','run','--root',grid,'--devices',args.device)
    assert hashes=={f.name:digest(f) for f in (grid/'raw').glob('*.npz')}
    run('revision_pipeline.py','summarize','--root',grid)
    run('revision_pipeline.py','status','--root',grid)
    atomic_json(args.output/'check_summary.json',dict(passed=True,smoke=True,device=args.device,
        cases=8,frames_per_case=8,devices=devices,
        training='2+2 epochs on synthetic data; resumes in both stages match uninterrupted training',
        launcher='subset then detached remainder, repeated completed-run check',
        note='Software integration only; never use smoke checkpoints or metrics in the paper'))


if __name__=='__main__': main()
