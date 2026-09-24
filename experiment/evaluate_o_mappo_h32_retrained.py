#!/usr/bin/env python3
"""Parallel exact evaluation of fixed checkpoints, with case-level resume."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.revision_pipeline import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--timeline', type=Path, required=True)
    parser.add_argument('--policies', type=Path, nargs='+', required=True)
    parser.add_argument('--rates', default='1,5,15,21,25,29,35')
    parser.add_argument('--seeds', default='101')
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    policies = {f'{p.parent.name}_{p.stem}': p.resolve() for p in args.policies}
    if len(policies) != len(args.policies):
        raise ValueError('Checkpoint labels collide')
    manifest = dict(timeline=str(args.timeline.resolve()), timeline_sha256=digest(args.timeline),
        policies={k:dict(path=str(p), sha256=digest(p)) for k,p in policies.items()},
        rates=list(map(int,args.rates.split(','))), seeds=list(map(int,args.seeds.split(','))),
        seconds=args.seconds,
        code={str(p.relative_to(ROOT)):digest(p) for p in
              [ROOT/'experiment/train_o_mappo_hierarchical32.py',ROOT/'utils/o_mappo.py',
               ROOT/'utils/o_mappo_sim.py',ROOT/'utils/hierarchical_beam.py']})
    path = args.root/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Evaluation inputs changed; use a new root')
    atomic_json(path,manifest)
    jobs = [(label,rate,seed) for label in policies for rate in manifest['rates'] for seed in manifest['seeds']]
    devices=args.devices.split(',')
    (args.root/'logs').mkdir(exist_ok=True)
    def worker(device, jobs):
        for label,rate,seed in jobs:
            name=f'{label}_rate{rate}_seed{seed}'
            output=args.root/'runs'/f'{name}.json'
            if output.exists():
                result=json.loads(output.read_text())
                if result['checkpoint_sha256'] != manifest['policies'][label]['sha256']:
                    raise ValueError('Checkpoint changed')
                continue
            command=[sys.executable,'-u',str(ROOT/'experiment/train_o_mappo_hierarchical32.py'),
                'exact','--timeline',str(args.timeline.resolve()),'--policy',str(policies[label]),
                '--rate',str(rate),'--seed',str(seed),'--device',device,
                '--seconds',str(args.seconds),'--output',str(output.resolve())]
            with (args.root/'logs'/f'{name}.log').open('a') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                    cwd=ROOT,env=dict(os.environ,PYTHONHASHSEED='0'))
            print('DONE',name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(worker,d,jobs[i::len(devices)]) for i,d in enumerate(devices)]
        for future in futures: future.result()
    atomic_json(args.root/'complete.json',dict(cases=len(jobs)))


if __name__ == '__main__':
    main()
