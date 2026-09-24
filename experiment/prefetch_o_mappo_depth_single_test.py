#!/usr/bin/env python3
"""Cache the already-selected control's test cases while the treatment trains.

This does not select or modify either policy. The main pipeline later checks
checkpoint hashes and skips these exact same single-layer cases. The separate
evaluation manifest preserves input/source provenance for the prefetched cases.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.o_mappo_actor_depth_experiment import evaluate, TEST_TIMELINE, TEST_RATES
from experiment.revision_pipeline import atomic_json, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'experiment/results/o_mappo_actor_depth_20260924')
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    args=p.parse_args()
    selection=json.loads((args.root/'validation_single/selection.json').read_text())
    policy=Path(selection['selected']['checkpoint'])
    cache=args.root/'prefetched_single_test'
    evaluate(cache,[policy],TEST_TIMELINE,TEST_RATES,[1,2,3],30,args.devices)
    destination=args.root/'paired_test/runs'
    destination.mkdir(parents=True,exist_ok=True)
    provenance=[]
    for source in sorted((cache/'runs').glob('*.json')):
        case=json.loads(source.read_text())
        if case['checkpoint_sha256']!=digest(policy) or case['frames']!=300:
            raise ValueError('Unexpected prefetch case')
        for src in (source.with_suffix('.npz'),source):
            target=destination/src.name
            if target.exists():
                if digest(target)!=digest(src):
                    raise ValueError('Conflicting existing test case')
            else:
                temp=target.with_suffix(target.suffix+'.prefetch')
                shutil.copyfile(src,temp)
                os.replace(temp,target)
        provenance.append(dict(source=str(source),destination=str(destination/source.name),
                               sha256=digest(source),traffic_sha256=case['traffic_sha256']))
    atomic_json(args.root/'paired_test/prefetch_provenance.json',dict(
        reference_manifest=str(cache/'protocol.json'),cases=provenance))
    print('CACHED',len(provenance),'paired single-layer test cases',flush=True)


if __name__=='__main__':
    main()
