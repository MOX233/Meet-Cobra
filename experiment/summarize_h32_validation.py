#!/usr/bin/env python3
"""Rank held-out exact validation checkpoints using one rule for all loads."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('experiment/results/o_mappo_h32_retrained_20260924/exact_reference'))
    args=parser.parse_args()
    ref={(r['rate_mbps'],r['seed'],r['frames']):r for r in
         [read(f) for f in (args.reference/'runs').glob('*.json')]}
    protocol=read(args.root/'protocol.json')
    expected=len(protocol['rates'])*len(protocol['seeds'])
    groups={}
    for f in (args.root/'runs').glob('*.json'):
        r=read(f);p=Path(r['checkpoint']);label=f'{p.parent.name}_{p.stem}'
        reference=ref[(r['rate_mbps'],r['seed'],r['frames'])]
        assert r['traffic_sha256']==reference['traffic_sha256']
        m=r['metrics'];b=reference['metrics']
        cost=m['violation_percent']+.02*m['power_w']
        baseline=b['violation_percent']+.02*b['power_w']
        groups.setdefault(label,[]).append(dict(rate=r['rate_mbps'],seed=r['seed'],
            cost_ratio=cost/max(baseline,1e-3),metrics=m,reference=b))
    result={}
    for label,rows in groups.items():
        ratio=np.array([r['cost_ratio'] for r in rows])
        result[label]=dict(complete=len(rows)==expected,cases=len(rows),
            score=float(.5*(ratio.mean()+ratio.max())),mean_ratio=float(ratio.mean()),
            worst_ratio=float(ratio.max()),rows=sorted(rows,key=lambda r:(r['rate'],r['seed'])))
    (args.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for label,row in sorted(result.items(),key=lambda x:x[1]['score']):
        print(label,{k:v for k,v in row.items() if k!='rows'})


if __name__=='__main__':main()
