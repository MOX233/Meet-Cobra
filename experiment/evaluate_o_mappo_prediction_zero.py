#!/usr/bin/env python3
"""Retain the unfinetuned prediction-input control, without test-set selection.

This supplementary protocol is declared while training is still in progress.
It cannot change the frozen training run or its selected positive-update model.
"""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.train_o_mappo_predicted_cross5 import DEFAULT,RATES,grid,old
from experiment.revision_pipeline import atomic_json,digest
import numpy as np


def run(root,devices):
    policy=root/'training/seed11/round0000.pt'
    protocol=dict(policy=str(policy),policy_sha256=digest(policy),rates=RATES,
        validation_interval=[710,720],validation_seed=101,test_interval=[800,830],test_seeds=[1,2,3],
        purpose='Unfinetuned prediction-input reference; never selects a checkpoint using test results',
        selection='Compare zero and trained candidates only with the same fixed all-load validation criterion')
    target=root/'zero_control_protocol.json'
    if target.exists() and old.read(target)!=protocol:raise ValueError('Zero-control protocol changed')
    atomic_json(target,protocol)
    policies={'predicted_zero':policy}
    rows=grid(root/'zero_selection',policies,RATES,[101],'selection',devices)
    reference=np.array(old.read(root/'validation_reference.json')['costs'])
    rows=sorted(rows,key=lambda r:r['rate'])
    ratios=np.array([r['metrics']['violation_percent']+.02*r['metrics']['power_w'] for r in rows])/reference
    atomic_json(root/'zero_selection_score.json',dict(policy=str(policy),policy_sha256=digest(policy),
        score=float(.5*(ratios.mean()+ratios.max())),criterion='same score as trained candidate selection'))
    grid(root/'zero_test',policies,RATES,[1,2,3],'test',devices)
    print('UNFINETUNED CONTROL COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','launch'])
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    a=p.parse_args()
    if a.command=='run':run(a.root,a.devices.split(','))
    else:
        argv=[sys.executable,'-u',str(Path(__file__)),'run','--root',str(a.root),'--devices',a.devices]
        with (a.root/'zero_control.log').open('a') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,PYTHONHASHSEED='0'))
        atomic_json(a.root/'zero_control_launch.json',dict(pid=child.pid,argv=argv,started_unix=time.time()))
        print('LAUNCHED UNFINETUNED CONTROL',child.pid,flush=True)


if __name__=='__main__':main()
