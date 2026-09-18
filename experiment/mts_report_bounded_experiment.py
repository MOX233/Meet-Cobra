#!/usr/bin/env python3
"""Additional diagnostic: bounded physical occupancy, unchanged MTS scores."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import mts_report_experiment as base
import utils.mts_report_sim as simulator
from utils.mts_report_bounded import estimate_bounded_report_load

OUTPUT=ROOT/'experiment/results/mts_report_bounded_20260918'
ADDED=('utils/mts_report_bounded.py','test_mts_report_bounded.py',
       'experiment/mts_report_bounded_experiment.py')


def prepare():
    protocol,sha=base.validate()
    protocol['parent_experiment_protocol_sha256']=sha
    protocol['variant']='capacity-bounded occupancy feedback'
    protocol['code_sha256'].update({p:base.grid.digest(ROOT/p) for p in ADDED})
    protocol['information']['load']='Same predicted and historical measured gains; ten-step load estimate with physical occupancy bounded at every step. RA uses bounded occupancy; GS preserves demand/capacity overload cue clipped to 1.5.'
    protocol['interpretation']='Additional numerical-robustness diagnostic after observing unbounded RB estimates; no retuning of NN or MTS preference weights. Parent results preserved.'
    path=OUTPUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError('Protocol changed; do not overwrite existing results')
    base.grid.write_json(path,protocol)
    print('PROTOCOL READY',base.grid.digest(path),flush=True)


def activate():
    # Process-local dependency injection keeps every frozen source/result intact.
    # Each worker is a separate Python process; no concurrent in-process runs.
    base.OUTPUT=OUTPUT
    simulator.estimate_report_load=estimate_bounded_report_load


def run_case(rate,gpu):
    activate()
    row=base.run_case(rate,gpu)
    row['variant']='capacity-bounded occupancy feedback'
    row['label']='MTS-GS-HBF-adapted (prediction reports; bounded occupancy)'
    base.grid.write_json(OUTPUT/'runs'/f'mts_report_rate{rate}_seed1.json',row)


def queue(gpus):
    activate()
    base.validate()
    assert len(gpus)==4 and len(set(gpus))==4
    (OUTPUT/'logs').mkdir(exist_ok=True)
    def worker(rate,gpu):
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'case','--rate',str(rate),'--gpu',str(gpu)]
        with (OUTPUT/'logs'/f'mts_report_rate{rate}_seed1.log').open('a') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONHASHSEED='0'))
        print('CASE',rate,'EXIT',result.returncode,flush=True)
        return result.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        codes=list(executor.map(worker,base.RATES,gpus))
    if any(codes): raise RuntimeError(str(codes))
    base.summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','case','queue','summarize'])
    parser.add_argument('--rate',type=int,choices=base.RATES)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='case': run_case(args.rate,args.gpu)
    elif args.command=='queue': queue(args.gpus)
    else:
        activate()
        base.summarize()
