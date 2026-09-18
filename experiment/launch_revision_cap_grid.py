#!/usr/bin/env python3
"""Detached, bounded 450-case manager with preflight and final raw audits."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'experiment/results/revision_cap_mts_full_20260918_v2'
RERUN=['meet_cobra','oracle_mc','wo_pet_bf','wo_otr_ra','mts_report']


def write_json(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)


def progress():
    counts={method:0 for method in RERUN}
    for method in RERUN:
        counts[method]=sum((OUTPUT/'runs'/f'{method}_rate{rate}_seed{seed}.json').exists()
                          for rate in range(1,36,2) for seed in (1,2,3,4,5))
    return dict(completed_new=sum(counts.values()),expected_new=450,by_method=counts)


def status(phase,**kwargs):
    row=dict(phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**progress(),**kwargs)
    write_json(OUTPUT/'pipeline_status.json',row)
    print(json.dumps(row),flush=True)


def command(script,*args):
    return [sys.executable,'-u',str(ROOT/'experiment'/script),*args]


def run_stage(phase,cmd):
    with (OUTPUT/f'{phase}.log').open('a') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONHASHSEED='0'))
        while process.poll() is None:
            status(phase,child_pid=process.pid)
            time.sleep(30)
    if process.returncode:
        raise RuntimeError(f'{phase} failed ({process.returncode}); inspect {phase}.log')


def run(args):
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            common=['--gpus',args.gpus,'--workers-per-gpu',str(args.workers_per_gpu)]
            run_stage('control_check',command('revision_cap_grid.py','case','--method','wo_gap_ho',
                '--rate','13','--seed','1','--gpu',args.gpus.split(',')[0],'--control'))
            run_stage('preflight',command('revision_cap_grid.py','queue','--rates','1,13,35','--seeds','1',*common))
            run_stage('preflight_audit',command('summarize_revision_cap_grid.py','--preflight'))
            run_stage('reference_import',command('revision_cap_grid.py','import-unaffected'))
            run_stage('full_grid',command('revision_cap_grid.py','queue',*common))
            run_stage('final_audit',command('summarize_revision_cap_grid.py'))
            status('completed',new_cases=450,reused_cases=180,
                note='Results and statistics only; manuscript and figures unchanged; timing/common-RA caveats retained in protocol.')
        except Exception:
            status('failed',error=traceback.format_exc())
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--gpus',default='1,5,6')
    parser.add_argument('--workers-per-gpu',type=int,choices=[1,2],default=2)
    args=parser.parse_args()
    if args.worker:
        run(args)
    else:
        if not (OUTPUT/'protocol.json').exists(): raise RuntimeError('Prepare and freeze protocol first')
        with (OUTPUT/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
                 '--gpus',args.gpus,'--workers-per-gpu',str(args.workers_per_gpu)]
            with (OUTPUT/'pipeline.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONHASHSEED='0'))
        manifest=dict(pid=child.pid,command=cmd,launched_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUTPUT/'pipeline_launch.json',manifest)
        print(json.dumps(manifest,indent=2),flush=True)
