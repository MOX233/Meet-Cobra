#!/usr/bin/env python3
"""Paired, frozen-checkpoint pilot study of prediction-report MTS."""
import argparse
import concurrent.futures
import csv
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import revision_grid as grid
from utils.mts_report_sim import run_sim_mts_report
import numpy as np
import torch

OUTPUT = ROOT / 'experiment/results/mts_report_20260918'
RATES = (1, 13, 23, 29)
CODE = ('utils/mts_report.py', 'utils/mts_report_sim.py',
        'test_mts_report.py', 'experiment/mts_report_experiment.py')
REFERENCES = ('meet_cobra', 'mts', 'oracle_mc')


def prepare():
    frozen, sha = grid.validate_protocol()
    assert grid.digest(grid.CACHE) == frozen['frontend']['test']['cache_sha256']
    protocol = dict(version=1, rollback_tag='pre-mts-report-20260918',
        rates_mbps=list(RATES), seeds=[1], reference_protocol_sha256=sha,
        code_sha256={p:grid.digest(ROOT/p) for p in CODE},
        prediction_cache_sha256=grid.digest(grid.CACHE),
        frontend=frozen['frontend']['frontend'],
        config=dataclasses.asdict(dataclasses.replace(grid.candidate_configs()['pressure_early'],ho_interruption_ms=10)),
        evaluation=dict(service_frames=300,warmup_frames=2,interval_s=[800,830],
            frame_ms=100,slot_ms=1,ho_ms=10,latency_ms=20,fading='paired FP64 GPU Rician'),
        information=dict(matching='position, predicted optimal/interference gains, queues, mean arrival rates, estimated load, association',
            beam='five ordered predicted candidates, paid measurements at first available slot, fixed for remaining frame',
            load='predicted interference and predicted desired gains; last serving measurement replaces serving optimum when BS unchanged',
            slot_ra='measured serving beam and same physical interference convention as frozen MEET/MTS',
            forbidden='no full H or oracle-best-pair lookup in matching controller'),
        timing=dict(association_period_frames=5,beam_period_frames=1,ra_period_slots=1,
            matching='report[x] drives association applied in x+1',
            beams='report[x] candidate list is probed in x+1, after interruption if applicable',
            reference_caveat='frozen MEET BF uses cache[x] during x, although prediction target is x+1; reference not modified'),
        overhead=dict(candidates=5,tracking_pilots=1,
            per_micro_frame_without_ho=104,per_micro_frame_with_ho=94,
            explanation='5 at first active slot; 1 in each other active slot; zero during HO'),
        interpretation='Preliminary one-seed comparison, not an isolated information ablation; no tuning on these points.')
    path=OUTPUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Protocol changed; use a new output directory instead of overwriting results')
    grid.write_json(path,protocol)
    print('PROTOCOL READY',grid.digest(path),flush=True)


def validate():
    _,reference_sha=grid.validate_protocol()
    path=OUTPUT/'protocol.json'
    protocol=json.loads(path.read_text())
    assert protocol['reference_protocol_sha256']==reference_sha
    for name,sha in protocol['code_sha256'].items():
        assert grid.digest(ROOT/name)==sha, name
    assert grid.digest(grid.CACHE)==protocol['prediction_cache_sha256']
    return protocol,grid.digest(path)


def run_case(rate,gpu):
    protocol,sha=validate()
    assert rate in RATES
    seed=1
    key=f'mts_report_rate{rate}_seed{seed}'
    destination=OUTPUT/'raw'/f'{key}.npz'
    saved=OUTPUT/'runs'/f'{key}.json'
    if saved.exists():
        row=json.loads(saved.read_text())
        assert row['protocol_sha256']==sha and grid.digest(destination)==row['raw_sha256']
        return row
    torch.set_num_threads(1)
    grid.shared.single_thread_solvers()
    timeline=grid.shared.temporal_slice(grid.shared.read_pickle(grid.CACHE),800,830)
    assert len(timeline)==301
    args=grid.shared.paper_args(rate*1e6)
    args.device=torch.device('cpu')
    traffic=grid.make_paired_traffic(args,timeline,seed)
    reference_hashes={}
    for method in REFERENCES:
        path=grid.OUTPUT/'runs'/f'{method}_rate{rate}_seed{seed}.json'
        reference=json.loads(path.read_text())
        assert reference['traffic_sha256']==traffic['sha256']
        assert reference['protocol_sha256']==protocol['reference_protocol_sha256']
        reference_hashes[str(path.relative_to(ROOT))]=grid.digest(path)
    np.random.seed(seed)
    diagnostics=[]
    start=time.monotonic()
    print('START',key,'GPU',gpu,flush=True)
    result=run_sim_mts_report(args,grid.shared.MICRO_BS_LOCATIONS,timeline,
        grid.candidate_configs()['pressure_early'],traffic,seed=seed,physics_device=f'cuda:{gpu}',
        diagnostics=diagnostics,
        progress_callback=lambda n,total:print('FRAME',n,total,flush=True) if n%50==0 else None)
    metrics,raw=grid.extract(args,result,diagnostics,traffic)
    for name in ('beam_switch_record','proposal_record','unassigned_record',
                 'association_epoch_record','optimizer_time_record','full_sweep_record','local_sweep_record'):
        raw[name]=getattr(result,name)
    assert len(diagnostics)==300 and result.full_sweep_record.sum()==0
    assert sum(d['blocked_vehicle_slots'] for d in diagnostics)==10*result.handover_record.sum()
    for fi,d in enumerate(diagnostics):
        micro=sum(bs>0 for bs in d['association'].values())
        micro_ho=sum(d['association'][v]>0 for v in d['switched'])
        expected=104*micro-10*micro_ho
        np.testing.assert_allclose(result.pilot_record[fi],expected/d['active_vehicle_slots'],atol=1e-12)
        assert d['total_probe_count']==5*micro
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp,**raw)
    os.replace(tmp,destination)
    diagpath=OUTPUT/'diagnostics'/f'{key}.json'
    grid.write_json(diagpath,diagnostics)
    row=dict(method='mts_report',label='MTS-GS-HBF-adapted (prediction reports)',rate_mbps=rate,seed=seed,
        gpu=gpu,frames=300,retained_frames=298,metrics=metrics,traffic_sha256=traffic['sha256'],
        protocol_sha256=sha,raw_sha256=grid.digest(destination),diagnostics_sha256=grid.digest(diagpath),
        reference_sha256=reference_hashes,elapsed_s=time.monotonic()-start,
        diagnostics=dict(full_sweeps=int(result.full_sweep_record.sum()),
            candidate_probes=sum(d['total_probe_count'] for d in diagnostics),
            matching_epochs=int(result.association_epoch_record.sum()),
            unassigned_at_matching=int(result.unassigned_record.sum()),
            max_estimated_capacity_overflow=float(result.optimizer_overflow_record.max())))
    grid.write_json(saved,row)
    print('DONE',key,json.dumps(metrics),flush=True)
    return row


def summarize():
    _,sha=validate()
    rows=[]
    for rate in RATES:
        new=json.loads((OUTPUT/'runs'/f'mts_report_rate{rate}_seed1.json').read_text())
        assert new['protocol_sha256']==sha
        for method in (*REFERENCES,'mts_report'):
            base=OUTPUT if method=='mts_report' else grid.OUTPUT
            path=base/'runs'/f'{method}_rate{rate}_seed1.json'
            row=json.loads(path.read_text())
            assert row['traffic_sha256']==new['traffic_sha256']
            if method!='mts_report':
                assert grid.digest(path)==new['reference_sha256'][str(path.relative_to(ROOT))]
            assert grid.digest(base/'raw'/f'{method}_rate{rate}_seed1.npz')==row['raw_sha256']
            rows.append(row)
    grid.write_json(OUTPUT/'comparison.json',dict(protocol_sha256=sha,rows=rows))
    columns=['rate_mbps','seed','method',*rows[0]['metrics'].keys()]
    with (OUTPUT/'comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k:r[k] for k in columns[:3]},**r['metrics']})
            print(r['rate_mbps'],r['method'],json.dumps(r['metrics']),flush=True)


def queue(gpus):
    validate()
    if len(gpus)!=len(RATES) or len(set(gpus))!=len(gpus):
        raise ValueError('Supply four distinct idle GPUs for this four-point pilot study')
    (OUTPUT/'logs').mkdir(exist_ok=True)
    def worker(rate,gpu):
        path=OUTPUT/'logs'/f'mts_report_rate{rate}_seed1.log'
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'case','--rate',str(rate),'--gpu',str(gpu)]
        with path.open('a') as log:
            completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONHASHSEED='0'))
        print('CASE',rate,'EXIT',completed.returncode,flush=True)
        return completed.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        codes=list(executor.map(worker,RATES,gpus))
    if any(codes):
        raise RuntimeError(f'Failed cases: {codes}; inspect separate logs')
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','case','queue','summarize'])
    parser.add_argument('--rate',type=int,choices=RATES)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='case': run_case(args.rate,args.gpu)
    elif args.command=='queue': queue(args.gpus)
    else: summarize()
