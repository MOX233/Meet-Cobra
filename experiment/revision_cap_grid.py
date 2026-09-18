#!/usr/bin/env python3
"""Versioned full-load rerun after GAP usage saturation, plus report-input MTS."""
import argparse
import ast
import concurrent.futures
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue as queue_module
import shutil
import subprocess
import sys
import threading
import time

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import revision_grid as old
from utils.gap_refinement import GAPRefinementConfig
from utils.mts_report_bounded import estimate_bounded_report_load
import utils.mts_report_sim as mts_sim
import numpy as np
import torch

OUTPUT=ROOT/'experiment/results/revision_cap_mts_full_20260918'
RATES=list(range(1,36,2))
SEEDS=[1,2,3,4,5]
RERUN=['meet_cobra','oracle_mc','wo_pet_bf','wo_otr_ra','mts_report']
REUSE=['wo_gap_ho','o_mappo']
METHODS=['meet_cobra','oracle_mc','wo_gap_ho','wo_pet_bf','wo_otr_ra','o_mappo','mts_report']
LABELS=dict(zip(old.METHODS,old.LABELS))|{'mts_report':'MTS-GS-HBF-adapted (prediction reports)'}
PREFLIGHT_RATES=[1,13,35]
PILOT=ROOT/'experiment/results/mts_report_bounded_20260918'
ROLLBACK='pre-full-cap-mts-grid-20260918'
CODE=sorted(set(old.CODE)|{str(p.relative_to(ROOT)) for p in (ROOT/'utils').glob('*.py')}|
    {'experiment/revision_cap_grid.py','experiment/launch_revision_cap_grid.py',
     'experiment/summarize_revision_cap_grid.py','test_revision_cap_grid.py',
     'experiment/pql_ba_experiment.py','experiment/o_mappo_shared_frontend.py',
     'experiment/ho_interruption_experiment.py','experiment/summarize_revision_grid.py'})


def reference_row(method,rate,seed,check_raw=False):
    path=old.OUTPUT/'runs'/f'{method}_rate{rate}_seed{seed}.json'
    row=json.loads(path.read_text())
    assert row['protocol_sha256']==old.digest(old.OUTPUT/'protocol.json')
    raw=old.OUTPUT/'raw'/f'{method}_rate{rate}_seed{seed}.npz'
    if check_raw:
        assert old.digest(raw)==row['raw_sha256']
    return row,path,raw


def audit_unaffected_sources():
    """Allow only the documented GAP changes in previously shared source files."""
    frozen=json.loads((old.OUTPUT/'protocol.json').read_text())
    exceptions={'utils/alg_utils.py','utils/sim_utils.py','utils/gap_refinement.py'}
    for name,sha in frozen['code_sha256'].items():
        if name not in exceptions:
            assert old.digest(ROOT/name)==sha,name
    def prior(name):
        text=subprocess.check_output(['git','show',f'pre-gap-occupancy-cap-20260918:{name}'],cwd=ROOT,text=True)
        assert hashlib.sha256(text.encode()).hexdigest()==frozen['code_sha256'][name]
        return text
    def without_gap(text):
        tree=ast.parse(text)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and
            n.name=='HO_EE_GAP_APX_SINR_conservative_adaptive')]
        return ast.dump(tree,include_attributes=False)
    assert without_gap(prior('utils/alg_utils.py'))==without_gap((ROOT/'utils/alg_utils.py').read_text())
    forwarding="        if 'gap_cap_rb_usage' in kwargs:\n            ho_options['gap_cap_rb_usage'] = kwargs['gap_cap_rb_usage']\n"
    current=(ROOT/'utils/sim_utils.py').read_text()
    assert current.count(forwarding)==1
    assert current.replace(forwarding,'')==prior('utils/sim_utils.py')
    for method_file in ('utils/o_mappo.py','utils/o_mappo_sim.py'):
        assert 'HO_EE_GAP_APX_SINR_conservative_adaptive' not in (ROOT/method_file).read_text()
    return dict(unchanged_methods=REUSE,
        verification='All frozen files unchanged except documented GAP function/refinement and optional simulator kwarg forwarding; no GAP call in O-MAPPO; w/o GAP-HO uses unchanged greedy function.')


def prepare():
    audit=audit_unaffected_sources()
    if shutil.disk_usage(OUTPUT.parent).free < 25*2**30:
        raise RuntimeError('Need at least 25 GiB free for this retained-raw grid')
    protocol=old.protocol()
    protocol.update(version=2,rollback_git='c5ea7cf',rollback_tag=ROLLBACK,
        methods=METHODS,rerun_methods=RERUN,reuse_methods=REUSE,
        expected_new_cases=450,expected_reused_cases=180,rates=RATES,seeds=SEEDS,
        preflight=dict(rates=PREFLIGHT_RATES,seeds=[1],methods=RERUN),
        gap_cap_rb_usage=True,
        old_protocol_sha256=old.digest(old.OUTPUT/'protocol.json'),
        mts_pilot_protocol_sha256=old.digest(PILOT/'protocol.json'),
        mts_information='Shared frozen predictions; target/serving BS only, five paid probes per frame; original GS weights; bounded occupied-RB estimates with separate demand cue.',
        mts_load_estimator='utils/mts_report_bounded.py:estimate_bounded_report_load',
        prediction_timing='MEET retains existing BF cache[x] in x; MTS probes report[x-1] in x. No unapproved timing change; not a strict timing-controlled ablation.',
        common_ra_estimator='Unchanged for MEET and original baselines; only GAP feedback has been capped. MTS uses its separately confirmed bounded estimator.',
        unaffected_source_audit=audit,
        reference_status='Reactive-OBRA input and Oracle-CR-LB definition remain outside this run; original true-CSI MTS retained only as historical comparison.',
        code_sha256={name:old.digest(ROOT/name) for name in CODE})
    destination=OUTPUT/'protocol.json'
    if destination.exists() and json.loads(destination.read_text())!=protocol:
        raise RuntimeError('Changed protocol: preserve this directory and explicitly version the new run')
    old.write_json(destination,protocol)
    print('PROTOCOL READY',old.digest(destination),flush=True)


def validate(check_inputs=False):
    path=OUTPUT/'protocol.json'
    protocol=json.loads(path.read_text())
    for name,sha in protocol['code_sha256'].items():
        assert old.digest(ROOT/name)==sha,f'Code changed during run: {name}'
    assert protocol['old_protocol_sha256']==old.digest(old.OUTPUT/'protocol.json')
    assert protocol['mts_pilot_protocol_sha256']==old.digest(PILOT/'protocol.json')
    if check_inputs:
        assert old.digest(old.CACHE)==protocol['frontend']['test']['cache_sha256']
        source=protocol['frontend']['test']
        assert old.digest(source['source'])==source['source_sha256']
        for model in protocol['frontend']['frontend'].values():
            assert old.digest(model['checkpoint'])==model['sha256']
        assert old.digest(old.shared.LEGACY)==protocol['legacy_policy']['sha256']
    return protocol,old.digest(path)


class GapAudit(list):
    def append(self,record):
        if record['iterations']:
            caps=np.asarray(record['physical_capacity'])
            assert record['cap_rb_usage'] and record['iterations']==2
            for step in record['traces']:
                demand=np.asarray(step['implied_demand'])
                np.testing.assert_array_equal(step['implied_load'],np.minimum(demand,caps))
                assert (np.asarray(step['input_load'])<=caps).all()
            assert (np.asarray(record['post_repair_frame_average_load'])<=caps).all()
        super().append(record)


def simulate(method,rate,seed,gpu):
    torch.set_num_threads(1)
    old.shared.single_thread_solvers()
    from utils.compiled_matching import km_algorithm_compiled
    old.alg.km_algorithm=km_algorithm_compiled
    km_algorithm_compiled(np.zeros((2,2)))
    timeline=old.shared.temporal_slice(old.shared.read_pickle(old.CACHE),800,830)
    assert len(timeline)==301
    args=old.shared.paper_args(rate*1e6)
    args.device=torch.device('cpu')
    traffic=old.make_paired_traffic(args,timeline,seed)
    ref,refpath,_=reference_row('mts' if method=='mts_report' else method,rate,seed)
    assert traffic['sha256']==ref['traffic_sha256']
    np.random.seed(seed)
    diag=old.Diagnostics()
    gap=GapAudit()
    if method=='mts_report':
        # Worker-local injection; no other method runs in this child process.
        mts_sim.estimate_report_load=estimate_bounded_report_load
        result=mts_sim.run_sim_mts_report(args,old.shared.MICRO_BS_LOCATIONS,timeline,
            old.candidate_configs()['pressure_early'],traffic,seed=seed,physics_device=f'cuda:{gpu}',
            ho_interruption_ms=10,k=5,diagnostics=diag)
        caps=np.array([args.num_RB_macro]+[args.num_RB_micro]*4)
        for i,d in enumerate(diag):
            assert (np.array(d['estimated_rb'])<=caps).all()
            micro=sum(bs>0 for bs in d['association'].values())
            micro_ho=sum(d['association'][v]>0 for v in d['switched'])
            np.testing.assert_allclose(result.pilot_record[i],(104*micro-10*micro_ho)/d['active_vehicle_slots'],atol=1e-12)
            assert d['total_probe_count']==5*micro
        assert result.full_sweep_record.sum()==0
    else:
        cache={f:{v:r['shared_prediction'] for v,r in records.items()} for f,records in timeline.items()}
        oracle=method=='oracle_mc'
        random_bf=method=='wo_pet_bf'
        ho=old.alg.HO_EE_Greedy_offload if method=='wo_gap_ho' else old.alg.HO_EE_GAP_APX_SINR_conservative_adaptive
        ra=old.alg.RA_OTR3_SINR if method=='wo_otr_ra' else old.alg.RA_OTR_SINR
        extra=dict(oracle_ho_cache=old.oracle_cache(args,old.shared.MICRO_BS_LOCATIONS,timeline),
                   oracle_hold_macro_position=True) if oracle else {}
        result=old.run_sim_withUMa(args,old.shared.MICRO_BS_LOCATIONS,timeline,None,
            None if oracle else True,None if oracle else True,None if oracle else True,
            HO_func=ho,RA_func=ra,save_pilot=not random_bf,
            BF_func='topKbeam_NoPred' if random_bf else 'topKbeam_savePilot',K_BF=5,
            prediction_cache=None if oracle else cache,ho_capacity_correction=True,gap_cap_rb_usage=True,
            gap_refinement_config=GAPRefinementConfig(max_iterations=2,tolerance_rb=None,relaxation_factor=1.1),
            gap_refinement_diagnostics=gap,vectorized_pet_measurement=not random_bf,
            correct_random_beam_index=True,ho_diagnostics=diag,
            prt=False,rician_fading=True,traffic_trace=traffic,ho_interruption_ms=10,
            paired_fading_seed=seed,physics_device=f'cuda:{gpu}',**extra)
        assert len(gap)==(0 if method=='wo_gap_ho' else 300)
    metrics,raw=old.extract(args,result,diag,traffic)
    assert len(diag)==300
    assert sum(d['blocked_vehicle_slots'] for d in diag)==10*sum(raw['handover_count'])
    if method=='mts_report':
        for name in ('proposal_record','unassigned_record','full_sweep_record','local_sweep_record',
                     'association_epoch_record','optimizer_time_record','beam_switch_record'):
            raw[name]=getattr(result,name)
    return metrics,raw,dict(ho=diag,gap=gap),traffic,refpath


def run_case(method,rate,seed,gpu,control=False):
    protocol,sha=validate(check_inputs=True)
    assert method in (REUSE if control else RERUN) and rate in RATES and seed in SEEDS
    if control:
        assert method=='wo_gap_ho'
    prefix=OUTPUT/'control' if control else OUTPUT
    key=f'{method}_rate{rate}_seed{seed}'
    saved=prefix/'runs'/f'{key}.json'
    rawpath=prefix/'raw'/f'{key}.npz'
    (OUTPUT/'locks').mkdir(exist_ok=True)
    with (OUTPUT/'locks'/f'{"control_" if control else ""}{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if saved.exists():
            row=json.loads(saved.read_text())
            assert row['protocol_sha256']==sha and old.digest(rawpath)==row['raw_sha256']
            return row
        if shutil.disk_usage(OUTPUT).free < 8*2**30:
            raise RuntimeError('Disk reserve below 8 GiB; stop without deleting results')
        started=time.monotonic()
        print('START',key,'gpu',gpu,flush=True)
        metrics,raw,diagnostics,traffic,refpath=simulate(method,rate,seed,gpu)
        if control:
            previous,_,reference_raw=reference_row(method,rate,seed,True)
            with np.load(reference_raw,allow_pickle=False) as old_raw:
                assert set(raw)==set(old_raw.files)
                for name in raw:
                    np.testing.assert_array_equal(raw[name],old_raw[name])
            assert metrics==previous['metrics']
        if method=='mts_report' and seed==1 and rate in (1,13,23,29):
            pilot=json.loads((PILOT/'runs'/f'mts_report_rate{rate}_seed1.json').read_text())
            assert pilot['traffic_sha256']==traffic['sha256']
            np.testing.assert_allclose(list(metrics.values()),[pilot['metrics'][k] for k in metrics],rtol=1e-12,atol=1e-12)
        rawpath.parent.mkdir(parents=True,exist_ok=True)
        tmp=rawpath.with_suffix(f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(tmp,**raw)
        os.replace(tmp,rawpath)
        diagpath=prefix/'diagnostics'/f'{key}.json'
        old.write_json(diagpath,diagnostics)
        row=dict(method=method,label=LABELS[method],rate_mbps=rate,seed=seed,gpu=gpu,
            frames=300,retained_frames=298,metrics=metrics,traffic_sha256=traffic['sha256'],
            protocol_sha256=sha,method_configuration_sha256=None,raw_sha256=old.digest(rawpath),
            diagnostics_sha256=old.digest(diagpath),origin='control_rerun' if control else 'new_run',
            previous_reference=dict(path=str(refpath.relative_to(ROOT)),sha256=old.digest(refpath)),
            elapsed_s=time.monotonic()-started)
        old.write_json(saved,row)
        print('DONE',key,json.dumps(metrics),flush=True)
        return row


def queue(methods,rates,seeds,gpus,workers_per_gpu=2):
    _,sha=validate(check_inputs=True)
    assert set(methods)<=set(RERUN) and set(rates)<=set(RATES) and set(seeds)<=set(SEEDS)
    assert len(set(gpus))==len(gpus) and workers_per_gpu in (1,2)
    jobs=queue_module.Queue()
    for rate in rates:
        for seed in seeds:
            for method in methods:
                path=OUTPUT/'runs'/f'{method}_rate{rate}_seed{seed}.json'
                if path.exists():
                    row=json.loads(path.read_text())
                    assert row['protocol_sha256']==sha
                    assert old.digest(OUTPUT/'raw'/f'{path.stem}.npz')==row['raw_sha256']
                else:
                    jobs.put((method,rate,seed))
    failures=[]
    stop=threading.Event()
    (OUTPUT/'logs').mkdir(exist_ok=True)
    def worker(gpu):
        while not stop.is_set():
            try: method,rate,seed=jobs.get_nowait()
            except queue_module.Empty: return
            key=f'{method}_rate{rate}_seed{seed}'
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'case','--method',method,
                     '--rate',str(rate),'--seed',str(seed),'--gpu',str(gpu)]
            try:
                with (OUTPUT/'logs'/f'{key}.log').open('a') as log:
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,PYTHONHASHSEED='0'),timeout=3600)
                if result.returncode: raise RuntimeError(f'exit {result.returncode}')
            except Exception as error:
                failures.append(dict(key=key,error=str(error)))
                stop.set()
            print('QUEUE',key,'remaining',jobs.qsize(),'failures',len(failures),flush=True)
    with (OUTPUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        devices=[gpu for gpu in gpus for _ in range(workers_per_gpu)]
        with concurrent.futures.ThreadPoolExecutor(len(devices)) as executor:
            list(executor.map(worker,devices))
    old.write_json(OUTPUT/'queue_last_result.json',dict(methods=methods,rates=rates,seeds=seeds,
        failures=failures,unstarted=jobs.qsize(),protocol_sha256=sha))
    if failures: raise RuntimeError(f'Queue stopped after {len(failures)} failure(s); inspect logs')


def import_unaffected():
    _,sha=validate(check_inputs=True)
    audit_unaffected_sources()
    control=json.loads((OUTPUT/'control/runs/wo_gap_ho_rate13_seed1.json').read_text())
    assert control['protocol_sha256']==sha and control['origin']=='control_rerun'
    for method in REUSE:
        for rate in RATES:
            for seed in SEEDS:
                reference,source,rawsource=reference_row(method,rate,seed,True)
                rawtarget=OUTPUT/'raw'/rawsource.name
                rawtarget.parent.mkdir(exist_ok=True)
                if not rawtarget.exists(): os.link(rawsource,rawtarget)
                assert old.digest(rawtarget)==reference['raw_sha256']
                row=dict(reference,protocol_sha256=sha,origin='reused_unaffected',
                    previous_reference=dict(path=str(source.relative_to(ROOT)),sha256=old.digest(source)))
                destination=OUTPUT/'runs'/source.name
                if destination.exists(): assert json.loads(destination.read_text())==row
                else: old.write_json(destination,row)
    print('IMPORTED',len(REUSE)*len(RATES)*len(SEEDS),'unaffected reference cases',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','case','queue','import-unaffected'])
    parser.add_argument('--method',choices=METHODS,default='meet_cobra')
    parser.add_argument('--rate',type=int,default=13)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--methods',default=','.join(RERUN))
    parser.add_argument('--rates',default=','.join(map(str,RATES)))
    parser.add_argument('--seeds',default='1,2,3,4,5')
    parser.add_argument('--gpus',default='1,5,6')
    parser.add_argument('--workers-per-gpu',type=int,default=2)
    parser.add_argument('--control',action='store_true')
    cli=parser.parse_args()
    if cli.phase=='prepare': prepare()
    elif cli.phase=='case': run_case(cli.method,cli.rate,cli.seed,cli.gpu,cli.control)
    elif cli.phase=='import-unaffected': import_unaffected()
    else:
        queue(cli.methods.split(','),list(map(int,cli.rates.split(','))),
              list(map(int,cli.seeds.split(','))),list(map(int,cli.gpus.split(','))),cli.workers_per_gpu)
