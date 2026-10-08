#!/usr/bin/env python3
"""Evaluate selected R1 baselines beside a NEW six-method main grid.

Delegates algorithms to the existing final simulators. Does not retrain/select
policies, reuse pilot cases, alter old protocols, or substitute legacy baselines.
"""
import argparse
import concurrent.futures
from contextlib import ExitStack
import csv
import fcntl
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from experiment import revision_pipeline as grid

METHODS = ('o_mappo', 'mts_report')
POLICY = ROOT/'experiment/results/o_mappo_predicted_cross5_20260925/training/seed11/best_positive.pt'


def validate(root):
    p = grid.read(root/'protocol.json')
    for name, checksum in p['code'].items():
        if grid.digest(ROOT/name) != checksum:
            raise ValueError(f'Source changed: {name}; use a new run root')
    base, sha = grid.validate(Path(p['base_grid']))
    assert sha == p['base_protocol_sha256']
    assert grid.digest(p['policy']) == p['policy_sha256']
    return p, base


def prepare(args):
    root = args.root.resolve()
    if root.parent != ROOT/'experiment/results' or not root.name.startswith('r1_baselines_'):
        raise ValueError('Use experiment/results/r1_baselines_NAME')
    base, checksum = grid.validate(args.base_grid)
    required = set(grid.METHODS)-set(METHODS)
    if not required <= set(base['methods']):
        raise ValueError('Main grid must contain all six non-literature methods')
    sources = [Path(__file__), *sorted((ROOT/'utils').glob('*.py')), *sorted((ROOT/'experiment').glob('*.py'))]
    p = dict(base_grid=str(args.base_grid.resolve()), base_protocol_sha256=checksum,
             cache=base['cache'], policy=str(args.policy.resolve()), policy_sha256=grid.digest(args.policy),
             methods=METHODS, rates=base['rates'], seeds=base['seeds'], warmup_frames=base['warmup_frames'],
             code={str(path.relative_to(ROOT)):grid.digest(path) for path in sources},
             scope='Selected R1 policies, prediction-only association, H32/cross5, physical per-RB service')
    dest = root/'protocol.json'
    if dest.exists():
        assert grid.read(dest) == grid.native(p), 'Run configuration changed'
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Nonempty unversioned output directory')
        grid.atomic_json(dest, p)


def case(args):
    p, base = validate(args.root)
    assert args.method in METHODS and args.rate in p['rates'] and args.seed in p['seeds']
    assert args.device.split(':')[0] == base['backend']
    name = grid.key(args.method, args.rate, args.seed)
    checksum = grid.digest(args.root/'protocol.json')
    (args.root/'locks').mkdir(exist_ok=True)
    with (args.root/'locks'/f'{name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if grid.completed(args.root, name, checksum):
            print('SKIP', name, flush=True)
            return
        import torch
        from experiment import mts_hierarchical_tracking as mts
        from experiment import o_mappo_predicted_cross5 as predicted
        from utils.o_mappo import OMAPPPolicy
        from utils.compiled_matching import km_algorithm_compiled
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(1)
        mts.shared.single_thread_solvers()
        mts.alg.km_algorithm = km_algorithm_compiled
        km_algorithm_compiled(np.zeros((2,2)))
        with Path(p['cache']).open('rb') as stream:
            timeline = pickle.load(stream)
        params = mts.shared.paper_args(args.rate*1e6)
        params.device = torch.device('cpu')
        def progress(n, total):
            grid.atomic_json(args.root/'progress'/f'{name}.json', dict(frame=n,total=total,pid=os.getpid()))
        started = time.monotonic()
        if args.method == 'o_mappo':
            policy = OMAPPPolicy.load(p['policy'])
            result, traffic, _, diagnostics = predicted.simulate(timeline, policy, args.rate, args.seed,
                                                                 args.device, learn=False, progress=progress)
            diagnostics.pop('probes')
        else:
            traffic = mts.make_paired_traffic(params, timeline, args.seed)
            np.random.seed(args.seed)
            adapter, service = mts.BeamAdapter('hier32_cross5'), []
            with ExitStack() as stack:
                adapter.activate(stack)
                result = mts.sim.run_sim_mts_report(params, mts.shared.MICRO_BS_LOCATIONS, timeline,
                    mts.candidate_configs()['pressure_early'], traffic, seed=args.seed, physics_device=args.device,
                    ho_interruption_ms=10, k=5, directional_service=True, predicted_ra_interference=True,
                    service_diagnostics=service, progress_callback=progress)
            diagnostics = dict(beam=adapter.frames, service=service)
        metrics, raw = grid.extract(params, result, traffic, p['warmup_frames'])
        for folder in ('raw','diagnostics','runs'):
            (args.root/folder).mkdir(exist_ok=True)
        raw_path, diag_path = args.root/'raw'/f'{name}.npz', args.root/'diagnostics'/f'{name}.json'
        temp = raw_path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **raw)
        os.replace(temp, raw_path)
        grid.atomic_json(diag_path, grid.native(diagnostics))
        grid.atomic_json(args.root/'runs'/f'{name}.json', dict(method=args.method,rate_mbps=args.rate,seed=args.seed,
            metrics=metrics,frames=len(result.energy_record),protocol_sha256=checksum,
            traffic_sha256=traffic['sha256'],raw_sha256=grid.digest(raw_path),diagnostics_sha256=grid.digest(diag_path),
            device=args.device,elapsed_s=time.monotonic()-started,comparison={}))
        print('COMPLETE', name, metrics, flush=True)


def run(args):
    p, base = validate(args.root)
    jobs = [(m,r,s) for m in METHODS for r in p['rates'] for s in p['seeds']]
    devices = args.devices.split(',')
    if not devices or any(d.split(':')[0] != base['backend'] for d in devices):
        raise ValueError('Device/backend mismatch')
    (args.root/'logs').mkdir(exist_ok=True)
    with (args.root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        def worker(device, subset):
            for m,r,s in subset:
                name = grid.key(m,r,s)
                if grid.completed(args.root,name,grid.digest(args.root/'protocol.json')):
                    print('SKIP',name,flush=True)
                    continue
                with (args.root/'logs'/f'{name}.log').open('a') as log:
                    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'case','--root',str(args.root),
                        '--method',m,'--rate',str(r),'--seed',str(s),'--device',device],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,PYTHONHASHSEED='0'))
                print('COMPLETE', name, flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker,d,jobs[j::len(devices)]) for j,d in enumerate(devices)]
            for future in futures:
                future.result()


def plot(args):
    p, base = validate(args.root)
    from experiment import plot_revision_system_results as plotting
    rows, curves = [], {}
    main = Path(p['base_grid'])
    for method in plotting.METHODS:
        directory = args.root if method in METHODS else main
        checksum = grid.digest(directory/'protocol.json')
        for rate in p['rates']:
            for seed in p['seeds']:
                name = grid.key(method,rate,seed)
                assert grid.completed(directory,name,checksum), name
                row = grid.read(directory/'runs'/f'{name}.json')
                ref = grid.read(main/'runs'/f'meet_cobra_rate{rate}_seed{seed}.json')
                assert row['traffic_sha256'] == ref['traffic_sha256'] and row['frames'] == ref['frames']
                rows.append(row)
        curves[method] = {metric:np.array([[next(x['metrics'][metric] for x in rows
            if x['method']==method and x['rate_mbps']==rate and x['seed']==seed) for seed in p['seeds']]
            for rate in p['rates']]) for metric in plotting.METRICS}
    # Never target the manuscript or submission figure directories.
    output = args.root/'plots'
    output.mkdir(exist_ok=True)
    with (output/'figure_data.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method','rate_mbps','metric','mean','seed_min','seed_max'])
        for m in plotting.METHODS:
            for metric in plotting.METRICS:
                for rate, values in zip(p['rates'],curves[m][metric]):
                    writer.writerow([m,rate,metric,values.mean(),values.min(),values.max()])
    plotting.plots(curves, {}, p['rates'], output)
    grid.atomic_json(output/'manifest.json',dict(cases=len(rows),rates=p['rates'],seeds=p['seeds'],
        smoke=base['cache_manifest']['smoke'],frames=rows[0]['frames'],paired_traffic_verified=True,
        baseline_protocol_sha256=grid.digest(args.root/'protocol.json'),base_protocol_sha256=p['base_protocol_sha256']))
    print('PLOTS', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare','run','case','status','plot'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base-grid',type=Path)
    parser.add_argument('--policy',type=Path,default=POLICY)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--rate',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--devices',default='cuda:0')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.phase == 'prepare':
        if args.base_grid is None:
            parser.error('--base-grid is required for prepare')
        prepare(args)
    elif args.phase == 'case':
        case(args)
    elif args.phase == 'run':
        run(args)
    elif args.phase == 'plot':
        plot(args)
    else:
        p = grid.read(args.root/'protocol.json')
        sha = grid.digest(args.root/'protocol.json')
        jobs = [grid.key(m,r,s) for m in METHODS for r in p['rates'] for s in p['seeds']]
        print(dict(completed=sum(grid.completed(args.root,n,sha) for n in jobs),expected=len(jobs)))


if __name__ == '__main__':
    main()
