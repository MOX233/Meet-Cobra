#!/usr/bin/env python3
"""R1C6: repeat the paired iteration study using the frozen directional grid.

No frozen simulator source or historical result is modified. A process-local
wrapper replaces only the GAPRefinementConfig supplied to each HO call.
Fixed-two runs must reproduce the corresponding published grid arrays exactly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from unittest.mock import patch

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiment import revision_pipeline as pipeline
from experiment.revision_training import atomic_json
from utils.gap_refinement import GAPRefinementConfig

DEFAULT_BASE = ROOT / 'experiment/results/revision_directional_20260922/grid'
DEFAULT_ROOT = ROOT / 'experiment/results/gap_refinement_directional_20260922'
CONFIGS = dict(fixed2=GAPRefinementConfig(2, None, 1.1),
               upto3=GAPRefinementConfig(3, .1, 1.1),
               upto5=GAPRefinementConfig(5, .1, 1.1))


@contextmanager
def iteration_override(config):
    """Change just the iteration settings and restore the original function."""
    from utils import alg_utils
    original = alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive

    def configured(*args, **kwargs):
        assert kwargs['gap_refinement_config'] == CONFIGS['fixed2']
        assert kwargs['gap_cap_rb_usage'] and kwargs['ho_capacity_correction']
        kwargs['gap_refinement_config'] = config
        return original(*args, **kwargs)

    with patch.object(alg_utils, 'HO_EE_GAP_APX_SINR_conservative_adaptive', configured):
        yield


def prepare(args):
    base, sha = pipeline.validate(args.base_grid)
    assert all(r in base['rates'] for r in (21, 27, 35))
    assert all(s in base['seeds'] for s in (1, 2, 3))
    document = dict(base_grid=str(args.base_grid.resolve()), base_protocol_sha256=sha,
                    experiment_sha256=pipeline.digest(Path(__file__)), method='oracle_mc',
                    rates=[21, 27, 35], seeds=[1, 2, 3], frames=300,
                    warmup_frames=base['warmup_frames'], ho_ms=base['ho_ms'],
                    top_k=base['top_k'], service=base['service'],
                    configs={k: asdict(v) for k, v in CONFIGS.items()},
                    manipulation='Only gap_refinement_config changes; all other frozen simulator inputs and code remain unchanged.',
                    git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    target = args.root / 'protocol.json'
    if target.exists():
        assert pipeline.read(target) == document, 'Changed protocol; choose a new root'
    else:
        assert not args.root.exists() or not any(args.root.iterdir()), 'Nonempty unversioned root'
        atomic_json(target, document)
    print(json.dumps(document, indent=2), flush=True)


def validate(root, device=None):
    p = pipeline.read(root / 'protocol.json')
    assert pipeline.digest(Path(__file__)) == p['experiment_sha256'], 'Study source changed'
    base, sha = pipeline.validate(Path(p['base_grid']), device)
    assert sha == p['base_protocol_sha256']
    return p, base, pipeline.digest(root / 'protocol.json')


def key(rate, seed, mode):
    return f'rate{rate}_seed{seed}_{mode}'


def complete(root, name, sha):
    result = root / 'runs' / f'{name}.json'
    if not result.exists():
        return False
    r = pipeline.read(result)
    assert r['protocol_sha256'] == sha
    for folder, suffix in (('raw', '.npz'), ('diagnostics', '.json')):
        assert pipeline.digest(root / folder / (name + suffix)) == r[folder + '_sha256']
    return True


def case(args):
    import torch
    from experiment import o_mappo_shared_frontend as shared
    from utils import alg_utils
    from utils.compiled_matching import km_algorithm_compiled
    from utils.ho_utils import make_paired_traffic
    from utils.revision_meet_sim import run_revised_meet
    p, base, sha = validate(args.root, args.device)
    assert args.rate in p['rates'] and args.seed in p['seeds'] and args.mode in p['configs']
    name = key(args.rate, args.seed, args.mode)
    dest = args.root / ('smoke' if args.smoke else '')
    if complete(dest, name, sha):
        print('SKIP', name, flush=True)
        return
    assert torch.cuda.is_available(), 'CUDA required for parity with the frozen grid'
    torch.set_num_threads(1)
    shared.single_thread_solvers()
    alg_utils.km_algorithm = km_algorithm_compiled
    km_algorithm_compiled(np.zeros((2, 2)))
    with Path(base['cache']).open('rb') as f:
        full_timeline = pickle.load(f)
    params = shared.paper_args(args.rate * 1e6)
    params.device = torch.device('cpu')
    # Generate the full paired trace even for prefix regression tests.
    traffic = make_paired_traffic(params, full_timeline, args.seed)
    reference_name = pipeline.key(p['method'], args.rate, args.seed)
    reference_root = Path(p['base_grid'])
    reference = pipeline.read(reference_root / 'runs' / f'{reference_name}.json')
    assert traffic['sha256'] == reference['traffic_sha256']
    if args.smoke:
        timeline = dict(list(full_timeline.items())[:9])
    else:
        timeline = full_timeline
    gap, ho, service = [], [], []

    def progress(n, total):
        atomic_json(dest / 'progress' / f'{name}.json', dict(frame=n, total=total,
                                                           pid=os.getpid(), device=args.device))
        if n % 50 == 0 or n == total:
            print('FRAME', name, n, total, flush=True)

    started = time.monotonic()
    with iteration_override(CONFIGS[args.mode]):
        result = run_revised_meet(params, shared.MICRO_BS_LOCATIONS, timeline, p['method'],
                                  traffic, args.seed, args.device, ho_ms=p['ho_ms'], k=p['top_k'],
                                  diagnostics=ho, service_diagnostics=service,
                                  gap_diagnostics=gap, progress_callback=progress)
    metrics, raw = pipeline.extract(params, result, traffic, p['warmup_frames'])
    frames = len(result.energy_record)
    assert frames == (8 if args.smoke else p['frames'])
    assert len(gap) == len(ho) == len(service) == frames
    assert all(d['slots'] == params.slots_per_frame for d in service)
    assert all(d['cap_rb_usage'] for d in gap)
    for d in gap:
        assert 1 <= d['iterations'] <= CONFIGS[args.mode].max_iterations
        if args.mode == 'fixed2':
            assert d['iterations'] == 2 and d['stop_reason'] == 'iteration_limit'
        elif d['stop_reason'] == 'tolerance':
            assert d['pre_repair_residual_rb'] <= .1
        for trace in d['traces']:
            assert np.all(np.array(trace['implied_load']) <= d['physical_capacity'])
    regression = None
    if args.mode == 'fixed2':
        assert pipeline.completed(reference_root, reference_name, p['base_protocol_sha256'])
        with np.load(reference_root / 'raw' / f'{reference_name}.npz', allow_pickle=False) as original:
            for field, values in raw.items():
                if field.startswith('queue_'):
                    expected = original[field][original['queue_frame'] < frames]
                else:
                    expected = original[field][:frames]
                np.testing.assert_array_equal(values, expected, err_msg=f'{name}: {field}')
        regression = 'all raw arrays exactly equal to frozen grid' if not args.smoke else 'all prefix raw arrays exactly equal to frozen grid'
    selected = gap[p['warmup_frames']:]
    iteration_metrics = dict(mean_iterations=float(np.mean([d['iterations'] for d in selected])),
        tolerance_stop_percent=float(100*np.mean([d['stop_reason'] == 'tolerance' for d in selected])),
        residual_mean_rb=float(np.mean([d['pre_repair_residual_rb'] for d in selected])),
        ho_median_ms=float(np.median([d['elapsed_s'] for d in selected])*1000),
        ho_p95_ms=float(np.percentile([d['elapsed_s'] for d in selected], 95)*1000))
    for folder in ('raw', 'diagnostics', 'runs'):
        (dest / folder).mkdir(parents=True, exist_ok=True)
    raw_path = dest / 'raw' / f'{name}.npz'
    temporary = raw_path.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, **raw)
    os.replace(temporary, raw_path)
    diagnostic_path = dest / 'diagnostics' / f'{name}.json'
    atomic_json(diagnostic_path, pipeline.native(dict(ho=ho, gap=gap, service=service)))
    row = dict(rate_mbps=args.rate, seed=args.seed, mode=args.mode, method=p['method'],
               protocol_sha256=sha, traffic_sha256=traffic['sha256'],
               raw_sha256=pipeline.digest(raw_path), diagnostics_sha256=pipeline.digest(diagnostic_path),
               frames=frames, smoke=args.smoke, metrics=metrics, iteration_metrics=iteration_metrics,
               fixed2_regression=regression, device=args.device, elapsed_s=time.monotonic()-started)
    atomic_json(dest / 'runs' / f'{name}.json', row)
    print('DONE', name, json.dumps(row), flush=True)


def run(args):
    p, _, sha = validate(args.root)
    smoke = args.root / 'smoke/runs/rate27_seed1_fixed2.json'
    assert smoke.exists() and pipeline.read(smoke)['fixed2_regression'], 'Run the fixed-two smoke regression first'
    devices = args.devices.split(',')
    tasks = [(r, s, m) for m in CONFIGS for r in p['rates'] for s in p['seeds']
             if not complete(args.root, key(r, s, m), sha)]
    (args.root / 'logs').mkdir(exist_ok=True)
    workers = len(devices) * args.workers_per_device
    batches = [tasks[i::workers] for i in range(workers)]

    def worker(index):
        for r, s, mode in batches[index]:
            name = key(r, s, mode)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'case',
                       '--root', str(args.root), '--rate', str(r), '--seed', str(s),
                       '--mode', mode, '--device', devices[index % len(devices)]]
            print('START', name, devices[index % len(devices)], flush=True)
            with (args.root / 'logs' / f'{name}.log').open('a') as output:
                process = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f'{name} failed; see its log')
            print('FINISHED', name, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))
    summarize(args)


def summarize(args):
    import csv
    from scipy.stats import t
    p, _, sha = validate(args.root)
    rows = []
    for r in p['rates']:
        for s in p['seeds']:
            for m in CONFIGS:
                name = key(r, s, m)
                assert complete(args.root, name, sha), f'Missing {name}'
                rows.append(pipeline.read(args.root / 'runs' / f'{name}.json'))
    for r in p['rates']:
        for s in p['seeds']:
            assert len({x['traffic_sha256'] for x in rows if x['rate_mbps'] == r and x['seed'] == s}) == 1
    groups, paired = [], []
    for r in p['rates']:
        for m in CONFIGS:
            group = [x for x in rows if x['rate_mbps'] == r and x['mode'] == m]
            statistics = {}
            for category in ('metrics', 'iteration_metrics'):
                for k in group[0][category]:
                    a = np.array([x[category][k] for x in group])
                    statistics[k] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                                         ci95_halfwidth=float(t.ppf(.975, 2)*a.std(ddof=1)/np.sqrt(3)))
            groups.append(dict(rate_mbps=r, mode=m, statistics=statistics))
            if m != 'fixed2':
                differences = []
                for new in group:
                    old = next(x for x in rows if x['rate_mbps'] == r and x['seed'] == new['seed'] and x['mode'] == 'fixed2')
                    differences.append({k: new['metrics'][k]-old['metrics'][k] for k in old['metrics']})
                paired.append(dict(rate_mbps=r, mode=m, mean_differences={
                    k: float(np.mean([x[k] for x in differences])) for k in differences[0]}))
    atomic_json(args.root / 'summary.json', dict(protocol_sha256=sha, runs=len(rows),
                groups=groups, paired=paired, all_fixed2_regressions_passed=all(
                    x['fixed2_regression'] for x in rows if x['mode'] == 'fixed2')))
    with (args.root / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        fields = ('power_w', 'violation_percent', 'mean_iterations', 'tolerance_stop_percent', 'ho_median_ms')
        writer.writerow(['rate_mbps', 'mode', *fields])
        for g in groups:
            writer.writerow([g['rate_mbps'], g['mode'], *[g['statistics'][k]['mean'] for k in fields]])
    print('COMPLETE', len(rows), 'runs', json.dumps(groups), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'case', 'run', 'summarize'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--base-grid', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--rate', type=int, default=27)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--mode', choices=tuple(CONFIGS), default='fixed2')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    parser.add_argument('--workers-per-device', type=int, default=2)
    args = parser.parse_args()
    {'prepare': prepare, 'case': case, 'run': run, 'summarize': summarize}[args.phase](args)


if __name__ == '__main__':
    main()
