#!/usr/bin/env python3
"""Paired gain-report perturbations with the unchanged formal MEET simulator.

No retraining, channel perturbation, or edits to frozen simulation sources.
The same noisy report is reused for next-frame HO and the later BF/RA calls.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiment import revision_pipeline as pipeline
from experiment.revision_training import atomic_json

BASE = ROOT / 'experiment/results/revision_directional_20260922/grid'
OUTPUT = ROOT / 'experiment/results/gain_error_sensitivity_20260929'
KINDS = {'desired': 'gain', 'interfering': 'interference'}
NOISE_NAMESPACE = 2026092901


def report_rows(timeline):
    return [(frame, v) for frame in sorted(timeline) for v in sorted(timeline[frame], key=str)]


def standard_noise(count, seed, kind):
    """Independent of global RNG, traffic/fading RNG, load, and noise amplitude."""
    stream = np.random.SeedSequence([NOISE_NAMESPACE, int(seed), list(KINDS).index(kind)])
    return np.random.default_rng(stream).standard_normal((count, 4))


def perturb_reports(timeline, kind, sigma, noise):
    if kind not in KINDS or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Invalid gain perturbation')
    rows = report_rows(timeline)
    if noise.shape != (len(rows), 4) or not np.isfinite(noise).all():
        raise ValueError('Noise must have one value per report and micro-BS link')
    if sigma == 0:
        return timeline  # Preserve original dtype and bitwise values exactly.
    out = {frame: dict(records) for frame, records in timeline.items()}
    field = KINDS[kind]
    for i, (frame, v) in enumerate(rows):
        record = timeline[frame][v]
        prediction = record['shared_prediction']
        changed = dict(prediction)
        changed[field] = np.asarray(prediction[field], dtype=np.float64) + sigma * noise[i]
        out[frame][v] = dict(record, shared_prediction=changed)
        # Physical channels, position, pilot observations and candidate indices
        # are shared immutable inputs, not regenerated or modified here.
        assert out[frame][v]['h'] is record['h']
        assert changed['beam'] is prediction['beam']
    return out


def function_ast(source, name):
    return ast.dump(next(n for n in ast.parse(source).body
                         if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def check_base(base_root):
    """Audit MEET dependencies, without invalidating old unrelated RL updates."""
    p = pipeline.read(base_root / 'protocol.json')
    exceptions = {'experiment/revision_pipeline.py', 'utils/o_mappo.py', 'utils/o_mappo_sim.py'}
    changed = []
    for name, sha in p['code_sha256'].items():
        if pipeline.digest(ROOT / name) != sha:
            if name not in exceptions:
                raise ValueError(f'Changed formal MEET dependency: {name}')
            changed.append(name)
    # The pipeline only gained new O-MAPPO modes; its metric extractor must
    # remain syntactically identical to the original formal-grid extractor.
    old = subprocess.check_output(['git', 'show', p['git_revision'] + ':experiment/revision_pipeline.py'],
                                  cwd=ROOT, text=True)
    current = (ROOT / 'experiment/revision_pipeline.py').read_text()
    if function_ast(old, 'extract') != function_ast(current, 'extract'):
        raise ValueError('Formal metric extraction changed')
    if pipeline.digest(p['cache']) != p['cache_sha256']:
        raise ValueError('Formal prediction cache changed')
    for model in p['cache_manifest']['models'].values():
        if pipeline.digest(model['checkpoint']) != model['sha256']:
            raise ValueError('Formal checkpoint changed')
    return p, changed


def prediction_audit(timeline, seeds, sigmas):
    from utils.directional_service import beam_average_gain_db
    rows = report_rows(timeline)
    frames = sorted(timeline)
    following = dict(zip(frames[:-1], frames[1:]))
    predicted = {kind: [] for kind in KINDS}
    labels = {kind: [] for kind in KINDS}
    eligible = []
    for i, (frame, v) in enumerate(rows):
        report = timeline[frame][v]['shared_prediction']
        if not np.isclose(report['source_frame'], frame, rtol=0, atol=1e-7):
            raise ValueError('Incorrect prediction source frame')
        if not np.isclose(report['target_frame'], frame + .1, rtol=0, atol=1e-7):
            raise ValueError('Incorrect prediction target frame')
        target = timeline.get(following.get(frame), {}).get(v)
        if target is None:
            continue
        eligible.append(i)
        for kind, field in KINDS.items():
            predicted[kind].append(np.asarray(report[field], dtype=np.float64))
        labels['desired'].append(np.asarray(target['g_opt_beam'], dtype=np.float64))
        labels['interfering'].append(beam_average_gain_db(target['h']))
    eligible = np.asarray(eligible)
    errors = {kind: np.asarray(predicted[kind]) - np.asarray(labels[kind]) for kind in KINDS}
    if any(values.shape != (len(eligible), 4) for values in errors.values()):
        raise ValueError('Unexpected gain-label shape')
    stats = []
    for seed in seeds:
        for kind in KINDS:
            z = standard_noise(len(rows), seed, kind)
            for sigma in sigmas:
                error = errors[kind] + sigma * z[eligible]
                stats.append(dict(seed=seed, kind=kind, sigma_db=sigma,
                    reports=len(rows), labeled_reports=len(eligible), link_samples=error.size,
                    noise_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
                    injected_mean_db=float(sigma*z.mean()), injected_std_db=float(sigma*z.std()),
                    gain_mae_db=float(np.abs(error).mean()), gain_bias_db=float(error.mean()),
                    gain_rmse_db=float(np.sqrt(np.square(error).mean()))))
    return dict(scope='Target-frame labels, all four micro links; only vehicles present at source and target; final unlabeled report excluded.',
                injection='Independent zero-mean Gaussian noise in dB, no clipping, independent between links and reports; same normalized noise across loads and sigma.',
                original_mae_db={kind: float(np.abs(value).mean()) for kind, value in errors.items()},
                rows=stats)


def prepare(args):
    if (args.root / 'protocol.json').exists():
        validate(args.root)
        print('Existing validated protocol; use run to resume', flush=True)
        return
    if args.root.exists() and any(args.root.iterdir()):
        raise FileExistsError('Use a new empty study directory')
    base, changed = check_base(args.base_grid)
    rates, seeds, sigmas = [9, 19, 29, 35], [1, 2, 3], list(range(11))
    assert set(rates) <= set(base['rates']) and set(seeds) <= set(base['seeds'])
    with Path(base['cache']).open('rb') as f:
        timeline = pickle.load(f)
    assert len(timeline) == 301
    stats = prediction_audit(timeline, seeds, sigmas)
    atomic_json(args.root / 'prediction_statistics.json', stats)
    sources = set(base['code_sha256']) | {str(Path(__file__).relative_to(ROOT)), 'test_gain_error_sensitivity.py'}
    references = {}
    for rate in rates:
        for seed in seeds:
            name = pipeline.key('meet_cobra', rate, seed)
            row = pipeline.read(args.base_grid / 'runs' / (name + '.json'))
            assert pipeline.completed(args.base_grid, name, pipeline.digest(args.base_grid / 'protocol.json'))
            references[name] = dict(run_sha256=pipeline.digest(args.base_grid / 'runs' / (name+'.json')),
                raw_sha256=row['raw_sha256'], traffic_sha256=row['traffic_sha256'])
    document = dict(version=1, base_grid=str(args.base_grid.resolve()),
        base_protocol_sha256=pipeline.digest(args.base_grid / 'protocol.json'),
        audited_unrelated_source_changes=changed,
        audit_note='MEET simulator/physics/HO/RA sources match the formal grid; only O-MAPPO files and pipeline extensions differ. The metric extractor AST matches the formal version.',
        cache=base['cache'], cache_sha256=base['cache_sha256'],
        models=base['cache_manifest']['models'], rates=rates, seeds=seeds, sigmas_db=sigmas,
        kinds=KINDS, cases=252, frames=300, warmup_frames=2, ho_ms=10., top_k=5,
        gap_iterations=2, gap_capacity_correction=True, gap_rb_usage_capped=True,
        noise_namespace=NOISE_NAMESPACE, distribution='N(0, sigma_db**2) in dB',
        noise_scope='Only one micro-BS gain-report channel at a time, before observed-gain smoothing; reused consistently in all consumers. No macro gain changes.',
        pairing='Fixed cache/beam candidates/physical matrices; same arrivals/fading/RB-permutation streams per rate and seed. Noise has its own seed and is shared across sigma and loads.',
        statistics_sha256=pipeline.digest(args.root / 'prediction_statistics.json'),
        references=references, environment=base['environment'],
        code_sha256={name: pipeline.digest(ROOT / name) for name in sorted(sources)},
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    atomic_json(args.root / 'protocol.json', document)
    print(json.dumps(dict(root=str(args.root), cases=252, original_mae_db=stats['original_mae_db']), indent=2), flush=True)


def validate(root):
    p = pipeline.read(root / 'protocol.json')
    for name, sha in p['code_sha256'].items():
        if pipeline.digest(ROOT / name) != sha:
            raise ValueError(f'Frozen study source changed: {name}')
    if pipeline.digest(p['cache']) != p['cache_sha256']:
        raise ValueError('Prediction cache changed')
    if pipeline.digest(Path(p['base_grid']) / 'protocol.json') != p['base_protocol_sha256']:
        raise ValueError('Base protocol changed')
    if pipeline.digest(root / 'prediction_statistics.json') != p['statistics_sha256']:
        raise ValueError('Prediction statistics changed')
    return p, pipeline.digest(root / 'protocol.json')


def key(kind, sigma, rate, seed):
    mode = 'control' if sigma == 0 else f'{kind}_sigma{sigma}'
    return f'{mode}_rate{rate}_seed{seed}'


def completed(root, name, sha):
    path = root / 'runs' / (name + '.json')
    if not path.exists():
        return False
    row = pipeline.read(path)
    if row['protocol_sha256'] != sha:
        raise ValueError('Foreign result: ' + name)
    for folder, suffix in (('raw', '.npz'), ('diagnostics', '.json')):
        if pipeline.digest(root / folder / (name + suffix)) != row[folder + '_sha256']:
            raise ValueError('Corrupt result: ' + name)
    return True


def case(args):
    import torch
    from experiment import o_mappo_shared_frontend as shared
    from utils import alg_utils
    from utils.compiled_matching import km_algorithm_compiled
    from utils.ho_utils import make_paired_traffic
    from utils.revision_meet_sim import run_revised_meet
    p, sha = validate(args.root)
    assert args.kind in KINDS and args.sigma in p['sigmas_db']
    assert args.rate in p['rates'] and args.seed in p['seeds']
    assert args.smoke_frames == 0 or 3 <= args.smoke_frames < p['frames']
    assert args.device.startswith('cuda:') and torch.cuda.is_available()
    name = key(args.kind, args.sigma, args.rate, args.seed)
    dest = args.root / ('smoke' if args.smoke_frames else '')
    (dest / 'locks').mkdir(parents=True, exist_ok=True)
    with (dest / 'locks' / (name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(dest, name, sha):
            print('SKIP', name, flush=True)
            return
        if shutil.disk_usage(args.root).free < 3*2**30:
            raise RuntimeError('Insufficient output space')
        torch.set_num_threads(1)
        shared.single_thread_solvers()
        alg_utils.km_algorithm = km_algorithm_compiled
        km_algorithm_compiled(np.zeros((2, 2)))
        with Path(p['cache']).open('rb') as f:
            original = pickle.load(f)
        params = shared.paper_args(args.rate * 1e6)
        params.device = torch.device('cpu')
        traffic = make_paired_traffic(params, original, args.seed)
        reference_name = pipeline.key('meet_cobra', args.rate, args.seed)
        assert traffic['sha256'] == p['references'][reference_name]['traffic_sha256']
        z = standard_noise(len(report_rows(original)), args.seed, args.kind)
        timeline = perturb_reports(original, args.kind, args.sigma, z)
        if args.smoke_frames:
            timeline = dict(list(timeline.items())[:args.smoke_frames+1])
        ho, gap, service = [], [], []
        started = time.monotonic()

        def progress(n, total):
            atomic_json(dest / 'progress' / (name+'.json'), dict(frame=n, total=total,
                pid=os.getpid(), device=args.device, elapsed_s=time.monotonic()-started))
            if n % 50 == 0 or n == total:
                print('FRAME', name, n, total, round(time.monotonic()-started, 1), flush=True)

        result = run_revised_meet(params, shared.MICRO_BS_LOCATIONS, timeline, 'meet_cobra',
            traffic, args.seed, args.device, ho_ms=p['ho_ms'], k=p['top_k'],
            diagnostics=ho, service_diagnostics=service, gap_diagnostics=gap, progress_callback=progress)
        metrics, raw = pipeline.extract(params, result, traffic, p['warmup_frames'])
        frames = len(result.energy_record)
        assert frames == (args.smoke_frames or p['frames'])
        assert len(ho) == len(gap) == len(service) == frames
        assert all(d['slots'] == params.slots_per_frame for d in service)
        assert all(d['iterations'] == 2 and d['cap_rb_usage'] for d in gap)
        regression = None
        if args.sigma == 0:
            path = Path(p['base_grid']) / 'raw' / (reference_name+'.npz')
            assert pipeline.digest(path) == p['references'][reference_name]['raw_sha256']
            with np.load(path, allow_pickle=False) as old:
                for field, value in raw.items():
                    expected = old[field][old['queue_frame'] < frames] if field.startswith('queue_') else old[field][:frames]
                    np.testing.assert_array_equal(value, expected, err_msg=name+':'+field)
            regression = 'all saved raw arrays exactly match the formal MEET-COBRA run'
        # Preserve per-vehicle decisions as well as the usual aggregate records.
        raw['serving_bs'] = np.array([result.association_record[fi][v]
            for fi, rows in result.queue_per_vehicle_record.items() for v in sorted(rows, key=str)], dtype=np.int16)
        metrics['mean_handovers_per_frame'] = float(result.handover_record[p['warmup_frames']:].mean())
        for folder in ('raw', 'diagnostics', 'runs'):
            (dest / folder).mkdir(parents=True, exist_ok=True)
        raw_path = dest / 'raw' / (name+'.npz')
        temp = raw_path.with_suffix(f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(temp, **raw)
        os.replace(temp, raw_path)
        dpath = dest / 'diagnostics' / (name+'.json')
        atomic_json(dpath, pipeline.native(dict(ho=ho, gap=gap, service=service)))
        atomic_json(dest / 'runs' / (name+'.json'), dict(kind='control' if args.sigma == 0 else args.kind,
            sigma_db=args.sigma, rate_mbps=args.rate, seed=args.seed, protocol_sha256=sha,
            traffic_sha256=traffic['sha256'], noise_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
            raw_sha256=pipeline.digest(raw_path), diagnostics_sha256=pipeline.digest(dpath),
            frames=frames, smoke_frames=args.smoke_frames, metrics=metrics,
            zero_noise_regression=regression, device=args.device, elapsed_s=time.monotonic()-started))
        print('COMPLETE', name, json.dumps(metrics), flush=True)


def all_tasks(p):
    controls = [('desired', 0, r, s) for r in p['rates'] for s in p['seeds']]
    # Early small/medium/large-noise coverage; does not change paired draws.
    others = [(k, sigma, r, s) for sigma in (1, 5, 10, 2, 3, 4, 6, 7, 8, 9)
              for k in KINDS for r in p['rates'] for s in p['seeds']]
    return controls, others


def status(root):
    p = pipeline.read(root / 'protocol.json')
    tasks = sum(all_tasks(p), [])
    done, active = [], []
    for task in tasks:
        name = key(*task)
        if (root / 'runs' / (name+'.json')).exists():
            done.append(name)
        elif (root / 'progress' / (name+'.json')).exists():
            active.append(dict(case=name, **pipeline.read(root / 'progress' / (name+'.json'))))
    output = dict(completed=len(done), total=len(tasks), incomplete_progress=active)
    if (root / 'queue_status.json').exists():
        output['queue'] = pipeline.read(root / 'queue_status.json')
    print(json.dumps(output, indent=2), flush=True)
    return output


def run(args):
    if args.detach:
        with (args.root / 'queue.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), 'run', '--root', str(args.root),
                '--devices', args.devices, '--workers-per-device', str(args.workers_per_device)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, log=str(args.root / 'queue.log'))), flush=True)
        return
    p, sha = validate(args.root)
    devices = args.devices.split(',') * args.workers_per_device
    assert devices and args.workers_per_device >= 1
    (args.root / 'logs').mkdir(exist_ok=True)
    with (args.root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Never launch perturbed runs until all full-length controls reproduce.
        for phase, tasks in zip(('zero_noise_regression', 'perturbations'), all_tasks(p)):
            waiting = queue.Queue()
            for task in tasks:
                if not completed(args.root, key(*task), sha):
                    waiting.put(task)
            stop = threading.Event()
            errors = []
            atomic_json(args.root / 'queue_status.json', dict(phase=phase, state='running',
                pid=os.getpid(), workers=len(devices), pending=waiting.qsize()))

            def worker(device):
                while not stop.is_set():
                    try:
                        kind, sigma, rate, seed = waiting.get_nowait()
                    except queue.Empty:
                        return
                    name = key(kind, sigma, rate, seed)
                    command = [sys.executable, '-u', str(Path(__file__)), 'case', '--root', str(args.root),
                        '--kind', kind, '--sigma', str(sigma), '--rate', str(rate), '--seed', str(seed), '--device', device]
                    with (args.root / 'logs' / (name+'.log')).open('a') as log:
                        try:
                            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
                            if result.returncode:
                                raise RuntimeError(f'exit={result.returncode}')
                        except Exception as exc:
                            errors.append(dict(case=name, error=str(exc)))
                            stop.set()
                    print('FINISHED' if not stop.is_set() else 'ERROR', name, 'remaining', waiting.qsize(), flush=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
                list(pool.map(worker, devices))
            if errors:
                atomic_json(args.root / 'queue_status.json', dict(phase=phase, state='failed', errors=errors))
                raise RuntimeError(errors)
        atomic_json(args.root / 'queue_status.json', dict(state='complete', cases=p['cases']))
        summarize(args.root)


def summarize(root, allow_partial=False):
    import csv
    p, sha = validate(root)
    rows = []
    for task in sum(all_tasks(p), []):
        name = key(*task)
        if completed(root, name, sha):
            rows.append(pipeline.read(root / 'runs' / (name+'.json')))
        elif not allow_partial:
            raise ValueError('Incomplete: '+name)
    stats = pipeline.read(root / 'prediction_statistics.json')
    error_rows = {(r['kind'], r['sigma_db'], r['seed']): r for r in stats['rows']}
    aggregate = []
    metrics = ('violation_percent', 'power_w', 'p99_proxy_ms', 'macro_association_percent',
               'pilots_per_vehicle_slot', 'mean_handovers_per_frame')
    for kind in KINDS:
        for rate in p['rates']:
            for sigma in p['sigmas_db']:
                selected = [r for r in rows if r['rate_mbps'] == rate and r['sigma_db'] == sigma
                            and r['kind'] == ('control' if sigma == 0 else kind)]
                if not selected:
                    continue
                record = dict(kind=kind, rate_mbps=rate, sigma_db=sigma, seeds=len(selected))
                for metric in metrics:
                    values = np.array([r['metrics'][metric] for r in selected])
                    for suffix, value in (('mean', values.mean()), ('min', values.min()), ('max', values.max())):
                        record[metric+'_'+suffix] = float(value)
                record['gain_mae_db_mean'] = float(np.mean([error_rows[kind, sigma, r['seed']]['gain_mae_db'] for r in selected]))
                aggregate.append(record)
    atomic_json(root / 'summary.json', dict(complete=len(rows)==p['cases'], cases=len(rows), expected=p['cases'],
        protocol_sha256=sha, noise_definition=stats['injection'], original_mae_db=stats['original_mae_db'],
        aggregate=aggregate, per_seed=rows))
    if aggregate:
        with (root / 'curves.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
            writer.writeheader(); writer.writerows(aggregate)
    print(json.dumps(dict(completed=len(rows), total=p['cases'], summary=str(root/'summary.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'case', 'run', 'status', 'summarize'):
        item = sub.add_parser(name)
        item.add_argument('--root', type=Path, default=OUTPUT)
        if name == 'prepare':
            item.add_argument('--base-grid', type=Path, default=BASE)
        elif name == 'case':
            item.add_argument('--kind', choices=KINDS, required=True)
            item.add_argument('--sigma', type=int, required=True)
            item.add_argument('--rate', type=int, required=True)
            item.add_argument('--seed', type=int, required=True)
            item.add_argument('--device', default='cuda:0')
            item.add_argument('--smoke-frames', type=int, default=0)
        elif name == 'run':
            item.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
            item.add_argument('--workers-per-device', type=int, default=2)
            item.add_argument('--detach', action='store_true')
        elif name == 'summarize':
            item.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.command == 'status': status(args.root)
    elif args.command == 'summarize': summarize(args.root, args.allow_partial)
    else: globals()[args.command](args)


if __name__ == '__main__':
    main()
