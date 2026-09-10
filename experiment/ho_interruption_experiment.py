#!/usr/bin/env python3
"""Paired Oracle experiment: unchanged GAP-HO versus HO-aware capacities.

All new simulator options are opt-in. See ho_interruption_experiment_report.md.
Use --regression-only before a sweep, then --seconds 30 --workers 4.
"""

import argparse
import collections
import concurrent.futures
import csv
import faulthandler
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import types

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARGV = sys.argv[:]

import numpy as np
import torch
from scipy.stats import t as student_t
from utils import alg_utils
from utils.beam_utils import generate_dft_codebook
from utils.channel_utils import get_g_macroBS_dict
from utils.ho_utils import make_paired_traffic
from utils.mox_utils import setup_seed
from utils.sim_utils import get_default_sim_params, run_sim_withUMa

sys.argv = ARGV
CHECKPOINT = '76078e5'
WARMUP = 2
CONTEXT = None


def initialize_worker(context):
    global CONTEXT
    CONTEXT = context
    torch.set_num_threads(1)


class ProgressDiagnostics(list):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def append(self, value):
        super().append(value)
        if len(self) % 50 == 0:
            print('FRAME', self.name, len(self), flush=True)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def oracle_cache(args, locations, timeline):
    tx, rx = generate_dft_codebook(args.M_t), generate_dft_codebook(args.M_r)
    cache = {}
    for frame, entries in timeline.items():
        positions = {v: d['pos'] for v, d in entries.items()}
        gains = {v: d['g_opt_beam'] for v, d in entries.items()}
        beams = {v: d['best_beam_pair_idx'].reshape(-1, 1).repeat(5, axis=1)
                 for v, d in entries.items()}
        _, interference, _, pilots = alg_utils.measure_gain(
            args, frame, set(entries), timeline, locations, beams, gains, tx, rx,
            'topKbeam_savePilot', {}, rician_fading=False, K_BF=5)
        macro = get_g_macroBS_dict(args, positions, [0, 0], fc_ghz=2.8, Gt_macro=0)
        cache[frame] = dict(
            positions=positions, pilots=pilots,
            gain={v: np.r_[macro[v], gains[v]] for v in entries},
            interference={v: np.r_[macro[v], interference[v]] for v in entries},
        )
    return cache


def invoke(args, locations, timeline, ho_func=None, simulator=run_sim_withUMa, **kwargs):
    return simulator(
        args, locations, timeline, None, None, None, None,
        HO_func=ho_func or alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive,
        RA_func=alg_utils.RA_OTR_SINR, prt=False, save_pilot=True, K_BF=5,
        **kwargs)


def assert_recursive_equal(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_recursive_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_recursive_equal(x, y)
    else:
        np.testing.assert_array_equal(a, b)


def regression(args, locations, timeline, output):
    """Execute historical source from Git without changing the working tree."""
    short = collections.OrderedDict(list(timeline.items())[:12])
    modules = {}
    for name in ('alg_utils', 'sim_utils'):
        source = subprocess.check_output(
            ['git', 'show', CHECKPOINT + ':utils/' + name + '.py'], cwd=ROOT, text=True)
        module = types.ModuleType('ho_regression_' + name)
        exec(compile(source, CHECKPOINT + ':' + name, 'exec'), module.__dict__)
        modules[name] = module
    args.data_rate = 21e6
    setup_seed(7)
    old = invoke(args, locations, short,
                 ho_func=modules['alg_utils'].HO_EE_GAP_APX_SINR_conservative_adaptive,
                 simulator=modules['sim_utils'].run_sim_withUMa)
    setup_seed(7)
    new = invoke(args, locations, short)
    assert_recursive_equal(old, new)
    setup_seed(7)
    zero = invoke(args, locations, short, ho_interruption_ms=0,
                  ho_capacity_correction=True)
    assert_recursive_equal(old, zero)
    cache = oracle_cache(args, locations, short)
    traffic = make_paired_traffic(args, short, 7)
    setup_seed(7)
    plain = invoke(args, locations, short, oracle_ho_cache=cache,
                   traffic_trace=traffic, measured_gain_gamma=0)
    setup_seed(7)
    corrected = invoke(args, locations, short, oracle_ho_cache=cache,
                       traffic_trace=traffic, measured_gain_gamma=0,
                       ho_interruption_ms=0, ho_capacity_correction=True)
    assert_recursive_equal(plain, corrected)
    # Force one retained vehicle to switch each frame, exercising both tiers.
    def force_switch(args, vehicles, q, rates, positions, gains, bs, **kwargs):
        current = kwargs['current_connection']
        commands = dict(current)
        veh = sorted(vehicles, key=repr)[0]
        commands[veh] = 1 if current[veh] == 0 else 0
        return commands, np.zeros(len(bs))
    diagnostic = []
    invoke(args, locations, short, ho_func=force_switch, traffic_trace=traffic,
           oracle_ho_cache=cache, ho_interruption_ms=10,
           ho_capacity_correction=True, ho_diagnostics=diagnostic)
    assert sum(d['blocked_vehicle_slots'] for d in diagnostic) > 0
    # Simulator assertions check no RBs, no pilots, and q_next = q + arrivals
    # in every blocked slot; arrivals must continue during interruption.
    result = dict(checkpoint=CHECKPOINT, frames=len(short) - 1,
                  default_vs_checkpoint='bitwise_equal_all_8_outputs',
                  zero_correction_vs_checkpoint='bitwise_equal_all_8_outputs',
                  paired_next_frame_oracle_zero_correction='bitwise_equal_all_8_outputs',
                  forced_switch_no_service_no_pilots_arrival_accumulation='passed')
    save_json(output / 'regression.json', result)
    print('REGRESSION', json.dumps(result), flush=True)


def extract_metrics(args, result, diagnostics):
    energy, ho, commands, violation, qavg, pilots, rb, queues = result
    dt = args.slots_per_frame * args.slot_len
    selected = slice(WARMUP, None)
    frames, vehicles, qrows = [], [], []
    for frame, frame_queues in queues.items():
        for veh in sorted(frame_queues, key=repr):
            frames.append(frame)
            vehicles.append(repr(veh))
            qrows.append(frame_queues[veh])
    qrows = np.asarray(qrows)
    samples = qrows[np.asarray(frames) >= WARMUP].ravel() / args.data_rate * 1000
    associations = np.array([
        np.bincount(list(d['association'].values()), minlength=len(rb[0]))
        for d in diagnostics])
    active = np.array([d['active_vehicle_slots'] for d in diagnostics])
    blocked = np.array([d['blocked_vehicle_slots'] for d in diagnostics])
    metrics = dict(
        power_w=float(np.mean(energy[selected]) / dt),
        violation_percent=float(np.mean(violation[selected]) * 100),
        mean_proxy_ms=float(np.mean(qavg[selected]) / args.data_rate * 1000),
        p99_proxy_ms=float(np.percentile(samples, 99)),
        handover_count=int(np.sum(ho[selected])),
        handovers_per_vehicle_s=float(np.sum(ho[selected]) / (np.sum(active[selected]) * args.slot_len)),
        blocked_vehicle_time_percent=float(100 * np.sum(blocked[selected]) / np.sum(active[selected])),
        pilots_per_vehicle_slot=float(np.mean(pilots[selected])),
        macro_association_percent=float(100 * np.sum(associations[selected, 0]) / np.sum(associations[selected])),
    )
    raw = dict(energy_j=energy, handover_count=ho, violation_probability=violation,
               mean_queue_bits=qavg, pilots=pilots, rb_per_bs=rb,
               association_counts=associations, blocked_vehicle_slots=blocked,
               active_vehicle_slots=active, queue_bits=qrows,
               queue_frame_index=np.array(frames), queue_vehicle=np.array(vehicles))
    return metrics, raw


def run_pair(task):
    rate, seed = task
    args, locations, timeline, cache, output, durations = CONTEXT
    args.data_rate = rate * 1e6
    traffic = make_paired_traffic(args, timeline, seed)
    rows = []
    for duration in durations:
        for corrected in (False, True):
            method = 'capacity_corrected' if corrected else 'original'
            name = f'rate{rate:g}_seed{seed}_ho{duration:g}ms_{method}'
            path = output / 'runs' / (name + '.json')
            raw_path = output / 'raw' / (name + '.npz')
            if path.exists() and raw_path.exists():
                payload = json.loads(path.read_text())
                assert payload['traffic_sha256'] == traffic['sha256']
                rows.append(payload)
                continue
            print('START', name, flush=True)
            setup_seed(seed)
            diagnostics = ProgressDiagnostics(name)
            started = time.monotonic()
            result = invoke(args, locations, timeline, oracle_ho_cache=cache,
                            traffic_trace=traffic, measured_gain_gamma=0,
                            ho_interruption_ms=duration,
                            ho_capacity_correction=corrected,
                            ho_diagnostics=diagnostics)
            metrics, raw = extract_metrics(args, result, diagnostics)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_path, **raw)
            payload = dict(rate_mbps=rate, seed=seed, ho_ms=duration, method=method,
                           traffic_sha256=traffic['sha256'], metrics=metrics,
                           elapsed_s=time.monotonic() - started)
            save_json(path, payload)
            save_json(output / 'associations' / (name + '.json'), diagnostics)
            rows.append(payload)
            print('DONE', name, json.dumps(metrics), f"{payload['elapsed_s']:.1f}s", flush=True)
    return rows


def aggregate(output):
    payloads = [json.loads(p.read_text()) for p in sorted((output / 'runs').glob('*.json'))]
    rows = [dict(rate_mbps=p['rate_mbps'], seed=p['seed'], ho_ms=p['ho_ms'],
                 method=p['method'], **p['metrics']) for p in payloads]
    save_csv(output / 'per_seed.csv', rows)
    groups = collections.defaultdict(list)
    by_key = {}
    for p in payloads:
        groups[(p['rate_mbps'], p['ho_ms'], p['method'])].append(p)
        by_key[(p['rate_mbps'], p['ho_ms'], p['seed'], p['method'])] = p
    means = []
    for (rate, duration, method), ps in sorted(groups.items()):
        row = dict(rate_mbps=rate, ho_ms=duration, method=method, n=len(ps))
        for metric in ps[0]['metrics']:
            values = [p['metrics'][metric] for p in ps]
            row[metric + '_mean'] = float(np.mean(values))
            row[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        means.append(row)
    save_csv(output / 'aggregate.csv', means)
    paired = []
    for (rate, duration, method), ps in sorted(groups.items()):
        if method != 'capacity_corrected':
            continue
        pairs = [(by_key[(rate, duration, p['seed'], 'original')], p) for p in ps
                 if (rate, duration, p['seed'], 'original') in by_key]
        if not pairs:
            continue
        for old, new in pairs:
            assert old['traffic_sha256'] == new['traffic_sha256']
        row = dict(rate_mbps=rate, ho_ms=duration, n=len(pairs))
        for metric in ps[0]['metrics']:
            differences = np.array([new['metrics'][metric] - old['metrics'][metric]
                                    for old, new in pairs])
            mean = float(differences.mean())
            half = (float(student_t.ppf(.975, len(pairs)-1) * differences.std(ddof=1)
                          / np.sqrt(len(pairs))) if len(pairs) > 1 else None)
            row[metric + '_delta_mean'] = mean
            row[metric + '_delta_ci95_half'] = half
            row[metric + '_delta_min'] = float(differences.min())
            row[metric + '_delta_max'] = float(differences.max())
        paired.append(row)
    save_csv(output / 'paired_differences.csv', paired)
    save_json(output / 'summary.json', dict(aggregate=means, paired_differences=paired))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rates', type=float, nargs='+', default=[5, 21, 35])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--durations-ms', type=float, nargs='+', default=[0, 5, 10])
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--start-method', choices=['spawn', 'fork'], default='spawn')
    parser.add_argument('--regression-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiment/results_ho_interruption')
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    if options.aggregate_only:
        aggregate(options.output)
        return
    if not 1.3 <= options.seconds <= 150:
        parser.error('--seconds must lie in [1.3, 150]')
    torch.set_num_threads(1)
    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        args, locations, timeline, *_ = get_default_sim_params(
            str(options.output / '_loader'), cut_ratio=options.seconds / 150,
            load_predictors=False)
    finally:
        sys.argv = saved
    if options.regression_only:
        regression(args, locations, timeline, options.output)
        return
    cache = oracle_cache(args, locations, timeline)
    sources = ['utils/sim_utils.py', 'utils/alg_utils.py', 'utils/ho_utils.py',
               'utils/queue_utils.py', 'experiment/ho_interruption_experiment.py']
    protocol = dict(
        checkpoint=CHECKPOINT, rates_mbps=options.rates, seeds=options.seeds,
        durations_ms=options.durations_ms, seconds=options.seconds,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        frame_count=len(timeline)-1, warmup_frames=WARMUP,
        evaluated_duration_s=(len(timeline)-1-WARMUP)*args.slot_len*args.slots_per_frame,
        oracle='exact next-frame gains/interference/positions for continuing vehicles',
        beamforming='Oracle best beam, existing PET search overhead retained',
        slot_channel='existing Oracle mode: RT frame channels, no extra Rician draws',
        correction='active-period RB capacity only; average power/interference unchanged',
        initial_access_interruption=False, all_bs_transition_types=True,
        traffic='paired independent Poisson arrivals and entry queues; hash checked',
        preparation='outside data resources; fixed successful execution interruption only',
        process_start_method=options.start_method,
        weights_or_inference_used=False, args=vars(args),
    )
    protocol_path = options.output / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == jsonable(protocol), 'Use a new output directory for a different protocol'
    else:
        save_json(protocol_path, protocol)
    global CONTEXT
    CONTEXT = (args, locations, timeline, cache, options.output, options.durations_ms)
    tasks = [(r, s) for r in options.rates for s in options.seeds]
    print('PROTOCOL', json.dumps({k: v for k, v in protocol.items() if k != 'args'}), flush=True)
    if options.workers == 1:
        for task in tasks:
            run_pair(task)
            aggregate(options.output)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=options.workers,
                mp_context=multiprocessing.get_context(options.start_method),
                initializer=initialize_worker, initargs=(CONTEXT,)) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run_pair, t) for t in tasks]):
                future.result()
                aggregate(options.output)
    print('COMPLETE', options.output, flush=True)


if __name__ == '__main__':
    main()
