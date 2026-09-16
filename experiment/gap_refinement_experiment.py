#!/usr/bin/env python3
"""Paired Oracle evaluation of opt-in GAP-HO fixed-point refinement.

Legacy reserve and pilot estimates are intentionally unchanged. Physical HO
interruption and capacity correction are identical across compared variants.
"""

import argparse
import collections
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import types

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import ho_interruption_experiment as exp
from utils.gap_refinement import GAPRefinementConfig

np = exp.np
CHECKPOINT = 'a5e50ff57d92cc96813d97266c89693379a9a109'
DATASET = ROOT / 'data4sim/lbd1.00_800_950_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl'
MODES = dict(
    legacy=None,
    iter1_off=GAPRefinementConfig(1, None),
    iter2_off=GAPRefinementConfig(2, None),
    iter3_off=GAPRefinementConfig(3, None),
    iter5_off=GAPRefinementConfig(5, None),
    iter10_off=GAPRefinementConfig(10, None),
    iter3_eps0p1=GAPRefinementConfig(3, .1),
    iter3_eps0p5=GAPRefinementConfig(3, .5),
    iter3_eps1=GAPRefinementConfig(3, 1.),
    iter5_eps0p1=GAPRefinementConfig(5, .1),
)
CONTEXT = None


def initialize_worker(context):
    global CONTEXT
    CONTEXT = context
    exp.torch.set_num_threads(1)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def regression(args, locations, timeline, cache, output):
    """Compare all eight full outputs, not only aggregate performance metrics."""
    timeline = collections.OrderedDict(list(timeline.items())[:13])
    cache = {k: cache[k] for k in timeline}
    args.data_rate = 27e6
    traffic = exp.make_paired_traffic(args, timeline, 7)
    source = subprocess.check_output(['git', 'show', CHECKPOINT+':utils/alg_utils.py'],
                                     cwd=ROOT, text=True)
    old_module = types.ModuleType('gap_refinement_legacy_checkpoint')
    exec(compile(source, CHECKPOINT+':utils/alg_utils.py', 'exec'), old_module.__dict__)
    rows = []
    for duration in (0, 10):
        options = dict(oracle_ho_cache=cache, traffic_trace=traffic,
                       measured_gain_gamma=0, ho_interruption_ms=duration,
                       ho_capacity_correction=True)
        exp.setup_seed(7)
        old = exp.invoke(args, locations, timeline,
                         ho_func=old_module.HO_EE_GAP_APX_SINR_conservative_adaptive, **options)
        exp.setup_seed(7)
        default = exp.invoke(args, locations, timeline, **options)
        exp.setup_seed(7)
        traces = []
        new = exp.invoke(args, locations, timeline, **options,
                         gap_refinement_config=asdict(MODES['iter2_off']),
                         gap_refinement_diagnostics=traces)
        exp.assert_recursive_equal(old, default)
        exp.assert_recursive_equal(old, new)
        assert all(d['iterations'] == 2 for d in traces)
        rows.append(dict(ho_ms=duration, frames=len(timeline)-1,
                         default_vs_checkpoint='bitwise_equal_all_8_outputs',
                         iter2_no_early_stop_vs_checkpoint='bitwise_equal_all_8_outputs'))
        print('REGRESSION PASSED', duration, flush=True)
    exp.save_json(output/'regression.json', dict(checkpoint=CHECKPOINT, cases=rows))


def refinement_metrics(records):
    selected = records[exp.WARMUP:]
    elapsed = np.array([d['elapsed_s']*1000 for d in selected])
    metrics = dict(
        gap_iterations_mean=float(np.mean([d['iterations'] for d in selected])),
        ho_cpu_median_ms=float(np.median(elapsed)),
        ho_cpu_p95_ms=float(np.percentile(elapsed, 95)),
        tolerance_stop_percent=float(100*np.mean([d['stop_reason']=='tolerance' for d in selected])),
    )
    if selected[0]['stop_reason'] != 'legacy':
        metrics.update(
            final_residual_median_rb=float(np.median([d['pre_repair_residual_rb'] for d in selected])),
            final_residual_p95_rb=float(np.percentile([d['pre_repair_residual_rb'] for d in selected], 95)),
            repair_changed_frame_percent=float(100*np.mean([d['repair_changed_vehicles']>0 for d in selected])),
            infeasible_after_repair_percent=float(100*np.mean([not d['capacity_feasible_after_repair'] for d in selected])),
            repeated_iterate_frame_percent=float(100*np.mean([any(t['repeated_load'] for t in d['traces']) for d in selected])),
        )
    return metrics


def run_case(task):
    rate, seed, mode = task
    args, locations, timeline, cache, output, duration = CONTEXT
    args.data_rate = rate*1e6
    traffic = exp.make_paired_traffic(args, timeline, seed)
    name = f'rate{rate:g}_seed{seed}_{mode}'
    result_path = output/'runs'/f'{name}.json'
    raw_path = output/'raw'/f'{name}.npz'
    trace_path = output/'diagnostics'/f'{name}.json'
    if result_path.exists() and raw_path.exists() and trace_path.exists():
        saved = json.loads(result_path.read_text())
        assert saved['traffic_sha256'] == traffic['sha256']
        return saved
    print('START', name, flush=True)
    exp.setup_seed(seed)
    diagnostics = []
    frame_diagnostics = []
    config = MODES[mode]

    def timed_handover(*positional, **kwargs):
        if config is not None:
            kwargs.update(gap_refinement_config=config,
                          gap_refinement_diagnostics=diagnostics)
        start = time.perf_counter()
        result = exp.alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(*positional, **kwargs)
        elapsed = time.perf_counter()-start
        if config is None:
            diagnostics.append(dict(iterations=2, stop_reason='legacy', elapsed_s=elapsed))
        else:
            diagnostics[-1]['elapsed_s'] = elapsed
        if len(diagnostics) % 50 == 0:
            print('FRAME', name, len(diagnostics), flush=True)
        return result

    started = time.monotonic()
    result = exp.invoke(
        args, locations, timeline, ho_func=timed_handover,
        oracle_ho_cache=cache, traffic_trace=traffic, measured_gain_gamma=0,
        ho_interruption_ms=duration, ho_capacity_correction=True,
        ho_diagnostics=frame_diagnostics)
    metrics, raw = exp.extract_metrics(args, result, frame_diagnostics)
    assert len(diagnostics) == len(frame_diagnostics)
    metrics.update(refinement_metrics(diagnostics))
    # Independently reconcile the saved violation metric with slot queues.
    chosen = raw['queue_frame_index'] >= exp.WARMUP
    proxy_violations = (raw['queue_bits'][chosen] > args.data_rate*args.lat_slot_ub*args.slot_len)
    # Frame means weight frames equally, not vehicles across all frames.
    frame_v = [proxy_violations[raw['queue_frame_index'][chosen] == f].mean()
               for f in np.unique(raw['queue_frame_index'][chosen])]
    np.testing.assert_allclose(np.mean(frame_v)*100, metrics['violation_percent'], atol=1e-10)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **raw)
    exp.save_json(trace_path, dict(refinement=diagnostics, service=frame_diagnostics))
    saved = dict(rate_mbps=rate, seed=seed, mode=mode, ho_ms=duration,
                 config=None if config is None else asdict(config),
                 traffic_sha256=traffic['sha256'], metrics=metrics,
                 elapsed_s=time.monotonic()-started)
    exp.save_json(result_path, saved)
    print('DONE', name, json.dumps(metrics), flush=True)
    return saved


def aggregate(output):
    payloads = [json.loads(p.read_text()) for p in sorted((output/'runs').glob('*.json'))]
    if not payloads:
        return
    common_metrics = sorted(set.intersection(*(set(p['metrics']) for p in payloads)))
    exp.save_csv(output/'per_seed.csv', [dict(rate_mbps=p['rate_mbps'], seed=p['seed'], mode=p['mode'],
                                           **{k:p['metrics'][k] for k in common_metrics}) for p in payloads])
    groups = collections.defaultdict(list)
    by_key = {(p['rate_mbps'],p['seed'],p['mode']):p for p in payloads}
    for p in payloads:
        groups[(p['rate_mbps'], p['mode'])].append(p)
    means, paired = [], []
    for (rate, mode), group in sorted(groups.items()):
        row = dict(rate_mbps=rate, mode=mode, n=len(group))
        for k in common_metrics:
            values = [p['metrics'][k] for p in group]
            row[k+'_mean'] = float(np.mean(values))
            row[k+'_std'] = float(np.std(values, ddof=1)) if len(values)>1 else 0.
        means.append(row)
        pairs = [(by_key[(rate,p['seed'],'legacy')],p) for p in group if (rate,p['seed'],'legacy') in by_key]
        if mode == 'legacy' or not pairs:
            continue
        for old, new in pairs:
            assert old['traffic_sha256'] == new['traffic_sha256']
        row = dict(rate_mbps=rate, mode=mode, n=len(pairs))
        for k in common_metrics:
            differences = [new['metrics'][k]-old['metrics'][k] for old,new in pairs]
            row[k+'_delta_mean'] = float(np.mean(differences))
        paired.append(row)
        if mode == 'iter2_off':
            for old,new in pairs:
                prefix = f"rate{rate:g}_seed{new['seed']}_"
                with np.load(output/'raw'/f'{prefix}legacy.npz') as a, np.load(output/'raw'/f'{prefix}iter2_off.npz') as b:
                    assert set(a.files) == set(b.files)
                    for k in a.files:
                        np.testing.assert_array_equal(a[k], b[k], err_msg=prefix+k)
    exp.save_csv(output/'aggregate.csv', means)
    exp.save_csv(output/'paired_differences.csv', paired)
    exp.save_json(output/'summary.json', dict(aggregate=means, paired_differences=paired,
                                            completed_runs=len(payloads)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rates', nargs='+', type=float, default=[5,21,27,35])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--modes', nargs='+', choices=tuple(MODES), default=list(MODES))
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--ho-ms', type=float, default=10)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--regression-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    if options.aggregate_only:
        aggregate(options.output)
        return
    if not 1.3 <= options.seconds <= 150:
        parser.error('--seconds must be between 1.3 and 150')
    if not DATASET.is_file():
        raise FileNotFoundError('Existing RT cache required; do not regenerate it in this study')
    print('Loading existing RT cache', flush=True)
    exp.torch.set_num_threads(1)
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0]]
    try:
        args, locations, timeline, *_ = exp.get_default_sim_params(
            str(options.output/'_loader'), cut_ratio=options.seconds/150, load_predictors=False)
    finally:
        sys.argv = old_argv
    cache = exp.oracle_cache(args, locations, timeline)
    if options.regression or options.regression_only:
        regression(args, locations, timeline, cache, options.output)
    if options.regression_only:
        return
    paths = ['utils/alg_utils.py','utils/sim_utils.py','utils/gap_refinement.py',
             'utils/ho_utils.py','utils/queue_utils.py','experiment/ho_interruption_experiment.py',
             'experiment/gap_refinement_experiment.py']
    protocol = dict(checkpoint=CHECKPOINT, seconds=options.seconds, ho_ms=options.ho_ms,
                    rates_mbps=options.rates, seeds=options.seeds, modes=options.modes,
                    configs={m:None if MODES[m] is None else asdict(MODES[m]) for m in options.modes},
                    frame_count=len(timeline)-1, warmup_frames=exp.WARMUP,
                    evaluated_seconds=(len(timeline)-1-exp.WARMUP)*args.slots_per_frame*args.slot_len,
                    source_sha256={p:file_hash(ROOT/p) for p in paths},
                    dataset_sha256=file_hash(DATASET), args=vars(args),
                    oracle='true next-frame desired and interfering gains; no NN inference',
                    fading='same existing Oracle mode, no extra slot Rician randomization',
                    pairing='identical per-vehicle arrivals and entry queues across all modes',
                    preserved='adaptive reserve, measured pilot counts, HO capacity correction, OTR-RA',
                    stopping='max absolute per-BS average-load change, before final repair',
                    initialization='physical RB capacity for all BSs; macro load does not enter interference',
                    cpu_threads_per_worker=1, workers=options.workers)
    path = options.output/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == exp.jsonable(protocol), 'Use a fresh output directory for changed protocol'
    else:
        exp.save_json(path, protocol)
    context = (args, locations, timeline, cache, options.output, options.ho_ms)
    tasks = [(r,s,m) for m in options.modes for r in options.rates for s in options.seeds]
    print('RUNS', len(tasks), 'WORKERS', options.workers, flush=True)
    if options.workers == 1:
        initialize_worker(context)
        for task in tasks:
            run_case(task)
            aggregate(options.output)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=options.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_worker, initargs=(context,)) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run_case,t) for t in tasks]):
                future.result()
                aggregate(options.output)
    aggregate(options.output)
    print('COMPLETE', options.output, flush=True)


if __name__ == '__main__':
    main()
