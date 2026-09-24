#!/usr/bin/env python3
"""Frozen-actor module-consistency ablation on held-out validation traffic.

Experiment-local adapters leave all production defaults and physical service
unchanged. C bounds iterative occupancy; E additionally uses beam-average
interference. Each is applied to optimizer, actor+optimizer, or all three
decision modules. No learning, checkpoint selection or test-set tuning.
"""
import argparse
import collections
import concurrent.futures
from contextlib import ExitStack
import dataclasses
import fcntl
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
import torch
from experiment.o_mappo_target_check import (
    POLICY, OptimizerHook, Observer, estimate_bounded, paper_args,
    MICRO_BS_LOCATIONS, single_thread_solvers, km_algorithm_compiled,
    atomic_json, digest, read, extract, om, sim, alg_utils, beam_average_gain_db,
    make_paired_traffic)

OUTPUT = ROOT / 'experiment/results/o_mappo_stage1_consistency_20260924'
TIMELINE = ROOT / 'experiment/results/o_mappo_h32_retrained_20260924_v4/exact_validation.pkl'
OLD_VALIDATION = ROOT / 'experiment/results/o_mappo_actor_depth_20260924/validation_actor2/runs'
RATES = (5, 13, 25)
SEEDS = (101, 102, 103)
VARIANTS = {
    'baseline': dict(route='original', scope='none', hook='baseline'),
    'C_optimizer': dict(route='C', scope='optimizer', hook='bounded'),
    'C_actor_optimizer': dict(route='C', scope='actor_optimizer', hook='bounded'),
    'C_all': dict(route='C', scope='all', hook='bounded'),
    'E_optimizer': dict(route='E', scope='optimizer', hook='bounded_average'),
    'E_actor_optimizer': dict(route='E', scope='actor_optimizer', hook='bounded_average'),
    'E_all': dict(route='E', scope='all', hook='bounded_average'),
}


def load_timeline(seconds):
    with TIMELINE.open('rb') as stream:
        full = pickle.load(stream)
    start = min(full)
    selected = collections.OrderedDict((t, full[t]) for t in sorted(full)
                                      if start - 1e-8 <= t <= start + seconds + 1e-8)
    if len(selected) != round(seconds * 10) + 1:
        raise ValueError('Missing chronological validation frames')
    return selected


class ModuleAdapter:
    """Capture frame context; override only the selected decision interfaces.

The legacy estimate is ALWAYS returned to the simulator, so untouched modules
continue receiving their original inputs. Separate corrected arrays are used
by the actor/RA wrappers. The post-actor optimizer hook independently recomputes
the same correction and is checked against these arrays on every call.
    """
    def __init__(self, variant, timeline):
        self.spec = VARIANTS[variant]
        self.timeline = timeline
        self.frames = list(timeline)[1:]
        self.index = -1
        self.state_index = 0
        self.frame_rows = []
        self.old_estimate = sim.estimate_num_RB_allocated_perBS
        self.old_state = sim.make_local_state
        self.old_ra = alg_utils.RA_OTR_SINR
        self.hook = OptimizerHook(self.spec['hook'])
        self.diagnostics = self.hook.diagnostics
        self.state_count = 0
        self.state_changed = 0
        self.state_abs_change = np.zeros(31)
        self.decision_counts = np.zeros((5, 2), dtype=int)
        self.ra_calls = 0
        self.ra_interference = np.zeros(3)  # original sum, supplied sum, count

    def estimate(self, args, connection, bs_locations, vehicles, gains, rates, infer_g_dict=None):
        if self.index >= 0 and self.state_index != len(self.ids):
            raise AssertionError('Not every vehicle state was visited')
        self.index += 1
        self.state_index = 0
        self.args = args
        self.frame = self.frames[self.index]
        self.records = self.timeline[self.frame]
        self.connection = connection
        self.ids = sorted(connection, key=str)
        self.serving = {v: gains[v][connection[v]] for v in self.ids}
        self.caps = np.array([args.num_RB_macro] + [args.num_RB_micro] * 4)
        legacy = self.old_estimate(args, connection, bs_locations, vehicles, gains,
                                   rates, infer_g_dict=infer_g_dict)
        self.corrected_gain = infer_g_dict
        self.corrected_rb = legacy
        if self.spec['route'] != 'original':
            if self.spec['route'] == 'E':
                self.corrected_gain = {v: np.r_[infer_g_dict[v][0],
                    beam_average_gain_db(self.records[v]['h'])] for v in self.ids}
            estimate_gain = {v: self.corrected_gain[v].copy() for v in self.ids}
            for v in self.ids:
                estimate_gain[v][connection[v]] = self.serving[v]
            self.corrected_rb = estimate_bounded(args, connection, estimate_gain,
                                                self.corrected_gain, rates)
            assert np.all((self.corrected_rb >= 0) & (self.corrected_rb <= self.caps))
        self.corrected_load = np.clip(self.corrected_rb / self.caps, 0, 1.5)
        self.frame_rows.append(dict(frame=float(self.frame),
            legacy_load=np.clip(legacy / self.caps, 0, 1.5).tolist(),
            corrected_load=self.corrected_load.tolist()))
        return legacy

    def state(self, *args, **kwargs):
        cfg = args[0]
        if cfg.state_variant != 'adapted' or cfg.trigger_gate != 'periodic' or cfg.recurrent:
            raise ValueError('This ablation is limited to the frozen periodic adapted MLP')
        vehicle = self.ids[self.state_index]
        self.state_index += 1
        np.testing.assert_array_equal(args[1], self.records[vehicle]['pos'])
        assert args[4] == self.connection[vehicle]
        original = self.old_state(*args, **kwargs)
        result = original
        if self.spec['scope'] in ('actor_optimizer', 'all'):
            new = list(args)
            interference = sim._interference_db(self.args, args[4],
                self.corrected_gain[vehicle], self.corrected_load)
            new[5] = sim.effective_sinr_db(self.args, args[4], self.serving[vehicle], interference)
            new[8] = self.corrected_load
            new[10] = interference
            result = self.old_state(*new, **kwargs)
            # No changes to queue, geometry, serving BS/beam, or past feedback.
            names = om.state_feature_names(cfg)
            allowed = {'serving_sinr', 'interference_to_noise',
                       *(f'bs_rb_load_{i}' for i in range(5))}
            unchanged = [i for i, name in enumerate(names) if name not in allowed]
            np.testing.assert_array_equal(result[unchanged], original[unchanged])
        self.state_count += 1
        self.state_changed += int(not np.array_equal(result, original))
        self.state_abs_change += np.abs(result - original)
        return result

    def optimizer(self, **context):
        assert context['frame'] == self.frame
        result = self.hook(**context)
        if self.spec['route'] != 'original':
            np.testing.assert_allclose(result['load'], self.corrected_load, rtol=1e-12, atol=1e-12)
        return result

    def ra(self, args, **kwargs):
        supplied = kwargs
        if self.spec['scope'] == 'all':
            supplied = dict(kwargs, infer_g_dict=self.corrected_gain,
                            est_num_RB_allocated_perBS=self.corrected_rb)
        self.ra_calls += 1
        if kwargs['slot_idx'] == 0 and kwargs['BS_id'] > 0:
            bs = kwargs['BS_id']
            for v in kwargs['veh_set']:
                old_load = np.minimum(kwargs['est_num_RB_allocated_perBS'], self.caps) / self.caps
                new_load = np.minimum(supplied['est_num_RB_allocated_perBS'], self.caps) / self.caps
                old_i = sim._interference_db(args, bs, kwargs['infer_g_dict'][v], old_load)
                new_i = sim._interference_db(args, bs, supplied['infer_g_dict'][v], new_load)
                self.ra_interference += [10. ** (old_i / 10), 10. ** (new_i / 10), 1]
        return self.old_ra(args, **supplied)

    def act_observer(self, original):
        def act(local, global_state, explore):
            assert not explore
            result = original(local, global_state, explore)
            serving = np.argmax(local[:, 8:13], axis=1)
            for bs, action in zip(serving, result[0]):
                self.decision_counts[int(bs), int(action)] += 1
            return result
        return act

    def summary(self):
        assert self.index == len(self.frames) - 1
        assert self.state_index == len(self.ids)
        return dict(state_count=self.state_count, changed_states=self.state_changed,
            mean_absolute_feature_change=(self.state_abs_change / max(self.state_count, 1)).tolist(),
            actor_decisions_by_source_bs=self.decision_counts.tolist(),
            ra_calls=self.ra_calls,
            ra_sampled_interference_ratio=(float(self.ra_interference[1] / self.ra_interference[0])
                if self.ra_interference[0] > 0 else None),
            ra_interference_sample_count=int(self.ra_interference[2]))


def simulate(timeline, variant, rate, seed, device, plain=False):
    torch.set_num_threads(1)
    single_thread_solvers()
    alg_utils.km_algorithm = km_algorithm_compiled
    args = paper_args(rate * 1e6)
    args.device = torch.device('cpu')
    traffic = make_paired_traffic(args, timeline, seed)
    policy = om.OMAPPPolicy.load(str(POLICY))
    assert policy.config.actor_hidden_sizes == (64, 64)
    assert policy.config.candidate_gain_mode == 'search'
    weights = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    adapter = ModuleAdapter(variant, timeline)
    observer = Observer(adapter.hook)
    started = time.monotonic()
    with ExitStack() as stack:
        extra = {}
        if not plain:
            stack.enter_context(patch.object(sim, 'estimate_num_RB_allocated_perBS', adapter.estimate))
            stack.enter_context(patch.object(sim, 'make_local_state', adapter.state))
            stack.enter_context(patch.object(om, '_candidate_links', observer.candidate))
            stack.enter_context(patch.object(sim, 'optimize_triggered_targets', observer.optimize))
            stack.enter_context(patch.object(policy, 'act', adapter.act_observer(policy.act)))
            extra = dict(ra_func=adapter.ra, optimizer_input_hook=adapter.optimizer)
        result = sim.run_sim_o_mappo(args, MICRO_BS_LOCATIONS, timeline, policy,
            seed=seed, prt=False, optimizer_solver='milp', traffic_trace=traffic,
            ho_interruption_ms=10, paired_fading_seed=seed, physics_device=device,
            directional_service=True, **extra,
            progress_callback=lambda n,t: print('FRAME', n, t, flush=True) if n % 100 == 0 else None)
    assert all(torch.equal(v, weights[k]) for k, v in policy.actor.state_dict().items())
    metrics, raw = extract(args, result, traffic, 2)
    metrics.update(handovers=int(result.handover_record.sum()),
        optimizer_failures=int(result.optimizer_failure_record.sum()),
        optimizer_calls=len(observer.events),
        mean_overflow_rb=float(result.optimizer_overflow_record[2:].mean()))
    choices = [c for e in observer.events for c in e['choices']]
    macro = [c for c in choices if c['target'] == 0]
    info = {} if plain else adapter.summary()
    info.update(target_assignments=len(choices), macro_targets=len(macro),
        macro_exits=sum(c['source'] == 0 for c in choices),
        macro_cheapest=sum(c['cheapest'] for c in macro),
        macro_with_lower_power_micro=sum(c['micro_lower_power'] for c in macro),
        actual_rb_utilization=(raw['rb_per_bs'][2:].mean(0) / adapter.caps).tolist() if not plain else [],
        macro_power_w=float(raw['rb_per_bs'][2:, 0].mean() * args.p_macro))
    row = dict(variant=variant, settings=VARIANTS[variant], rate_mbps=rate, seed=seed,
        metrics=metrics, diagnostics_summary=info, actor_sha256=digest(POLICY),
        traffic_sha256=traffic['sha256'], frames=len(result.energy_record),
        elapsed_s=time.monotonic() - started)
    diagnostics = dict(frames=adapter.frame_rows, optimizer_inputs=adapter.diagnostics,
                       optimization=observer.events)
    return row, raw, diagnostics, result


def protocol(seconds):
    sources = [Path(__file__), ROOT/'experiment/o_mappo_target_check.py',
        ROOT/'experiment/o_mappo_optimizer_information.py', ROOT/'experiment/pql_ba_experiment.py',
        ROOT/'experiment/o_mappo_shared_frontend.py', ROOT/'experiment/revision_pipeline.py',
        *sorted((ROOT/'utils').glob('*.py'))]
    return dict(rates=list(RATES), traffic_seeds=list(SEEDS), seconds=seconds,
        interval=[710, 710+seconds], warmup_frames=2, variants=VARIANTS,
        actor=str(POLICY), actor_sha256=digest(POLICY), timeline=str(TIMELINE),
        timeline_sha256=digest(TIMELINE), training=False,
        information='current ground-truth CSI; hierarchical candidate gains; no future CSI',
        service='unchanged directional explicit-RB service; common HO10ms and BF overhead',
        unchanged='weights, state dimension/scaling, actions, BF, RA rule, optimizer objective/candidates',
        code={str(p.relative_to(ROOT)): digest(p) for p in sources})


def name(variant, rate, seed):
    return f'{variant}_rate{rate}_seed{seed}'


def verified(root, label):
    path = root/'runs'/f'{label}.json'
    if not path.exists():
        return False
    row = read(path)
    assert row['protocol_sha256'] == digest(root/'protocol.json')
    assert row['actor_sha256'] == digest(POLICY)
    for suffix, field in [('.npz', 'raw_sha256'), ('_diagnostics.json', 'diagnostics_sha256')]:
        assert digest(path.with_name(label + suffix)) == row[field]
    return True


def case(root, variant, rate, seed, device):
    frozen = read(root/'protocol.json')
    if frozen != protocol(frozen['seconds']):
        raise ValueError('Frozen experiment inputs or source changed')
    assert rate in RATES and seed in SEEDS and variant in VARIANTS
    label = name(variant, rate, seed)
    (root/'runs').mkdir(exist_ok=True)
    (root/'locks').mkdir(exist_ok=True)
    with (root/'locks'/f'{label}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if verified(root, label):
            print('SKIP', label, flush=True)
            return
        row, raw, diagnostics, _ = simulate(load_timeline(frozen['seconds']), variant, rate, seed, device)
        base = root/'runs'/label
        np.savez_compressed(base.with_suffix('.npz'), **raw)
        atomic_json(base.with_name(label+'_diagnostics.json'), diagnostics)
        row.update(protocol_sha256=digest(root/'protocol.json'),
            raw_sha256=digest(base.with_suffix('.npz')),
            diagnostics_sha256=digest(base.with_name(label+'_diagnostics.json')))
        atomic_json(base.with_suffix('.json'), row)
        print('COMPLETE', label, row['metrics'], flush=True)


def preflight(root, device):
    """Real-data baseline parity, plus scoped-wrapper parity with prior C/E."""
    timeline = load_timeline(10)
    row, raw, _, _ = simulate(timeline, 'baseline', 13, 101, device)
    reference = OLD_VALIDATION/'actor2_seed33_selected_rate13_seed101.json'
    old = read(reference)
    assert row['traffic_sha256'] == old['traffic_sha256']
    for k, v in old['metrics'].items():
        np.testing.assert_allclose(row['metrics'][k], v, atol=1e-10, rtol=1e-10)
    with np.load(reference.with_suffix('.npz')) as archive:
        for k in raw:
            np.testing.assert_array_equal(raw[k], archive[k])
    small = collections.OrderedDict(list(timeline.items())[:8])
    for variant in ('C_optimizer', 'E_optimizer'):
        _, new, _, _ = simulate(small, variant, 13, 101, device)
        args = paper_args(13e6)
        args.device = torch.device('cpu')
        traffic = make_paired_traffic(args, small, 101)
        hook = OptimizerHook(VARIANTS[variant]['hook'])
        result = sim.run_sim_o_mappo(args, MICRO_BS_LOCATIONS, small,
            om.OMAPPPolicy.load(str(POLICY)), seed=101, prt=False, optimizer_solver='milp',
            traffic_trace=traffic, ho_interruption_ms=10, paired_fading_seed=101,
            physics_device=device, directional_service=True, optimizer_input_hook=hook)
        _, old_raw = extract(args, result, traffic, 2)
        for k in new:
            np.testing.assert_array_equal(new[k], old_raw[k])
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root/'preflight.json', dict(baseline_all_raw_arrays_match=True,
        optimizer_only_matches_previous_implementation=['C', 'E'],
        actor_sha256=digest(POLICY), code_sha256=digest(Path(__file__)),
        reference=str(reference), reference_sha256=digest(reference)))
    print('PREFLIGHT PASSED', flush=True)


def summarize(root):
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            paired = []
            for variant in VARIANTS:
                label = name(variant, rate, seed)
                assert verified(root, label)
                row = read(root/'runs'/f'{label}.json')
                rows.append(row)
                paired.append(row['traffic_sha256'])
            assert len(set(paired)) == 1
    aggregates = []
    for rate in RATES:
        for variant in VARIANTS:
            cases = [r for r in rows if r['rate_mbps'] == rate and r['variant'] == variant]
            values = {k: [r['metrics'][k] for r in cases] for k in cases[0]['metrics']}
            aggregates.append(dict(rate_mbps=rate, variant=variant,
                mean={k:float(np.mean(v)) for k,v in values.items()},
                minimum={k:float(np.min(v)) for k,v in values.items()},
                maximum={k:float(np.max(v)) for k,v in values.items()}))
    comparisons = []
    edges = [('baseline', x) for x in ('C_optimizer', 'E_optimizer')]
    edges += [(f'{r}_{a}', f'{r}_{b}') for r in ('C','E')
              for a,b in [('optimizer','actor_optimizer'), ('actor_optimizer','all')]]
    for rate in RATES:
        for before, after in edges:
            a = sorted((r for r in rows if r['rate_mbps']==rate and r['variant']==before), key=lambda r:r['seed'])
            b = sorted((r for r in rows if r['rate_mbps']==rate and r['variant']==after), key=lambda r:r['seed'])
            delta = [{k:y['metrics'][k]-x['metrics'][k] for k in
                ('power_w','violation_percent','p99_proxy_ms','macro_association_percent')}
                     for x,y in zip(a,b)]
            comparisons.append(dict(rate_mbps=rate, before=before, after=after,
                per_seed_delta=delta, double_improved=sum(d['power_w']<0 and d['violation_percent']<0 for d in delta)))
    output = dict(protocol=read(root/'protocol.json'), cases=len(rows), runs=rows,
        aggregates=aggregates, paired_comparisons=comparisons,
        scope='Frozen actor, validation interval only; no retraining or formal baseline replacement')
    atomic_json(root/'summary.json', output)
    for row in aggregates:
        print(row['rate_mbps'], row['variant'], {k:round(row['mean'][k],5) for k in
            ('power_w','violation_percent','macro_association_percent')}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight','run','case','summarize'])
    p.add_argument('--root', type=Path, default=OUTPUT)
    p.add_argument('--seconds', type=int, default=30, choices=[10,30])
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--rate', type=int, choices=RATES)
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    a = p.parse_args()
    if a.command == 'preflight':
        preflight(a.root, a.device)
        return
    if a.command == 'case':
        case(a.root, a.variant, a.rate, a.seed, a.device)
        return
    if a.command == 'summarize':
        summarize(a.root)
        return
    a.root.mkdir(parents=True, exist_ok=True)
    lock = (a.root/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check = read(a.root/'preflight.json')
    assert check['code_sha256'] == digest(Path(__file__)) and check['baseline_all_raw_arrays_match']
    frozen = protocol(a.seconds)
    if (a.root/'protocol.json').exists() and read(a.root/'protocol.json') != frozen:
        raise ValueError('Protocol changed; use a new root')
    atomic_json(a.root/'protocol.json', frozen)
    (a.root/'logs').mkdir(exist_ok=True)
    jobs = [(v,r,s) for r in RATES for s in SEEDS for v in VARIANTS]
    devices = a.devices.split(',')
    def worker(device, tasks):
        for variant, rate, seed in tasks:
            label = name(variant, rate, seed)
            if verified(a.root, label):
                continue
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'case',
                '--root', str(a.root), '--variant', variant, '--rate', str(rate),
                '--seed', str(seed), '--device', device]
            with (a.root/'logs'/f'{label}.log').open('a') as log:
                subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONHASHSEED='0'))
            print('DONE', label, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, d, jobs[i::len(devices)]) for i,d in enumerate(devices)]
        for future in futures:
            future.result()
    summarize(a.root)


if __name__ == '__main__':
    main()
