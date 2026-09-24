#!/usr/bin/env python3
"""Frozen-actor, optimizer-only ablations at 13 Mbps / traffic seed 1.

No production source/default is changed. A post-actor hook supplies causal
optimizer estimates. An experiment-local wrapper optionally changes only
candidate costs. BF, actor observation construction, RA and physical service
remain unchanged; closed-loop observations/actions can naturally diverge.
"""
import argparse
import concurrent.futures
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
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from experiment.o_mappo_shared_frontend import single_thread_solvers
from experiment.o_mappo_optimizer_information import fixed_allocation
from experiment.revision_pipeline import atomic_json, digest, extract
from utils import o_mappo as om, o_mappo_sim as sim, alg_utils
from utils.compiled_matching import km_algorithm_compiled
from utils.directional_service import beam_average_gain_db
from utils.hierarchical_beam import hierarchical_beam_pair
from utils.ho_utils import make_paired_traffic
from utils.mox_utils import dB2lin

OUTPUT = ROOT / 'experiment/results/o_mappo_target_check_20260924'
POLICY = ROOT / 'experiment/results/o_mappo_actor_depth_20260924/checkpoints/actor2_seed33/selected.pt'
TIMELINE = ROOT / 'experiment/results/revision_directional_20260922/test_predictions.pkl'
REFERENCE = ROOT / 'experiment/results/o_mappo_actor2_full_grid_20260924/runs/actor2_seed33_selected_rate13_seed1.json'
VARIANTS = {
    'baseline': dict(load='legacy', gain='max', objective='mixed', candidates=3),
    'clip_only': dict(load='clip', gain='max', objective='mixed', candidates=3),
    'bounded': dict(load='bounded', gain='max', objective='mixed', candidates=3),
    'beam_average': dict(load='legacy', gain='mean', objective='mixed', candidates=3),
    'bounded_average': dict(load='bounded', gain='mean', objective='mixed', candidates=3),
    'bounded_average_power': dict(load='bounded', gain='mean', objective='power', candidates=3),
    'bounded_average_all': dict(load='bounded', gain='mean', objective='mixed', candidates=4),
    'bounded_average_power_all': dict(load='bounded', gain='mean', objective='power', candidates=4),
}


def read(path):
    return json.loads(Path(path).read_text())


def estimate_bounded(args, connection, gains, interference, rates):
    """Keep the legacy rate equations/10-round limit, but bound each iterate.

Demand is unconstrained; occupied RBs used in the interference feedback are
clipped to capacity. Initial occupancies and stopping tolerance match the
legacy helper, apart from the physically necessary capacity bound.
    """
    ids = list(connection)
    caps = np.array([args.num_RB_macro] + [args.num_RB_micro] * 4, dtype=float)
    occupied = np.minimum(np.full(5, args.num_RB_micro, dtype=float), caps)
    if not ids:
        return np.zeros(5)
    association = np.array([connection[v] for v in ids])
    gain = np.array([gains[v] for v in ids])
    arrival = np.array([rates[v] for v in ids])
    for _ in range(10):
        demand = np.zeros((len(ids), 5))
        for bs in range(5):
            bandwidth = args.RB_intervel_micro if bs else args.RB_intervel_macro
            power = args.p_micro if bs else args.p_macro
            nf = args.NF_micro_dB if bs else args.NF_macro_dB
            inter = np.array([sum(dB2lin(interference[v][j]) * args.p_micro
                        * occupied[j] / args.num_RB_micro
                        for j in range(1, 5) if j != bs) for v in ids])
            if bs == 0:
                inter *= 0
            demand[:, bs] = arrival / (1e-10 + bandwidth * np.log2(1 + power
                * dB2lin(gain[:, bs]) / (args.N0 * bandwidth * dB2lin(nf) + inter)) + 1e-10)
        totals = np.bincount(association, weights=demand[np.arange(len(ids)), association], minlength=5)
        updated = np.clip(totals, 0, caps)
        converged = np.allclose(occupied, updated, atol=1)
        occupied = updated
        if converged:
            break
    return occupied


def optimizer_records(c, interference):
    """Transport identical current-frame search gains, not NN predictions.

The existing optimizer's scalar-input interface is reused locally. The
simulation/actor config and original cached records are never modified.
    """
    result = {}
    for v, record in c['records'].items():
        gains = np.array([hierarchical_beam_pair(record['h'], m, c['dft_tx'], c['dft_rx'])[2]
                          for m in range(4)])
        result[v] = dict(record, shared_prediction=dict(gain=gains, interference=interference[v][1:]))
    return result


def power_candidates(original, *args, **kwargs):
    """Rerank ALL alternatives by 0.2*estimated power before top-k filtering.

The power coefficient and normalized overflow penalty are unchanged, so
this removes the occupancy/load terms without changing their relative scale.
    """
    if kwargs:
        raise ValueError('Diagnostic wrapper expects positional candidate inputs')
    pars, _, _, _, _, _, cfg, *_ = args
    expanded = list(args)
    expanded[6] = dataclasses.replace(cfg, candidate_count=4)
    candidates = original(*expanded)
    duration = pars.slots_per_frame * pars.slot_len
    fraction = 1 - cfg.ho_interruption_ms / (1000 * duration)
    for item in candidates:
        power = pars.p_macro if item.bs == 0 else pars.p_micro
        item.base_cost = cfg.optimizer_energy_weight * item.required_rb * fraction * power
    return sorted(candidates, key=lambda c: (c.base_cost, c.bs))[:cfg.candidate_count]


class OptimizerHook:
    def __init__(self, variant):
        self.spec = VARIANTS[variant]
        self.diagnostics = []

    def __call__(self, **c):
        spec = self.spec
        args, cfg = c['args'], c['config']
        caps = np.array([args.num_RB_macro] + [args.num_RB_micro] * 4)
        records, load, fixed = c['records'], c['load'], c['allocated_rb']
        no_bf = c['no_bf_gain']
        if spec['gain'] == 'mean':
            no_bf = {v: np.r_[c['no_bf_gain'][v][0], beam_average_gain_db(r['h'])]
                     for v, r in records.items()}
            records = optimizer_records(c, no_bf)
            cfg = dataclasses.replace(cfg, information_mode='shared_prediction')
        if spec['load'] == 'clip':
            load = np.clip(load, 0, 1)  # Deliberately leaves fixed-user reservation unchanged.
        if spec['load'] == 'bounded' or spec['gain'] == 'mean':
            connection = {v: c['learners'][v].action for v in c['records']}
            gains = {v: no_bf[v].copy() for v in connection}
            for v, bs in connection.items():
                gains[v][bs] = c['serving_gain'][v]
            rates = {v: args.data_rate for v in connection}
            if spec['load'] == 'bounded':
                rb = estimate_bounded(args, connection, gains, no_bf, rates)
                load = rb / caps
            else:
                rb = alg_utils.estimate_num_RB_allocated_perBS(args, connection,
                    np.zeros((5, 2)), list(connection), gains, rates, infer_g_dict=no_bf)
                load = np.clip(rb / caps, 0, 1.5)
            fixed = fixed_allocation(args, c['learners'], c['backlog'], c['serving_gain'],
                                     no_bf, load, rb, cfg)
        cfg = dataclasses.replace(cfg, candidate_count=spec['candidates'])
        self.diagnostics.append(dict(frame=float(c['frame']), load=np.asarray(load).tolist(),
            original_load=np.asarray(c['load']).tolist(),
            previous_realized_load=np.asarray(c['feedback_load']).tolist(),
            fixed_per_bs=[sum(fixed[v] for v in fixed if c['learners'][v].action == bs)
                          for bs in range(5)]))
        return dict(records=records, load=load, allocated_rb=fixed, config=cfg)


class Observer:
    def __init__(self, hook):
        self.hook = hook
        self.candidates = {}
        self.events = []
        self.original_candidates = om._candidate_links
        self.original_optimize = sim.optimize_triggered_targets

    def candidate(self, *args, **kwargs):
        if self.hook.spec['objective'] == 'power':
            result = power_candidates(self.original_candidates, *args, **kwargs)
        else:
            result = self.original_candidates(*args, **kwargs)
        self.candidates[args[1]] = result
        return result

    def optimize(self, *args, **kwargs):
        self.candidates.clear()
        result = self.original_optimize(*args, **kwargs)
        pars, records, learners, triggered, backlog, allocated, load, cfg = args[:8]
        if not triggered:
            return result
        fixed = np.zeros(5)
        for v, state in learners.items():
            if v not in triggered and v in records:
                fixed[state.action] += allocated.get(v, 0)
        cap = np.array([pars.num_RB_macro] + [pars.num_RB_micro] * 4)
        fraction = 1 - cfg.ho_interruption_ms / (1000 * pars.slots_per_frame * pars.slot_len)
        choices = []
        for v, target in result.targets.items():
            cs = self.candidates[v]
            selected = next(x for x in cs if x.bs == target)
            estimated_power = selected.required_rb * fraction * (pars.p_micro if target else pars.p_macro)
            choices.append(dict(vehicle=str(v), source=int(learners[v].action), target=int(target),
                cheapest=bool(target == cs[0].bs),
                micro_lower_power=any(x.bs > 0 and x.required_rb * fraction * pars.p_micro < estimated_power for x in cs),
                micro_fits=any(x.bs > 0 and fixed[x.bs] + x.required_rb <= cap[x.bs] for x in cs),
                candidates=[dataclasses.asdict(x) for x in cs]))
        self.events.append(dict(frame=self.hook.diagnostics[-1]['frame'],
            success=bool(result.solver_success), elapsed_s=result.elapsed_s,
            overflow=result.overflow.tolist(), choices=choices))
        return result


def protocol():
    source = [Path(__file__), ROOT/'experiment/pql_ba_experiment.py',
              ROOT/'experiment/o_mappo_shared_frontend.py', ROOT/'experiment/o_mappo_optimizer_information.py',
              ROOT/'experiment/revision_pipeline.py', *sorted((ROOT/'utils').glob('*.py'))]
    return dict(rate_mbps=13, traffic_seed=1, interval=[800, 830], warmup_frames=2,
        actor=str(POLICY), actor_sha256=digest(POLICY), timeline=str(TIMELINE),
        timeline_sha256=digest(TIMELINE), variants=VARIANTS,
        reference=str(REFERENCE), reference_sha256=digest(REFERENCE),
        information='Current ground-truth CSI; optimizer-only intervention; no future CSI',
        unchanged='actor weights/input mapping, hierarchical32 BF, OTR-RA inputs/rule, HO10ms, explicit-RB service',
        code={str(p.relative_to(ROOT)): digest(p) for p in source})


def case(root, variant, device):
    frozen = read(root/'protocol.json')
    if frozen != protocol():
        raise ValueError('Frozen experiment changed')
    torch.set_num_threads(1)
    single_thread_solvers()
    alg_utils.km_algorithm = km_algorithm_compiled
    with TIMELINE.open('rb') as f:
        timeline = pickle.load(f)
    args = paper_args(13e6)
    args.device = torch.device('cpu')
    traffic = make_paired_traffic(args, timeline, 1)
    policy = om.OMAPPPolicy.load(str(POLICY))
    assert policy.config.actor_hidden_sizes == (64, 64)
    assert policy.config.candidate_gain_mode == 'search'
    weights = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    hook = OptimizerHook(variant)
    observer = Observer(hook)
    start = time.monotonic()
    with patch.object(om, '_candidate_links', observer.candidate), \
         patch.object(sim, 'optimize_triggered_targets', observer.optimize):
        result = sim.run_sim_o_mappo(args, MICRO_BS_LOCATIONS, timeline, policy,
            seed=1, prt=False, optimizer_solver='milp', traffic_trace=traffic,
            ho_interruption_ms=10, paired_fading_seed=1, physics_device=device,
            directional_service=True, optimizer_input_hook=hook,
            progress_callback=lambda n,t: print('FRAME',n,t,flush=True) if n%100==0 else None)
    for k, v in policy.actor.state_dict().items():
        assert torch.equal(v, weights[k])
    metrics, raw = extract(args, result, traffic, 2)
    metrics.update(handovers=int(result.handover_record.sum()),
        optimizer_failures=int(result.optimizer_failure_record.sum()),
        optimizer_calls=len(observer.events),
        mean_overflow_rb=float(result.optimizer_overflow_record[2:].mean()))
    assert traffic['sha256'] == read(REFERENCE)['traffic_sha256']
    if variant == 'baseline':
        for k, v in read(REFERENCE)['metrics'].items():
            np.testing.assert_allclose(metrics[k], v, rtol=1e-10, atol=1e-10)
        with np.load(REFERENCE.with_suffix('.npz')) as old:
            for k in raw:
                np.testing.assert_equal(raw[k], old[k])
    folder = root/'runs'
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{variant}.npz', **raw)
    atomic_json(folder/f'{variant}_diagnostics.json', dict(inputs=hook.diagnostics, optimization=observer.events))
    row = dict(variant=variant, settings=VARIANTS[variant], metrics=metrics,
        protocol_sha256=digest(root/'protocol.json'), actor_sha256=digest(POLICY),
        traffic_sha256=traffic['sha256'], frames=len(result.energy_record),
        raw_sha256=digest(folder/f'{variant}.npz'),
        diagnostics_sha256=digest(folder/f'{variant}_diagnostics.json'),
        elapsed_s=time.monotonic()-start, actual_rb_utilization=(raw['rb_per_bs'][2:].mean(0)
            / np.array([133,66,66,66,66])).tolist())
    atomic_json(folder/f'{variant}.json', row)
    print('COMPLETE', variant, metrics, flush=True)


def summarize(root):
    rows = []
    sha = digest(root/'protocol.json')
    for variant in VARIANTS:
        row = read(root/'runs'/f'{variant}.json')
        assert row['protocol_sha256'] == sha
        assert row['actor_sha256'] == digest(POLICY)
        for suffix, field in [('.npz', 'raw_sha256'), ('_diagnostics.json', 'diagnostics_sha256')]:
            assert digest(root/'runs'/f'{variant}{suffix}') == row[field]
        d = read(root/'runs'/f'{variant}_diagnostics.json')
        loads = np.array([x['load'] for x in d['inputs']])
        choices = [c for e in d['optimization'] for c in e['choices']]
        macro = [c for c in choices if c['target'] == 0]
        row['diagnostics_summary'] = dict(mean_input_load=loads.mean(0).tolist(),
            micro_load_gt1_fraction=float((loads[:,1:]>1+1e-10).mean()),
            target_assignments=len(choices), macro_targets=len(macro),
            macro_cheapest=sum(c['cheapest'] for c in macro),
            macro_with_lower_power_micro=sum(c['micro_lower_power'] for c in macro))
        rows.append(row)
    assert len({r['traffic_sha256'] for r in rows}) == 1
    meet = read(ROOT/'experiment/results/revision_directional_20260922/grid/runs/meet_cobra_rate13_seed1.json')
    assert meet['traffic_sha256'] == rows[0]['traffic_sha256']
    atomic_json(root/'summary.json', dict(runs=rows, meet_cobra_reference=meet,
        scope='13 Mbps / traffic seed 1 only; frozen actor; optimizer-only closed-loop ablation',
        baseline_raw_arrays_match=True))
    for row in rows:
        print(row['variant'], {k:row['metrics'][k] for k in ('power_w','violation_percent','p99_proxy_ms','macro_association_percent')}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['run','case','summarize'])
    p.add_argument('--root', type=Path, default=OUTPUT)
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    a = p.parse_args()
    if a.command == 'case':
        case(a.root, a.variant, a.device)
        return
    if a.command == 'summarize':
        summarize(a.root)
        return
    a.root.mkdir(parents=True, exist_ok=True)
    lock = (a.root/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = protocol()
    if (a.root/'protocol.json').exists() and read(a.root/'protocol.json') != frozen:
        raise ValueError('Protocol changed; use a new root')
    atomic_json(a.root/'protocol.json', frozen)
    (a.root/'logs').mkdir(exist_ok=True)
    devices = a.devices.split(',')
    def worker(device, variants):
        for variant in variants:
            path = a.root/'runs'/f'{variant}.json'
            if path.exists():
                row = read(path)
                assert row['protocol_sha256'] == digest(a.root/'protocol.json')
                assert row['raw_sha256'] == digest(path.with_suffix('.npz'))
                continue
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'case',
                '--root', str(a.root), '--variant', variant, '--device', device]
            with (a.root/'logs'/f'{variant}.log').open('a') as log:
                subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONHASHSEED='0'))
            print('DONE', variant, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, list(VARIANTS)[i::len(devices)]) for i,device in enumerate(devices)]
        for future in futures:
            future.result()
    summarize(a.root)


if __name__ == '__main__':
    main()
