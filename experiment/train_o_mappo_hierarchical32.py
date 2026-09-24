#!/usr/bin/env python3
"""Balanced, multi-start PPO with search-consistent target evaluation.

Training uses the existing frame surrogate; convergence is checked per load
on held-out times, and selected policies must additionally pass exact GPU
evaluation. No test-time checkpoint selection is performed.
"""
import argparse
import collections
import concurrent.futures
import dataclasses
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['KMP_AFFINITY'] = 'disabled'
CPU_POOL = tuple(sorted(os.sched_getaffinity(0)))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiment.pql_ba_experiment import DEFAULT_TRAIN_PATH, paper_args, temporal_slice
from experiment.o_mappo_shared_frontend import single_thread_solvers, MICRO_BS_LOCATIONS
from experiment.revision_pipeline import atomic_json, digest, extract, native
from utils.o_mappo import (OMAPPOConfig, OMAPPPolicy, OMAPPOMemory,
    o_mappo_reward_presets, run_fluid_o_mappo_episode)
from utils.o_mappo_sim import run_sim_o_mappo
from utils.ho_utils import make_paired_traffic

RATES = list(range(1, 36, 2))
DEFAULT_ROOT = ROOT / 'experiment/results/o_mappo_h32_retrained_20260924_v5'
SOURCE = ROOT / DEFAULT_TRAIN_PATH
DATA = None


def read(path):
    return json.loads(Path(path).read_text())


def save_policy(policy, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.pt')
    policy.save(str(temp))
    os.replace(temp, path)


def configuration():
    return OMAPPOConfig(beam_search_variant='hierarchical32', candidate_gain_mode='search',
        reward_sweep_reference_pilots=256, ho_interruption_ms=10, optimizer_solver='milp',
        torch_threads=1, actor_learning_rate=3e-4, critic_learning_rate=3e-4,
        hidden_sizes=(64,), batch_size=256, ppo_epochs=4)


def initialize_worker(root):
    global DATA
    torch.set_num_threads(1)
    single_thread_solvers()
    root=Path(root)
    with np.load(root/'channel_index.npz',allow_pickle=True) as archive:
        index={key:archive[key] for key in archive.files}
    DATA=(np.load(root/'channels.npy',mmap_mode='r'),index)


def prepare_arrays(root):
    """Shared read-only mmap allows spawn without copying 5 GB per worker."""
    if (root/'arrays_complete.json').exists():
        return
    with SOURCE.open('rb') as stream:
        timeline=pickle.load(stream)
    timeline=temporal_slice(timeline,200,740)
    with (root/'exact_validation.pkl').open('wb') as stream:
        pickle.dump(temporal_slice(timeline,710,740),stream,protocol=4)
    frames=np.array(list(timeline))
    counts=np.array([len(records) for records in timeline.values()])
    offsets=np.r_[0,np.cumsum(counts)]
    total=int(offsets[-1])
    first=next(iter(next(iter(timeline.values())).values()))['h']
    channels=np.lib.format.open_memmap(root/'channels.npy',mode='w+',dtype=first.dtype,
                                      shape=(total,*first.shape))
    names=[]
    pos=np.empty((total,2)); speed=np.empty(total); angle=np.empty(total)
    index=0
    for records in timeline.values():
        for vehicle,record in records.items():
            names.append(vehicle)
            channels[index]=record['h']; pos[index]=record['pos']
            speed[index]=record.get('v',0); angle[index]=record.get('angle',0)
            index+=1
    channels.flush()
    np.savez(root/'channel_index.npz',frames=frames,offsets=offsets,names=np.array(names,dtype=object),
             positions=pos,speeds=speed,angles=angle)
    atomic_json(root/'arrays_complete.json',dict(frames=len(frames),records=total,
                channel_sha256=digest(root/'channels.npy'),index_sha256=digest(root/'channel_index.npz')))


def array_slice(start,length):
    channels,index=DATA
    frames=index['frames']; offsets=index['offsets']; names=index['names']
    positions=index['positions']; speeds=index['speeds']; angles=index['angles']
    selected=np.flatnonzero((frames>=start-1e-8)&(frames<=start+length-.1+1e-8))
    return collections.OrderedDict((float(frames[i]),collections.OrderedDict(
        (names[j],dict(h=channels[j],pos=positions[j],v=float(speeds[j]),angle=float(angles[j])))
        for j in range(offsets[i],offsets[i+1]))) for i in selected)


def bind_worker():
    # Some OpenMP builds pin every forked worker to CPU 0 despite a larger
    # allowed mask. Assign distinct cores AFTER checkpoint/thread setup.
    identity = mp.current_process()._identity
    if identity:
        core = CPU_POOL[(identity[-1]-1) % len(CPU_POOL)]
        os.sched_setaffinity(0, {core})


def rollout(job):
    policy_path, rate, start, length, seed, learn = job
    policy = OMAPPPolicy.load(str(policy_path), seed=seed)
    bind_worker()
    timeline = array_slice(start,length)
    result, memory = run_fluid_o_mappo_episode(paper_args(rate * 1e6), timeline, policy,
        o_mappo_reward_presets()['qos_energy020_load1'], rate, seed=seed, learn=learn, collect_only=True)
    if learn:
        for t in memory.transitions:
            t.vehicle = (rate, start, seed, t.vehicle)
        return result, memory.transitions
    # Fixed validation trajectories provide naturally distributed probe states.
    states = np.stack([t.local_state for t in memory.transitions])
    return result, states[np.linspace(0, len(states)-1, min(256, len(states)), dtype=int)]


def score(rows, reference):
    costs = np.array([r['queue_violation_percent'] + .02 * r['average_system_power_w'] for r in rows])
    ratios = costs / reference
    return float(.5 * ratios.mean() + .5 * ratios.max()), costs


def convergence(history):
    if len(history) < 5:
        return dict(passed=False, reason='fewer than five validation checkpoints')
    tail = history[-5:]
    costs = np.array([x['costs'] for x in tail])
    relative_span = np.ptp(costs, axis=0) / np.maximum(costs.mean(0), 1e-3)
    scores = np.array([x['score'] for x in tail])
    score_span = float(np.ptp(scores) / max(abs(scores.mean()), 1e-3))
    flips = max(x.get('probe_flip_fraction', 1) for x in tail[1:])
    drift = max(x.get('probe_probability_change', 1) for x in tail[1:])
    return dict(passed=bool(tail[-1]['round'] >= 60 and relative_span.max() < .05
                           and score_span < .02 and flips < .01 and drift < .005),
                maximum_per_load_cost_span=float(relative_span.max()), score_span=score_span,
                maximum_probe_flip_fraction=flips, maximum_probe_probability_change=drift,
                window_rounds=[x['round'] for x in tail])


def train(args):
    global DATA
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = configuration()
    data_root = args.data_root or args.root
    seeds = [int(x) for x in args.training_seeds.split(',')]
    manifest = dict(config=dataclasses.asdict(config), training_seeds=seeds, rates=RATES,
        source=str(SOURCE), source_sha256=digest(SOURCE), train_interval=[200, 650],
        data_root=str(data_root.resolve()),
        validation_interval=[700, 710], exact_validation_interval=[710, 740],
        test_interval=[800, 830], rollout_seconds=args.rollout_seconds,
        validation_every=args.validate_every,
        balanced_update='one equal-duration rollout for each of all 18 loads, jointly updated',
        reward='qos_energy020_load1; sweep term retains fixed 256-probe normalization',
        convergence_criteria=dict(min_rounds=60, validation_checkpoints=5,
            per_load_cost_relative_span=.05, balanced_score_relative_span=.02,
            probe_action_flip_fraction=.01, probe_probability_mean_change=.005),
        selection='mean and worst normalized per-load cost on held-out time; exact validation before test',
        code={str(p.relative_to(ROOT)):digest(p) for p in
              [Path(__file__), ROOT/'utils/o_mappo.py', ROOT/'utils/o_mappo_sim.py', ROOT/'utils/hierarchical_beam.py']})
    path = args.root / 'training_protocol.json'
    if path.exists():
        if read(path) != native(manifest):
            raise ValueError('Training protocol changed; use a new root')
    else:
        atomic_json(path, native(manifest))
    print('PREPARE SHARED ARRAYS', SOURCE, flush=True)
    prepare_arrays(data_root)
    if data_root.resolve() != args.root.resolve():
        shutil.copyfile(data_root/'exact_validation.pkl',args.root/'exact_validation.pkl')
    # Fresh interpreters avoid this server's Intel OpenMP post-fork assertion.
    # Immutable channel samples remain shared through the file-backed mmap.
    torch.set_num_threads(1)
    single_thread_solvers()
    policies, histories, probes, last_probs, best_scores = {}, {}, {}, {}, {}
    for seed in seeds:
        folder = args.root / f'train_seed{seed}'
        folder.mkdir(exist_ok=True)
        if (folder / 'last.pt').exists():
            policies[seed] = OMAPPPolicy.load(str(folder / 'last.pt'), seed=seed, load_optimizers=True)
            histories[seed] = read(folder / 'validation_history.json')
            probes[seed] = np.load(folder / 'probes.npy')
            best_scores[seed] = min(x['score'] for x in histories[seed])
            with torch.no_grad():
                last_probs[seed] = policies[seed].actor(torch.from_numpy(probes[seed])).softmax(-1).numpy()
        else:
            policies[seed] = OMAPPPolicy(config=dataclasses.replace(config), seed=seed)
            histories[seed] = []
            best_scores[seed] = float('inf')
            save_policy(policies[seed], folder / 'round0000.pt')
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=mp.get_context('spawn'), initializer=initialize_worker,
            initargs=(str(data_root.resolve()),)) as pool:
        reference_path = args.root / 'reference.json'
        if reference_path.exists():
            reference = np.array(read(reference_path)['costs'])
        else:
            anchor = OMAPPPolicy.load(str(ROOT / 'experiment/results/o_mappo/final_load1/final_policy.pt'))
            anchor.config = dataclasses.replace(anchor.config, beam_search_variant='hierarchical32',
                candidate_gain_mode='search', ho_interruption_ms=10, optimizer_solver='milp',
                reward_sweep_reference_pilots=256)
            save_policy(anchor, args.root / 'reference.pt')
            jobs = [(args.root / 'reference.pt', r, 700, 10, 9900+r, False) for r in RATES]
            anchor_outputs = list(pool.map(rollout, jobs))
            anchor_rows = [x[0] for x in anchor_outputs]
            np.save(args.root/'anchor_probes.npy', np.concatenate([x[1] for x in anchor_outputs]))
            reference = np.maximum([r['queue_violation_percent'] + .02*r['average_system_power_w']
                                    for r in anchor_rows], 1e-3)
            atomic_json(reference_path, dict(costs=reference.tolist(), rows=anchor_rows))
        for round_no in range(0, args.rounds + 1):
            unfinished = [s for s in seeds if not convergence(histories[s])['passed']]
            if not unfinished:
                break
            active = [s for s in unfinished if round_no > max([x['round'] for x in histories[s]], default=-1)]
            if not active:
                continue
            if round_no:
                jobs = []
                for seed in active:
                    folder = args.root / f'train_seed{seed}'
                    policy = policies[seed]
                    # Decay to a nonzero LR, not an artificial zero-LR plateau.
                    fraction = min(round_no / 120, 1)
                    learning_rate = 3e-5 + (3e-4-3e-5)*.5*(1+np.cos(np.pi*fraction))
                    policy.config.entropy_coefficient = .002 + .008*(1-fraction)
                    for optimizer in (policy.actor_optimizer, policy.critic_optimizer):
                        for group in optimizer.param_groups:
                            group['lr'] = learning_rate
                    save_policy(policy, folder / 'rollout.pt')
                    for rate in RATES:
                        rng = np.random.default_rng(np.random.SeedSequence([seed, round_no, rate]))
                        start = float(rng.integers(2000, int((650-args.rollout_seconds)*10))) / 10
                        jobs.append((folder/'rollout.pt', rate, start, args.rollout_seconds,
                                     seed*100000 + round_no*100 + rate, True))
                outputs = list(pool.map(rollout, jobs))
                for index, seed in enumerate(active):
                    group = outputs[index*len(RATES):(index+1)*len(RATES)]
                    memory = OMAPPOMemory()
                    for _, transitions in group:
                        memory.transitions.extend(transitions)
                    update = policies[seed].update(memory)
                    row = dict(round=round_no, seed=seed, learning_rate=learning_rate,
                               elapsed_s=time.monotonic()-started, update=update,
                               loads=[x[0] for x in group])
                    atomic_json(args.root/f'train_seed{seed}/updates'/f'{round_no:04d}.json', row)
                    save_policy(policies[seed], args.root/f'train_seed{seed}/latest_update.pt')
                    print('TRAIN', seed, round_no, 'transitions', len(memory),
                          'reward', np.mean([x[0]['mean_event_reward'] for x in group]), flush=True)
                del outputs
            if round_no % args.validate_every == 0:
                jobs = []
                for seed in active:
                    checkpoint = args.root/f'train_seed{seed}/round{round_no:04d}.pt'
                    save_policy(policies[seed], checkpoint)
                    jobs.extend([(checkpoint, r, 700, 10, 9900+r, False) for r in RATES])
                outputs = list(pool.map(rollout, jobs))
                for index, seed in enumerate(active):
                    group = outputs[index*len(RATES):(index+1)*len(RATES)]
                    rows = [x[0] for x in group]
                    value, costs = score(rows, reference)
                    if seed not in probes:
                        probes[seed] = np.concatenate([np.load(args.root/'anchor_probes.npy')]
                                                     + [x[1] for x in group])
                        np.save(args.root/f'train_seed{seed}/probes.npy', probes[seed])
                    with torch.no_grad():
                        probabilities = policies[seed].actor(torch.from_numpy(probes[seed])).softmax(-1).numpy()
                    previous = last_probs.get(seed, probabilities)
                    row = dict(round=round_no, rows=rows, score=value, costs=costs.tolist(),
                        probe_flip_fraction=float(np.mean(probabilities.argmax(-1)!=previous.argmax(-1))),
                        probe_probability_change=float(np.abs(probabilities-previous).mean()))
                    last_probs[seed] = probabilities
                    histories[seed].append(row)
                    if value < best_scores[seed]:
                        best_scores[seed] = value
                        save_policy(policies[seed], args.root/f'train_seed{seed}/best.pt')
                        atomic_json(args.root/f'train_seed{seed}/best.json', row)
                    save_policy(policies[seed], args.root/f'train_seed{seed}/last.pt')
                    atomic_json(args.root/f'train_seed{seed}/validation_history.json', histories[seed])
                    state = convergence(histories[seed])
                    atomic_json(args.root/f'train_seed{seed}/convergence.json', state)
                    print('VALIDATE', seed, round_no, 'score', value, 'convergence', state, flush=True)
                atomic_json(args.root/'progress.json', dict(round=round_no, elapsed_s=time.monotonic()-started,
                    seeds={s:dict(best_score=best_scores[s], convergence=convergence(histories[s])) for s in seeds}))
    atomic_json(args.root/'training_complete.json', dict(elapsed_s=time.monotonic()-started,
        seeds={s:convergence(histories[s]) for s in seeds}))


def exact(args):
    torch.set_num_threads(1)
    single_thread_solvers()
    from utils.compiled_matching import km_algorithm_compiled
    from utils import alg_utils
    alg_utils.km_algorithm = km_algorithm_compiled
    with args.timeline.open('rb') as stream:
        timeline = pickle.load(stream)
    if args.seconds:
        timeline = temporal_slice(timeline, min(timeline), min(timeline)+args.seconds)
    params = paper_args(args.rate*1e6)
    params.device = torch.device('cpu')
    traffic = make_paired_traffic(params, timeline, args.seed)
    result = run_sim_o_mappo(params, MICRO_BS_LOCATIONS, timeline,
        OMAPPPolicy.load(str(args.policy)), seed=args.seed, prt=False, optimizer_solver='milp',
        traffic_trace=traffic, ho_interruption_ms=10, paired_fading_seed=args.seed,
        physics_device=args.device, directional_service=True,
        progress_callback=lambda n,t: print('FRAME', n,t,flush=True) if n%100==0 else None)
    metrics, raw = extract(params, result, traffic, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), **raw)
    atomic_json(args.output, dict(metrics=metrics, rate_mbps=args.rate, seed=args.seed,
        checkpoint=str(args.policy), checkpoint_sha256=digest(args.policy),
        traffic_sha256=traffic['sha256'], frames=len(result.energy_record)))
    print('COMPLETE', args.output, metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--training-seeds', default='11,22,33')
    p.add_argument('--rounds', type=int, default=160)
    p.add_argument('--rollout-seconds', type=float, default=10)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--validate-every', type=int, default=10)
    p = sub.add_parser('exact')
    p.add_argument('--timeline', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--rate', type=int, required=True)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=float, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'train': train(args)
    else: exact(args)


if __name__ == '__main__':
    main()
