#!/usr/bin/env python3
"""Shared-observation O-MAPPO: frozen stateful frontend, retraining and paired tests.

Separate prepare/train/evaluate commands make long experiments resumable.
The test trace is never used for checkpoint selection. Existing checkpoints
and published figures are not overwritten. Raw H belongs to the environment;
the new target optimizer accepts only predicted gains and current position.
"""
import argparse
import collections
import concurrent.futures
import dataclasses
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARGV = list(sys.argv)
import numpy as np
import torch
from experiment.pql_ba_experiment import paper_args, temporal_slice, MICRO_BS_LOCATIONS
from experiment.benchmark_nn_overhead import digest
from experiment.benchmark_stateful_nn_overhead import load_selected_models, DEFAULT_CHECKPOINT_ROOT
from experiment.compare_stateful_prediction import StatefulPredictor
from experiment.prepare_stateful_trajectories import DEFAULT_SOURCE, frame_values
from utils.data_utils import preprocess_input_np
from utils.ho_utils import make_paired_traffic
from utils.o_mappo import (OMAPPOConfig, OMAPPPolicy, o_mappo_reward_presets,
                           run_fluid_o_mappo_episode)
from utils.o_mappo_sim import run_sim_o_mappo
from utils.sim_utils import run_sim_withUMa
from utils.alg_utils import HO_EE_GAP_APX_SINR_conservative_adaptive
sys.argv = ARGV

OUTPUT = ROOT / "experiment/results/o_mappo_shared_frontend_20260917"
TEST = ROOT / "data4sim/lbd1.00_800_830_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl"
LEGACY = ROOT / "experiment/results/o_mappo/final_load1/final_policy.pt"
RATES = (1, 7, 13, 19, 27, 35)
CONTEXT = None


class FrameProgress:
    def __init__(self, name, total):
        self.name, self.total, self.count = name, total, 0
        self.started = time.monotonic()

    def tick(self, count, total):
        self.count = count
        if count % 50 == 0 or count == total:
            print(f"PROGRESS {self.name} {count}/{total} "
                  f"elapsed={time.monotonic()-self.started:.1f}s", flush=True)

    def append(self, diagnostic):
        self.tick(self.count + 1, self.total)


def single_thread_solvers():
    # HiGHS otherwise creates 128 threads per worker on this server, despite
    # OMP_NUM_THREADS=1. Pass its native option through SciPy's wrappers.
    import utils.alg_utils as alg
    import utils.o_mappo as om
    if getattr(alg.linprog, "_shared_single_thread", False):
        return
    def wrap(function):
        def solve(*args, **kwargs):
            kwargs["options"] = dict(kwargs.get("options", {}), threads=1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unrecognized options.*")
                return function(*args, **kwargs)
        solve._shared_single_thread = True
        return solve
    alg.linprog = wrap(alg.linprog)
    om.milp = wrap(om.milp)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path, value):
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def prepare(output, gpu):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    models, inventory = load_selected_models(DEFAULT_CHECKPOINT_ROOT)
    for model in models.values():
        model.to(device)
    manifest = dict(rollback_git="cdd095a58dc99504629522532a0f7cb751a99f81",
                    frontend=inventory, inference_device=str(device),
                    training_interval=[200, 700], validation_interval=[700.1, 710],
                    validation_cache_interval=[700.1, 730],
                    test_interval=[800, 830], observation_noise_seed=20260917,
                    prediction_alignment="CSI at x predicts x+1; no future label input",
                    test_selection=False)
    for name, source in (("train", DEFAULT_SOURCE), ("test", TEST)):
        destination = output / f"{name}_prepared.pkl"
        if destination.exists():
            print("REUSE", destination, flush=True)
            continue
        timeline = read_pickle(source)
        if name == "train":
            timeline = temporal_slice(timeline, 200, 730)
        rng = np.random.default_rng(20260917)
        streams = {task: StatefulPredictor(model) for task, model in models.items()}
        prepared = collections.OrderedDict()
        started = time.monotonic()
        previous_frame = None
        with torch.inference_mode():
            for fi, (frame, records) in enumerate(timeline.items()):
                ids = sorted(records, key=str)
                if previous_frame is not None and not np.isclose(frame - previous_frame, .1):
                    for stream in streams.values():
                        stream.reset()
                if name == "train":
                    clean, _, _, _ = frame_values([records[v] for v in ids])
                    noise = (rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)) * np.sqrt(1e-14 / 2)
                    pilots = preprocess_input_np((clean + noise).astype(np.complex64)).astype(np.float32)
                else:
                    pilots = np.stack([records[v]["CSI_preprocessed"][-1] for v in ids]).astype(np.float32)
                latest = torch.as_tensor(pilots[:, None, :], device=device)
                predictions = {}
                for task, stream in streams.items():
                    value = stream.step(ids, latest)
                    if task == "beam":
                        value = value.topk(5, dim=-1, sorted=True).indices
                    else:
                        scale, offset = models[task].params_norm
                        value = scale * (value - offset)
                    predictions[task] = value.cpu().numpy()
                prepared[frame] = {}
                for i, v in enumerate(ids):
                    record = dict(records[v])
                    record["CSI_preprocessed"] = pilots[i:i+1]
                    record["shared_prediction"] = dict(
                        gain=predictions["desired_gain"][i],
                        interference=predictions["interfering_gain"][i],
                        beam=predictions["beam"][i])
                    prepared[frame][v] = record
                previous_frame = frame
                if fi % 500 == 0:
                    print(f"PREP {name} {fi}/{len(timeline)} {time.monotonic()-started:.1f}s", flush=True)
        save_pickle(destination, prepared)
        manifest[name] = dict(source=str(source), source_sha256=digest(source),
                              frames=len(prepared), cache_sha256=digest(destination))
        write_json(output / "frontend_manifest.json", manifest)
        del timeline, prepared
    print("PREP COMPLETE", flush=True)


def train(output, seed, episodes, state_variant="pilot", cache_root=None):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root or output
    timeline = read_pickle(cache_root / "train_prepared.pkl")
    training = temporal_slice(timeline, 200, 700)
    validation = temporal_slice(timeline, 700.1, 710)
    config = OMAPPOConfig(state_variant=state_variant, information_mode="shared_prediction",
                          hidden_sizes=(64,), torch_threads=1, ho_interruption_ms=10)
    policy = OMAPPPolicy(config, seed=seed)
    reward = o_mappo_reward_presets()["qos_energy020_load1"]
    args = paper_args()
    destination = output / f"training_seed{seed}"
    destination.mkdir(exist_ok=True)
    if (destination / "training.json").exists():
        raise FileExistsError(f"Preserve completed training; choose another output: {destination}")
    write_json(destination / "protocol.json", dict(
        state_variant=state_variant, cache_root=str(cache_root.resolve()),
        frontend_manifest=json.loads((cache_root / "frontend_manifest.json").read_text()),
        training_seed=seed, episodes=episodes, actor_input_dim=policy.local_dim,
        critic_input_dim=policy.global_dim,
        actor_initialization=("matched gain-report common weights; zero derived columns"
                              if state_variant == "gain_derived" else "default random initialization"),
        test_used_for_selection=False))
    rng = np.random.default_rng(seed)
    history, validation_history = [], []
    best = float("inf")
    started = time.monotonic()
    # Each round covers all training loads; independent random 30-s segments.
    for episode in range(episodes):
        start_tick = int(rng.integers(2000, 6700))
        sample = temporal_slice(training, start_tick / 10, start_tick / 10 + 30)
        rate = RATES[episode % len(RATES)]
        result = run_fluid_o_mappo_episode(args, sample, policy, reward, rate,
                                         seed=seed * 1000 + episode, learn=True)
        result.update(episode=episode + 1, start=start_tick / 10)
        history.append(result)
        print(f"TRAIN seed{seed} ep{episode+1}/{episodes} rate{rate} "
              f"U={result['queue_violation_percent']:.3f} P={result['average_system_power_w']:.3f} "
              f"trigger={result['trigger_ratio']:.3f} elapsed={time.monotonic()-started:.0f}s", flush=True)
        if (episode + 1) % 12 == 0 or episode + 1 == episodes:
            rows = [run_fluid_o_mappo_episode(args, validation, policy, reward, rate,
                                             seed=2026, learn=False) for rate in (7, 19, 35)]
            score = max(x["queue_violation_percent"] for x in rows) + .01 * np.mean(
                [x["average_system_power_w"] for x in rows])
            validation_history.append(dict(episode=episode+1, score=float(score), results=rows))
            if score < best:
                best = score
                policy.save(str(destination / "best_policy.pt"))
                write_json(destination / "selection.json", dict(seed=seed, episode=episode+1,
                           score=float(score), metric="maximum validation fluid U[%] + 0.01 * mean P[W]",
                           validation_interval=[700.1, 710], checkpoint_sha256=digest(destination / "best_policy.pt")))
            policy.save(str(destination / "last_policy.pt"))
            write_json(destination / "training.json", dict(config=dataclasses.asdict(config),
                       reward=dataclasses.asdict(reward), history=history,
                       validation=validation_history, elapsed_s=time.monotonic()-started))
            print(f"VALID seed{seed} ep{episode+1} score={score:.4f} best={best:.4f}", flush=True)


def metric_arrays(args, timeline, result, proposed=False):
    if proposed:
        energy, handover, _, violation, qavg, pilot, rb, queues = result
    else:
        energy, handover, violation, qavg, pilot, rb, queues = (
            result.energy_record, result.handover_record, result.violation_probability_record,
            result.average_queue_record, result.pilot_record, result.rb_allocated_record,
            result.queue_per_vehicle_record)
    mask = slice(2, None)
    rows = [(fi, str(v), q) for fi, values in queues.items() for v, q in values.items()]
    qdata = np.stack([row[2] for row in rows])
    frame = np.array([row[0] for row in rows])
    active = sum(len(records) for records in list(timeline.values())[3:])
    proxy = qdata[frame >= 2].ravel() / args.data_rate * 1000
    metrics = dict(power_w=float(np.mean(energy[mask]) / .1),
                   violation_percent=float(np.mean(violation[mask]) * 100),
                   mean_proxy_ms=float(np.mean(qavg[mask]) / args.data_rate * 1000),
                   p99_proxy_ms=float(np.percentile(proxy, 99)),
                   pilots_per_vehicle_slot=float(np.mean(pilot[mask])),
                   handovers_per_vehicle_s=float(np.sum(handover[mask]) / (active * .1)))
    if not proposed:
        metrics["trigger_ratio"] = float(np.sum(result.trigger_record[mask]) / max(1, np.sum(result.decision_record[mask])))
        metrics["macro_association_percent"] = float(100 * sum(
            sum(bs == 0 for bs in values.values()) for fi, values in result.association_record.items() if fi >= 2) / active)
    raw = dict(energy_j=energy, handover_count=handover, violation_probability=violation,
               mean_queue_bits=qavg, pilots=pilot, rb_per_bs=rb, queue_bits=qdata,
               queue_frame=frame, vehicle=np.array([row[1] for row in rows]))
    return metrics, raw


def evaluate_one(task):
    method, rate, seed, ho_ms = task
    output, timeline, selected, rician, *options = CONTEXT
    vectorized = bool(options[0]) if options else False
    physics_gpu = options[1] if len(options) > 1 else None
    compiled_matching = bool(options[2]) if len(options) > 2 else False
    torch.set_num_threads(1)
    single_thread_solvers()
    if compiled_matching:
        import utils.alg_utils as alg
        from utils.compiled_matching import km_algorithm_compiled
        alg.km_algorithm = km_algorithm_compiled
    np.random.seed(seed)
    args = paper_args(rate * 1e6)
    args.device = torch.device("cpu")
    traffic = make_paired_traffic(args, timeline, seed)
    name = f"{method}_rate{rate}_seed{seed}_ho{ho_ms:g}_t1"
    if not np.isclose(list(timeline)[-1], 830):
        name += f"_end{list(timeline)[-1]:g}"
    if rician:
        name += "_rician"
    if method == "meet_cobra" and vectorized:
        name += "_batch"
    if physics_gpu is not None:
        name += f"_cuda{physics_gpu}"
    path = output / "runs" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    print("EVAL START", name, flush=True)
    started = time.monotonic()
    progress = FrameProgress(name, len(timeline) - 1)
    if method == "meet_cobra":
        cache = {f: {v: r["shared_prediction"] for v, r in records.items()} for f, records in timeline.items()}
        # Non-None sentinels activate cached predictions, never Oracle labels.
        result = run_sim_withUMa(args, MICRO_BS_LOCATIONS, timeline, None, True, True, True,
            HO_func=HO_EE_GAP_APX_SINR_conservative_adaptive, save_pilot=True, K_BF=5,
            prt=False, prediction_cache=cache, traffic_trace=traffic, rician_fading=rician,
            paired_fading_seed=seed if rician else None,
            vectorized_pet_measurement=vectorized,
            physics_device=f"cuda:{physics_gpu}" if physics_gpu is not None else None,
            ho_diagnostics=progress,
            ho_interruption_ms=ho_ms, ho_capacity_correction=True)
    else:
        checkpoint = LEGACY if method == "legacy" else selected
        policy = OMAPPPolicy.load(str(checkpoint))
        torch.set_num_threads(1)
        result = run_sim_o_mappo(args, MICRO_BS_LOCATIONS, timeline, policy, seed=seed,
            prt=False, rician_fading=rician, traffic_trace=traffic, ho_interruption_ms=ho_ms,
            paired_fading_seed=seed if rician else None,
            physics_device=f"cuda:{physics_gpu}" if physics_gpu is not None else None,
            progress_callback=progress.tick,
            optimizer_solver="milp")
    metrics, raw = metric_arrays(args, timeline, result, method == "meet_cobra")
    for sub in ("runs", "raw"):
        (output / sub).mkdir(exist_ok=True)
    np.savez_compressed(output / "raw" / f"{name}.npz", **raw)
    row = dict(method=method, rate_mbps=rate, seed=seed, ho_interruption_ms=ho_ms,
               traffic_sha256=traffic["sha256"], rician_fading=rician,
               vectorized_pet_measurement=(method == "meet_cobra" and vectorized),
               solver_threads=1,
               physics_gpu=physics_gpu,
               fading_generator="torch_float64_cuda" if physics_gpu is not None else "numpy_float64",
               matching_backend=("numba_original_order" if compiled_matching else "python_original_order") if method == "meet_cobra" else "not_applicable",
               metrics=metrics, elapsed_s=time.monotonic()-started,
               actor_state_variant=None if method == "meet_cobra" else policy.config.state_variant,
               checkpoint_sha256=None if method == "meet_cobra" else digest(checkpoint))
    write_json(path, row)
    print("EVAL DONE", name, metrics, flush=True)
    return row


def evaluate(output, methods, rates, seeds, workers, end, ho_ms, rician, vectorized, physics_gpu, compiled_matching, cache_root=None):
    global CONTEXT
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    timeline = temporal_slice(read_pickle((cache_root or output) / "test_prepared.pkl"), 800, end)
    if not set(methods) <= {"legacy", "shared", "report", "gain_report", "gain_derived", "meet_cobra"}:
        raise ValueError("unknown comparison method")
    choices = [json.loads(path.read_text()) | {"path": str(path.parent / "best_policy.pt")}
               for path in output.glob("training_seed*/selection.json")]
    selected = Path(min(choices, key=lambda x: x["score"])["path"]) if choices else None
    needs_policy = bool({"shared", "report", "gain_report", "gain_derived"}.intersection(methods))
    if needs_policy and selected is None:
        raise RuntimeError("No validation-selected shared policy")
    if needs_policy:
        variant = OMAPPPolicy.load(str(selected)).config.state_variant
        expected = {"shared": "pilot", "report": "report", "gain_report": "gain_report", "gain_derived": "gain_derived"}
        if any(variant != expected[m] for m in methods if m in expected):
            raise ValueError("method label does not match selected actor input")
        write_json(output / "selected_policy.json", min(choices, key=lambda x: x["score"]))
    CONTEXT = (output, timeline, selected, rician, vectorized, physics_gpu, compiled_matching)
    if compiled_matching:
        if workers != 1:
            raise ValueError("compiled matching requires --workers 1; do not fork a JIT-initialized process")
        from utils.compiled_matching import km_algorithm_compiled
        km_algorithm_compiled(np.zeros((2, 2)))
    tasks = [(m, r, s, ho_ms) for r in rates for s in seeds for m in methods]
    # GPU inference has already finished in a separate command; workers fork
    # only the read-only CPU timeline, never a live CUDA context.
    if workers == 1:
        results = [evaluate_one(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(evaluate_one, tasks))
    run_set = "rates" + "-".join(map(str, rates)) + "_seeds" + "-".join(map(str, seeds))
    write_json(output / f"summary_{'_'.join(methods)}_ho{ho_ms:g}_{'rician' if rician else 'block'}_{run_set}.json", results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--state-variant", choices=("pilot", "report", "gain_report", "gain_derived"), default="pilot")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--training-seed", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=72)
    parser.add_argument("--methods", default="legacy,shared,meet_cobra")
    parser.add_argument("--rates", default="1,13,27,35")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--test-end", type=float, default=830)
    parser.add_argument("--ho-ms", type=float, default=10)
    parser.add_argument("--rician", action="store_true")
    parser.add_argument("--vectorized-pet", action="store_true")
    parser.add_argument("--physics-gpu", type=int)
    parser.add_argument("--compiled-matching", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output, args.gpu)
    elif args.phase == "train":
        train(args.output, args.training_seed, args.episodes, args.state_variant, args.cache_root)
    else:
        evaluate(args.output, args.methods.split(","), [int(x) for x in args.rates.split(",")],
                 [int(x) for x in args.seeds.split(",")], args.workers, args.test_end, args.ho_ms, args.rician, args.vectorized_pet, args.physics_gpu, args.compiled_matching, args.cache_root)


if __name__ == "__main__":
    main()
