#!/usr/bin/env python3
"""Time the selected two-stage checkpoints on real, batch-one CSI streams.

Reuses the state-exposing forward pass from the prediction-accuracy audit.
No training or system simulation is performed. Existing checkpoints, legacy
benchmarks, and result directories are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.benchmark_nn_overhead import (
    DEFAULT_DATA, BeamPredictionLSTMModel, BestGainPredictionLSTMModel,
    command_output, digest, json_write, matrix_arithmetic, summarize,
    verify_profiler,
)
from experiment.compare_stateful_prediction import forward_with_state
import numpy as np
import torch

DEFAULT_CHECKPOINT_ROOT = ROOT / (
    "experiment/results/stateful_tbptt_unified_split_20260913/stage2_stateful_tbptt"
)
TASKS = ("beam", "desired_gain", "interfering_gain")


class VehicleStatefulPipeline:
    """One vehicle's three predictors, with separate persistent LSTM states."""

    def __init__(self, models, top_k=5):
        self.models = models
        self.top_k = top_k
        self.reset()

    def reset(self):
        self.states = {name: None for name in TASKS}

    def step(self, current_csi, new_trajectory=False):
        if new_trajectory:
            self.reset()
        # One shared conversion, from preprocessed FP32 NumPy CSI to a tensor.
        x = torch.as_tensor(current_csi, dtype=torch.float32).reshape(1, 1, 128)
        outputs = {}
        for name in TASKS:
            output, self.states[name] = forward_with_state(
                self.models[name], x, self.states[name]
            )
            if name == "beam":
                outputs[name] = output.topk(
                    self.top_k, dim=-1, largest=True, sorted=True
                ).indices.cpu().numpy()
            else:
                scale, offset = self.models[name].params_norm
                outputs[name] = (scale * (output - offset)).cpu().numpy()
        return outputs


class StatefulStepModule(torch.nn.Module):
    """Expose one step with a nonzero carried state to the FLOP counter."""

    def __init__(self, model, state):
        super().__init__()
        self.model = model
        self.state = state

    def forward(self, x):
        return forward_with_state(self.model, x, self.state)[0]


def load_selected_models(root):
    models = {
        "beam": BeamPredictionLSTMModel(128, 4, 256),
        "desired_gain": BestGainPredictionLSTMModel(128, 4),
        "interfering_gain": BestGainPredictionLSTMModel(128, 4),
    }
    inventory = {}
    for name, model in models.items():
        path = (root / name / "best.pth").resolve()
        metadata = json.loads((path.parent / "metadata.json").read_text())
        sha = digest(path)
        if sha != metadata["best_checkpoint_sha256"]:
            raise ValueError(f"Selected-checkpoint hash mismatch: {path}")
        if metadata.get("input_normalization") != "paper":
            raise ValueError("This audit requires the paper CSI preprocessing.")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.float().eval()
        inventory[name] = {
            "checkpoint": str(path), "sha256": sha,
            "selected_stage2_epoch": metadata["best_epoch"],
            "selection_metric": "maximum validation Top-1" if name == "beam" else "minimum validation MAE",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "state_dict_bytes": sum(p.numel() * p.element_size() for p in model.state_dict().values()),
            "architecture": str(model),
        }
    return models, inventory


def extract_trajectories(timeline, frame_s=0.1):
    """Split on vehicle departures or missing frames; copy only latest CSI."""
    trajectories, previous = [], {}
    for frame in sorted(timeline):
        tick = int(round(float(frame) / frame_s))
        if abs(tick * frame_s - float(frame)) > 1e-6:
            raise ValueError(f"Unexpected frame grid: {frame}")
        for vehicle, record in sorted(timeline[frame].items(), key=lambda item: str(item[0])):
            previous_tick, index = previous.get(vehicle, (None, None))
            if previous_tick is None or tick != previous_tick + 1:
                index = len(trajectories)
                trajectories.append({"vehicle": str(vehicle), "frames": [], "csi": []})
            value = np.asarray(record["CSI_preprocessed"][-1], dtype=np.float32).copy()
            if value.shape != (128,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid preprocessed CSI for {vehicle} at {frame}")
            trajectories[index]["frames"].append(float(frame))
            trajectories[index]["csi"].append(value)
            previous[vehicle] = (tick, index)
    for trajectory in trajectories:
        trajectory["frames"] = np.asarray(trajectory["frames"])
        trajectory["csi"] = np.stack(trajectory["csi"])
    return trajectories


def validate_pipeline(models, trajectory, top_k=5):
    """Compare incremental outputs to the native model on the full prefix."""
    pipeline = VehicleStatefulPipeline(models, top_k)
    checks, errors = [], {name: 0.0 for name in TASKS}
    with torch.inference_mode():
        for i, csi in enumerate(trajectory["csi"]):
            outputs = pipeline.step(csi, new_trajectory=(i == 0))
            if i not in {0, 9, 10, 20, 99, len(trajectory["csi"]) - 1}:
                continue
            prefix = torch.from_numpy(trajectory["csi"][:i+1][None])
            for name in TASKS:
                expected = models[name](prefix)
                if name == "beam":
                    expected = expected.topk(top_k, dim=-1, sorted=True).indices.numpy()
                    np.testing.assert_array_equal(outputs[name], expected)
                else:
                    scale, offset = models[name].params_norm
                    expected = (scale * (expected - offset)).numpy()
                    np.testing.assert_allclose(outputs[name], expected, rtol=2e-5, atol=2e-4)
                    errors[name] = max(errors[name], float(np.max(np.abs(outputs[name] - expected))))
            checks.append(i + 1)
        # A new trajectory must discard ALL prior recurrent states.
        outputs = pipeline.step(trajectory["csi"][0], new_trajectory=True)
        fresh = VehicleStatefulPipeline(models, top_k).step(trajectory["csi"][0])
        for name in TASKS:
            np.testing.assert_array_equal(outputs[name], fresh[name])
    return {"prefix_lengths_checked": checks, "ranked_indices_match": True,
            "max_gain_error_db": {k: v for k, v in errors.items() if k != "beam"},
            "trajectory_reset_matches_fresh_state": True}


def measure_streams(pipeline, trajectories, rounds, warmup, seed):
    """Warm software before each round; never reset inside a trajectory."""
    rows, round_stats = [], []
    with torch.inference_mode():
        for repeat in range(rounds):
            order = np.random.default_rng(seed + repeat).permutation(len(trajectories))
            warm = trajectories[int(order[0])]["csi"]
            for i in range(warmup):
                pipeline.step(warm[i % len(warm)], new_trajectory=(i % len(warm) == 0))
            values = []
            for index in order:
                trajectory = trajectories[int(index)]
                for i, csi in enumerate(trajectory["csi"]):
                    started = time.perf_counter_ns()
                    outputs = pipeline.step(csi, new_trajectory=(i == 0))
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    values.append(elapsed)
                    rows.append((repeat, int(index), i, elapsed))
                if not all(np.isfinite(value).all() for value in outputs.values()):
                    raise ValueError("Nonfinite model output")
            stats = summarize(values)
            round_stats.append(stats)
            print(json.dumps({"round": repeat + 1, **stats}), flush=True)
    raw = np.asarray(rows, dtype=np.float64)
    history_groups = {}
    for label, lower, upper in (("1_to_10", 1, 10), ("11_to_100", 11, 100),
                                 ("101_to_200", 101, 200), ("201_plus", 201, np.inf)):
        mask = (raw[:, 2] + 1 >= lower) & (raw[:, 2] + 1 <= upper)
        if mask.any():
            history_groups[label] = summarize(raw[mask, 3])
    summary = {**summarize(raw[:, 3]), "rounds": round_stats,
               "by_observed_history_length": history_groups,
               "trajectory_starts": summarize(raw[raw[:, 2] == 0, 3])}
    return summary, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--trajectories", type=int, default=16)
    parser.add_argument("--min-frames", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--affinity", default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if min(args.trajectories, args.min_frames, args.rounds, args.warmup, args.threads) < 1:
        parser.error("All counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {int(x) for x in args.affinity.split(",")})
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    models, inventory = load_selected_models(args.checkpoint_root)
    initial_states = {name: {k: v.clone() for k, v in model.state_dict().items()}
                      for name, model in models.items()}
    with args.data.open("rb") as handle:
        all_trajectories = extract_trajectories(pickle.load(handle))
    candidates = [t for t in all_trajectories if len(t["csi"]) >= args.min_frames]
    if len(candidates) < args.trajectories:
        raise ValueError(f"Only {len(candidates)} sufficiently long trajectories")
    selected = np.random.default_rng(args.seed).choice(len(candidates), args.trajectories, replace=False)
    trajectories = [candidates[int(i)] for i in selected]
    metadata = {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
        "platform": platform.platform(), "lscpu": command_output(["lscpu"]),
        "torch_configuration": torch.__config__.show(),
        "affinity": sorted(os.sched_getaffinity(0)), "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "batch_size": 1,
        "precision": "FP32", "device": "cpu", "oneDNN_enabled": torch.backends.mkldnn.enabled,
        "data_path": str(args.data.resolve()), "data_sha256": digest(args.data),
        "trajectory_count": len(trajectories), "candidate_trajectory_count": len(candidates),
        "frames_per_round": sum(len(t["csi"]) for t in trajectories),
        "trajectories": [{"vehicle": t["vehicle"], "frames": t["frames"].tolist()} for t in trajectories],
        "timing_scope": "Preprocessed FP32 NumPy CSI to ranked top-5 indices and two gains in dB; includes shared tensor conversion, three sequential forward passes, state retention, top-K, gain denormalization and NumPy outputs.",
        "excluded": "CSI acquisition and preprocessing, model/data loading, reporting serialization and transmission, HO/BF/RA decisions, trajectory selection and result logging.",
        "history_policy": "Each trajectory starts with zero LSTM states; h and c are retained after every frame without any ten-frame reset. Software warmup is excluded; first trajectory-frame inference is included.",
        "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), ROOT / "experiment/compare_stateful_prediction.py",
            ROOT / "experiment/benchmark_nn_overhead.py", ROOT / "utils/NN_utils.py")},
    }
    json_write(args.output / "metadata.json", metadata)
    json_write(args.output / "model_inventory.json", inventory)
    correctness = validate_pipeline(models, trajectories[0])
    json_write(args.output / "correctness.json", correctness)
    arithmetic = {}
    sample = torch.from_numpy(trajectories[0]["csi"][:1][None])
    with torch.inference_mode():
        for name, model in models.items():
            _, state = forward_with_state(model, sample)
            step = StatefulStepModule(model, state)
            arithmetic[name] = matrix_arithmetic(step, sample)
            arithmetic[name]["profiler_crosscheck"] = verify_profiler(
                step, sample, arithmetic[name]["matrix_flops"]
            )
    arithmetic["total_matrix_flops"] = sum(arithmetic[name]["matrix_flops"] for name in TASKS)
    json_write(args.output / "arithmetic.json", arithmetic)
    print(json.dumps({"frames_per_round": metadata["frames_per_round"],
                      "correctness": correctness, "matrix_flops": arithmetic["total_matrix_flops"]}), flush=True)
    timing, raw = measure_streams(VehicleStatefulPipeline(models), trajectories,
                                 args.rounds, args.warmup, args.seed)
    for name, model in models.items():
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial_states[name][key], rtol=0, atol=0)
        if digest(inventory[name]["checkpoint"]) != inventory[name]["sha256"]:
            raise AssertionError("Checkpoint changed during timing")
    timing["model_weights_and_buffers_unchanged"] = True
    json_write(args.output / "timings.json", timing)
    np.savez_compressed(args.output / "raw_latency_ms.npz", round_index=raw[:, 0].astype(np.int32),
                        trajectory_index=raw[:, 1].astype(np.int32),
                        frame_index=raw[:, 2].astype(np.int32), latency_ms=raw[:, 3])
    print(json.dumps({"saved": str(args.output), "samples": timing["samples"],
                      "median_ms": timing["median_ms"], "p95_ms": timing["p95_ms"]}), flush=True)


if __name__ == "__main__":
    main()
