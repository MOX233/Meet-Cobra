#!/usr/bin/env python3
"""Reproducible, batch-one NN overhead audit for the R2-C6 revision.

Uses the unmodified model classes and paper checkpoints. Does not run network
simulations or change their predictors. Timing begins with preprocessed CSI.
FLOPs cover dense/LSTM matrix arithmetic (two FLOPs per MAC), not all CPU
instructions. Native (unfused) PyTorch profiling independently checks the count.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="meet-cobra-mpl-")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from utils.NN_utils import BeamPredictionLSTMModel, BestGainPredictionLSTMModel

CHECKPOINT_DIR = ROOT / (
    "NN_result/200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10/models"
)
CHECKPOINTS = {
    "beam": "beampred_lstm_valAcc89.73%_2025-09-19_21:48:48.pth",
    "desired_gain": "gainpred_lstm_valMae4.07dB_2025-09-25_02:04:34.pth",
    "interfering_gain": "inferpred_lstm_valMae3.20dB_2025-11-05_01:24:36.pth",
}
DEFAULT_DATA = ROOT / (
    "data4sim/lbd1.00_800_830_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl"
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def load_inputs(path, sample_count, seed):
    with path.open("rb") as handle:
        timeline = pickle.load(handle)
    candidates, lengths, vehicles = [], collections.Counter(), []
    for frame, records in timeline.items():
        vehicles.append(len(records))
        for vehicle, record in records.items():
            value = record["CSI_preprocessed"]
            lengths[value.shape[0]] += 1
            if value.shape == (10, 128):
                candidates.append((frame, vehicle, value))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=min(sample_count, len(candidates)), replace=False)
    pool, identities = [], []
    for idx in selected:
        frame, vehicle, value = candidates[int(idx)]
        pool.append(np.ascontiguousarray(value[None, ...], dtype=np.float32))
        identities.append({"frame": float(frame), "vehicle": str(vehicle)})
    assert pool and all(np.isfinite(x).all() for x in pool)
    metadata = {
        "path": str(path), "sha256": digest(path), "frames": len(timeline),
        "active_vehicles_mean": float(np.mean(vehicles)),
        "active_vehicles_max": int(max(vehicles)),
        "history_length_counts": dict(lengths), "selected_records": identities,
    }
    return pool, metadata


def load_models(device):
    models = {
        "beam": BeamPredictionLSTMModel(128, 4, 256),
        "desired_gain": BestGainPredictionLSTMModel(128, 4),
        "interfering_gain": BestGainPredictionLSTMModel(128, 4),
    }
    inventory = {}
    for name, model in models.items():
        path = CHECKPOINT_DIR / CHECKPOINTS[name]
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()
        inventory[name] = {
            "checkpoint": str(path.relative_to(ROOT)), "sha256": digest(path),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "state_dict_bytes": sum(p.numel() * p.element_size() for p in model.state_dict().values()),
            "checkpoint_file_bytes": path.stat().st_size,
            "architecture": str(model),
        }
    return models, inventory


def matrix_arithmetic(model, sample):
    """Count executed Linear branches and four-gate LSTM matrix products."""
    rows, handles = [], []

    def hook(name):
        def count(module, inputs, output):
            if isinstance(module, nn.Linear):
                macs = output.numel() * module.in_features
            else:
                assert module.num_layers == 1 and not module.bidirectional and module.proj_size == 0
                batch, length, features = inputs[0].shape
                macs = batch * length * 4 * module.hidden_size * (features + module.hidden_size)
            rows.append({"module": name, "type": type(module).__name__, "macs": macs})
        return count

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LSTM)):
            handles.append(module.register_forward_hook(hook(name)))
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    total = sum(row["macs"] for row in rows)
    return {"macs": total, "matrix_flops": 2 * total, "modules": rows}


def verify_profiler(model, sample, expected):
    # oneDNN fuses the LSTM; its opaque op is not counted by with_flops.
    # Disable it only for this count cross-check, not for the timing tests.
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        model(sample)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True, with_flops=True) as prof:
            model(sample)
    counts = {event.key: int(event.flops) for event in prof.key_averages() if event.flops}
    matrix_operators = {"aten::mm", "aten::addmm", "aten::bmm", "aten::matmul"}
    observed = sum(value for name, value in counts.items() if name in matrix_operators)
    assert observed == expected, (observed, expected, counts)
    return {"matrix_flops": observed, "reported_operators": counts,
            "other_counted_flops": sum(counts.values()) - observed,
            "matches_analytic_count": True}


def summarize(samples):
    value = np.asarray(samples, dtype=np.float64)
    return {
        "samples": int(value.size), "mean_ms": float(value.mean()),
        "std_ms": float(value.std(ddof=1)), "median_ms": float(np.median(value)),
        "p95_ms": float(np.percentile(value, 95)), "p99_ms": float(np.percentile(value, 99)),
        "min_ms": float(value.min()), "max_ms": float(value.max()),
    }


def report_bits(scalar_bits=32, top_k=5, frame_s=0.1):
    bs, tx, rx, pilots = 4, 32, 8, 8
    index_bits = math.ceil(math.log2(tx * rx))
    beam_bits = bs * top_k * index_bits
    gain_bits = bs * 2 * scalar_bits
    reports = {
        "ranked_candidates_and_two_gains": beam_bits + gain_bits,
        "full_prediction_output": bs * (tx * rx + 2) * scalar_bits,
        "full_complex_microcell_CSI": 2 * bs * tx * rx * scalar_bits,
        "new_superposed_CSI_input_for_central_NN": 2 * rx * pilots * scalar_bits,
    }
    return {
        "scalar_bits": scalar_bits, "frame_s": frame_s,
        "primary_report": "ranked_candidates_and_two_gains",
        "beam_candidates_per_BS": top_k, "bits_per_beam_pair_index": index_bits,
        "primary_report_components_bit": {"ranked_indices": beam_bits, "two_gains_per_BS": gain_bits},
        "bits_per_vehicle_per_frame": reports,
        "kbit_per_vehicle_per_s": {k: v / frame_s / 1000 for k, v in reports.items()},
        "prediction_to_full_CSI_ratio": reports["ranked_candidates_and_two_gains"] / reports["full_complex_microcell_CSI"],
        "prediction_to_superposed_input_ratio": reports["ranked_candidates_and_two_gains"] / reports["new_superposed_CSI_input_for_central_NN"],
        "pilot_probes_per_frame_superposed": pilots,
        "pilot_probes_per_frame_separate_BS_tx_sweep": bs * tx,
        "assumptions": [
            "Ranked beam-pair indices and two gain values per micro BS; gain values use scalar_bits each. Not measured radio traffic.",
            "Full-CSI and central-NN comparisons use the same per-real-component precision and 100-ms reporting interval.",
            "Central NN retains prior reports; only the newly acquired CSI vector is uploaded each frame.",
            "Top-K indices are packed in descending predicted-probability order; probabilities and separate rank fields are not transmitted.",
            "full_prediction_output is a counterfactual reference, not the adopted reporting payload.",
            "Excludes protocol headers, identifiers, coding, retransmission, HO signalling and common macro-channel reports.",
            "CSI vectors/matrices and probability outputs differ in information content; these are acquisition architectures, not equivalent algorithms.",
            "Pilot comparison assumes identical receive-side sampling capability and counts probing occasions, not full on-air symbol/energy cost.",
        ],
    }


def measure(fn, pool, warmup, samples, rounds, synchronize, inference=True):
    context = torch.inference_mode if inference else contextlib.nullcontext
    raw, round_stats = [], []
    with context():
        for _ in range(rounds):
            for i in range(warmup):
                result = fn(pool[i % len(pool)])
            synchronize()
            round_values = []
            for i in range(samples):
                synchronize()
                started = time.perf_counter_ns()
                result = fn(pool[i % len(pool)])
                synchronize()
                round_values.append((time.perf_counter_ns() - started) / 1e6)
            raw.extend(round_values)
            round_stats.append(summarize(round_values))
        del result
    return {**summarize(raw), "rounds": round_stats}, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--affinity", default="0,1,2,3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lengths", default="1,5,10")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.affinity:
        os.sched_setaffinity(0, {int(x) for x in args.affinity.split(",")})
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no GPU timing can be reported")
    pool, data_info = load_inputs(args.data, 32, args.seed)
    models, inventory = load_models(device)
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "torch_configuration": torch.__config__.show(), "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "nvidia_smi": command_output(["nvidia-smi"]),
        "lscpu": command_output(["lscpu"]), "affinity": sorted(os.sched_getaffinity(0)),
        "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "sources_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                           (Path(__file__), ROOT / "utils/NN_utils.py", ROOT / "utils/sim_utils.py")},
        "batch_size": 1, "precision": "FP32", "oneDNN_enabled": torch.backends.mkldnn.enabled,
        "timing_scope": "Warm process, preprocessed CSI to model outputs; excludes sensing, CSI preprocessing, network transport and checkpoint loading.",
        "history_policy": "Replay T input frames with zero initial hidden/cell states, exactly as predict(); no cached-state speedup is assumed.",
        "flop_scope": "LSTM and Linear matrix products, 1 MAC = 2 FLOPs; excludes bias, normalization, activations, residual additions and output postprocessing.",
        "data": data_info,
    }
    json_write(args.output / "metadata.json", metadata)
    json_write(args.output / "model_inventory.json", inventory)
    budget = report_bits()
    budget["aggregate_prediction_Mbit_s_at_dataset_mean_vehicles"] = (
        budget["kbit_per_vehicle_per_s"][budget["primary_report"]] * data_info["active_vehicles_mean"] / 1000
    )
    json_write(args.output / "control_payload.json", budget)
    timings, arithmetic, raw = {}, {}, {}
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)

    def pipeline(value, full_distribution=False):
        gain = models["desired_gain"].predict(value, device)
        if full_distribution:
            tensor = torch.tensor(value).to(device)
            beam = torch.softmax(models["beam"](tensor), dim=-1).detach().cpu().numpy()
        else:
            beam = models["beam"].predict(value, device, K=5)
        interference = models["interfering_gain"].predict(value, device)
        return beam, gain, interference

    for length in [int(x) for x in args.lengths.split(",")]:
        assert 1 <= length <= 10
        np_pool = [np.ascontiguousarray(x[:, -length:, :]) for x in pool]
        tensor_pool = [torch.from_numpy(x).to(device) for x in np_pool]
        for name, model in models.items():
            key = f"T{length}_{name}_forward"
            arithmetic[key] = matrix_arithmetic(model, tensor_pool[0])
            if device.type == "cpu":
                arithmetic[key]["profiler_crosscheck"] = verify_profiler(
                    model, tensor_pool[0], arithmetic[key]["matrix_flops"]
                )
            timings[key], raw[key] = measure(model, tensor_pool, args.warmup, args.samples,
                                             args.rounds, sync)
            print(key, timings[key]["median_ms"], "ms", flush=True)
        for name, fn, inference in (
            ("three_models_top5_predict", pipeline, True),
            ("three_models_full_distribution", lambda x: pipeline(x, True), True),
            ("legacy_autograd_top5_predict", pipeline, False),
        ):
            key = f"T{length}_{name}"
            timings[key], raw[key] = measure(fn, np_pool, args.warmup, args.samples,
                                            args.rounds, sync, inference)
            print(key, timings[key]["median_ms"], "ms", flush=True)
        with torch.inference_mode():
            top = pipeline(np_pool[0])
            full = pipeline(np_pool[0], True)
            assert full[0].shape == (1, 4, 256)
            assert top[0].shape == (1, 4, 5)
            np.testing.assert_allclose(full[0].sum(-1), 1, atol=1e-6)
            np.testing.assert_array_equal(top[0], np.argsort(-full[0], axis=-1)[..., :5])
            np.testing.assert_array_equal(top[1], full[1])
            np.testing.assert_array_equal(top[2], full[2])
        json_write(args.output / "timings.json", timings)
        json_write(args.output / "arithmetic.json", arithmetic)
        np.savez_compressed(args.output / "raw_latency_ms.npz", **raw)
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
