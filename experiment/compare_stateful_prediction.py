#!/usr/bin/env python3
"""Paired, next-frame test of frozen paper NNs: rolling window vs carried state.

No simulator, data preparation, training, or manuscript changes are performed.
Input: chronological cached CSI. Labels: the NEXT frame, using training targets.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import pickle
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.benchmark_nn_overhead import digest, json_write, load_models, command_output
from utils.beam_utils import generate_dft_codebook
import numpy as np
import torch

DEFAULT_DATA = ROOT / "data4sim/lbd1.00_800_950_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl"
TOP_K = (1, 3, 5, 10, 18)
METHODS = ("window", "stateful")


def forward_with_state(model, x, hc=None):
    """Exactly the original forward modules, additionally exposing (h, c)."""
    out, hc = model.lstm_layers(x, hc)
    features = model.shared_layers(out[:, -1, :])
    heads = [head(features) for head in model.output_heads]
    result = torch.stack(heads, dim=-2) if hasattr(model, "num_class") else torch.cat(heads, dim=-1)
    return result, hc


class StatefulPredictor:
    """State is isolated by vehicle and discarded on departure or a frame gap."""
    def __init__(self, model):
        self.model = model
        self.previous = {}
        self.hc = None

    def reset(self):
        self.previous = {}
        self.hc = None

    def step(self, ids, latest):
        assert len(set(ids)) == len(ids)
        shape = (self.model.lstm_layers.num_layers, len(ids), self.model.lstm_layers.hidden_size)
        hc = (latest.new_zeros(shape), latest.new_zeros(shape))
        if self.hc is not None:
            dst = [i for i, v in enumerate(ids) if v in self.previous]
            src = [self.previous[ids[i]] for i in dst]
            if dst:
                for new, old in zip(hc, self.hc):
                    new[:, dst, :] = old[:, src, :]
        output, self.hc = forward_with_state(self.model, latest, hc)
        self.previous = {v: i for i, v in enumerate(ids)}
        return output


def sliding_predictions(model, histories, device):
    groups = collections.defaultdict(list)
    for i, history in enumerate(histories):
        groups[len(history)].append(i)
    result = None
    for indices in groups.values():
        x = torch.as_tensor(np.stack([histories[i] for i in indices]), device=device)
        out = model(x)  # hc=None, same as original simulator, no padded zeros
        if result is None:
            result = out.new_empty((len(histories), *out.shape[1:]))
        result[indices] = out
    return result


def next_frame_ids(current, following):
    return [v for v in current if v in following]


def targets(records):
    channel = np.stack([r["h"] for r in records])  # (vehicle, rx, BS, tx)
    maximum = np.abs(channel).max(axis=(1, 3))
    return {
        "true_beam": np.stack([r["best_beam_pair_idx"] for r in records]).astype(np.int16),
        "true_desired_gain": np.stack([r["g_opt_beam"] for r in records]),
        # train_inferpred_lstm_model.py: maximum ELEMENT magnitude, not g_avg.
        "true_interfering_gain": (20 * np.log10(maximum + 1e-9)).astype(np.float32),
        "nonzero_link": maximum > 0,
    }


def audit_cached_labels(timeline, frames, count=128):
    """Independently check cached beam/desired labels against the training code."""
    tx, rx = generate_dft_codebook(32), generate_dft_codebook(8)
    candidates = [(f, v) for f in frames for v in timeline[f]]
    rng = np.random.default_rng(20)
    error = 0.0
    selected = rng.choice(len(candidates), min(count, len(candidates)), replace=False)
    for index in selected:
        frame, vehicle = candidates[index]
        record = timeline[frame][vehicle]
        h = record["h"]
        beam = np.abs(((h @ tx).T.conj() @ rx).transpose(1, 0, 2).reshape(4, -1)).argmax(-1)
        np.testing.assert_array_equal(beam, record["best_beam_pair_idx"])
        gain = np.zeros(4, dtype=np.float32)
        for b, k in enumerate(beam):
            gain[b] = np.abs((h[:, b, :] @ tx[:, k // 8]).T.conj() @ rx[:, k % 8]) / 16
            gain[b] = 20 * np.log10(gain[b] + 1e-9)
        error = max(error, float(np.max(np.abs(gain - record["g_opt_beam"]))))
        np.testing.assert_allclose(gain, record["g_opt_beam"], rtol=0, atol=3e-5)
    return {"records_checked": len(selected), "beam_labels_match": True,
            "desired_gain_max_abs_difference_db": error}


def metric_values(raw, method):
    result = {f"top{k}_accuracy_pct": 100.0 * (raw[f"{method}_beam"][..., :k] ==
              raw["true_beam"][..., None]).any(-1) for k in TOP_K}
    for name in ("desired_gain", "interfering_gain"):
        result[f"{name}_mae_db"] = np.abs(raw[f"{method}_{name}"] - raw[f"true_{name}"]).astype(np.float64)
    return result


def aggregate(raw, row_mask, link_mask):
    mask = row_mask[:, None] & link_mask
    result = {"vehicle_frames": int(row_mask.sum()), "links": int(mask.sum())}
    if not mask.any():
        return result
    for method in METHODS:
        value = {k: float(v[mask].mean()) for k, v in metric_values(raw, method).items()}
        for name in ("desired_gain", "interfering_gain"):
            err = (raw[f"{method}_{name}"] - raw[f"true_{name}"]).astype(np.float64)[mask]
            value.update({f"{name}_rmse_db": float(np.sqrt(np.mean(err**2))),
                          f"{name}_bias_db": float(err.mean()),
                          f"{name}_p95_abs_error_db": float(np.percentile(np.abs(err), 95))})
        result[method] = value
    result["stateful_minus_window"] = {k: result["stateful"][k] - v for k, v in result["window"].items()}
    result["top1_prediction_disagreement_pct"] = float(100 * (
        raw["window_beam"][..., 0] != raw["stateful_beam"][..., 0])[mask].mean())
    return result


def paired_bootstrap(raw, mask, replicates, seed=20):
    """Resample whole vehicles; maintain temporal/BS pairing within a vehicle."""
    _, clusters = np.unique(raw["vehicle"], return_inverse=True)
    n = clusters.max() + 1
    counts = np.bincount(clusters, weights=mask.sum(-1), minlength=n)
    valid = counts > 0
    counts = counts[valid]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(counts), size=(replicates, len(counts)))
    denom = counts[indices].sum(-1)
    a, b = (metric_values(raw, m) for m in METHODS)
    result = {}
    for key in a:
        sums = np.bincount(clusters, weights=((b[key] - a[key]) * mask).sum(-1), minlength=n)[valid]
        boot = sums[indices].sum(-1) / denom
        result[key] = {"difference": float(sums.sum() / counts.sum()),
                       "paired_vehicle_bootstrap_95pct": np.percentile(boot, [2.5, 97.5]).tolist()}
    return {"clusters": len(counts), "replicates": replicates, "seed": seed, "metrics": result}


def save_summaries(output, raw, bootstrap_replicates):
    age = raw["age"]
    all_rows = np.ones(len(age), dtype=bool)
    all_links = np.ones_like(raw["nonzero_link"])
    cohorts = {"all": all_rows, "age_le_10": age <= 10, "age_gt_10": age > 10,
               "age_11_20": (age > 10) & (age <= 20), "age_21_50": (age > 20) & (age <= 50),
               "age_51_100": (age > 50) & (age <= 100), "age_101_300": (age > 100) & (age <= 300),
               "age_gt_300": age > 300}
    summary = {kind: {name: aggregate(raw, rows, links) for name, rows in cohorts.items()}
               for kind, links in (("all_links", all_links), ("nonzero_links", raw["nonzero_link"]))}
    summary["by_bs"] = {str(b + 1): aggregate(raw, all_rows, all_links & (np.arange(4) == b)) for b in range(4)}
    summary["paired_uncertainty"] = {
        kind: paired_bootstrap(raw, links, bootstrap_replicates)
        for kind, links in (("all_links", all_links), ("nonzero_links", raw["nonzero_link"]))}
    summary["counts"] = {"vehicle_frames": len(age), "links": int(all_links.sum()),
                         "vehicles": len(np.unique(raw["vehicle"])), "segments": len(np.unique(raw["segment"])),
                         "maximum_age_frames": int(age.max()), "nonzero_links": int(raw["nonzero_link"].sum())}
    json_write(output / "summary.json", summary)
    rows = []
    for kind in ("all_links", "nonzero_links"):
        for cohort, data in summary[kind].items():
            for method in METHODS:
                if method in data:
                    rows.append({"links_subset": kind, "age_subset": cohort, "method": method,
                                 "vehicle_frames": data["vehicle_frames"], "links": data["links"], **data[method]})
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-prediction-frames", type=int, default=0, help="0 = complete test trace")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20)
    device = torch.device(args.device)
    models, inventory = load_models(device)
    print("Loading test trace", args.data, flush=True)
    with args.data.open("rb") as handle:
        timeline = pickle.load(handle)
    all_frames = sorted(timeline)
    frames = all_frames[:args.max_prediction_frames + 1] if args.max_prediction_frames else all_frames
    label_audit = audit_cached_labels(timeline, frames)
    print(f"Loaded {len(all_frames)} frames; evaluating {len(frames)-1} next-frame transitions", flush=True)
    streaming = {k: StatefulPredictor(m) for k, m in models.items()}
    raw = collections.defaultdict(list)
    ages, segments, prev_records = {}, {}, {}
    next_segment = 0
    controls = {k: {"max_abs_output_diff_age_le_10": 0.0, "full_prefix_max_abs_output_diff": 0.0,
                    "full_prefix_max_relative_l2_diff": 0.0, "full_prefix_checks": 0,
                    "batch_vs_single_max_abs_output_diff": 0.0} for k in models}
    histories_checked = 0
    first_ids = list(timeline[frames[0]])[:4]
    prefix = {v: [] for v in first_ids}
    prior_frame = None
    with torch.inference_mode():
        for fi, frame in enumerate(frames[:-1]):
            current = timeline[frame]
            if prior_frame is not None and not np.isclose(frame - prior_frame, 0.1, atol=1e-7, rtol=0):
                prev_records, ages, segments = {}, {}, {}
                for predictor in streaming.values():
                    predictor.reset()
            ids = list(current)
            new_ages, new_segments = {}, {}
            for v in ids:
                x = current[v]["CSI_preprocessed"]
                assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 128 and np.isfinite(x).all()
                if v in prev_records:
                    expected = np.concatenate((prev_records[v]["CSI_preprocessed"], x[-1:]), axis=0)[-10:]
                    np.testing.assert_array_equal(x, expected)
                    new_ages[v], new_segments[v] = ages[v] + 1, segments[v]
                else:
                    assert len(x) == 1, "Trace begins mid-history; must explicitly warm up both policies."
                    new_ages[v], new_segments[v] = 1, next_segment
                    next_segment += 1
                assert len(x) == min(new_ages[v], 10)
                histories_checked += 1
            ages, segments = new_ages, new_segments
            if not ids:
                for predictor in streaming.values():
                    predictor.reset()
                prev_records, prior_frame = current, frame
                continue
            history = [current[v]["CSI_preprocessed"] for v in ids]
            latest = torch.as_tensor(np.stack([x[-1:] for x in history]), device=device)
            following = timeline[frames[fi + 1]]
            score_ids = next_frame_ids(current, following) if np.isclose(frames[fi + 1] - frame, 0.1) else []
            index = {v: i for i, v in enumerate(ids)}
            scored = [index[v] for v in score_ids]
            young = [i for i, v in enumerate(ids) if ages[v] <= 10]
            for v in list(prefix):
                if v not in current:
                    del prefix[v]
                elif fi < 25:
                    prefix[v].append(current[v]["CSI_preprocessed"][-1])
            for name, model in models.items():
                window = sliding_predictions(model, history, device)
                stateful = streaming[name].step(ids, latest)
                assert torch.isfinite(window).all() and torch.isfinite(stateful).all()
                c = controls[name]
                if young:
                    diff = (window[young] - stateful[young]).abs().max().item()
                    c["max_abs_output_diff_age_le_10"] = max(c["max_abs_output_diff_age_le_10"], diff)
                if fi < 3:
                    for i in range(min(3, len(ids))):
                        single = model(torch.as_tensor(history[i][None], device=device))[0]
                        c["batch_vs_single_max_abs_output_diff"] = max(c["batch_vs_single_max_abs_output_diff"],
                                                                                  (single - window[i]).abs().max().item())
                if fi < 25:
                    for v, sequence in prefix.items():
                        full = model(torch.as_tensor(np.stack(sequence)[None], device=device))[0]
                        diff = full - stateful[index[v]]
                        absolute, relative = diff.abs().max().item(), (diff.norm() / full.norm().clamp_min(1)).item()
                        assert relative < 2e-5, (name, v, frame, absolute, relative)
                        c["full_prefix_max_abs_output_diff"] = max(c["full_prefix_max_abs_output_diff"], absolute)
                        c["full_prefix_max_relative_l2_diff"] = max(c["full_prefix_max_relative_l2_diff"], relative)
                        c["full_prefix_checks"] += 1
                if scored:
                    for method, out in (("window", window), ("stateful", stateful)):
                        out = out[scored]
                        value = (out.topk(max(TOP_K), dim=-1).indices.cpu().numpy().astype(np.int16) if name == "beam"
                                 else (model.params_norm[0] * (out - model.params_norm[1])).cpu().numpy())
                        raw[f"{method}_{name}"].append(value)
            if scored:
                for key, value in targets([following[v] for v in score_ids]).items():
                    raw[key].append(value)
                raw["vehicle"].append(np.asarray([str(v) for v in score_ids]))
                raw["frame"].append(np.full(len(scored), frame))
                raw["target_frame"].append(np.full(len(scored), frames[fi + 1]))
                raw["age"].append(np.asarray([ages[v] for v in score_ids], dtype=np.int32))
                raw["segment"].append(np.asarray([segments[v] for v in score_ids], dtype=np.int32))
            prev_records, prior_frame = current, frame
            if (fi + 1) % 100 == 0 or fi == len(frames) - 2:
                print(f"Predicted {fi+1}/{len(frames)-1} frames, elapsed {time.monotonic()-started:.1f}s", flush=True)
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    np.testing.assert_allclose(raw["target_frame"] - raw["frame"], .1, rtol=0, atol=1e-7)
    np.savez_compressed(args.output / "predictions.npz", **raw)
    summary = save_summaries(args.output, raw, args.bootstrap_replicates)
    metadata = {
        "data": str(args.data.resolve()), "data_sha256": digest(args.data),
        "script_sha256": digest(Path(__file__)), "checkpoints": inventory,
        "source_frame_range": [all_frames[0], all_frames[-1]],
        "input_frame_range": [float(raw["frame"].min()), float(raw["frame"].max())],
        "target_frame_range": [float(raw["target_frame"].min()), float(raw["target_frame"].max())],
        "prediction_frames": len(frames) - 1, "histories_checked": histories_checked,
        "label_audit": label_audit, "implementation_checks": controls,
        "device": str(device), "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "threads": args.threads,
        "nvidia_smi": command_output(["nvidia-smi", "-L"]),
        "cpu": command_output(["lscpu"]), "command": sys.argv,
        "elapsed_seconds": time.monotonic() - started,
        "protocol": {
            "window": "Per call zero hidden/cell state, last min(age,10) frames, no padding.",
            "stateful": "Separate h/c per vehicle and model, latest frame only; zero at test start, entry or gap.",
            "horizon": "CSI through x predicts labels at x+1, 0.1 s ahead; score shared vehicles only.",
            "desired_gain": "Cached g_opt_beam, verified against normalized DFT optimal pair training target: 20log10(amplitude+1e-9).",
            "interfering_gain": "20log10(max(abs(h_next), over RX/TX elements)+1e-9), per BS, matching training script; NOT g_avg.",
            "zero_channels": "Keep original argmax=0 labels in all-links metrics; also report nonzero links to exclude arbitrary zero-channel beam labels.",
            "statistics": "Link-weighted means over matched samples; paired bootstrap resamples whole vehicles, not correlated frames.",
            "scope": "Frozen checkpoints trained on 200-800 s; chronological 800-950 s test trace; no retraining or network simulation.",
            "note": "This chronological test uses cached per-frame noisy pilots, not a shuffled validation-window split; filename validation scores are not test scores.",
        },
    }
    json_write(args.output / "metadata.json", metadata)
    print(json.dumps({"counts": summary["counts"], "all_links": summary["all_links"]["all"],
                      "nonzero_links": summary["nonzero_links"]["all"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
