#!/usr/bin/env python
"""Evaluate the two frozen joint HO--BF baselines on the paper load grid.

The driver is deliberately resumable: every completed traffic point is saved
before the next simulation starts.  It exports compact curve-ready JSON/CSV
files as well as compressed per-frame/per-sample records for later auditing.
No policy is trained or adapted by this script.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, Iterable, MutableMapping, Sequence

import numpy as np

sys.path.append(os.getcwd())

from experiment.dql_hbt_experiment import summarize_exact as summarize_dql
from experiment.o_mappo_experiment import summarize_exact as summarize_o_mappo
from experiment.pql_ba_experiment import (
    DEFAULT_TEST_PATH,
    MICRO_BS_LOCATIONS,
    load_pickle,
    paper_args,
    temporal_slice,
)
from utils.alg_utils import RA_OTR_SINR
from utils.dql_hbt import DQLHBTPolicy
from utils.dql_hbt_sim import DQLHBTSimulationResult, run_sim_dql_hbt
from utils.o_mappo import OMAPPPolicy
from utils.o_mappo_sim import OMAPPOSimulationResult, run_sim_o_mappo


DEFAULT_OUTPUT_DIR = Path(
    "experiment/results_joint_baselines/full_grid_30s_seed1"
)
DEFAULT_DQL_POLICY = Path(
    "experiment/results_dql_hbt/final_adapted_8ep/final_policy.pt"
)
DEFAULT_O_MAPPO_POLICY = Path(
    "experiment/results/o_mappo/final_load1/final_policy.pt"
)
PAPER_RATES_MBPS = tuple(float(rate) for rate in range(1, 36, 2))
WARMUP_FRAMES = 2

METHOD_LABELS = collections.OrderedDict(
    (
        ("dql", "DQL-HBT-adapted"),
        ("o_mappo", "O-MAPPO-adapted"),
    )
)
METHOD_DIRS = {
    "dql": "dql_hbt_adapted",
    "o_mappo": "o_mappo_adapted",
}


def _json_ready(value):
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_rates(value: str) -> Sequence[float]:
    rates = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not rates:
        raise ValueError("at least one traffic rate is required")
    return rates


def _queue_samples(result, data_rate_bps: float) -> np.ndarray:
    samples = []
    for frame_record in result.queue_per_vehicle_record.values():
        samples.extend(np.asarray(queue, dtype=np.float64) for queue in frame_record.values())
    if not samples:
        return np.empty(0, dtype=np.float64)
    # paper_args fixes the per-vehicle traffic variation to zero, hence the
    # nominal rate equals every vehicle's lambda_v in d_v^Q = q_v/lambda_v.
    return np.concatenate(samples) / float(data_rate_bps) * 1000.0


def _association_counts(result, num_bs: int) -> np.ndarray:
    counts = np.zeros((len(result.association_record), num_bs), dtype=np.int32)
    for row, associations in enumerate(result.association_record.values()):
        for bs in associations.values():
            counts[row, int(bs)] += 1
    return counts


def _manuscript_metrics(args, result) -> Dict[str, float]:
    """Compute metrics using the same averaging rules as paper_exp1.py."""

    frame_duration_s = args.slots_per_frame * args.slot_len
    warmup = min(WARMUP_FRAMES, max(len(result.energy_record) - 1, 0))
    queue_ms = _queue_samples(result, args.data_rate)
    association_counts = _association_counts(
        result, len(MICRO_BS_LOCATIONS) + 1
    )
    total_associations = float(association_counts.sum())
    return collections.OrderedDict(
        (
            (
                "average_system_power_w",
                float(result.energy_record.mean() / frame_duration_s),
            ),
            (
                "queue_violation_percent",
                float(
                    100.0
                    * result.violation_probability_record[warmup:].mean()
                ),
            ),
            (
                "average_queueing_proxy_ms",
                float(
                    1000.0
                    * result.average_queue_record[warmup:].mean()
                    / args.data_rate
                ),
            ),
            (
                "queueing_proxy_p90_ms",
                float(np.percentile(queue_ms, 90)) if queue_ms.size else float("nan"),
            ),
            (
                "queueing_proxy_p99_ms",
                float(np.percentile(queue_ms, 99)) if queue_ms.size else float("nan"),
            ),
            (
                "macro_association_ratio",
                float(association_counts[:, 0].sum() / max(total_associations, 1.0)),
            ),
        )
    )


def _raw_arrays(args, result) -> Dict[str, np.ndarray]:
    arrays = {
        "energy_j_per_frame": np.asarray(result.energy_record),
        "handover_count_per_frame": np.asarray(result.handover_record),
        "beam_switch_count_per_frame": np.asarray(result.beam_switch_record),
        "queue_violation_probability_per_frame": np.asarray(
            result.violation_probability_record
        ),
        "average_queue_bits_per_frame_slot": np.asarray(result.average_queue_record),
        "average_pilots_per_frame_slot": np.asarray(result.pilot_record),
        "rb_allocated_per_bs_per_frame": np.asarray(result.rb_allocated_record),
        "association_count_per_bs_per_frame": _association_counts(
            result, len(MICRO_BS_LOCATIONS) + 1
        ),
        "normalized_backlog_proxy_ms_samples": _queue_samples(
            result, args.data_rate
        ),
        "decision_count_per_frame": np.asarray(result.decision_record),
        "full_sweep_count_per_frame": np.asarray(result.full_sweep_record),
        "local_sweep_count_per_frame": np.asarray(result.local_sweep_record),
        "inference_time_s_per_frame": np.asarray(result.inference_time_record),
    }
    if isinstance(result, DQLHBTSimulationResult):
        arrays.update(
            {
                "tracking_decision_count_per_frame": np.asarray(
                    result.tracking_decision_record
                ),
                "skipped_trigger_count_per_frame": np.asarray(
                    result.skipped_trigger_record
                ),
            }
        )
    else:
        arrays.update(
            {
                "trigger_count_per_frame": np.asarray(result.trigger_record),
                "skipped_gate_count_per_frame": np.asarray(result.skipped_gate_record),
                "optimizer_time_s_per_frame": np.asarray(result.optimizer_time_record),
                "optimizer_failure_count_per_frame": np.asarray(
                    result.optimizer_failure_record
                ),
                "optimizer_overflow_rb_per_frame": np.asarray(
                    result.optimizer_overflow_record
                ),
            }
        )
    return arrays


def _rate_key(rate: float, seed: int = 1) -> str:
    return "rate_{:g}Mbps_seed_{}".format(rate, seed)


def _run_one(method: str, common_args, timeline, policy, rate: float, seed: int):
    common_args.data_rate = rate * 1e6
    if method == "dql":
        result = run_sim_dql_hbt(
            common_args,
            MICRO_BS_LOCATIONS,
            timeline,
            policy,
            ra_func=RA_OTR_SINR,
            seed=seed,
            prt=True,
            rician_fading=True,
        )
        diagnostic = summarize_dql(common_args, result, timeline)
    else:
        result = run_sim_o_mappo(
            common_args,
            MICRO_BS_LOCATIONS,
            timeline,
            policy,
            ra_func=RA_OTR_SINR,
            seed=seed,
            prt=True,
            rician_fading=True,
            optimizer_solver="milp",
        )
        diagnostic = summarize_o_mappo(common_args, result, timeline)
    return result, diagnostic


def _load_run_summaries(
    method_dir: Path, seed: int = 1
) -> "collections.OrderedDict[str, Dict]":
    summaries = collections.OrderedDict()
    for path in sorted(
        (method_dir / "runs").glob("rate_*Mbps_seed_{}.json".format(seed))
    ):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries[_rate_key(float(summary["data_rate_mbps"]), seed)] = summary
    return collections.OrderedDict(
        sorted(summaries.items(), key=lambda item: float(item[1]["data_rate_mbps"]))
    )


def _curve_rows(summaries: MutableMapping[str, Dict]) -> Iterable[Dict]:
    for summary in summaries.values():
        yield collections.OrderedDict(
            (
                ("method", summary["method_label"]),
                ("data_rate_mbps", summary["data_rate_mbps"]),
                ("seed", summary["seed"]),
                *summary["manuscript_metrics"].items(),
            )
        )


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    if rows:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        temporary.touch()
    os.replace(temporary, path)


def _refresh_method_exports(method: str, method_dir: Path, seed: int = 1) -> None:
    summaries = _load_run_summaries(method_dir, seed)
    curve_rows = list(_curve_rows(summaries))
    payload = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "traffic_rates_mbps": [row["data_rate_mbps"] for row in curve_rows],
        "num_completed_points": len(curve_rows),
        "runs": summaries,
        "curve_data": curve_rows,
    }
    _write_json_atomic(method_dir / "curve_results.json", payload)
    _write_csv(method_dir / "curve_data.csv", curve_rows)


def evaluate_method(args, method: str) -> None:
    method_dir = args.output_dir / METHOD_DIRS[method]
    runs_dir = method_dir / "runs"
    raw_dir = method_dir / "raw"
    runs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    policy_path = args.dql_policy if method == "dql" else args.o_mappo_policy
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    policy = (
        DQLHBTPolicy.load(str(policy_path), seed=args.seed)
        if method == "dql"
        else OMAPPPolicy.load(str(policy_path), seed=args.seed)
    )
    protocol = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "policy_path": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "policy_config": dataclasses.asdict(policy.config),
        "test_path": str(args.test_path),
        "test_interval_s": [args.test_start, args.test_end],
        "simulation_duration_s": args.test_end - args.test_start,
        "slots_per_frame": 100,
        "warmup_frames_excluded_for_violation_and_mean_queue": WARMUP_FRAMES,
        "power_and_percentile_averaging": "all frames, matching paper_exp1.py",
        "traffic_rates_mbps": list(args.rates),
        "evaluation_seed": args.seed,
        "rician_fading": True,
        "resource_allocation": "OTR-RA (RA_OTR_SINR)",
        "o_mappo_exact_optimizer_solver": "milp" if method == "o_mappo" else None,
        "policy_frozen_during_test": True,
        "random_factor_range4data_rate": 0.0,
    }
    _write_json_atomic(method_dir / "protocol.json", protocol)

    timeline_all = load_pickle(str(args.test_path))
    timeline = temporal_slice(timeline_all, args.test_start, args.test_end)
    common_args = paper_args()
    total_start = time.time()
    for index, rate in enumerate(args.rates, start=1):
        key = _rate_key(rate, args.seed)
        summary_path = runs_dir / (key + ".json")
        raw_path = raw_dir / (key + ".npz")
        if summary_path.is_file() and raw_path.is_file() and not args.force:
            print("[{:02d}/{:02d}] Resume: {} already complete".format(index, len(args.rates), key))
            continue
        print(
            "\n[{:02d}/{:02d}] {} exact test at {:g} Mbps (seed={})".format(
                index, len(args.rates), METHOD_LABELS[method], rate, args.seed
            )
        )
        started = time.time()
        result, diagnostic = _run_one(
            method, common_args, timeline, policy, float(rate), args.seed
        )
        elapsed = time.time() - started
        manuscript = _manuscript_metrics(common_args, result)
        summary = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "data_rate_mbps": float(rate),
            "seed": args.seed,
            "elapsed_s": elapsed,
            "manuscript_metrics": manuscript,
            "diagnostic_metrics": diagnostic,
            "raw_npz": str(raw_path),
        }
        np.savez_compressed(raw_path, **_raw_arrays(common_args, result))
        _write_json_atomic(summary_path, summary)
        _refresh_method_exports(method, method_dir, args.seed)
        print("Completed {} in {:.1f} s: {}".format(key, elapsed, manuscript))

    _refresh_method_exports(method, method_dir, args.seed)
    completion = {
        "method": method,
        "completed_at_unix_s": time.time(),
        "elapsed_this_invocation_s": time.time() - total_start,
        "completed_points": len(_load_run_summaries(method_dir, args.seed)),
        "expected_points": len(args.rates),
    }
    _write_json_atomic(method_dir / "completion.json", completion)


def combine_outputs(output_dir: Path) -> None:
    all_rows = []
    methods = collections.OrderedDict()
    for method in METHOD_LABELS:
        method_dir = output_dir / METHOD_DIRS[method]
        curve_path = method_dir / "curve_results.json"
        if not curve_path.is_file():
            continue
        with curve_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        methods[method] = payload
        all_rows.extend(payload["curve_data"])
    all_rows.sort(key=lambda row: (row["method"], float(row["data_rate_mbps"])))
    combined = {
        "paper_traffic_rates_mbps": list(PAPER_RATES_MBPS),
        "methods": methods,
        "curve_data": all_rows,
    }
    _write_json_atomic(output_dir / "joint_baselines_curve_results.json", combined)
    _write_csv(output_dir / "joint_baselines_curve_data.csv", all_rows)


def merge_shards(output_dir: Path) -> None:
    """Collect non-overlapping parallel shards into the canonical directory."""

    shard_root = output_dir / "shards"
    for method in METHOD_LABELS:
        target_method_dir = output_dir / METHOD_DIRS[method]
        (target_method_dir / "runs").mkdir(parents=True, exist_ok=True)
        (target_method_dir / "raw").mkdir(parents=True, exist_ok=True)
        for shard in sorted(shard_root.glob("*")):
            source_method_dir = shard / METHOD_DIRS[method]
            if not source_method_dir.is_dir():
                continue
            for folder, pattern in (("runs", "*.json"), ("raw", "*.npz")):
                for source in sorted((source_method_dir / folder).glob(pattern)):
                    target = target_method_dir / folder / source.name
                    if target.exists():
                        if target.stat().st_size != source.stat().st_size:
                            raise RuntimeError(
                                "conflicting shard artifact: {}".format(target)
                            )
                        continue
                    shutil.copy2(source, target)
        # Shard summaries originally point to their shard-local NPZ path.
        # Rebase those references so the merged directory is self-contained.
        for summary_path in sorted((target_method_dir / "runs").glob("*.json")):
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            canonical_raw = target_method_dir / "raw" / (
                summary_path.stem + ".npz"
            )
            if not canonical_raw.is_file():
                raise RuntimeError("missing raw artifact: {}".format(canonical_raw))
            if summary.get("raw_npz") != str(canonical_raw):
                summary["raw_npz"] = str(canonical_raw)
                _write_json_atomic(summary_path, summary)
        _refresh_method_exports(method, target_method_dir)
        completed = len(_load_run_summaries(target_method_dir))
        _write_json_atomic(
            target_method_dir / "completion.json",
            {
                "method": method,
                "completed_at_unix_s": time.time(),
                "completed_points": completed,
                "expected_points": len(PAPER_RATES_MBPS),
                "merged_from_parallel_shards": True,
            },
        )
    combine_outputs(output_dir)


def validate_outputs(output_dir: Path) -> None:
    """Verify grid completeness and recompute curve metrics from every NPZ."""

    expected_rates = list(PAPER_RATES_MBPS)
    report = {
        "expected_rates_mbps": expected_rates,
        "expected_points_per_method": len(expected_rates),
        "methods": {},
    }
    for method in METHOD_LABELS:
        method_dir = output_dir / METHOD_DIRS[method]
        summaries = _load_run_summaries(method_dir)
        rates = [float(item["data_rate_mbps"]) for item in summaries.values()]
        if rates != expected_rates:
            raise RuntimeError("{} has incomplete or misordered rates: {}".format(method, rates))
        raw_paths = sorted((method_dir / "raw").glob("*.npz"))
        if len(raw_paths) != len(expected_rates):
            raise RuntimeError("{} has {} raw files".format(method, len(raw_paths)))
        with (method_dir / "protocol.json").open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
        if [float(x) for x in protocol["traffic_rates_mbps"]] != expected_rates:
            raise RuntimeError("{} canonical protocol has the wrong grid".format(method))
        policy_path = Path(protocol["policy_path"])
        if _sha256(policy_path) != protocol["policy_sha256"]:
            raise RuntimeError("{} policy hash mismatch".format(method))

        maximum_error = 0.0
        sample_count = 0
        artifact_bytes = 0
        for summary in summaries.values():
            rate = float(summary["data_rate_mbps"])
            raw_path = method_dir / "raw" / (
                _rate_key(rate) + ".npz"
            )
            if summary["raw_npz"] != str(raw_path):
                raise RuntimeError("non-canonical raw path in {}".format(rate))
            manuscript = summary["manuscript_metrics"]
            with np.load(raw_path) as raw:
                energy = raw["energy_j_per_frame"]
                violation = raw["queue_violation_probability_per_frame"]
                average_queue = raw["average_queue_bits_per_frame_slot"]
                queue_ms = raw["normalized_backlog_proxy_ms_samples"]
                association = raw["association_count_per_bs_per_frame"]
                if len(energy) != 300 or len(violation) != 300:
                    raise RuntimeError("{} {:g} Mbps does not have 300 frames".format(method, rate))
                if not all(
                    np.isfinite(value).all()
                    for value in (energy, violation, average_queue, queue_ms, association)
                ):
                    raise RuntimeError("non-finite raw result in {} {:g} Mbps".format(method, rate))
                recomputed = {
                    "average_system_power_w": float(energy.mean() / 0.1),
                    "queue_violation_percent": float(100.0 * violation[2:].mean()),
                    "average_queueing_proxy_ms": float(
                        1000.0 * average_queue[2:].mean() / (rate * 1e6)
                    ),
                    "queueing_proxy_p90_ms": float(np.percentile(queue_ms, 90)),
                    "queueing_proxy_p99_ms": float(np.percentile(queue_ms, 99)),
                    "macro_association_ratio": float(
                        association[:, 0].sum() / association.sum()
                    ),
                }
                sample_count += int(queue_ms.size)
            for metric, value in recomputed.items():
                error = abs(value - float(manuscript[metric]))
                maximum_error = max(maximum_error, error)
                if error > 1e-8 * max(1.0, abs(value)):
                    raise RuntimeError(
                        "metric mismatch: {} {:g} Mbps {}".format(method, rate, metric)
                    )
            artifact_bytes += raw_path.stat().st_size

        curve_path = method_dir / "curve_data.csv"
        with curve_path.open("r", encoding="utf-8", newline="") as handle:
            curve_rows = list(csv.DictReader(handle))
        if len(curve_rows) != len(expected_rates):
            raise RuntimeError("{} curve CSV row-count mismatch".format(method))
        report["methods"][method] = {
            "method_label": METHOD_LABELS[method],
            "completed_points": len(summaries),
            "raw_npz_files": len(raw_paths),
            "curve_csv_rows": len(curve_rows),
            "raw_queue_samples": sample_count,
            "raw_npz_bytes": artifact_bytes,
            "maximum_metric_recompute_abs_error": maximum_error,
            "policy_sha256_verified": True,
            "status": "passed",
        }

    with (output_dir / "joint_baselines_curve_data.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        combined_rows = list(csv.DictReader(handle))
    if len(combined_rows) != 2 * len(expected_rates):
        raise RuntimeError("combined curve CSV row-count mismatch")
    report["combined_curve_csv_rows"] = len(combined_rows)
    report["all_checks_passed"] = True
    _write_json_atomic(output_dir / "validation_report.json", report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["dql", "o_mappo", "combine", "merge", "validate"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-path", type=Path, default=Path(DEFAULT_TEST_PATH))
    parser.add_argument("--test-start", type=float, default=800.0)
    parser.add_argument("--test-end", type=float, default=830.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--rates", type=_parse_rates, default=PAPER_RATES_MBPS,
        help="comma-separated Mbps values (default: 1,3,...,35)",
    )
    parser.add_argument("--dql-policy", type=Path, default=DEFAULT_DQL_POLICY)
    parser.add_argument("--o-mappo-policy", type=Path, default=DEFAULT_O_MAPPO_POLICY)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "combine":
        combine_outputs(args.output_dir)
        return
    if args.method == "merge":
        merge_shards(args.output_dir)
        return
    if args.method == "validate":
        validate_outputs(args.output_dir)
        return
    evaluate_method(args, args.method)
    combine_outputs(args.output_dir)


if __name__ == "__main__":
    main()
