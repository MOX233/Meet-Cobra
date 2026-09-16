#!/usr/bin/env python3
"""Validate completeness and recompute manuscript metrics for all multi-seed runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from experiment.aggregate_multiseed_results import METHODS, RATES, ROOT, SEEDS


OUTPUT = ROOT / "experiment/results_multiseed/validation_report.json"


def directories(method, seed):
    base = ROOT / "experiment/results_multiseed"
    if method in tuple(METHODS)[:7]:
        root = base / "original_methods" / method / "seed_{}".format(seed)
        return root, 299
    if method == "mts_gs_hbf":
        root = base / "mts_gs_hbf/pressure_early/exact_800_830_seed{}".format(seed)
        return root, 300
    root = base / "baselines/seed_{}".format(seed) / "o_mappo_adapted"
    return root, 300


def close(expected, actual):
    return abs(expected - actual) <= 1e-8 * max(1.0, abs(expected))


def main():
    report = {"methods": {}, "all_checks_passed": False}
    total_points = 0
    max_error = 0.0
    for method, label in METHODS.items():
        method_points = 0
        method_samples = 0
        for seed in SEEDS:
            root, expected_frames = directories(method, seed)
            summaries = sorted(
                (root / "runs").glob("rate_*Mbps_seed_{}.json".format(seed))
            )
            if len(summaries) != len(RATES):
                raise RuntimeError(
                    "{} seed {} has {} summaries".format(method, seed, len(summaries))
                )
            observed_rates = []
            for summary_path in summaries:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                rate = float(payload["data_rate_mbps"])
                observed_rates.append(rate)
                raw_path = root / "raw" / (summary_path.stem + ".npz")
                if not raw_path.is_file():
                    raise RuntimeError("missing {}".format(raw_path))
                stored = payload["manuscript_metrics"]
                with np.load(raw_path) as raw:
                    energy = raw["energy_j_per_frame"]
                    violation = raw["queue_violation_probability_per_frame"]
                    if len(energy) != expected_frames or len(violation) != expected_frames:
                        raise RuntimeError("wrong frame count in {}".format(raw_path))
                    recomputed = {
                        "average_system_power_w": float(energy.mean() / 0.1),
                        "queue_violation_percent": float(100.0 * violation[2:].mean()),
                    }
                    if method != "oracle_cr_lb":
                        average_queue = raw["average_queue_bits_per_frame_slot"]
                        queue_ms = raw["normalized_backlog_proxy_ms_samples"]
                        association = raw["association_count_per_bs_per_frame"]
                        if not all(
                            np.isfinite(item).all()
                            for item in (energy, violation, average_queue, queue_ms, association)
                        ):
                            raise RuntimeError("non-finite data in {}".format(raw_path))
                        recomputed.update(
                            {
                                "average_queueing_proxy_ms": float(
                                    1000.0 * average_queue[2:].mean() / (rate * 1e6)
                                ),
                                "queueing_proxy_p90_ms": float(np.percentile(queue_ms, 90)),
                                "queueing_proxy_p99_ms": float(np.percentile(queue_ms, 99)),
                                "macro_association_ratio": float(
                                    association[:, 0].sum() / max(float(association.sum()), 1.0)
                                ),
                            }
                        )
                        method_samples += int(queue_ms.size)
                for metric, value in recomputed.items():
                    error = abs(value - float(stored[metric]))
                    max_error = max(max_error, error)
                    if not close(value, float(stored[metric])):
                        raise RuntimeError(
                            "metric mismatch {} seed {} rate {} {}".format(
                                method, seed, rate, metric
                            )
                        )
                method_points += 1
            if sorted(observed_rates) != list(RATES):
                raise RuntimeError(
                    "wrong rate grid for {} seed {}: {}".format(method, seed, observed_rates)
                )
        report["methods"][method] = {
            "label": label,
            "completed_points": method_points,
            "expected_points": len(SEEDS) * len(RATES),
            "raw_queue_samples": method_samples,
            "status": "passed",
        }
        total_points += method_points
    report.update(
        {
            "total_completed_points": total_points,
            "expected_total_points": len(METHODS) * len(SEEDS) * len(RATES),
            "maximum_metric_recompute_abs_error": max_error,
            "all_checks_passed": True,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
