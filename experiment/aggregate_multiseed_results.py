#!/usr/bin/env python3
"""Aggregate five-seed paper curves and report 95% seed-level confidence intervals."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiment/results_multiseed/aggregate"
RATES = tuple(float(value) for value in range(1, 36, 2))
SEEDS = (1, 2, 3, 4, 5)
T95_DF4 = 2.7764451051977987
METHODS = collections.OrderedDict(
    (
        ("proposed", "MEET-COBRA"),
        ("oracle_mc", "Oracle-MC"),
        ("oracle_cr_lb", "Oracle-CR-LB"),
        ("reactive_obra", "Reactive-OBRA"),
        ("wo_gap_ho", "w/o GAP-HO"),
        ("wo_pet_bf", "w/o PET-BF"),
        ("wo_otr_ra", "w/o OTR-RA"),
        ("mts_gs_hbf", "MTS-GS-HBF-adapted"),
        ("o_mappo", "O-MAPPO-adapted"),
    )
)
METRICS = (
    "average_system_power_w",
    "queue_violation_percent",
    "average_queueing_proxy_ms",
    "queueing_proxy_p90_ms",
    "queueing_proxy_p99_ms",
    "macro_association_ratio",
)


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_path(method: str, seed: int) -> Path:
    base = ROOT / "experiment/results_multiseed"
    if method in tuple(METHODS)[:7]:
        return base / "original_methods" / method / "seed_{}".format(seed) / "curve_data.csv"
    if method == "mts_gs_hbf":
        return (
            base
            / "mts_gs_hbf/pressure_early/exact_800_830_seed{}".format(seed)
            / "curve_data.csv"
        )
    return (
        base
        / "baselines/seed_{}".format(seed)
        / "o_mappo_adapted/curve_data.csv"
    )


def numeric_or_none(row, key):
    value = row.get(key)
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def aggregate(args):
    seed_rows = []
    source_files = []
    for method, label in METHODS.items():
        for seed in args.seeds:
            path = source_path(method, seed)
            rows = read_csv(path)
            source_files.append(str(path))
            rates = tuple(float(row["data_rate_mbps"]) for row in rows)
            if rates != RATES:
                raise ValueError("{} has an incomplete rate grid: {}".format(path, rates))
            for row in rows:
                normalized = {
                    "method_key": method,
                    "method": label,
                    "data_rate_mbps": float(row["data_rate_mbps"]),
                    "seed": seed,
                }
                for metric in METRICS:
                    normalized[metric] = numeric_or_none(row, metric)
                seed_rows.append(normalized)

    aggregated = []
    for method, label in METHODS.items():
        for rate in RATES:
            selected = [
                row
                for row in seed_rows
                if row["method_key"] == method
                and row["data_rate_mbps"] == rate
            ]
            if len(selected) != len(args.seeds):
                raise ValueError("missing seeds for {} at {} Mbps".format(method, rate))
            output = {
                "method_key": method,
                "method": label,
                "data_rate_mbps": rate,
                "num_seeds": len(args.seeds),
            }
            for metric in METRICS:
                values = np.asarray(
                    [row[metric] for row in selected if row[metric] is not None],
                    dtype=float,
                )
                prefix = metric + "_"
                if values.size == 0:
                    for suffix in ("mean", "std", "sem", "ci95_halfwidth", "min", "max", "n"):
                        output[prefix + suffix] = ""
                    continue
                if values.size != len(args.seeds):
                    raise ValueError(
                        "partial metric {} for {} at {} Mbps".format(metric, method, rate)
                    )
                standard_deviation = float(values.std(ddof=1))
                standard_error = standard_deviation / math.sqrt(values.size)
                output.update(
                    {
                        prefix + "mean": float(values.mean()),
                        prefix + "std": standard_deviation,
                        prefix + "sem": standard_error,
                        prefix + "ci95_halfwidth": T95_DF4 * standard_error,
                        prefix + "min": float(values.min()),
                        prefix + "max": float(values.max()),
                        prefix + "n": int(values.size),
                    }
                )
            aggregated.append(output)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "seed_level_curve_data.csv", seed_rows)
    write_csv(args.output / "multiseed_curve_data.csv", aggregated)
    write_json(
        args.output / "protocol.json",
        {
            "evaluation_seeds": list(args.seeds),
            "num_seeds": len(args.seeds),
            "traffic_rates_mbps": list(RATES),
            "test_interval_s": [800.0, 830.0],
            "aggregation_unit": "one complete 30-s simulation per seed",
            "center": "arithmetic mean of seed-level metrics",
            "uncertainty": "two-sided 95% Student-t confidence interval across seeds",
            "student_t_critical_value": T95_DF4,
            "trained_models_and_policies": "frozen across evaluation seeds",
            "randomized_components": "traffic arrivals and Rician fading",
            "source_files": source_files,
        },
    )
    print("wrote {} seed rows and {} aggregate rows".format(len(seed_rows), len(aggregated)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    aggregate(parse_args())
