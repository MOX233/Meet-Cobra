#!/usr/bin/env python3
"""Rebuild all per-seed curve exports from canonical point JSON files."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiment/results_multiseed"
RATES = [float(value) for value in range(1, 36, 2)]
ORIGINAL = (
    "proposed",
    "oracle_mc",
    "oracle_cr_lb",
    "reactive_obra",
    "wo_gap_ho",
    "wo_pet_bf",
    "wo_otr_ra",
)


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def payloads(directory: Path, seed: int):
    result = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (directory / "runs").glob("rate_*Mbps_seed_{}.json".format(seed))
    ]
    result.sort(key=lambda item: float(item["data_rate_mbps"]))
    rates = [float(item["data_rate_mbps"]) for item in result]
    if rates != RATES:
        raise RuntimeError("incomplete point JSON grid in {}: {}".format(directory, rates))
    return result


def refresh_original(method: str, seed: int) -> None:
    directory = BASE / "original_methods" / method / "seed_{}".format(seed)
    items = payloads(directory, seed)
    rows = [
        {
            "method": item["method_label"],
            "method_key": method,
            "data_rate_mbps": item["data_rate_mbps"],
            "seed": seed,
            **item["manuscript_metrics"],
        }
        for item in items
    ]
    write_csv(directory / "curve_data.csv", rows)
    write_json(directory / "curve_results.json", {"curve_data": rows})


def refresh_o_mappo(seed: int) -> None:
    directory = BASE / "baselines" / "seed_{}".format(seed) / "o_mappo_adapted"
    items = payloads(directory, seed)
    rows = [
        {
            "method": item["method_label"],
            "data_rate_mbps": item["data_rate_mbps"],
            "seed": seed,
            **item["manuscript_metrics"],
        }
        for item in items
    ]
    keyed_runs = {
        "rate_{:g}Mbps_seed_{}".format(float(item["data_rate_mbps"]), seed): item
        for item in items
    }
    write_csv(directory / "curve_data.csv", rows)
    write_json(
        directory / "curve_results.json",
        {
            "method": "o_mappo",
            "method_label": "O-MAPPO-adapted",
            "traffic_rates_mbps": RATES,
            "num_completed_points": len(rows),
            "runs": keyed_runs,
            "curve_data": rows,
        },
    )
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["traffic_rates_mbps"] = RATES
    write_json(protocol_path, protocol)


def refresh_mts(seed: int) -> None:
    directory = (
        BASE
        / "mts_gs_hbf/pressure_early/exact_800_830_seed{}".format(seed)
    )
    items = payloads(directory, seed)
    rows = []
    for item in items:
        rows.append(
            {
                "method": item["method"],
                "candidate": item["candidate"],
                "data_rate_mbps": item["data_rate_mbps"],
                "seed": seed,
                **item["manuscript_metrics"],
                "queueing_proxy_p995_ms": item["tail_metrics"]["queueing_proxy_p995_ms"],
                "queueing_proxy_max_ms": item["tail_metrics"]["queueing_proxy_max_ms"],
                "handover_per_vehicle_per_s": item["diagnostic_metrics"]["handover_per_vehicle_per_s"],
                "beam_switch_per_vehicle_per_s": item["diagnostic_metrics"]["beam_switch_per_vehicle_per_s"],
                "trigger_ratio": item["diagnostic_metrics"]["trigger_ratio"],
                "optimizer_mean_overflow_rb": item["diagnostic_metrics"]["optimizer_mean_overflow_rb"],
                "mean_fallbacks_per_association_epoch": item["mts_metrics"]["mean_fallbacks_per_association_epoch"],
            }
        )
    write_csv(directory / "curve_data.csv", rows)
    write_json(directory / "curve_results.json", {"curve_data": rows})
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["rates"] = RATES
    write_json(protocol_path, protocol)


def main() -> None:
    for seed in range(1, 6):
        for method in ORIGINAL:
            refresh_original(method, seed)
        refresh_o_mappo(seed)
        refresh_mts(seed)
    print("rebuilt 45 complete per-seed curve exports")


if __name__ == "__main__":
    main()
