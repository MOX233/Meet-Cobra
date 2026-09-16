#!/usr/bin/env python3
"""Screen and evaluate the non-RL MTS-GS-HBF-adapted baseline."""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, MutableMapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiment.o_mappo_experiment import summarize_exact
from experiment.pql_ba_experiment import (
    DEFAULT_TEST_PATH,
    MICRO_BS_LOCATIONS,
    load_pickle,
    paper_args,
    parse_number_list,
    temporal_slice,
)
from experiment.run_joint_baseline_full_grid import (
    _manuscript_metrics,
    _raw_arrays,
)
from utils.alg_utils import RA_OTR_SINR
from utils.mts_gs_hbf import MTSGSHBFConfig, candidate_configs
from utils.mts_gs_hbf_sim import run_sim_mts_gs_hbf


DEFAULT_OUTPUT = Path("experiment/results_mts_gs_hbf")
DEFAULT_SCREEN_RATES = (1.0, 19.0, 27.0)
DEFAULT_EVAL_RATES = (1.0, 19.0, 27.0)


def json_ready(value):
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def extra_diagnostics(result) -> Dict[str, float]:
    association_mask = result.association_epoch_record > 0
    return {
        "association_epoch_ratio": float(result.association_epoch_record.mean()),
        "full_sweep_epoch_ratio": float(result.full_sweep_epoch_record.mean()),
        "local_tracking_epoch_ratio": float(result.local_tracking_epoch_record.mean()),
        "mean_proposals_per_association_epoch": float(
            result.proposal_record[association_mask].mean()
            if np.any(association_mask)
            else 0.0
        ),
        "mean_fallbacks_per_association_epoch": float(
            result.unassigned_record[association_mask].mean()
            if np.any(association_mask)
            else 0.0
        ),
    }


def simulate_one(
    common_args,
    timeline: MutableMapping,
    config: MTSGSHBFConfig,
    rate: float,
    seed: int,
):
    common_args.data_rate = float(rate) * 1e6
    started = time.time()
    result = run_sim_mts_gs_hbf(
        common_args,
        MICRO_BS_LOCATIONS,
        timeline,
        config,
        ra_func=RA_OTR_SINR,
        seed=seed,
        prt=False,
        rician_fading=True,
    )
    return result, time.time() - started


def screen_score(rows: Sequence[Dict]) -> float:
    by_rate = {float(row["data_rate_mbps"]): row for row in rows}
    reliability = sum(
        float(by_rate[rate]["queue_violation_percent"])
        for rate in (19.0, 27.0)
        if rate in by_rate
    )
    mean_power = float(np.mean([x["average_system_power_w"] for x in rows]))
    mean_queue = float(np.mean([x["average_queueing_proxy_ms"] for x in rows]))
    return reliability + 0.02 * mean_power + 0.002 * mean_queue


def run_screen(args) -> None:
    configs = candidate_configs()
    if args.screen_candidates:
        requested = [x.strip() for x in args.screen_candidates.split(",") if x.strip()]
        unknown = sorted(set(requested).difference(configs))
        if unknown:
            raise ValueError("unknown screening candidates: {}".format(unknown))
        configs = {name: configs[name] for name in requested}
    timeline_all = load_pickle(args.test_path)
    timeline = temporal_slice(timeline_all, args.screen_start, args.screen_end)
    common_args = paper_args()
    summaries = []
    for index, (name, config) in enumerate(configs.items(), start=1):
        print("\n[{}/{}] screening {}".format(index, len(configs), name), flush=True)
        rows = []
        for rate in args.screen_rates:
            result, elapsed = simulate_one(
                common_args, timeline, config, rate, args.seed
            )
            metrics = _manuscript_metrics(common_args, result)
            diagnostic = summarize_exact(common_args, result, timeline)
            row = {
                "candidate": name,
                "data_rate_mbps": float(rate),
                **metrics,
                "handover_per_vehicle_per_s": diagnostic[
                    "handover_per_vehicle_per_s"
                ],
                "beam_switch_per_vehicle_per_s": diagnostic[
                    "beam_switch_per_vehicle_per_s"
                ],
                "trigger_ratio": diagnostic["trigger_ratio"],
                "optimizer_mean_overflow_rb": diagnostic[
                    "optimizer_mean_overflow_rb"
                ],
                "elapsed_s": elapsed,
                **extra_diagnostics(result),
            }
            rows.append(row)
            print(
                "  {:g} Mbps: {:.2f} W, vio {:.3f}%, queue {:.2f} ms, "
                "HO {:.3f}/veh/s".format(
                    rate,
                    metrics["average_system_power_w"],
                    metrics["queue_violation_percent"],
                    metrics["average_queueing_proxy_ms"],
                    diagnostic["handover_per_vehicle_per_s"],
                ),
                flush=True,
            )
        summary = {
            "candidate": name,
            "selection_score": screen_score(rows),
            "config": dataclasses.asdict(config),
            "results": rows,
        }
        summaries.append(summary)
        write_json(args.output / "screen" / name / "summary.json", summary)
    summaries.sort(key=lambda x: x["selection_score"])
    payload = {
        "selected_candidate": summaries[0]["candidate"],
        "test_path": args.test_path,
        "screen_interval": [args.screen_start, args.screen_end],
        "seed": args.seed,
        "ranked_candidates": summaries,
    }
    write_json(args.output / "screen" / "screening_summary.json", payload)
    print("\nScreening ranking:")
    for summary in summaries:
        print(
            "  {}: {:.4f}".format(
                summary["candidate"], summary["selection_score"]
            )
        )


def run_payload(
    common_args,
    timeline,
    config: MTSGSHBFConfig,
    rate: float,
    seed: int,
    raw_path: Path,
) -> Dict:
    result, elapsed = simulate_one(common_args, timeline, config, rate, seed)
    raw = _raw_arrays(common_args, result)
    raw.update(
        {
            "association_epoch_per_frame": result.association_epoch_record,
            "full_sweep_epoch_per_frame": result.full_sweep_epoch_record,
            "local_tracking_epoch_per_frame": result.local_tracking_epoch_record,
            "proposal_count_per_frame": result.proposal_record,
            "fallback_count_per_frame": result.unassigned_record,
        }
    )
    queue_ms = raw["normalized_backlog_proxy_ms_samples"]
    np.savez_compressed(raw_path, **raw)
    return {
        "method": "MTS-GS-HBF-adapted",
        "candidate": config.name,
        "data_rate_mbps": float(rate),
        "seed": int(seed),
        "elapsed_s": elapsed,
        "manuscript_metrics": _manuscript_metrics(common_args, result),
        "diagnostic_metrics": summarize_exact(common_args, result, timeline),
        "mts_metrics": extra_diagnostics(result),
        "tail_metrics": {
            "queueing_proxy_p995_ms": float(np.percentile(queue_ms, 99.5)),
            "queueing_proxy_max_ms": float(queue_ms.max()),
        },
        "raw_npz": str(raw_path),
    }


def curve_rows(payloads: Iterable[Dict]) -> List[Dict]:
    rows = []
    for payload in sorted(payloads, key=lambda x: x["data_rate_mbps"]):
        metrics = payload["manuscript_metrics"]
        rows.append(
            {
                "method": payload["method"],
                "candidate": payload["candidate"],
                "data_rate_mbps": payload["data_rate_mbps"],
                "seed": payload["seed"],
                **metrics,
                "queueing_proxy_p995_ms": payload["tail_metrics"][
                    "queueing_proxy_p995_ms"
                ],
                "queueing_proxy_max_ms": payload["tail_metrics"][
                    "queueing_proxy_max_ms"
                ],
                "handover_per_vehicle_per_s": payload["diagnostic_metrics"][
                    "handover_per_vehicle_per_s"
                ],
                "beam_switch_per_vehicle_per_s": payload["diagnostic_metrics"][
                    "beam_switch_per_vehicle_per_s"
                ],
                "trigger_ratio": payload["diagnostic_metrics"]["trigger_ratio"],
                "optimizer_mean_overflow_rb": payload["diagnostic_metrics"][
                    "optimizer_mean_overflow_rb"
                ],
                "mean_fallbacks_per_association_epoch": payload["mts_metrics"][
                    "mean_fallbacks_per_association_epoch"
                ],
            }
        )
    return rows


def run_evaluate(args) -> None:
    configs = candidate_configs()
    if args.candidate not in configs:
        raise ValueError("unknown candidate {}".format(args.candidate))
    config = configs[args.candidate]
    timeline_all = load_pickle(args.test_path)
    timeline = temporal_slice(timeline_all, args.test_start, args.test_end)
    common_args = paper_args()
    output = args.output / args.candidate / "exact_{:g}_{:g}_seed{}".format(
        args.test_start, args.test_end, args.seed
    )
    runs_dir = output / "runs"
    raw_dir = output / "raw"
    runs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for index, rate in enumerate(args.eval_rates, start=1):
        tag = "rate_{:g}Mbps_seed_{}".format(rate, args.seed)
        summary_path = runs_dir / (tag + ".json")
        raw_path = raw_dir / (tag + ".npz")
        if summary_path.is_file() and raw_path.is_file() and not args.force:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            canonical_raw_path = str(raw_path)
            if payload.get("raw_npz") != canonical_raw_path:
                payload["raw_npz"] = canonical_raw_path
                write_json(summary_path, payload)
            payloads.append(payload)
            print("[{}/{}] resume {}".format(index, len(args.eval_rates), tag))
            continue
        print(
            "\n[{}/{}] {} exact test at {:g} Mbps".format(
                index, len(args.eval_rates), config.name, rate
            ),
            flush=True,
        )
        payload = run_payload(
            common_args, timeline, config, rate, args.seed, raw_path
        )
        write_json(summary_path, payload)
        payloads.append(payload)
        rows = curve_rows(payloads)
        write_json(output / "curve_results.json", {"curve_data": rows})
        write_csv(output / "curve_data.csv", rows)
        print(
            "  completed in {:.1f}s: {}".format(
                payload["elapsed_s"], payload["manuscript_metrics"]
            ),
            flush=True,
        )
    rows = curve_rows(payloads)
    write_json(output / "curve_results.json", {"curve_data": rows})
    write_csv(output / "curve_data.csv", rows)
    write_json(
        output / "protocol.json",
        {
            "method": "MTS-GS-HBF-adapted",
            "candidate": config.name,
            "config": dataclasses.asdict(config),
            "test_path": args.test_path,
            "test_interval": [args.test_start, args.test_end],
            "rates": args.eval_rates,
            "seed": args.seed,
            "optimizer": "capacity-aware Gale-Shapley",
            "resource_allocator": "OTR-RA",
            "rician_fading": True,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "evaluate"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--screen-start", type=float, default=800.0)
    parser.add_argument("--screen-end", type=float, default=803.0)
    parser.add_argument(
        "--screen-rates",
        type=parse_number_list,
        default=list(DEFAULT_SCREEN_RATES),
    )
    parser.add_argument(
        "--screen-candidates",
        default="",
        help="optional comma-separated subset of candidate configuration names",
    )
    parser.add_argument("--candidate", default="balanced")
    parser.add_argument("--test-start", type=float, default=800.0)
    parser.add_argument("--test-end", type=float, default=805.0)
    parser.add_argument(
        "--eval-rates",
        type=parse_number_list,
        default=list(DEFAULT_EVAL_RATES),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "screen":
        run_screen(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
