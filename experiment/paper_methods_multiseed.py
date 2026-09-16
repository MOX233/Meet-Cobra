#!/usr/bin/env python3
"""Resumable multi-seed evaluation of the original paper methods.

Unlike ``paper_exp1.py``, this driver saves each method/rate/seed point
independently and resets the requested evaluation seed before every method.
It otherwise calls the same simulators, HO/BF routines, trained predictors,
and metric definitions used by the original experiment.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
ORIGINAL_ARGV = tuple(sys.argv)

from experiment.pql_ba_experiment import parse_number_list
from utils.alg_utils import (
    HO_EE_GAP_APX_SINR_conservative_adaptive,
    HO_EE_Greedy_offload,
    HO_LowerBound_SINR,
    RA_OTR3_SINR,
    RA_OTR_SINR,
)
from utils.mox_utils import setup_seed
from utils.sim_utils import (
    get_default_sim_params,
    run_sim_withUMa,
    run_sim_withUMa_analyzed_lowerbound,
)

# ``utils.sim_utils`` resets ``sys.argv`` at import time for legacy notebook
# compatibility.  Restore this driver's command line after importing it.
sys.argv = list(ORIGINAL_ARGV)


PAPER_RATES_MBPS = tuple(float(x) for x in range(1, 36, 2))
METHOD_LABELS = collections.OrderedDict(
    (
        ("proposed", "MEET-COBRA"),
        ("oracle_mc", "Oracle-MC"),
        ("oracle_cr_lb", "Oracle-CR-LB"),
        ("reactive_obra", "Reactive-OBRA"),
        ("wo_gap_ho", "w/o GAP-HO"),
        ("wo_pet_bf", "w/o PET-BF"),
        ("wo_otr_ra", "w/o OTR-RA"),
    )
)
DEFAULT_OUTPUT = Path("experiment/results_multiseed/original_methods")
WARMUP_FRAMES = 2


def json_ready(value):
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


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_common(
    seed: int,
    output: Path,
    test_start: float,
    test_end: float,
    gpu: int,
):
    if abs(test_start - 800.0) > 1e-9:
        raise ValueError("the original test loader starts at 800 s")
    if not 800.0 < test_end <= 950.0:
        raise ValueError("test_end must lie in (800, 950]")
    cut_ratio = (test_end - 800.0) / 150.0
    saved_argv = sys.argv
    original_torch_load = torch.load

    def portable_torch_load(*load_args, **load_kwargs):
        target = torch.device("cuda:{}".format(gpu)) if torch.cuda.is_available() else torch.device("cpu")
        load_kwargs.setdefault("map_location", target)
        return original_torch_load(*load_args, **load_kwargs)

    try:
        sys.argv = [saved_argv[0], "--seed", str(seed)]
        torch.load = portable_torch_load
        values = get_default_sim_params(
            str(output / "_loader"), gpu=gpu, lbd=1, cut_ratio=cut_ratio
        )
    finally:
        torch.load = original_torch_load
        sys.argv = saved_argv
    return values


def method_specs(beampred_model, gainpred_model, inferpred_model) -> Dict[str, Dict]:
    common_predictors = {
        "gainpred_model": gainpred_model,
        "beampred_model": beampred_model,
        "inferpred_model": inferpred_model,
    }
    return {
        "proposed": {
            "RA": RA_OTR_SINR,
            "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
            "BF": "topKbeam_savePilot",
            "save_pilot": True,
            "NoBF": False,
            "K_BF": 5,
            **common_predictors,
        },
        "oracle_mc": {
            "RA": RA_OTR_SINR,
            "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
            "BF": "topKbeam_savePilot",
            "save_pilot": True,
            "gainpred_model": None,
            "beampred_model": None,
            "inferpred_model": None,
            "NoBF": False,
            "K_BF": 5,
        },
        "oracle_cr_lb": {
            "RA": RA_OTR_SINR,
            "HO": HO_LowerBound_SINR,
            "BF": "topKbeam_savePilot",
            "save_pilot": True,
            "gainpred_model": None,
            "beampred_model": None,
            "inferpred_model": None,
            "NoBF": False,
            "K_BF": 1,
        },
        "reactive_obra": {
            "RA": RA_OTR3_SINR,
            "HO": HO_EE_Greedy_offload,
            "BF": "topKbeam_NoPred",
            "save_pilot": False,
            "NoBF": False,
            "K_BF": 5,
            **common_predictors,
        },
        "wo_gap_ho": {
            "RA": RA_OTR_SINR,
            "HO": HO_EE_Greedy_offload,
            "BF": "topKbeam_savePilot",
            "save_pilot": True,
            "NoBF": False,
            "K_BF": 5,
            **common_predictors,
        },
        "wo_pet_bf": {
            "RA": RA_OTR_SINR,
            "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
            "BF": "topKbeam_NoPred",
            "save_pilot": False,
            "NoBF": False,
            "K_BF": 5,
            **common_predictors,
        },
        "wo_otr_ra": {
            "RA": RA_OTR3_SINR,
            "HO": HO_EE_GAP_APX_SINR_conservative_adaptive,
            "BF": "topKbeam_savePilot",
            "save_pilot": True,
            "NoBF": False,
            "K_BF": 5,
            **common_predictors,
        },
    }


def queue_samples_ms(queue_record: Mapping, rate_bps: float) -> np.ndarray:
    samples = []
    for frame in queue_record.values():
        samples.extend(np.asarray(queue, dtype=np.float64).reshape(-1) for queue in frame.values())
    if not samples:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(samples) / rate_bps * 1000.0


def association_counts(command_record: Mapping, num_bs: int) -> np.ndarray:
    keys = list(command_record)
    counts = np.zeros((max(len(keys) - 1, 0), num_bs), dtype=np.int32)
    for row, key in enumerate(keys[1:]):
        for bs in command_record[key].values():
            counts[row, int(bs)] += 1
    return counts


def run_one(
    args,
    common_args,
    bs_locations,
    timeline,
    models,
    spec,
    rate: float,
    prediction_cache=None,
):
    common_args.data_rate = rate * 1e6
    setup_seed(args.seed)
    started = time.time()
    if spec["HO"] == HO_LowerBound_SINR:
        energy, violation = run_sim_withUMa_analyzed_lowerbound(
            common_args,
            bs_locations,
            timeline,
            models["pospred_model"],
            beampred_model=spec["beampred_model"],
            gainpred_model=spec["gainpred_model"],
            inferpred_model=spec["inferpred_model"],
            RA_func=spec["RA"],
            HO_func=spec["HO"],
            prt=False,
            No_BF=spec["NoBF"],
            K_BF=spec["K_BF"],
            batch_prediction=args.batch_prediction,
            prediction_batch_size=args.prediction_batch_size,
            prediction_cache=prediction_cache,
            measured_gain_gamma=args.measured_gain_gamma,
        )
        raw = {
            "energy_j_per_frame": np.asarray(energy),
            "queue_violation_probability_per_frame": np.asarray(violation),
        }
        metrics = {
            "average_system_power_w": float(
                np.asarray(energy).mean()
                / (common_args.slots_per_frame * common_args.slot_len)
            ),
            "queue_violation_percent": float(
                100.0 * np.asarray(violation)[WARMUP_FRAMES:].mean()
            ),
        }
    else:
        (
            energy,
            handover,
            commands,
            violation,
            average_queue,
            pilots,
            rb_allocated,
            queue_record,
        ) = run_sim_withUMa(
            common_args,
            bs_locations,
            timeline,
            models["pospred_model"],
            beampred_model=spec["beampred_model"],
            gainpred_model=spec["gainpred_model"],
            inferpred_model=spec["inferpred_model"],
            RA_func=spec["RA"],
            HO_func=spec["HO"],
            BF_func=spec["BF"],
            prt=False,
            save_pilot=spec["save_pilot"],
            No_BF=spec["NoBF"],
            K_BF=spec["K_BF"],
            batch_prediction=args.batch_prediction,
            prediction_batch_size=args.prediction_batch_size,
            prediction_cache=prediction_cache,
            measured_gain_gamma=args.measured_gain_gamma,
        )
        energy = np.asarray(energy)
        handover = np.asarray(handover)
        violation = np.asarray(violation)
        average_queue = np.asarray(average_queue)
        pilots = np.asarray(pilots)
        rb_allocated = np.asarray(rb_allocated)
        queue_ms = queue_samples_ms(queue_record, common_args.data_rate)
        association = association_counts(commands, len(bs_locations) + 1)
        association_total = float(association.sum())
        frame_duration = common_args.slots_per_frame * common_args.slot_len
        metrics = {
            "average_system_power_w": float(energy.mean() / frame_duration),
            "queue_violation_percent": float(
                100.0 * violation[WARMUP_FRAMES:].mean()
            ),
            "average_queueing_proxy_ms": float(
                1000.0
                * average_queue[WARMUP_FRAMES:].mean()
                / common_args.data_rate
            ),
            "queueing_proxy_p90_ms": float(np.percentile(queue_ms, 90)),
            "queueing_proxy_p99_ms": float(np.percentile(queue_ms, 99)),
            "macro_association_ratio": float(
                association[:, 0].sum() / max(association_total, 1.0)
            ),
        }
        raw = {
            "energy_j_per_frame": energy,
            "handover_count_per_frame": handover,
            "queue_violation_probability_per_frame": violation,
            "average_queue_bits_per_frame_slot": average_queue,
            "average_pilots_per_frame_slot": pilots,
            "rb_allocated_per_bs_per_frame": rb_allocated,
            "association_count_per_bs_per_frame": association,
            "normalized_backlog_proxy_ms_samples": queue_ms,
        }
    return metrics, raw, time.time() - started


def refresh_exports(output: Path, method: str, seed: int) -> None:
    seed_dir = output / method / "seed_{}".format(seed)
    rows = []
    for path in sorted((seed_dir / "runs").glob("rate_*Mbps_seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": payload["method_label"],
                "method_key": method,
                "data_rate_mbps": payload["data_rate_mbps"],
                "seed": seed,
                **payload["manuscript_metrics"],
            }
        )
    rows.sort(key=lambda row: float(row["data_rate_mbps"]))
    if rows:
        write_csv(seed_dir / "curve_data.csv", rows)
        write_json(seed_dir / "curve_results.json", {"curve_data": rows})


def evaluate(args) -> None:
    if args.method not in METHOD_LABELS:
        raise ValueError("unknown method {}".format(args.method))
    torch.set_num_threads(args.torch_threads)
    (
        common_args,
        bs_locations,
        timeline,
        pospred_model,
        beampred_model,
        gainpred_model,
        inferpred_model,
        _,
    ) = load_common(
        args.seed, args.output, args.test_start, args.test_end, args.gpu
    )
    models = {
        "pospred_model": pospred_model,
        "beampred_model": beampred_model,
        "gainpred_model": gainpred_model,
        "inferpred_model": inferpred_model,
    }
    spec = method_specs(beampred_model, gainpred_model, inferpred_model)[args.method]
    prediction_cache = None
    if spec["gainpred_model"] is not None and args.prediction_cache is not None:
        if not args.prediction_cache.is_file():
            raise FileNotFoundError(
                "prediction cache does not exist: {}".format(args.prediction_cache)
            )
        with args.prediction_cache.open("rb") as handle:
            cache_payload = pickle.load(handle)
        prediction_cache = cache_payload["predictions"]
        missing_frames = set(timeline).difference(prediction_cache)
        if missing_frames:
            raise ValueError(
                "prediction cache lacks {} requested frames".format(len(missing_frames))
            )
    seed_dir = args.output / args.method / "seed_{}".format(args.seed)
    runs_dir = seed_dir / "runs"
    raw_dir = seed_dir / "raw"
    runs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        seed_dir / "protocol.json",
        {
            "method": args.method,
            "method_label": METHOD_LABELS[args.method],
            "seed": args.seed,
            "rates_mbps": list(PAPER_RATES_MBPS),
            "requested_rates_this_process": args.rates,
            "test_interval_s": [args.test_start, args.test_end],
            "warmup_frames": WARMUP_FRAMES,
            "common_random_seed_reset_before_each_method_rate": True,
            "source_simulator": "utils.sim_utils.run_sim_withUMa",
            "device": str(common_args.device),
            "batch_prediction": args.batch_prediction,
            "prediction_batch_size": args.prediction_batch_size,
            "prediction_cache": (
                str(args.prediction_cache) if prediction_cache is not None else None
            ),
            "measured_gain_gamma": args.measured_gain_gamma,
        },
    )
    for index, rate in enumerate(args.rates, start=1):
        tag = "rate_{:g}Mbps_seed_{}".format(rate, args.seed)
        summary_path = runs_dir / (tag + ".json")
        raw_path = raw_dir / (tag + ".npz")
        if summary_path.is_file() and raw_path.is_file() and not args.force:
            print("[{}/{}] resume {}".format(index, len(args.rates), tag), flush=True)
            continue
        print(
            "[{}/{}] {} at {:g} Mbps, seed {}".format(
                index, len(args.rates), METHOD_LABELS[args.method], rate, args.seed
            ),
            flush=True,
        )
        metrics, raw, elapsed = run_one(
            args,
            common_args,
            bs_locations,
            timeline,
            models,
            spec,
            float(rate),
            prediction_cache=prediction_cache,
        )
        np.savez_compressed(raw_path, **raw)
        write_json(
            summary_path,
            {
                "method": args.method,
                "method_label": METHOD_LABELS[args.method],
                "data_rate_mbps": float(rate),
                "seed": args.seed,
                "elapsed_s": elapsed,
                "manuscript_metrics": metrics,
                "raw_npz": str(raw_path),
            },
        )
        refresh_exports(args.output, args.method, args.seed)
        print("completed in {:.1f}s: {}".format(elapsed, metrics), flush=True)
    refresh_exports(args.output, args.method, args.seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_LABELS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rates", type=parse_number_list, default=list(PAPER_RATES_MBPS))
    parser.add_argument("--test-start", type=float, default=800.0)
    parser.add_argument("--test-end", type=float, default=830.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-prediction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--measured-gain-gamma", type=float, default=0.9)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
