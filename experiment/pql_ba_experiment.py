#!/usr/bin/env python
"""Train, tune, and evaluate the adapted PQL-BA baseline.

Examples
--------
Quick protocol check::

    python experiment/pql_ba_experiment.py --phase tune --train-start 200 \
        --train-end 260 --validation-start 260 --validation-end 270 --epochs 1

Full experiment::

    python experiment/pql_ba_experiment.py --phase all --epochs 3 \
        --final-epochs 5 --rates 19,25,29,35
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import datetime
import json
import os
import pickle
import sys
import time
from typing import Dict, List, MutableMapping, Tuple

import numpy as np

sys.path.append(os.getcwd())

from utils.alg_utils import RA_OTR_SINR
from utils.options import args_parser
from utils.pql_ba import PQLBAConfig, PQLBAPolicy, rollout_link_policy, train_pql_ba
from utils.pql_ba_sim import PQLBASimulationResult, run_sim_pql_ba


DEFAULT_TRAIN_PATH = (
    "sionna_result/trajectoryInfo_lbd1.00_200_800_3Dbeam_"
    "tx(1,32)_rx(1,8)_freq2.8e+10.pkl"
)
DEFAULT_TEST_PATH = (
    "data4sim/lbd1.00_800_950_tx(1,32)_rx(1,8)_"
    "freq2.8e+10_Np8_mode0_lookahead10.pkl"
)
MICRO_BS_LOCATIONS = [
    np.array([300.0, 300.0]),
    np.array([-300.0, 300.0]),
    np.array([300.0, -300.0]),
    np.array([-300.0, -300.0]),
]


def paper_args(data_rate_bps: float = 19e6):
    # ``utils.options.args_parser`` parses the process argv directly.  Isolate
    # it from this experiment driver's own command-line flags.
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        args = args_parser()
    finally:
        sys.argv = saved_argv
    args.from_sionna = True
    args.M_t = 32
    args.M_r = 8
    args.N_bs = 4
    args.slots_per_frame = 100
    args.frames_per_sample = 10
    args.num_RB_macro = 133
    args.num_RB_micro = 66
    args.RB_intervel_macro = 0.36e6
    args.RB_intervel_micro = 1.44e6
    args.p_macro = 1.0
    args.p_micro = 0.2
    args.NF_macro_dB = 5.0
    args.NF_micro_dB = 10.0
    args.random_factor_range4data_rate = 0.0
    args.lat_slot_ub = 20
    args.K = 5
    args.Lambda = 1.0
    args.data_rate = float(data_rate_bps)
    return args


def load_pickle(path: str):
    start = time.time()
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    print("Loaded {} in {:.1f} s".format(path, time.time() - start))
    return value


def temporal_slice(timeline: MutableMapping, start: float, end: float):
    sliced = collections.OrderedDict(
        (frame, records)
        for frame, records in timeline.items()
        if frame >= start - 1e-9 and frame <= end + 1e-9
    )
    if len(sliced) < 2:
        raise ValueError("empty or one-frame slice [{}, {}]".format(start, end))
    return sliced


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


def write_json(path: str, value) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)


def candidate_configs(args) -> collections.OrderedDict:
    candidates = collections.OrderedDict()
    candidates["source_state_z10"] = PQLBAConfig(
        zone_size_m=10.0,
        include_heading=False,
        epsilon_decay_decisions=args.epsilon_decay,
    )
    for zone in args.zone_sizes:
        candidates["heading4_z{:g}".format(zone)] = PQLBAConfig(
            zone_size_m=float(zone),
            include_heading=True,
            num_heading_bins=4,
            epsilon_decay_decisions=args.epsilon_decay,
        )
    for location_bin in args.location_bin_sizes:
        for zone in args.location_zone_sizes:
            candidates[
                "location{:g}_heading4_z{:g}".format(location_bin, zone)
            ] = PQLBAConfig(
                zone_size_m=float(zone),
                include_heading=True,
                num_heading_bins=4,
                location_bin_size_m=float(location_bin),
                epsilon_decay_decisions=args.epsilon_decay,
            )
    return candidates


def tune_candidates(args, output_dir: str) -> Tuple[str, Dict[str, Dict]]:
    timeline = load_pickle(args.train_path)
    train_timeline = temporal_slice(timeline, args.train_start, args.train_end)
    validation_timeline = temporal_slice(
        timeline, args.validation_start, args.validation_end
    )
    common_args = paper_args()
    results = collections.OrderedDict()
    best_name = None
    best_rate = -np.inf

    for candidate_index, (name, config) in enumerate(candidate_configs(args).items()):
        print("\nTraining candidate: {}".format(name))
        start = time.time()
        policy, history = train_pql_ba(
            common_args,
            train_timeline,
            config,
            epochs=args.epochs,
            seed=args.training_seed,
            verbose=True,
        )
        validation = rollout_link_policy(
            common_args,
            validation_timeline,
            policy,
            seed=args.evaluation_seeds[0],
        )
        elapsed = time.time() - start
        policy_path = os.path.join(output_dir, "{}_policy.pkl".format(name))
        policy.save(policy_path)
        results[name] = {
            "config": dataclasses.asdict(config),
            "history": history,
            "validation": validation,
            "elapsed_s": elapsed,
            "policy_path": policy_path,
        }
        print("Validation {}: {}".format(name, validation))
        # PQL-BA's source objective is long-term average data rate.  Candidate
        # selection therefore uses only validation rate, never test queues.
        if validation["average_full_band_rate_mbps"] > best_rate:
            best_rate = validation["average_full_band_rate_mbps"]
            best_name = name
        write_json(os.path.join(output_dir, "tuning_results.json"), results)

    summary = {
        "selection_metric": "validation average_full_band_rate_mbps",
        "selected_candidate": best_name,
        "training_interval": [args.train_start, args.train_end],
        "validation_interval": [args.validation_start, args.validation_end],
        "test_data_used_for_selection": False,
        "candidates": results,
    }
    write_json(os.path.join(output_dir, "tuning_results.json"), summary)
    return str(best_name), results


def final_training(
    args,
    output_dir: str,
    selected_name: str,
    selected_config: PQLBAConfig,
) -> str:
    timeline = load_pickle(args.train_path)
    full_training_timeline = temporal_slice(
        timeline, args.final_train_start, args.final_train_end
    )
    common_args = paper_args()
    print("\nFinal training: {}".format(selected_name))
    start = time.time()
    policy, history = train_pql_ba(
        common_args,
        full_training_timeline,
        selected_config,
        epochs=args.final_epochs,
        seed=args.training_seed,
        verbose=True,
    )
    policy_path = os.path.join(output_dir, "final_policy.pkl")
    policy.save(policy_path)
    write_json(
        os.path.join(output_dir, "final_training.json"),
        {
            "selected_candidate": selected_name,
            "config": dataclasses.asdict(selected_config),
            "training_interval": [args.final_train_start, args.final_train_end],
            "history": history,
            "elapsed_s": time.time() - start,
            "policy_path": policy_path,
        },
    )
    return policy_path


def summarize_loaded_simulation(
    args,
    result: PQLBASimulationResult,
    timeline: MutableMapping,
    warmup_frames: int = 2,
) -> Dict[str, float]:
    warmup = min(warmup_frames, max(len(result.energy_record) - 1, 0))
    sl = slice(warmup, None)
    frame_duration_s = args.slots_per_frame * args.slot_len
    evaluated_connections = [
        connection
        for frame, connection in result.association_record.items()
        if frame >= warmup
    ]
    total_associations = sum(len(x) for x in evaluated_connections)
    macro_associations = sum(
        sum(bs == 0 for bs in x.values()) for x in evaluated_connections
    )
    active_vehicle_counts = np.asarray(
        [len(timeline[frame]) for frame in list(timeline.keys())[1 + warmup :]],
        dtype=float,
    )
    average_vehicle_count = float(active_vehicle_counts.mean())
    return {
        # Match the manuscript experiment scripts: power averages all frames,
        # whereas queue, HO, and pilot statistics discard two warm-up frames.
        "average_system_power_w": float(result.energy_record.mean() / frame_duration_s),
        "queue_violation_percent": float(
            100.0 * result.violation_probability_record[sl].mean()
        ),
        "average_queueing_proxy_ms": float(
            1000.0 * result.average_queue_record[sl].mean() / args.data_rate
        ),
        "handover_per_vehicle_per_s": float(
            result.handover_record[sl].mean()
            / frame_duration_s
            / average_vehicle_count
        ),
        "beam_switch_per_vehicle_per_s": float(
            result.beam_switch_record[sl].mean()
            / frame_duration_s
            / average_vehicle_count
        ),
        "average_pilots_per_micro_link_slot": float(result.pilot_record[sl].mean()),
        "known_state_decision_ratio": float(
            result.known_decision_record[sl].sum()
            / max(result.decision_record[sl].sum(), 1.0)
        ),
        "macro_association_ratio": float(
            macro_associations / max(total_associations, 1)
        ),
        "average_vehicle_count": average_vehicle_count,
        "evaluated_frames": float(len(result.energy_record) - warmup),
    }


def evaluate_policy(args, output_dir: str, policy_path: str) -> Dict[str, Dict]:
    policy = PQLBAPolicy.load(policy_path)
    timeline_all = load_pickle(args.test_path)
    timeline = temporal_slice(timeline_all, args.test_start, args.test_end)
    common_args = paper_args()
    results = collections.OrderedDict()

    link_rollout = rollout_link_policy(
        common_args, timeline, policy, seed=args.evaluation_seeds[0]
    )
    write_json(os.path.join(output_dir, "test_link_rollout.json"), link_rollout)
    print("Frozen-policy test link rollout: {}".format(link_rollout))

    for data_rate_mbps in args.rates:
        for seed in args.evaluation_seeds:
            key = "rate_{:g}Mbps_seed_{}".format(data_rate_mbps, seed)
            common_args.data_rate = float(data_rate_mbps) * 1e6
            print("\nPQL-BA loaded simulation: {}".format(key))
            start = time.time()
            simulation = run_sim_pql_ba(
                common_args,
                MICRO_BS_LOCATIONS,
                timeline,
                policy,
                ra_func=RA_OTR_SINR,
                seed=seed,
                prt=True,
                rician_fading=not args.no_rician,
            )
            summary = summarize_loaded_simulation(common_args, simulation, timeline)
            summary["data_rate_mbps"] = float(data_rate_mbps)
            summary["seed"] = int(seed)
            summary["elapsed_s"] = time.time() - start
            summary["rician_fading"] = not args.no_rician
            results[key] = summary
            print("Summary {}: {}".format(key, summary))
            write_json(os.path.join(output_dir, "loaded_network_results.json"), results)

    csv_path = os.path.join(output_dir, "loaded_network_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        if results:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(results.values())).keys()))
            writer.writeheader()
            writer.writerows(results.values())
    return results


def parse_number_list(raw: str, converter=float) -> List:
    return [converter(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["tune", "final", "evaluate", "all"], default="all")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--selected-candidate", default=None)
    parser.add_argument("--train-start", type=float, default=200.0)
    parser.add_argument("--train-end", type=float, default=740.0)
    parser.add_argument("--validation-start", type=float, default=740.1)
    parser.add_argument("--validation-end", type=float, default=800.0)
    parser.add_argument("--final-train-start", type=float, default=200.0)
    parser.add_argument("--final-train-end", type=float, default=800.0)
    parser.add_argument("--test-start", type=float, default=800.0)
    parser.add_argument("--test-end", type=float, default=830.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=5)
    parser.add_argument("--epsilon-decay", type=float, default=5e4)
    parser.add_argument("--zone-sizes", default="5,10,20")
    parser.add_argument("--location-bin-sizes", default="25,50,100")
    parser.add_argument(
        "--location-zone-sizes",
        default="10",
        help="decision-zone sizes crossed with every location-bin candidate",
    )
    parser.add_argument("--rates", default="19,25,29,35")
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--evaluation-seeds", default="1")
    parser.add_argument("--no-rician", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.zone_sizes = parse_number_list(args.zone_sizes, float)
    args.location_bin_sizes = parse_number_list(args.location_bin_sizes, float)
    args.location_zone_sizes = parse_number_list(args.location_zone_sizes, float)
    args.rates = parse_number_list(args.rates, float)
    args.evaluation_seeds = parse_number_list(args.evaluation_seeds, int)
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("experiment", "results_pql_ba", timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "protocol.json"), vars(args))

    selected_name = args.selected_candidate
    selected_config = None
    policy_path = args.policy_path

    if args.phase in ("tune", "all"):
        selected_name, tuning = tune_candidates(args, args.output_dir)
        selected_config = PQLBAConfig(**tuning[selected_name]["config"])
        if args.phase == "tune":
            print("Selected candidate: {}".format(selected_name))
            return

    if args.phase in ("final", "all"):
        if selected_config is None:
            configs = candidate_configs(args)
            if selected_name is None or selected_name not in configs:
                parser.error("--selected-candidate is required for phase=final")
            selected_config = configs[selected_name]
        policy_path = final_training(
            args, args.output_dir, str(selected_name), selected_config
        )
        if args.phase == "final":
            print("Final policy: {}".format(policy_path))
            return

    if args.phase in ("evaluate", "all"):
        if policy_path is None:
            parser.error("--policy-path is required for phase=evaluate")
        evaluate_policy(args, args.output_dir, policy_path)


if __name__ == "__main__":
    main()
