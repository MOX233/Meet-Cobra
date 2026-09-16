#!/usr/bin/env python
"""Train and evaluate queue/load/interference/energy-aware PQL-BA variants."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import sys
import time
from typing import Dict

import numpy as np

sys.path.append(os.getcwd())

from experiment.pql_ba_experiment import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    MICRO_BS_LOCATIONS,
    load_pickle,
    paper_args,
    parse_number_list,
    summarize_loaded_simulation,
    temporal_slice,
    write_json,
)
from utils.alg_utils import RA_OTR_SINR
from utils.pql_ba import PQLBAPolicy
from utils.pql_ba_adapted import (
    AdaptedRewardConfig,
    contextual_config,
    reward_presets,
    run_fluid_pql_episode,
    train_contextual_pql_ba,
)
from utils.pql_ba_sim import run_sim_pql_ba


def _selected_rewards(raw: str) -> "collections.OrderedDict[str, AdaptedRewardConfig]":
    presets = reward_presets()
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in presets]
    if unknown:
        raise ValueError("unknown reward presets: {}".format(", ".join(unknown)))
    return collections.OrderedDict((name, presets[name]) for name in names)


def _selection_score(results: Dict[str, Dict]) -> float:
    """QoS-first score with power only breaking near-QoS ties."""

    violations = [item["queue_violation_percent"] for item in results.values()]
    powers = [item["average_system_power_w"] for item in results.values()]
    return float(max(violations) + 0.01 * np.mean(powers))


def tune(args) -> None:
    timeline_all = load_pickle(args.train_path)
    training = temporal_slice(timeline_all, args.train_start, args.train_end)
    validation = temporal_slice(
        timeline_all, args.validation_start, args.validation_end
    )
    common_args = paper_args()
    rewards = _selected_rewards(args.reward_presets)
    schedule = args.training_rates
    all_results = collections.OrderedDict()

    for name, reward_config in rewards.items():
        print("\nTraining adapted candidate: {}".format(name))
        config = contextual_config(
            zone_size_m=args.zone_size,
            location_bin_size_m=(
                None if args.location_bin_size <= 0 else args.location_bin_size
            ),
            epsilon_decay_decisions=args.epsilon_decay,
            include_traffic_state=args.include_traffic_state,
            hierarchical_bs_action=args.hierarchical_bs_action,
        )
        started = time.time()
        policy, history = train_contextual_pql_ba(
            common_args,
            training,
            config,
            reward_config,
            data_rate_schedule_mbps=schedule,
            epochs=args.epochs,
            seed=args.training_seed,
            verbose=True,
        )
        proxy_validation = collections.OrderedDict()
        for rate in args.validation_rates:
            proxy_validation["rate_{:g}Mbps".format(rate)] = run_fluid_pql_episode(
                common_args,
                validation,
                policy,
                reward_config,
                data_rate_mbps=rate,
                seed=args.training_seed,
                learn=False,
            )
        policy_path = os.path.join(args.output_dir, "{}_policy.pkl".format(name))
        policy.save(policy_path)
        all_results[name] = {
            "config": dataclasses.asdict(config),
            "reward": dataclasses.asdict(reward_config),
            "history": history,
            "proxy_validation": proxy_validation,
            "selection_score": _selection_score(proxy_validation),
            "elapsed_s": time.time() - started,
            "policy_path": policy_path,
        }
        write_json(os.path.join(args.output_dir, "tuning_results.json"), all_results)
        print("Proxy validation {}: {}".format(name, proxy_validation))

    selected = min(all_results, key=lambda name: all_results[name]["selection_score"])
    write_json(
        os.path.join(args.output_dir, "tuning_results.json"),
        {
            "selection_rule": "minimize worst validation violation + 0.01 * mean power",
            "selected_candidate": selected,
            "test_data_used_for_selection": False,
            "training_interval": [args.train_start, args.train_end],
            "validation_interval": [args.validation_start, args.validation_end],
            "candidates": all_results,
        },
    )
    print("Proxy-selected candidate: {}".format(selected))


def exact_validate(args) -> None:
    timeline_all = load_pickle(args.train_path)
    validation = temporal_slice(
        timeline_all, args.exact_validation_start, args.exact_validation_end
    )
    common_args = paper_args()
    results = collections.OrderedDict()
    rewards = _selected_rewards(args.reward_presets)
    for name in rewards:
        policy_path = os.path.join(args.policy_dir, "{}_policy.pkl".format(name))
        policy = PQLBAPolicy.load(policy_path)
        candidate_results = collections.OrderedDict()
        for rate in args.validation_rates:
            common_args.data_rate = rate * 1e6
            print("\nExact validation: {}, {:g} Mbps".format(name, rate))
            started = time.time()
            simulation = run_sim_pql_ba(
                common_args,
                MICRO_BS_LOCATIONS,
                validation,
                policy,
                ra_func=RA_OTR_SINR,
                seed=args.evaluation_seed,
                prt=True,
                rician_fading=True,
            )
            summary = summarize_loaded_simulation(
                common_args, simulation, validation
            )
            summary["data_rate_mbps"] = rate
            summary["elapsed_s"] = time.time() - started
            candidate_results["rate_{:g}Mbps".format(rate)] = summary
        results[name] = {
            "metrics": candidate_results,
            "selection_score": _selection_score(candidate_results),
            "policy_path": policy_path,
        }
        write_json(os.path.join(args.output_dir, "exact_validation.json"), results)
    selected = min(results, key=lambda name: results[name]["selection_score"])
    write_json(
        os.path.join(args.output_dir, "exact_validation.json"),
        {
            "selection_rule": "minimize worst exact-validation violation + 0.01 * mean power",
            "selected_candidate": selected,
            "test_data_used_for_selection": False,
            "validation_interval": [
                args.exact_validation_start,
                args.exact_validation_end,
            ],
            "candidates": results,
        },
    )
    print("Exact-validation-selected candidate: {}".format(selected))


def final_train(args) -> None:
    rewards = _selected_rewards(args.reward_presets)
    if args.selected_candidate not in rewards:
        raise ValueError("--selected-candidate must be one of --reward-presets")
    reward_config = rewards[args.selected_candidate]
    timeline_all = load_pickle(args.train_path)
    timeline = temporal_slice(
        timeline_all, args.final_train_start, args.final_train_end
    )
    common_args = paper_args()
    config = contextual_config(
        zone_size_m=args.zone_size,
        location_bin_size_m=(
            None if args.location_bin_size <= 0 else args.location_bin_size
        ),
        epsilon_decay_decisions=args.epsilon_decay,
        include_traffic_state=args.include_traffic_state,
        hierarchical_bs_action=args.hierarchical_bs_action,
    )
    started = time.time()
    policy, history = train_contextual_pql_ba(
        common_args,
        timeline,
        config,
        reward_config,
        data_rate_schedule_mbps=args.training_rates,
        epochs=args.final_epochs,
        seed=args.training_seed,
        verbose=True,
    )
    policy_path = os.path.join(args.output_dir, "final_policy.pkl")
    policy.save(policy_path)
    write_json(
        os.path.join(args.output_dir, "final_training.json"),
        {
            "selected_candidate": args.selected_candidate,
            "config": dataclasses.asdict(config),
            "reward": dataclasses.asdict(reward_config),
            "history": history,
            "training_interval": [args.final_train_start, args.final_train_end],
            "training_rate_schedule_mbps": args.training_rates,
            "elapsed_s": time.time() - started,
            "policy_path": policy_path,
        },
    )
    print("Final adapted policy: {}".format(policy_path))


def evaluate(args) -> None:
    policy = PQLBAPolicy.load(args.policy_path)
    timeline_all = load_pickle(args.test_path)
    timeline = temporal_slice(timeline_all, args.test_start, args.test_end)
    common_args = paper_args()
    results = collections.OrderedDict()
    for rate in args.test_rates:
        common_args.data_rate = rate * 1e6
        print("\nAdapted PQL-BA test: {:g} Mbps".format(rate))
        started = time.time()
        simulation = run_sim_pql_ba(
            common_args,
            MICRO_BS_LOCATIONS,
            timeline,
            policy,
            ra_func=RA_OTR_SINR,
            seed=args.evaluation_seed,
            prt=True,
            rician_fading=True,
        )
        summary = summarize_loaded_simulation(common_args, simulation, timeline)
        summary["data_rate_mbps"] = rate
        summary["elapsed_s"] = time.time() - started
        results["rate_{:g}Mbps".format(rate)] = summary
        write_json(os.path.join(args.output_dir, "loaded_network_results.json"), results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["tune", "exact-validate", "final", "evaluate"], required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--policy-dir", default=None)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--selected-candidate", default=None)
    parser.add_argument(
        "--reward-presets",
        default=",".join(reward_presets().keys()),
    )
    parser.add_argument("--train-start", type=float, default=200.0)
    parser.add_argument("--train-end", type=float, default=740.0)
    parser.add_argument("--validation-start", type=float, default=740.1)
    parser.add_argument("--validation-end", type=float, default=800.0)
    parser.add_argument("--exact-validation-start", type=float, default=740.1)
    parser.add_argument("--exact-validation-end", type=float, default=750.0)
    parser.add_argument("--final-train-start", type=float, default=200.0)
    parser.add_argument("--final-train-end", type=float, default=800.0)
    parser.add_argument("--test-start", type=float, default=800.0)
    parser.add_argument("--test-end", type=float, default=830.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=12)
    parser.add_argument("--training-rates", default="1,7,13,19")
    parser.add_argument("--validation-rates", default="1,19")
    parser.add_argument("--test-rates", default="1,19")
    parser.add_argument("--zone-size", type=float, default=10.0)
    parser.add_argument("--location-bin-size", type=float, default=100.0)
    parser.add_argument("--epsilon-decay", type=float, default=1e5)
    parser.add_argument("--include-traffic-state", action="store_true")
    parser.add_argument("--hierarchical-bs-action", action="store_true")
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.training_rates = parse_number_list(args.training_rates, float)
    args.validation_rates = parse_number_list(args.validation_rates, float)
    args.test_rates = parse_number_list(args.test_rates, float)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "protocol.json"), vars(args))
    if args.phase == "tune":
        tune(args)
    elif args.phase == "exact-validate":
        if not args.policy_dir:
            raise ValueError("--policy-dir is required for exact validation")
        exact_validate(args)
    elif args.phase == "final":
        if not args.selected_candidate:
            raise ValueError("--selected-candidate is required for final training")
        final_train(args)
    else:
        if not args.policy_path:
            raise ValueError("--policy-path is required for evaluation")
        evaluate(args)


if __name__ == "__main__":
    main()
