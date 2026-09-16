#!/usr/bin/env python
"""Train, screen, and exactly evaluate O-MAPPO adaptations."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import sys
import time
from typing import Dict, List, MutableMapping, Tuple

import numpy as np

sys.path.append(os.getcwd())

from experiment.pql_ba_experiment import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    MICRO_BS_LOCATIONS,
    load_pickle,
    paper_args,
    parse_number_list,
    temporal_slice,
    write_json,
)
from utils.alg_utils import RA_OTR_SINR
from utils.o_mappo import (
    OMAPPOConfig,
    OMAPPORewardConfig,
    OMAPPPolicy,
    o_mappo_reward_presets,
    run_fluid_o_mappo_episode,
    train_o_mappo,
)
from utils.o_mappo_sim import OMAPPOSimulationResult, run_sim_o_mappo


def candidate_library(
    args,
) -> "collections.OrderedDict[str, Tuple[OMAPPOConfig, OMAPPORewardConfig]]":
    rewards = o_mappo_reward_presets()

    def config(state: str, gate: str, optimizer: str) -> OMAPPOConfig:
        return OMAPPOConfig(
            state_variant=state,
            trigger_gate=gate,
            optimizer_variant=optimizer,
            optimizer_solver="greedy",
            zone_size_m=args.zone_size,
            hidden_sizes=tuple(args.hidden_sizes),
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            entropy_coefficient=args.entropy_coefficient,
            torch_threads=args.torch_threads,
        )

    candidates = collections.OrderedDict(
        (
            (
                "source_gated",
                (config("source", "source", "source"), rewards["source"]),
            ),
            (
                "source_periodic",
                (config("source", "periodic", "source"), rewards["source"]),
            ),
            (
                "adapted_qos_sourceopt",
                (config("adapted", "periodic", "source"), rewards["qos"]),
            ),
            (
                "adapted_qos_loadopt",
                (config("adapted", "periodic", "load"), rewards["qos"]),
            ),
            (
                "adapted_qos_energy005",
                (
                    config("adapted", "periodic", "load_energy"),
                    rewards["qos_energy005"],
                ),
            ),
            (
                "adapted_qos_energy020",
                (
                    config("adapted", "periodic", "load_energy"),
                    rewards["qos_energy020"],
                ),
            ),
            (
                "adapted_qos_energy020_load1",
                (
                    config("adapted", "periodic", "load_energy"),
                    rewards["qos_energy020_load1"],
                ),
            ),
        )
    )
    if args.candidates:
        requested = [x.strip() for x in args.candidates.split(",") if x.strip()]
        unknown = [x for x in requested if x not in candidates]
        if unknown:
            raise ValueError("unknown candidates: {}".format(", ".join(unknown)))
        candidates = collections.OrderedDict((x, candidates[x]) for x in requested)
    return candidates


def proxy_selection_score(results: Dict[str, Dict]) -> float:
    violations = np.asarray(
        [float(x["queue_violation_percent"]) for x in results.values()]
    )
    powers = np.asarray([float(x["average_system_power_w"]) for x in results.values()])
    # Worst-load reliability is primary; power is a secondary tie-breaker.
    return float(violations.max() + 0.01 * powers.mean())


def summarize_exact(
    args,
    result: OMAPPOSimulationResult,
    timeline: MutableMapping,
    warmup_frames: int = 2,
) -> Dict[str, float]:
    warmup = min(warmup_frames, max(len(result.energy_record) - 1, 0))
    sl = slice(warmup, None)
    frame_duration = args.slots_per_frame * args.slot_len
    active_counts = np.asarray(
        [len(timeline[frame]) for frame in list(timeline)[1 + warmup :]],
        dtype=float,
    )
    average_vehicles = float(active_counts.mean())
    associations = [
        value
        for frame, value in result.association_record.items()
        if frame >= warmup
    ]
    total_associations = sum(len(x) for x in associations)
    macro_associations = sum(
        sum(bs == 0 for bs in values.values()) for values in associations
    )
    decisions = float(result.decision_record[sl].sum())
    triggers = float(result.trigger_record[sl].sum())
    duration_s = (len(result.energy_record) - warmup) * frame_duration
    vehicle_seconds = max(duration_s * average_vehicles, 1e-12)
    trigger_frame_mask = result.trigger_record[sl] > 0
    optimizer_calls = float(np.count_nonzero(trigger_frame_mask))
    return {
        "average_system_power_w": float(
            result.energy_record[sl].mean() / frame_duration
        ),
        "queue_violation_percent": float(
            100.0 * result.violation_probability_record[sl].mean()
        ),
        "average_queueing_proxy_ms": float(
            1000.0 * result.average_queue_record[sl].mean() / args.data_rate
        ),
        "handover_per_vehicle_per_s": float(
            result.handover_record[sl].sum() / vehicle_seconds
        ),
        "beam_switch_per_vehicle_per_s": float(
            result.beam_switch_record[sl].sum() / vehicle_seconds
        ),
        "decision_per_vehicle_per_s": float(decisions / vehicle_seconds),
        "trigger_ratio": float(triggers / max(decisions, 1.0)),
        "average_pilots_per_micro_link_slot": float(
            result.pilot_record[sl].mean()
        ),
        "full_sweep_per_vehicle_per_s": float(
            result.full_sweep_record[sl].sum() / vehicle_seconds
        ),
        "local_sweep_per_vehicle_per_s": float(
            result.local_sweep_record[sl].sum() / vehicle_seconds
        ),
        "actor_inference_ms_per_decision_frame": float(
            1000.0
            * result.inference_time_record[sl].sum()
            / max(np.count_nonzero(result.decision_record[sl]), 1)
        ),
        "optimizer_ms_per_trigger_frame": float(
            1000.0
            * result.optimizer_time_record[sl][trigger_frame_mask].sum()
            / max(optimizer_calls, 1)
        ),
        "optimizer_failure_count": float(
            result.optimizer_failure_record[sl][trigger_frame_mask].sum()
        ),
        "optimizer_mean_overflow_rb": float(
            result.optimizer_overflow_record[sl][trigger_frame_mask].sum()
            / max(optimizer_calls, 1)
        ),
        "skipped_gate_events": float(result.skipped_gate_record[sl].sum()),
        "macro_association_ratio": float(
            macro_associations / max(total_associations, 1)
        ),
        "average_vehicle_count": average_vehicles,
        "evaluated_frames": float(len(result.energy_record) - warmup),
    }


def tune(args) -> None:
    timeline_all = load_pickle(args.train_path)
    training = temporal_slice(timeline_all, args.train_start, args.train_end)
    validation = temporal_slice(
        timeline_all, args.validation_start, args.validation_end
    )
    common_args = paper_args()
    all_results = collections.OrderedDict()
    for name, (config, reward) in candidate_library(args).items():
        print("\nTraining O-MAPPO candidate: {}".format(name))
        started = time.time()
        policy, history = train_o_mappo(
            common_args,
            training,
            config,
            reward,
            args.training_rates,
            args.epochs,
            seed=args.training_seed,
            verbose=True,
        )
        validation_results = collections.OrderedDict()
        for rate in args.validation_rates:
            key = "rate_{:g}Mbps".format(rate)
            validation_results[key] = run_fluid_o_mappo_episode(
                common_args,
                validation,
                policy,
                reward,
                rate,
                seed=args.training_seed,
                learn=False,
            )
        policy_path = os.path.join(args.output_dir, "{}_policy.pt".format(name))
        policy.save(policy_path)
        all_results[name] = {
            "config": dataclasses.asdict(config),
            "reward": dataclasses.asdict(reward),
            "history": history,
            "proxy_validation": validation_results,
            "selection_score": proxy_selection_score(validation_results),
            "elapsed_s": time.time() - started,
            "policy_path": policy_path,
        }
        write_json(os.path.join(args.output_dir, "tuning_results_partial.json"), all_results)
        print("Validation {}: {}".format(name, validation_results))
    selected = min(all_results, key=lambda x: all_results[x]["selection_score"])
    write_json(
        os.path.join(args.output_dir, "tuning_results.json"),
        {
            "selection_rule": "minimize worst proxy violation + 0.01 * mean power",
            "selected_candidate": selected,
            "test_data_used_for_selection": False,
            "training_interval": [args.train_start, args.train_end],
            "validation_interval": [args.validation_start, args.validation_end],
            "candidates": all_results,
        },
    )
    print("Proxy-selected O-MAPPO candidate: {}".format(selected))


def exact_validate(args) -> None:
    timeline_all = load_pickle(args.train_path)
    validation = temporal_slice(
        timeline_all, args.exact_validation_start, args.exact_validation_end
    )
    common_args = paper_args()
    results = collections.OrderedDict()
    for name in candidate_library(args):
        policy_path = os.path.join(args.policy_dir, "{}_policy.pt".format(name))
        policy = OMAPPPolicy.load(policy_path, seed=args.evaluation_seeds[0])
        metrics = collections.OrderedDict()
        for rate in args.validation_rates:
            common_args.data_rate = rate * 1e6
            print("\nExact O-MAPPO validation: {}, {:g} Mbps".format(name, rate))
            started = time.time()
            simulation = run_sim_o_mappo(
                common_args,
                MICRO_BS_LOCATIONS,
                validation,
                policy,
                ra_func=RA_OTR_SINR,
                seed=args.evaluation_seeds[0],
                prt=True,
                rician_fading=not args.no_rician,
                optimizer_solver=args.exact_optimizer_solver,
            )
            summary = summarize_exact(common_args, simulation, validation)
            summary["elapsed_s"] = time.time() - started
            summary["data_rate_mbps"] = rate
            metrics["rate_{:g}Mbps".format(rate)] = summary
        results[name] = {
            "metrics": metrics,
            "selection_score": proxy_selection_score(metrics),
            "policy_path": policy_path,
        }
        write_json(os.path.join(args.output_dir, "exact_validation_partial.json"), results)
    selected = min(results, key=lambda x: results[x]["selection_score"])
    write_json(
        os.path.join(args.output_dir, "exact_validation.json"),
        {
            "selection_rule": "minimize worst exact violation + 0.01 * mean power",
            "selected_candidate": selected,
            "test_data_used_for_selection": False,
            "validation_interval": [
                args.exact_validation_start,
                args.exact_validation_end,
            ],
            "optimizer_solver": args.exact_optimizer_solver,
            "candidates": results,
        },
    )
    print("Exact-selected O-MAPPO candidate: {}".format(selected))


def final_train(args) -> None:
    library = candidate_library(args)
    if args.selected_candidate not in library:
        raise ValueError("selected candidate is not in candidate library")
    config, reward = library[args.selected_candidate]
    timeline_all = load_pickle(args.train_path)
    timeline = temporal_slice(
        timeline_all, args.final_train_start, args.final_train_end
    )
    common_args = paper_args()
    started = time.time()
    policy, history = train_o_mappo(
        common_args,
        timeline,
        config,
        reward,
        args.training_rates,
        args.final_epochs,
        seed=args.training_seed,
        verbose=True,
    )
    policy_path = os.path.join(args.output_dir, "final_policy.pt")
    policy.save(policy_path)
    write_json(
        os.path.join(args.output_dir, "final_training.json"),
        {
            "selected_candidate": args.selected_candidate,
            "config": dataclasses.asdict(config),
            "reward": dataclasses.asdict(reward),
            "history": history,
            "training_interval": [args.final_train_start, args.final_train_end],
            "training_rate_schedule_mbps": args.training_rates,
            "elapsed_s": time.time() - started,
            "policy_path": policy_path,
        },
    )
    print("Final O-MAPPO policy: {}".format(policy_path))


def _aggregate_seeds(raw: Dict[str, Dict]) -> Dict[str, Dict]:
    grouped: Dict[float, List[Dict]] = collections.defaultdict(list)
    for item in raw.values():
        grouped[float(item["data_rate_mbps"])].append(item)
    aggregate = collections.OrderedDict()
    metrics = [
        "average_system_power_w",
        "queue_violation_percent",
        "average_queueing_proxy_ms",
        "handover_per_vehicle_per_s",
        "beam_switch_per_vehicle_per_s",
        "trigger_ratio",
        "average_pilots_per_micro_link_slot",
        "macro_association_ratio",
    ]
    t95_by_df = {
        1: 12.706204736,
        2: 4.302652730,
        3: 3.182446305,
        4: 2.776445105,
        5: 2.570581836,
        6: 2.446911851,
        7: 2.364624252,
        8: 2.306004135,
        9: 2.262157163,
        10: 2.228138852,
    }
    for rate in sorted(grouped):
        items = grouped[rate]
        summary = {"data_rate_mbps": rate, "num_seeds": len(items)}
        for metric in metrics:
            values = np.asarray([x[metric] for x in items], dtype=float)
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            critical = t95_by_df.get(len(values) - 1, 1.96)
            summary[metric + "_ci95"] = (
                float(critical * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        aggregate["rate_{:g}Mbps".format(rate)] = summary
    return aggregate


def evaluate(args) -> None:
    policy = OMAPPPolicy.load(args.policy_path, seed=args.evaluation_seeds[0])
    timeline_all = load_pickle(args.test_path)
    timeline = temporal_slice(timeline_all, args.test_start, args.test_end)
    common_args = paper_args()
    results = collections.OrderedDict()
    for rate in args.test_rates:
        for seed in args.evaluation_seeds:
            common_args.data_rate = rate * 1e6
            key = "rate_{:g}Mbps_seed_{}".format(rate, seed)
            print("\nO-MAPPO exact test: {}".format(key))
            started = time.time()
            simulation = run_sim_o_mappo(
                common_args,
                MICRO_BS_LOCATIONS,
                timeline,
                policy,
                ra_func=RA_OTR_SINR,
                seed=seed,
                prt=True,
                rician_fading=not args.no_rician,
                optimizer_solver=args.exact_optimizer_solver,
            )
            summary = summarize_exact(common_args, simulation, timeline)
            summary.update(
                {
                    "data_rate_mbps": rate,
                    "seed": seed,
                    "elapsed_s": time.time() - started,
                    "rician_fading": not args.no_rician,
                    "optimizer_solver": args.exact_optimizer_solver,
                }
            )
            results[key] = summary
            write_json(
                os.path.join(args.output_dir, "loaded_network_results.json"),
                {"raw": results, "aggregate": _aggregate_seeds(results)},
            )
            print("Summary {}: {}".format(key, summary))


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
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--train-start", type=float, default=200.0)
    parser.add_argument("--train-end", type=float, default=240.0)
    parser.add_argument("--validation-start", type=float, default=240.1)
    parser.add_argument("--validation-end", type=float, default=250.0)
    parser.add_argument("--exact-validation-start", type=float, default=740.1)
    parser.add_argument("--exact-validation-end", type=float, default=745.0)
    parser.add_argument("--final-train-start", type=float, default=200.0)
    parser.add_argument("--final-train-end", type=float, default=800.0)
    parser.add_argument("--test-start", type=float, default=800.0)
    parser.add_argument("--test-end", type=float, default=830.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=8)
    parser.add_argument("--training-rates", default="1,7,13,19")
    parser.add_argument("--validation-rates", default="1,19")
    parser.add_argument("--test-rates", default="1,19")
    parser.add_argument("--zone-size", type=float, default=10.0)
    parser.add_argument("--hidden-sizes", default="64")
    parser.add_argument("--actor-learning-rate", type=float, default=5e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=5e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--evaluation-seeds", default="1")
    parser.add_argument("--exact-optimizer-solver", choices=["greedy", "milp"], default="milp")
    parser.add_argument("--no-rician", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.hidden_sizes = tuple(parse_number_list(args.hidden_sizes, int))
    args.training_rates = parse_number_list(args.training_rates, float)
    args.validation_rates = parse_number_list(args.validation_rates, float)
    args.test_rates = parse_number_list(args.test_rates, float)
    args.evaluation_seeds = parse_number_list(args.evaluation_seeds, int)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, "protocol.json"), vars(args))
    if args.phase == "tune":
        tune(args)
    elif args.phase == "exact-validate":
        if not args.policy_dir:
            raise ValueError("--policy-dir is required")
        exact_validate(args)
    elif args.phase == "final":
        if not args.selected_candidate:
            raise ValueError("--selected-candidate is required")
        final_train(args)
    else:
        if not args.policy_path:
            raise ValueError("--policy-path is required")
        evaluate(args)


if __name__ == "__main__":
    main()
