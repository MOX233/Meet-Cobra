#!/usr/bin/env python3
"""Train and evaluate two stronger O-MAPPO adaptations.

Experiment 1 adds a tail-queue CVaR reward, all four alternative BSs, a
two-layer 128-unit actor/critic, and a longer-horizon PPO configuration.
Experiment 2 additionally exposes candidate feasibility and optimizer feedback
and replaces both actor and critic encoders with fixed-window GRUs.

Every stage is resumable and writes to a directory separate from the frozen
formal baseline.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiment.o_mappo_experiment import summarize_exact
from experiment.pql_ba_experiment import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
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
from utils.o_mappo import (
    OMAPPOConfig,
    OMAPPORewardConfig,
    OMAPPPolicy,
    run_fluid_o_mappo_episode,
)
from utils.o_mappo_sim import run_sim_o_mappo


DEFAULT_OUTPUT_ROOT = Path("experiment/results/o_mappo_improvements")
SCREEN_WEIGHTS = (0.5, 1.0, 2.0, 3.0)
SCREEN_RATES = (1.0, 7.0, 9.0, 13.0, 19.0, 27.0)
FINAL_TRAINING_SCHEDULE = (
    1.0,
    7.0,
    13.0,
    19.0,
    25.0,
    31.0,
    7.0,
    19.0,
    1.0,
    31.0,
    13.0,
    25.0,
)
DEFAULT_EXACT_RATES = (1.0, 7.0, 9.0, 19.0, 27.0, 35.0)


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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def weight_tag(value: float) -> str:
    return ("{:g}".format(value)).replace(".", "p")


def build_reward(cvar_weight: float) -> OMAPPORewardConfig:
    return OMAPPORewardConfig(
        name="qos_energy020_load1_cvar{}".format(weight_tag(cvar_weight)),
        team_mix=0.5,
        service_weight=2.0,
        queue_weight=1.0,
        violation_weight=5.0,
        energy_weight=0.20,
        overload_weight=1.0,
        cvar_weight=float(cvar_weight),
        cvar_alpha=0.95,
        handover_weight=0.05,
        sweep_weight=0.02,
    )


def build_config(variant: str) -> OMAPPOConfig:
    common = dict(
        state_variant="adapted",
        trigger_gate="periodic",
        optimizer_variant="load_energy",
        optimizer_solver="greedy",
        candidate_count=4,
        hidden_sizes=(128, 128),
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        discount_factor=0.98,
        gae_lambda=0.90,
        clip_ratio=0.20,
        ppo_epochs=4,
        batch_size=256,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        gradient_clip_norm=10.0,
        optimizer_overflow_penalty=25.0,
        optimizer_load_weight=1.0,
        optimizer_energy_weight=0.20,
        torch_threads=4,
    )
    if variant == "exp1":
        return OMAPPOConfig(**common)
    if variant == "exp2":
        common.update(
            state_variant="feasibility",
            hidden_sizes=(128,),
            recurrent=True,
            recurrent_hidden_size=128,
            recurrent_sequence_length=4,
            # Sequence batches are substantially more expensive than MLP
            # samples.  Larger minibatches preserve three full PPO passes
            # while keeping the full-trace experiment tractable.
            ppo_epochs=3,
            batch_size=512,
        )
        return OMAPPOConfig(**common)
    raise ValueError("unknown variant {}".format(variant))


def train_resumable(
    output_dir: Path,
    variant: str,
    reward: OMAPPORewardConfig,
    timeline,
    rates: Sequence[float],
    epochs: int,
    seed: int,
) -> Tuple[OMAPPPolicy, List[Dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.pt"
    history_path = output_dir / "training_history.json"
    config = build_config(variant)
    if policy_path.is_file() and history_path.is_file():
        policy = OMAPPPolicy.load(str(policy_path), seed=seed, load_optimizers=True)
        history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
        if dataclasses.asdict(policy.config) != dataclasses.asdict(config):
            raise RuntimeError("saved policy config does not match requested config")
        print("Resuming {} from epoch {}".format(output_dir, len(history)))
    else:
        policy = OMAPPPolicy(config, seed=seed)
        history = []

    args = paper_args()
    started = time.time()
    for epoch_index in range(len(history), epochs):
        rate = float(rates[epoch_index % len(rates)])
        result = run_fluid_o_mappo_episode(
            args,
            timeline,
            policy,
            reward,
            rate,
            seed=seed + epoch_index,
            learn=True,
        )
        result["epoch"] = epoch_index + 1
        result["elapsed_total_s"] = time.time() - started
        history.append(result)
        policy.save(str(policy_path))
        write_json(
            history_path,
            {
                "variant": variant,
                "config": dataclasses.asdict(config),
                "reward": dataclasses.asdict(reward),
                "rates": rates,
                "epochs_requested": epochs,
                "history": history,
            },
        )
        print(
            "{} epoch {:02d}/{:02d} rate={:g}: power={:.2f} W, vio={:.3f}%, "
            "CVaR-excess={:.3f}, trigger={:.3f}, actor={:.4f}, critic={:.4f}".format(
                variant,
                epoch_index + 1,
                epochs,
                rate,
                result["average_system_power_w"],
                result["queue_violation_percent"],
                result["queue_cvar_excess_ratio"],
                result["trigger_ratio"],
                result["actor_loss"],
                result["critic_loss"],
            ),
            flush=True,
        )
    return policy, history


def fluid_validate(
    output_dir: Path,
    policy: OMAPPPolicy,
    reward: OMAPPORewardConfig,
    timeline,
    rates: Sequence[float],
    seed: int,
) -> List[Dict]:
    args = paper_args()
    rows = []
    for index, rate in enumerate(rates):
        result = run_fluid_o_mappo_episode(
            args,
            timeline,
            policy,
            reward,
            float(rate),
            seed=seed + 1000 + index,
            learn=False,
        )
        rows.append(result)
        print(
            "validate rate={:g}: power={:.2f} W, vio={:.3f}%, queue={:.2f} ms, "
            "CVaR-excess={:.3f}".format(
                rate,
                result["average_system_power_w"],
                result["queue_violation_percent"],
                result["average_queueing_proxy_ms"],
                result["queue_cvar_excess_ratio"],
            ),
            flush=True,
        )
    write_json(output_dir / "fluid_validation.json", {"results": rows})
    return rows


def screen_exp1(args) -> None:
    timeline_all = load_pickle(args.train_path)
    training = temporal_slice(timeline_all, args.screen_train_start, args.screen_train_end)
    validation = temporal_slice(
        timeline_all, args.screen_validation_start, args.screen_validation_end
    )
    summary = []
    for weight in SCREEN_WEIGHTS:
        print("\n=== Exp-1 CVaR weight {:g} ===".format(weight), flush=True)
        output = args.output_root / "screen_exp1" / ("cvar_" + weight_tag(weight))
        reward = build_reward(weight)
        policy, _ = train_resumable(
            output,
            "exp1",
            reward,
            training,
            args.screen_rates,
            args.screen_epochs,
            args.seed,
        )
        rows = fluid_validate(
            output, policy, reward, validation, SCREEN_RATES, args.seed
        )
        low_medium = [row for row in rows if row["data_rate_mbps"] <= 19.0]
        score = (
            max(row["queue_violation_percent"] for row in low_medium)
            + 0.25 * rows[-1]["queue_violation_percent"]
            + np.mean([row["queue_cvar_excess_ratio"] for row in rows])
            + 0.005 * np.mean([row["average_system_power_w"] for row in rows])
        )
        summary.append(
            {
                "cvar_weight": weight,
                "selection_score": float(score),
                "validation": rows,
                "output_dir": str(output),
            }
        )
    summary.sort(key=lambda row: row["selection_score"])
    write_json(
        args.output_root / "screen_exp1" / "screening_summary.json",
        {"ranked_candidates": summary},
    )
    print("\nExp-1 screening ranking:")
    for row in summary:
        print(
            "  CVaR {:g}: score {:.4f}".format(
                row["cvar_weight"], row["selection_score"]
            )
        )


def variant_output(args, variant: str) -> Path:
    name = (
        "exp1_cvar_full_strong"
        if variant == "exp1"
        else "exp2_feasibility_recurrent"
    )
    return args.output_root / name


def train_final(args) -> None:
    timeline_all = load_pickle(args.train_path)
    timeline = temporal_slice(timeline_all, args.train_start, args.train_end)
    output = variant_output(args, args.variant)
    reward = build_reward(args.cvar_weight)
    train_resumable(
        output,
        args.variant,
        reward,
        timeline,
        args.training_rates,
        args.epochs,
        args.seed,
    )
    write_json(
        output / "training_protocol.json",
        {
            "variant": args.variant,
            "train_path": args.train_path,
            "train_interval": [args.train_start, args.train_end],
            "training_rates": args.training_rates,
            "epochs": args.epochs,
            "seed": args.seed,
            "config": dataclasses.asdict(build_config(args.variant)),
            "reward": dataclasses.asdict(reward),
        },
    )


def _curve_rows(run_payloads: Iterable[Dict]) -> List[Dict]:
    rows = []
    for payload in sorted(run_payloads, key=lambda x: x["data_rate_mbps"]):
        row = {
            "variant": payload["variant"],
            "data_rate_mbps": payload["data_rate_mbps"],
            "seed": payload["seed"],
        }
        row.update(payload["manuscript_metrics"])
        row.update(
            {
                "queueing_proxy_p995_ms": payload["tail_metrics"][
                    "queueing_proxy_p995_ms"
                ],
                "queueing_proxy_max_ms": payload["tail_metrics"][
                    "queueing_proxy_max_ms"
                ],
                "optimizer_mean_overflow_rb": payload["diagnostic_metrics"][
                    "optimizer_mean_overflow_rb"
                ],
                "trigger_ratio": payload["diagnostic_metrics"]["trigger_ratio"],
            }
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_evaluate(args) -> None:
    output = variant_output(args, args.variant)
    policy_path = output / "policy.pt"
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    policy = OMAPPPolicy.load(str(policy_path), seed=args.seed)
    expected = build_config(args.variant)
    if dataclasses.asdict(policy.config) != dataclasses.asdict(expected):
        raise RuntimeError("policy/config mismatch")

    timeline_all = load_pickle(args.test_path)
    timeline = temporal_slice(timeline_all, args.test_start, args.test_end)
    evaluation_dir = output / "exact_{:g}_{:g}_seed{}".format(
        args.test_start, args.test_end, args.seed
    )
    runs_dir = evaluation_dir / "runs"
    raw_dir = evaluation_dir / "raw"
    runs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for index, rate in enumerate(args.exact_rates, start=1):
        tag = "rate_{:g}Mbps_seed_{}".format(rate, args.seed)
        summary_path = runs_dir / (tag + ".json")
        raw_path = raw_dir / (tag + ".npz")
        if summary_path.is_file() and raw_path.is_file() and not args.force:
            payloads.append(json.loads(summary_path.read_text(encoding="utf-8")))
            print("[{}/{}] resume {}".format(index, len(args.exact_rates), tag))
            continue
        common_args = paper_args()
        common_args.data_rate = float(rate) * 1e6
        print(
            "\n[{}/{}] {} exact test at {:g} Mbps".format(
                index, len(args.exact_rates), args.variant, rate
            ),
            flush=True,
        )
        started = time.time()
        result = run_sim_o_mappo(
            common_args,
            MICRO_BS_LOCATIONS,
            timeline,
            policy,
            ra_func=RA_OTR_SINR,
            seed=args.seed,
            prt=False,
            rician_fading=True,
            optimizer_solver="milp",
        )
        raw = _raw_arrays(common_args, result)
        queue_ms = raw["normalized_backlog_proxy_ms_samples"]
        payload = {
            "variant": args.variant,
            "data_rate_mbps": float(rate),
            "seed": args.seed,
            "elapsed_s": time.time() - started,
            "manuscript_metrics": _manuscript_metrics(common_args, result),
            "diagnostic_metrics": summarize_exact(
                common_args, result, timeline
            ),
            "tail_metrics": {
                "queueing_proxy_p995_ms": float(np.percentile(queue_ms, 99.5)),
                "queueing_proxy_max_ms": float(queue_ms.max()),
            },
            "raw_npz": str(raw_path),
        }
        np.savez_compressed(raw_path, **raw)
        write_json(summary_path, payload)
        payloads.append(payload)
        rows = _curve_rows(payloads)
        write_json(evaluation_dir / "curve_results.json", {"curve_data": rows})
        _write_csv(evaluation_dir / "curve_data.csv", rows)
        print(
            "completed in {:.1f}s: {}".format(
                payload["elapsed_s"], payload["manuscript_metrics"]
            ),
            flush=True,
        )
    rows = _curve_rows(payloads)
    write_json(evaluation_dir / "curve_results.json", {"curve_data": rows})
    _write_csv(evaluation_dir / "curve_data.csv", rows)
    write_json(
        evaluation_dir / "protocol.json",
        {
            "variant": args.variant,
            "policy_path": str(policy_path),
            "test_path": args.test_path,
            "test_interval": [args.test_start, args.test_end],
            "rates": args.exact_rates,
            "seed": args.seed,
            "optimizer_solver": "milp",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("screen-exp1", "train", "evaluate"), required=True
    )
    parser.add_argument("--variant", choices=("exp1", "exp2"), default="exp1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cvar-weight", type=float, default=2.0)
    parser.add_argument("--screen-train-start", type=float, default=200.0)
    parser.add_argument("--screen-train-end", type=float, default=320.0)
    parser.add_argument("--screen-validation-start", type=float, default=320.1)
    parser.add_argument("--screen-validation-end", type=float, default=360.0)
    parser.add_argument("--screen-epochs", type=int, default=6)
    parser.add_argument("--screen-rates", default="1,7,13,19,25,31")
    parser.add_argument("--train-start", type=float, default=200.0)
    parser.add_argument("--train-end", type=float, default=800.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--training-rates",
        default=",".join("{:g}".format(x) for x in FINAL_TRAINING_SCHEDULE),
    )
    parser.add_argument("--test-start", type=float, default=800.0)
    parser.add_argument("--test-end", type=float, default=830.0)
    parser.add_argument(
        "--exact-rates",
        default=",".join("{:g}".format(x) for x in DEFAULT_EXACT_RATES),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.screen_rates = parse_number_list(args.screen_rates, float)
    args.training_rates = parse_number_list(args.training_rates, float)
    args.exact_rates = parse_number_list(args.exact_rates, float)
    if args.mode == "screen-exp1":
        screen_exp1(args)
    elif args.mode == "train":
        train_final(args)
    else:
        exact_evaluate(args)


if __name__ == "__main__":
    main()
