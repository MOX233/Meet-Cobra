#!/usr/bin/env python3
"""Frozen-policy gain-interface diagnostic, on validation data only.

Oracle substitutions are diagnostic controls, not eligible baselines or
checkpoint selectors. They do not modify saved predictors or prepared data.
"""
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.o_mappo_shared_frontend import (
    OUTPUT, LEGACY, read_pickle, temporal_slice, paper_args, frame_values,
    OMAPPPolicy, o_mappo_reward_presets, run_fluid_o_mappo_episode, write_json, torch,
)


def main():
    torch.set_num_threads(1)
    data = temporal_slice(read_pickle(OUTPUT / "train_prepared.pkl"), 700.1, 710)
    selected = json.loads((OUTPUT / "selected_policy.json").read_text())
    shared = OMAPPPolicy.load(selected["path"])
    old = OMAPPPolicy.load(str(LEGACY))
    old.config = dataclasses.replace(old.config, ho_interruption_ms=10)
    torch.set_num_threads(1)
    labels = {}
    for frame, records in data.items():
        ids = sorted(records, key=str)
        _, _, desired, interference = frame_values([records[v] for v in ids])
        labels[frame] = {v: dict(gain=desired[i], interference=interference[i]) for i, v in enumerate(ids)}
    oracle_current, oracle_next = {}, {}
    frames = list(data)
    for i, frame in enumerate(frames):
        following = labels[frames[min(i+1, len(frames)-1)]]
        oracle_current[frame] = {v: dict(r, shared_prediction=labels[frame][v]) for v,r in data[frame].items()}
        oracle_next[frame] = {v: dict(r, shared_prediction=following.get(v, labels[frame][v])) for v,r in data[frame].items()}
    variants = (("legacy_actor_and_information", old, data),
                ("shared_actor_predicted_gains", shared, data),
                ("shared_actor_current_true_micro_gains", shared, oracle_current),
                ("shared_actor_next_true_micro_gains", shared, oracle_next))
    reward = o_mappo_reward_presets()["qos_energy020_load1"]
    rows = []
    for name, policy, timeline in variants:
        for rate in (7, 19, 35):
            started = time.monotonic()
            result = run_fluid_o_mappo_episode(paper_args(rate*1e6), timeline, policy, reward,
                                              rate, seed=2026, learn=False)
            rows.append(dict(variant=name, rate_mbps=rate, metrics=result, elapsed_s=time.monotonic()-started))
            print(name, rate, result["average_system_power_w"], result["queue_violation_percent"], flush=True)
    write_json(OUTPUT / "validation_information_ablation.json", dict(
        scope="validation 700.1--710 s, fluid approximation, frozen actor; NOT exact test or model selection",
        selected_checkpoint=selected, substitutions="micro desired and interfering gain only; macro location unchanged",
        results=rows))


if __name__ == "__main__":
    main()
