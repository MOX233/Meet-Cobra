#!/usr/bin/env python3
"""Launch the remaining disjoint multi-seed shards as detached CPU jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "experiment/results_multiseed/logs_host"
MANIFEST = ROOT / "experiment/results_multiseed/detached_jobs_manifest_host.json"
COMMON_RATES_MID = "13,15,17,19,21,23"
COMMON_RATES_HIGH = "25,27,29,31,33,35"
SEEDS = (1, 2, 3, 4, 5)


def original_command(method, seed, rates):
    return [
        sys.executable,
        "experiment/paper_methods_multiseed.py",
        "--method",
        method,
        "--seed",
        str(seed),
        "--rates",
        rates,
        "--test-end",
        "830",
        "--output",
        "experiment/results_multiseed/original_methods",
        "--prediction-cache",
        "experiment/results_multiseed/prediction_cache_800_830_k5.pkl",
        "--torch-threads",
        "1",
        "--no-batch-prediction",
        "--measured-gain-gamma",
        "0.9",
        "--force",
    ]


def o_mappo_command(seed):
    return [
        sys.executable,
        "experiment/run_joint_baseline_full_grid.py",
        "--method",
        "o_mappo",
        "--seed",
        str(seed),
        "--rates",
        COMMON_RATES_HIGH,
        "--output-dir",
        "experiment/results_multiseed/baselines/seed_{}".format(seed),
    ]


def mts_command(seed):
    return [
        sys.executable,
        "experiment/mts_gs_hbf_experiment.py",
        "--mode",
        "evaluate",
        "--candidate",
        "pressure_early",
        "--seed",
        str(seed),
        "--test-start",
        "800",
        "--test-end",
        "830",
        "--eval-rates",
        COMMON_RATES_HIGH,
        "--output",
        "experiment/results_multiseed/mts_gs_hbf",
    ]


def jobs():
    output = []
    # Mid-load shards not launched by the interactive first wave.
    output.append(("proposed_seed1_mid", original_command("proposed", 1, COMMON_RATES_MID)))
    for method in ("oracle_mc", "oracle_cr_lb", "wo_otr_ra"):
        for seed in SEEDS:
            output.append(
                (
                    "{}_seed{}_mid".format(method, seed),
                    original_command(method, seed, COMMON_RATES_MID),
                )
            )
    for seed in (3, 4, 5):
        output.append(
            (
                "wo_pet_bf_seed{}_mid".format(seed),
                original_command("wo_pet_bf", seed, COMMON_RATES_MID),
            )
        )

    # High-load shards not launched by the interactive first wave.
    for method, seeds in (
        ("oracle_mc", SEEDS),
        ("oracle_cr_lb", SEEDS),
        ("wo_gap_ho", (3, 4, 5)),
        ("wo_pet_bf", SEEDS),
        ("wo_otr_ra", SEEDS),
    ):
        for seed in seeds:
            output.append(
                (
                    "{}_seed{}_high".format(method, seed),
                    original_command(method, seed, COMMON_RATES_HIGH),
                )
            )

    # High-load shards for the two new baselines.
    for seed in SEEDS:
        output.append(("o_mappo_seed{}_high".format(seed), o_mappo_command(seed)))
        output.append(("mts_seed{}_high".format(seed), mts_command(seed)))
    if len(output) != 52:
        raise RuntimeError("expected 52 detached jobs, got {}".format(len(output)))
    return output


def main():
    if MANIFEST.exists():
        raise FileExistsError(
            "manifest already exists; inspect running jobs before relaunching: {}".format(
                MANIFEST
            )
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/meet_cobra_matplotlib",
        }
    )
    records = []
    for name, command in jobs():
        log_path = LOG_DIR / (name + ".log")
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        records.append(
            {
                "name": name,
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
            }
        )
    payload = {
        "launched_at_unix_s": time.time(),
        "num_jobs": len(records),
        "jobs": records,
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    print("launched {} detached jobs".format(len(records)))
    print("manifest: {}".format(MANIFEST))


if __name__ == "__main__":
    main()
