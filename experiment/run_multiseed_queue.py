#!/usr/bin/env python3
"""Run every missing five-seed paper point with bounded CPU concurrency.

Each child evaluates exactly one method/rate/seed point.  Completed point files
are never overwritten, so the queue can be stopped and resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment/results_multiseed"
RATES = tuple(range(1, 36, 2))
SEEDS = tuple(range(1, 6))
ORIGINAL_METHODS = (
    "proposed",
    "wo_pet_bf",
    "reactive_obra",
    "wo_gap_ho",
    "wo_otr_ra",
    "oracle_mc",
    "oracle_cr_lb",
)


def rate_text(rate: float) -> str:
    return "{:g}".format(rate)


def original_summary(method: str, seed: int, rate: float) -> Path:
    return (
        RESULTS
        / "original_methods"
        / method
        / "seed_{}".format(seed)
        / "runs"
        / "rate_{}Mbps_seed_{}.json".format(rate_text(rate), seed)
    )


def o_mappo_summary(seed: int, rate: float) -> Path:
    return (
        RESULTS
        / "baselines"
        / "seed_{}".format(seed)
        / "o_mappo_adapted/runs"
        / "rate_{}Mbps_seed_{}.json".format(rate_text(rate), seed)
    )


def mts_summary(seed: int, rate: float) -> Path:
    return (
        RESULTS
        / "mts_gs_hbf/pressure_early/exact_800_830_seed{}".format(seed)
        / "runs"
        / "rate_{}Mbps_seed_{}.json".format(rate_text(rate), seed)
    )


def make_task(name: str, command, expected: Path):
    return {"name": name, "command": list(command), "expected": expected, "attempt": 0}


def collect_tasks():
    tasks = []
    python = sys.executable
    cache = RESULTS / "prediction_cache_800_830_k5.pkl"
    for method in ORIGINAL_METHODS:
        for seed in SEEDS:
            for rate in RATES:
                expected = original_summary(method, seed, rate)
                if expected.is_file():
                    continue
                name = "{}_seed{}_rate{}".format(method, seed, rate_text(rate))
                tasks.append(
                    make_task(
                        name,
                        (
                            python,
                            "experiment/paper_methods_multiseed.py",
                            "--method",
                            method,
                            "--seed",
                            str(seed),
                            "--rates",
                            str(rate),
                            "--test-end",
                            "830",
                            "--output",
                            "experiment/results_multiseed/original_methods",
                            "--prediction-cache",
                            str(cache.relative_to(ROOT)),
                            "--torch-threads",
                            "1",
                            "--no-batch-prediction",
                            "--measured-gain-gamma",
                            "0.9",
                        ),
                        expected,
                    )
                )
    for seed in SEEDS:
        for rate in RATES:
            expected = mts_summary(seed, rate)
            if not expected.is_file():
                tasks.append(
                    make_task(
                        "mts_seed{}_rate{}".format(seed, rate_text(rate)),
                        (
                            python,
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
                            str(rate),
                            "--output",
                            "experiment/results_multiseed/mts_gs_hbf",
                        ),
                        expected,
                    )
                )
            expected = o_mappo_summary(seed, rate)
            if not expected.is_file():
                tasks.append(
                    make_task(
                        "o_mappo_seed{}_rate{}".format(seed, rate_text(rate)),
                        (
                            python,
                            "experiment/run_joint_baseline_full_grid.py",
                            "--method",
                            "o_mappo",
                            "--seed",
                            str(seed),
                            "--rates",
                            str(rate),
                            "--output-dir",
                            "experiment/results_multiseed/baselines/seed_{}".format(seed),
                        ),
                        expected,
                    )
                )
    return tasks


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".{}.tmp".format(os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parallel", type=int, default=80)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_parallel < 1:
        raise ValueError("max-parallel must be positive")

    pending = collect_tasks()
    initial_total = len(pending)
    log_dir = RESULTS / "logs_queue"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS / "queue_status.json"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": "/tmp/meet_cobra_mpl",
            "PYTHONUNBUFFERED": "1",
        }
    )
    Path(environment["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    running = {}
    completed = []
    failures = []
    started = time.time()

    def write_status(state="running"):
        atomic_json(
            status_path,
            {
                "state": state,
                "manager_pid": os.getpid(),
                "started_unix_s": started,
                "updated_unix_s": time.time(),
                "initial_missing_points": initial_total,
                "completed_this_run": len(completed),
                "pending": len(pending),
                "running": len(running),
                "failed": len(failures),
                "running_tasks": [
                    {"name": item["task"]["name"], "pid": process.pid}
                    for process, item in running.items()
                ],
                "failures": failures,
            },
        )

    while pending or running:
        while pending and len(running) < args.max_parallel:
            task = pending.pop(0)
            if task["expected"].is_file():
                completed.append(task["name"])
                continue
            task["attempt"] += 1
            log_path = log_dir / "{}_attempt{}.log".format(task["name"], task["attempt"])
            log_handle = log_path.open("ab", buffering=0)
            process = subprocess.Popen(
                task["command"],
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[process] = {"task": task, "log_handle": log_handle, "log": str(log_path)}
        write_status()
        if not running:
            continue
        time.sleep(args.poll_seconds)
        for process, item in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            item["log_handle"].close()
            task = item["task"]
            del running[process]
            if return_code == 0 and task["expected"].is_file():
                completed.append(task["name"])
            elif task["attempt"] < args.max_attempts:
                pending.append(task)
            else:
                failures.append(
                    {
                        "name": task["name"],
                        "return_code": return_code,
                        "expected": str(task["expected"]),
                        "log": item["log"],
                    }
                )

    write_status("failed" if failures else "complete")
    if failures:
        raise SystemExit("{} point(s) failed".format(len(failures)))


if __name__ == "__main__":
    main()
