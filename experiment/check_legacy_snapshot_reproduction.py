#!/usr/bin/env python3
"""Compare the 2025-12-21 simulator snapshot with saved 2025-12-22 records."""

import collections
from pathlib import Path
import sys

import numpy as np
import torch


LEGACY_ROOT = Path("/tmp/meet_cobra_08b_snapshot")
sys.path.insert(0, str(LEGACY_ROOT))

from utils.alg_utils import HO_EE_GAP_APX_SINR_conservative_adaptive, RA_OTR_SINR
from utils.mox_utils import setup_seed
from utils.sim_utils import get_default_sim_params, run_sim_withUMa


def flatten(frame):
    return np.concatenate(
        [np.asarray(frame[key]).reshape(-1) for key in sorted(frame)]
    )


def main():
    original_load = torch.load

    def portable_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cuda:0")
        return original_load(*args, **kwargs)

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], "--seed", "1"]
        torch.load = portable_load
        (
            args,
            locations,
            timeline,
            position_model,
            beam_model,
            gain_model,
            interference_model,
            _,
        ) = get_default_sim_params(
            "/tmp/meet_cobra_legacy_check", gpu=0, lbd=1, cut_ratio=0.5 / 150.0
        )
    finally:
        torch.load = original_load
        sys.argv = saved_argv
    args.data_rate = 1e6
    setup_seed(1)
    (
        energy,
        handover,
        commands,
        violation,
        average_queue,
        pilots,
        rb_allocated,
        queues,
    ) = run_sim_withUMa(
        args,
        locations,
        timeline,
        position_model,
        beampred_model=beam_model,
        gainpred_model=gain_model,
        inferpred_model=interference_model,
        RA_func=RA_OTR_SINR,
        HO_func=HO_EE_GAP_APX_SINR_conservative_adaptive,
        prt=False,
        save_pilot=True,
        No_BF=False,
        K_BF=5,
    )

    historical_path = Path(
        "experiment/results_paper_exp1/"
        "lbd1.00_800_830.0_2025-12-22 06:26:16/sim_result_dict.npy"
    )
    historical = np.load(historical_path, allow_pickle=True).item()["Proposed"]
    historical_queues = historical["queuelen_4eachVeh_record_list"][0]
    queue_checks = []
    for frame in queues:
        new = flatten(queues[frame])
        old = flatten(historical_queues[frame])
        queue_checks.append(
            {
                "frame": frame,
                "same_shape": new.shape == old.shape,
                "exact": np.array_equal(new, old),
                "max_abs_error": float(np.max(np.abs(new - old))),
            }
        )

    counts = np.zeros((max(len(commands) - 1, 0), len(locations) + 1))
    for row, frame in enumerate(list(commands)[1:]):
        for bs in commands[frame].values():
            counts[row, int(bs)] += 1
    historical_counts = np.asarray(historical["carnum_under_BS_list"][0])[
        : len(counts)
    ]
    print("frames", len(energy))
    print("queue_checks", queue_checks)
    print(
        "association_exact",
        np.array_equal(counts, historical_counts),
        "new",
        counts.tolist(),
        "old",
        historical_counts.tolist(),
    )
    print(
        "short_metrics",
        {
            "power_w": float(np.mean(energy) / 0.1),
            "violation_percent": float(100.0 * np.mean(violation)),
            "average_queue_ms": float(np.mean(average_queue) / 1e6 * 1000.0),
            "pilots": float(np.mean(pilots)),
            "handover": np.asarray(handover).tolist(),
            "rb_sum": np.asarray(rb_allocated).sum(axis=0).tolist(),
        },
    )


if __name__ == "__main__":
    main()
