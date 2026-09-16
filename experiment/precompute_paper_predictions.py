#!/usr/bin/env python3
"""Precompute exact single-sample LSTM predictions for paper simulations.

The legacy simulator invokes each predictor once per vehicle.  These outputs
depend only on the fixed CSI trace and trained checkpoints, not on offered load
or evaluation seed.  Persisting them once preserves the single-sample inference
path while avoiding repeated GPU calls in every method/load/seed run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.paper_methods_multiseed import load_common


DEFAULT_OUTPUT = Path(
    "experiment/results_multiseed/prediction_cache_800_830_k5.pkl"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-end", type=float, default=830.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main(args):
    if args.output.exists() and not args.force:
        print("cache already exists: {}".format(args.output))
        return
    (
        sim_args,
        _,
        timeline,
        _,
        beam_model,
        gain_model,
        interference_model,
        _,
    ) = load_common(1, args.output.parent, 800.0, args.test_end, args.gpu)
    predictions = {}
    record_count = 0
    started = time.time()
    with torch.inference_mode():
        for frame, frame_records in tqdm.tqdm(
            timeline.items(), desc="Precomputing exact predictions"
        ):
            frame_predictions = {}
            for vehicle, record in frame_records.items():
                value = record["CSI_preprocessed"].astype(np.float32)[None, ...]
                frame_predictions[vehicle] = {
                    "gain": gain_model.predict(value, sim_args.device)[0],
                    "beam": beam_model.predict(
                        value, sim_args.device, K=args.top_k
                    )[0],
                    "interference": interference_model.predict(
                        value, sim_args.device
                    )[0],
                }
                record_count += 1
            predictions[frame] = frame_predictions
    payload = {
        "metadata": {
            "test_interval_s": [800.0, args.test_end],
            "top_k": args.top_k,
            "device": str(sim_args.device),
            "inference_mode": "legacy single-vehicle calls",
            "num_frames": len(timeline),
            "num_vehicle_frame_records": record_count,
            "elapsed_s": time.time() - started,
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, args.output)
    print("saved {} records to {}".format(record_count, args.output))
    print(payload["metadata"])


if __name__ == "__main__":
    main(parse_args())
