#!/usr/bin/env python3
"""Create and validate one vehicle-disjoint split shared by both NN stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.benchmark_nn_overhead import digest, json_write
import numpy as np


def create_vehicle_split(vehicle_ids, train_fraction=0.7, seed=20):
    unique = np.unique(np.asarray(vehicle_ids).astype(str))
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    boundary = int(len(shuffled) * train_fraction)
    train, validation = shuffled[:boundary], shuffled[boundary:]
    assert len(train) and len(validation)
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(unique)
    return train, validation


def load_vehicle_split(path, available_vehicle_ids=None):
    with np.load(path) as archive:
        train = np.asarray(archive["train_vehicle_ids"]).astype(str)
        validation = np.asarray(archive["validation_vehicle_ids"]).astype(str)
    if len(np.unique(train)) != len(train) or len(np.unique(validation)) != len(validation):
        raise ValueError(f"Duplicate vehicle IDs in {path}")
    if set(train) & set(validation):
        raise ValueError(f"Training and validation vehicles overlap in {path}")
    if available_vehicle_ids is not None:
        available = set(np.asarray(available_vehicle_ids).astype(str))
        assigned = set(train) | set(validation)
        if assigned != available:
            raise ValueError(
                f"Split/data vehicle mismatch: {len(assigned - available)} unknown and "
                f"{len(available - assigned)} unassigned IDs"
            )
    return train, validation


def trajectory_indices(data, split_file):
    vehicles = np.asarray(data["vehicle_ids"]).astype(str)
    train_ids, validation_ids = load_vehicle_split(split_file, vehicles)
    train_set, validation_set = set(train_ids), set(validation_ids)
    train = np.flatnonzero(np.asarray([v in train_set for v in vehicles]))
    validation = np.flatnonzero(np.asarray([v in validation_set for v in vehicles]))
    if not len(train) or not len(validation):
        raise ValueError("Both partitions must contain at least one trajectory")
    if set(vehicles[train]) & set(vehicles[validation]):
        raise AssertionError("Vehicle-level split was not preserved")
    return train, validation, train_ids, validation_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must lie strictly between zero and one")
    with np.load(args.data) as archive:
        vehicle_ids = np.asarray(archive["vehicle_ids"]).astype(str)
    train, validation = create_vehicle_split(vehicle_ids, args.train_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, train_vehicle_ids=train, validation_vehicle_ids=validation)
    metadata = {
        "data": str(args.data.resolve()),
        "data_sha256": digest(args.data),
        "split": str(args.output.resolve()),
        "split_sha256": digest(args.output),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "usable_vehicles": int(len(np.unique(vehicle_ids))),
        "train_vehicles": int(len(train)),
        "validation_vehicles": int(len(validation)),
        "vehicle_overlap": 0,
        "protocol": "Split unique vehicle IDs before constructing finite-window or stateful samples.",
        "command": sys.argv,
    }
    json_write(args.output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
