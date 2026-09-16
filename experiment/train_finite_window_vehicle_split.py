#!/usr/bin/env python3
"""Finite-window NN pretraining with a fixed vehicle-disjoint split.

Each next-frame target is paired with the available CSI history, capped at ten
frames. As in the submitted training code, a second sample is generated for
every original sample by selecting a source window with replacement and
retaining a random suffix of its available history.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.benchmark_nn_overhead import digest, json_write
from experiment.train_stateful_tbptt import load_data, noisy_features
from experiment.vehicle_split import trajectory_indices
from utils.NN_utils import BeamPredictionLSTMModel, BestGainPredictionLSTMModel
import numpy as np
import torch
import torch.nn.functional as F


def finite_window_samples(data, trajectories, history_length=10, augmentation_ratio=2.0, seed=20):
    if history_length < 1 or augmentation_ratio < 1:
        raise ValueError("Invalid history length or augmentation ratio")
    original_targets, original_lengths = [], []
    for trajectory in trajectories:
        start, stop = data["offsets"][trajectory:trajectory + 2]
        length = int(stop - start)
        original_targets.append(np.arange(start, stop, dtype=np.int64))
        original_lengths.append(
            np.minimum(np.arange(1, length + 1), history_length).astype(np.int16)
        )
    original_targets = np.concatenate(original_targets)
    original_lengths = np.concatenate(original_lengths)
    augmented_count = int(len(original_targets) * augmentation_ratio) - len(original_targets)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(original_targets), augmented_count)
    augmented_targets = original_targets[selected]
    augmented_lengths = (
        np.floor(rng.random(augmented_count) * original_lengths[selected]).astype(np.int16) + 1
    )
    return (np.concatenate((original_targets, augmented_targets)),
            np.concatenate((original_lengths, augmented_lengths)))


def batch_arrays(data, task, targets, lengths, history_length):
    steps = np.arange(history_length, dtype=np.int64)[None, :]
    indices = targets[:, None] - lengths[:, None] + 1 + steps
    valid = steps < lengths[:, None]
    indices = np.where(valid, indices, 0)
    clean = data["clean_csi"][indices].copy()
    clean[~valid] = 0
    labels = data[task][targets]
    return clean, labels


def run_epoch(model, task, data, targets, lengths, device, history_length, batch_size,
              optimizer, shuffle, seed, noise_power=1e-14):
    training = optimizer is not None
    model.train(training)
    order = np.arange(len(targets))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 100000)
    totals = {"loss_sum": 0.0, "targets": 0, "correct1": 0, "correct3": 0,
              "abs_error_db": 0.0, "squared_error_db": 0.0, "optimizer_steps": 0}
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_start in range(0, len(order), batch_size):
            selected = order[batch_start:batch_start + batch_size]
            batch_targets, batch_lengths = targets[selected], lengths[selected]
            clean, label_np = batch_arrays(
                data, task, batch_targets, batch_lengths, history_length
            )
            inputs = noisy_features(clean, device, noise_generator, noise_power=noise_power)
            labels = torch.as_tensor(label_np, device=device)
            lengths_cpu = torch.as_tensor(batch_lengths.astype(np.int64))
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(inputs, lengths=lengths_cpu)
            if task == "beam":
                labels = labels.long()
                loss = F.cross_entropy(output.reshape(-1, 256), labels.reshape(-1))
                totals["correct1"] += int((output.argmax(-1) == labels).sum())
                totals["correct3"] += int(
                    (output.topk(3, -1).indices == labels[..., None]).any(-1).sum()
                )
                count = labels.numel()
            else:
                normalized = labels / 20 + 7
                loss = F.mse_loss(output, normalized)
                error = 20 * (output.detach() - normalized)
                totals["abs_error_db"] += float(error.abs().sum())
                totals["squared_error_db"] += float((error ** 2).sum())
                count = labels.numel()
            if training:
                loss.backward()
                optimizer.step()
                totals["optimizer_steps"] += 1
            totals["loss_sum"] += float(loss.detach()) * count
            totals["targets"] += count
    result = {
        "loss": totals["loss_sum"] / totals["targets"],
        "targets": totals["targets"],
        "optimizer_steps": totals["optimizer_steps"],
    }
    if task == "beam":
        result.update(
            top1_accuracy_pct=100 * totals["correct1"] / totals["targets"],
            top3_accuracy_pct=100 * totals["correct3"] / totals["targets"],
        )
    else:
        result.update(
            mae_db=totals["abs_error_db"] / totals["targets"],
            rmse_db=(totals["squared_error_db"] / totals["targets"]) ** 0.5,
        )
    return result


def make_model(task, device):
    model = (BeamPredictionLSTMModel(128, 4, 256) if task == "beam"
             else BestGainPredictionLSTMModel(128, 4))
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("beam", "desired_gain", "interfering_gain"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--augmentation-ratio", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Smoke-test limiter per partition; zero uses every sample")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    data = load_data(args.data)
    train_trajectories, validation_trajectories, train_ids, validation_ids = trajectory_indices(
        data, args.split_file
    )
    train_targets, train_lengths = finite_window_samples(
        data, train_trajectories, args.history_length, args.augmentation_ratio, args.seed
    )
    validation_targets, validation_lengths = finite_window_samples(
        data, validation_trajectories, args.history_length, args.augmentation_ratio, args.seed + 1
    )
    if args.max_samples:
        train_targets, train_lengths = train_targets[:args.max_samples], train_lengths[:args.max_samples]
        validation_targets = validation_targets[:max(2, args.max_samples // 3)]
        validation_lengths = validation_lengths[:max(2, args.max_samples // 3)]
    model = make_model(args.task, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    started = time.monotonic()
    best_value = -np.inf if args.task == "beam" else np.inf
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model, args.task, data, train_targets, train_lengths, device,
            args.history_length, args.batch_size, optimizer, True, args.seed + epoch
        )
        validation = run_epoch(
            model, args.task, data, validation_targets, validation_lengths, device,
            args.history_length, args.batch_size, None, False, args.seed
        )
        value = (validation["top1_accuracy_pct"] if args.task == "beam"
                 else validation["mae_db"])
        improved = value > best_value if args.task == "beam" else value < best_value
        if improved:
            best_value, best_epoch = value, epoch
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                args.output / "best.pth",
            )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.monotonic() - started,
            **{f"train_{k}": v for k, v in train.items()},
            **{f"val_{k}": v for k, v in validation.items()},
        }
        history.append(row)
        json_write(args.output / "history.json", history)
        print(json.dumps(row), flush=True)
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        args.output / "last.pth",
    )
    with (args.output / "history.csv").open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(history)
    metadata = {
        "task": args.task,
        "data": str(args.data.resolve()), "data_sha256": digest(args.data),
        "split_file": str(args.split_file.resolve()), "split_sha256": digest(args.split_file),
        "split_protocol": "vehicle-disjoint split fixed before finite-window construction",
        "train_vehicles": len(train_ids), "validation_vehicles": len(validation_ids),
        "vehicle_overlap": 0,
        "train_trajectories": len(train_trajectories),
        "validation_trajectories": len(validation_trajectories),
        "train_window_samples": len(train_targets),
        "validation_window_samples": len(validation_targets),
        "history_lengths": [1, args.history_length],
        "augmentation_ratio": args.augmentation_ratio,
        "random_initialization": True,
        "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "fresh_training_pilot_noise_each_epoch": True,
        "validation_noise_seed": args.seed + 100000,
        "best_epoch": best_epoch, "best_validation_metric": float(best_value),
        "epochs_completed": len(history), "elapsed_seconds": time.monotonic() - started,
        "best_checkpoint_sha256": digest(args.output / "best.pth"),
        "command": sys.argv,
    }
    json_write(args.output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
