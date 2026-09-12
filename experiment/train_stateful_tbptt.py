#!/usr/bin/env python3
"""Vehicle-stream stateful training with truncated BPTT.

State values cross 10-frame chunks; their autograd history does not. Each
vehicle trajectory starts from zero state. Existing paper weights are used as
the default initialization, with no architecture change.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.benchmark_nn_overhead import CHECKPOINTS, CHECKPOINT_DIR, digest, json_write
from utils.NN_utils import BeamPredictionLSTMModel, BestGainPredictionLSTMModel
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load_data(path):
    archive = np.load(path)
    return {k: np.asarray(archive[k]) for k in archive.files}


def split_vehicles(data, train_fraction=0.7, seed=20):
    vehicles = np.asarray(data["vehicle_ids"])
    unique = np.unique(vehicles)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    train_vehicles = set(shuffled[:int(len(shuffled) * train_fraction)])
    train = np.asarray([i for i, value in enumerate(vehicles) if value in train_vehicles])
    val = np.asarray([i for i, value in enumerate(vehicles) if value not in train_vehicles])
    assert len(train) and len(val) and not set(vehicles[train]) & set(vehicles[val])
    return train, val


def make_model(task, device, initialization):
    model = BeamPredictionLSTMModel(128, 4, 256) if task == "beam" else BestGainPredictionLSTMModel(128, 4)
    if initialization == "paper":
        key = {"beam": "beam", "desired_gain": "desired_gain", "interfering_gain": "interfering_gain"}[task]
        model.load_state_dict(torch.load(CHECKPOINT_DIR / CHECKPOINTS[key], map_location="cpu", weights_only=True))
    return model.to(device)


def forward_valid(model, lstm_output, mask, task):
    features = model.shared_layers(lstm_output[mask])
    heads = [head(features) for head in model.output_heads]
    return torch.stack(heads, dim=-2) if task == "beam" else torch.cat(heads, dim=-1)


def noisy_features(clean, device, generator, noise_power=1e-14):
    value = torch.as_tensor(clean, device=device)
    sigma = (noise_power / 2) ** 0.5
    real = torch.randn(value.shape, device=device, generator=generator)
    imag = torch.randn(value.shape, device=device, generator=generator)
    value = value + sigma * torch.complex(real, imag)
    return torch.cat((20 * torch.log10(value.abs() + 1e-9) / 20 + 7, torch.angle(value)), dim=-1).float()


def trajectory_groups(indices, offsets, batch_size, shuffle, rng):
    values = np.asarray(indices).copy()
    if shuffle:
        rng.shuffle(values)
    return [values[i:i + batch_size] for i in range(0, len(values), batch_size)]


def run_epoch(model, task, data, indices, device, chunk_length, batch_size, optimizer,
              shuffle, seed, gradient_clip, freeze_batch_norm=True):
    training = optimizer is not None
    model.train(training)
    if training and freeze_batch_norm:
        # Preserve the paper checkpoint's population statistics. Affine BN
        # parameters remain trainable, while short trajectory tails cannot
        # dominate running means/variances.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    rng = np.random.default_rng(seed)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 100000)
    totals = {"loss_sum": 0.0, "count": 0, "correct1": 0, "correct3": 0, "abs_error_db": 0.0,
              "squared_error_db": 0.0, "optimizer_steps": 0, "skipped_singleton_targets": 0}
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for group in trajectory_groups(indices, data["offsets"], batch_size, shuffle, rng):
            starts = data["offsets"][group]
            lengths = data["offsets"][group + 1] - starts
            batch = len(group)
            state = None
            for time_start in range(0, int(lengths.max()), chunk_length):
                valid_lengths = np.clip(lengths - time_start, 0, chunk_length)
                mask_np = np.arange(chunk_length)[None, :] < valid_lengths[:, None]
                valid_count = int(mask_np.sum())
                if valid_count == 0:
                    continue
                clean = np.zeros((batch, chunk_length, 64), dtype=np.complex64)
                if task == "beam":
                    target = np.zeros((batch, chunk_length, 4), dtype=np.int64)
                else:
                    target = np.zeros((batch, chunk_length, 4), dtype=np.float32)
                key = task
                for row, (start, count) in enumerate(zip(starts, valid_lengths)):
                    if count:
                        source = slice(start + time_start, start + time_start + count)
                        clean[row, :count] = data["clean_csi"][source]
                        target[row, :count] = data[key][source]
                x = noisy_features(clean, device, noise_generator)
                mask = torch.as_tensor(mask_np, device=device)
                labels = torch.as_tensor(target, device=device)[mask]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                lstm_output, state = model.lstm_layers(x, state)
                # The original prediction heads contain BatchNorm. A rare final
                # tail with one valid target cannot define training BN statistics.
                # Its recurrent state is advanced, but the singleton loss is omitted.
                if training and valid_count == 1:
                    totals["skipped_singleton_targets"] += 1
                    state = tuple(value.detach() for value in state)
                    continue
                output = forward_valid(model, lstm_output, mask, task)
                if task == "beam":
                    loss = F.cross_entropy(output.reshape(-1, 256), labels.reshape(-1))
                    totals["correct1"] += int((output.argmax(-1) == labels).sum())
                    totals["correct3"] += int((output.topk(3, -1).indices == labels[..., None]).any(-1).sum())
                    count = labels.numel()
                else:
                    normalized = labels / 20 + 7
                    loss = F.mse_loss(output, normalized)
                    error = 20 * (output.detach() - normalized)
                    totals["abs_error_db"] += float(error.abs().sum())
                    totals["squared_error_db"] += float((error ** 2).sum())
                    count = labels.numel()
                if training and valid_count >= 2:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    totals["optimizer_steps"] += 1
                state = tuple(value.detach() for value in state)  # TBPTT boundary: keep values, cut graph.
                totals["loss_sum"] += float(loss.detach()) * count
                totals["count"] += count
    result = {"loss": totals["loss_sum"] / totals["count"], "targets": totals["count"],
              "optimizer_steps": totals["optimizer_steps"],
              "skipped_singleton_targets": totals["skipped_singleton_targets"]}
    if task == "beam":
        result.update(top1_accuracy_pct=100 * totals["correct1"] / totals["count"],
                      top3_accuracy_pct=100 * totals["correct3"] / totals["count"])
    else:
        result.update(mae_db=totals["abs_error_db"] / totals["count"],
                      rmse_db=(totals["squared_error_db"] / totals["count"]) ** .5)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("beam", "desired_gain", "interfering_gain"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--initialization", choices=("paper", "scratch"), default="paper")
    parser.add_argument("--batch-norm-mode", choices=("frozen", "running"), default="frozen")
    parser.add_argument("--max-trajectories", type=int, default=0, help="Smoke-test limiter after split; 0 means all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    data = load_data(args.data)
    train_indices, val_indices = split_vehicles(data, args.train_fraction, args.seed)
    if args.max_trajectories:
        train_indices = train_indices[:args.max_trajectories]
        val_indices = val_indices[:max(2, args.max_trajectories // 3)]
    model = make_model(args.task, device, args.initialization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max" if args.task == "beam" else "min",
                                                           factor=.5, patience=2, min_lr=1e-6)
    started = time.monotonic()
    freeze_bn = args.batch_norm_mode == "frozen"
    initial_val = run_epoch(model, args.task, data, val_indices, device, args.chunk_length, args.batch_size,
                            None, False, args.seed, args.gradient_clip, freeze_bn)
    best_value = initial_val["top1_accuracy_pct"] if args.task == "beam" else initial_val["mae_db"]
    best_epoch, stale = 0, 0
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / "best.pth")
    history = [{"epoch": 0, "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - started,
                **{f"val_{k}": v for k, v in initial_val.items()}}]
    print(json.dumps(history[0]), flush=True)
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, args.task, data, train_indices, device, args.chunk_length, args.batch_size,
                          optimizer, True, args.seed + epoch, args.gradient_clip, freeze_bn)
        val = run_epoch(model, args.task, data, val_indices, device, args.chunk_length, args.batch_size,
                        None, False, args.seed, args.gradient_clip, freeze_bn)
        value = val["top1_accuracy_pct"] if args.task == "beam" else val["mae_db"]
        improved = value > best_value if args.task == "beam" else value < best_value
        if improved:
            best_value, best_epoch, stale = value, epoch, 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / "best.pth")
        else:
            stale += 1
        scheduler.step(value)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
               "elapsed_seconds": time.monotonic() - started,
               **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        json_write(args.output / "history.json", history)
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            print(f"Early stopping: best epoch {best_epoch}", flush=True)
            break
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.output / "last.pth")
    with (args.output / "history.csv").open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(history)
    metadata = {
        "task": args.task, "data": str(args.data.resolve()), "data_sha256": digest(args.data),
        "initialization": args.initialization,
        "initial_checkpoint": str((CHECKPOINT_DIR / CHECKPOINTS[{"beam":"beam","desired_gain":"desired_gain","interfering_gain":"interfering_gain"}[args.task]]).resolve()) if args.initialization == "paper" else None,
        "train_fraction_by_vehicle": args.train_fraction, "train_trajectories": len(train_indices),
        "validation_trajectories": len(val_indices), "split_seed": args.seed,
        "chunk_length": args.chunk_length, "batch_size": args.batch_size,
        "optimizer": "AdamW", "initial_learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "gradient_clip": args.gradient_clip,
        "batch_norm_mode": args.batch_norm_mode,
        "fresh_training_pilot_noise_each_epoch": True, "validation_noise_seed": args.seed + 100000,
        "best_epoch": best_epoch, "best_validation_metric": float(best_value),
        "epochs_completed": len(history) - 1, "elapsed_seconds": time.monotonic() - started,
        "best_checkpoint_sha256": digest(args.output / "best.pth"),
        "protocol": "vehicle-stream stateful forward pass; h/c detached, not zeroed, every 10 frames; zero only at trajectory start",
        "command": sys.argv,
    }
    json_write(args.output / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
