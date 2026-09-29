#!/usr/bin/env python3
"""Regenerate Fig. 4 from finite-window training and stateful fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="meet-cobra-mpl-"))
import matplotlib.pyplot as plt
import numpy as np

FONT_FAMILY = "Times New Roman"
AXIS_LABEL_SIZE = 8.5
IN_FIGURE_SIZE = 7.0
AXIS_LABEL_FONT = {
    "family": FONT_FAMILY,
    "size": AXIS_LABEL_SIZE,
    "weight": "normal",
    "style": "normal",
}
IN_FIGURE_FONT = {
    "family": FONT_FAMILY,
    "size": IN_FIGURE_SIZE,
    "weight": "normal",
    "style": "normal",
}

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": IN_FIGURE_SIZE,
        "font.weight": "normal",
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.labelweight": "normal",
        "xtick.labelsize": IN_FIGURE_SIZE,
        "ytick.labelsize": IN_FIGURE_SIZE,
        "legend.fontsize": IN_FIGURE_SIZE,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "experiment/results/stateful_tbptt_unified_split_20260913"
)
DEFAULT_STAGE1_RESULTS = DEFAULT_ROOT / "stage1_finite_window"
DEFAULT_STAGE2_RESULTS = DEFAULT_ROOT / "stage2_stateful_tbptt"
DEFAULT_FIGURES = ROOT / "latexCodes/figures"
PHASE_BOUNDARY = 100
EPOCH_RIGHT_LIMIT = 203  # Leave room for a best-checkpoint star at epoch 200.
TRAIN_COLOR = "#0072B2"
VALIDATION_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-results", type=Path, default=DEFAULT_STAGE1_RESULTS)
    parser.add_argument("--stage2-results", type=Path, default=DEFAULT_STAGE2_RESULTS)
    parser.add_argument("--interfering-stage1-results", type=Path,
                        help="Override only the interfering-gain stage-I results root")
    parser.add_argument("--interfering-stage2-results", type=Path,
                        help="Override only the interfering-gain stage-II results root")
    parser.add_argument("--summary", type=Path,
                        help="Write provenance and metrics here instead of the stage-II root")
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    overrides = (args.interfering_stage1_results, args.interfering_stage2_results)
    if any(overrides) and (not all(overrides) or args.summary is None):
        parser.error("Interfering-gain overrides require both stages and an explicit --summary")
    return args


def read_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    rows = [row for row in rows if 1 <= int(row["epoch"]) <= 100]
    if [int(row["epoch"]) for row in rows] != list(range(1, 101)):
        raise ValueError(f"Expected epochs 1--100 in {path}")
    return rows


def values(rows: list[dict], key: str) -> np.ndarray:
    result = np.asarray([row[key] for row in rows], dtype=np.float64)
    if result.shape != (100,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {key} values")
    return result


def best_validation_reference(ax, value: float, best_epoch: int, color: str) -> None:
    ax.axhline(value, color=color, linestyle=":", linewidth=1.0, alpha=0.9, zorder=1)
    ax.plot(
        PHASE_BOUNDARY + best_epoch,
        value,
        marker="*",
        markersize=6.2,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.45,
        linestyle="none",
        zorder=5,
    )


def plot_two_phases(
    ax,
    first: np.ndarray,
    second: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str,
) -> None:
    first_epochs = np.arange(1, PHASE_BOUNDARY + 1)
    second_epochs = np.arange(PHASE_BOUNDARY + 1, 2 * PHASE_BOUNDARY + 1)
    ax.plot(first_epochs, first, label=label, color=color, linestyle=linestyle, zorder=3)
    ax.plot(second_epochs, second, color=color, linestyle=linestyle, zorder=3)


def style_two_phase_axes(ax) -> None:
    ax.axvspan(PHASE_BOUNDARY + 0.5, EPOCH_RIGHT_LIMIT, color="0.94", zorder=0)
    ax.axvline(PHASE_BOUNDARY + 0.5, color="0.35", linestyle=":", linewidth=1.0, zorder=2)
    ax.text(0.25, 1.015, "Stage I: finite-window", transform=ax.transAxes,
            ha="center", va="bottom", color="0.25", fontdict=IN_FIGURE_FONT)
    ax.text(0.75, 1.015, "Stage II: stateful fine-tuning", transform=ax.transAxes,
            ha="center", va="bottom", color="0.25", fontdict=IN_FIGURE_FONT)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_xlim(1, EPOCH_RIGHT_LIMIT)
    ax.grid(axis="y", color="0.87", linewidth=0.5, zorder=0)
    ax.tick_params(direction="in", top=False, right=False)


def metric_box(ax, text: str, *, position: tuple[float, float]) -> None:
    ax.text(
        *position,
        text,
        transform=ax.transAxes,
        ha="right",
        va="center",
        linespacing=1.25,
        fontdict=IN_FIGURE_FONT,
        color="0.15",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
              "edgecolor": "0.75", "linewidth": 0.6, "alpha": 0.94},
        zorder=6,
    )


def plot_beam(first: list[dict], rows: list[dict], output: Path) -> dict:
    first_train_top1 = values(first, "train_top1_accuracy_pct")
    first_val_top1 = values(first, "val_top1_accuracy_pct")
    first_train_top3 = values(first, "train_top3_accuracy_pct")
    first_val_top3 = values(first, "val_top3_accuracy_pct")
    train_top1 = values(rows, "train_top1_accuracy_pct")
    val_top1 = values(rows, "val_top1_accuracy_pct")
    train_top3 = values(rows, "train_top3_accuracy_pct")
    val_top3 = values(rows, "val_top3_accuracy_pct")

    fig, ax = plt.subplots(figsize=(3.45, 2.55), dpi=240)
    style_two_phase_axes(ax)
    plot_two_phases(ax, first_train_top1, train_top1, label="Train, Top-1",
                    color=TRAIN_COLOR, linestyle="-")
    plot_two_phases(ax, first_val_top1, val_top1, label="Validation, Top-1",
                    color=VALIDATION_COLOR, linestyle="-")
    plot_two_phases(ax, first_train_top3, train_top3, label="Train, Top-3",
                    color=TRAIN_COLOR, linestyle="--")
    plot_two_phases(ax, first_val_top3, val_top3, label="Validation, Top-3",
                    color=VALIDATION_COLOR, linestyle="--")

    val_top1_epoch = int(val_top1.argmax() + 1)
    val_top3_epoch = int(val_top3.argmax() + 1)
    best_validation_reference(ax, float(val_top1.max()), val_top1_epoch, VALIDATION_COLOR)
    best_validation_reference(ax, float(val_top3.max()), val_top3_epoch, VALIDATION_COLOR)
    metric_box(
        ax,
        "Best validation accuracy\n"
        f"Top-1: {val_top1.max():.2f}%   Top-3: {val_top3.max():.2f}%",
        position=(0.975, 0.46),
    )

    ax.legend(loc="lower center", bbox_to_anchor=(0.50, 0.035), ncol=2,
              frameon=True, framealpha=0.94, borderpad=0.35,
              handlelength=2.2, columnspacing=1.0, labelspacing=0.25,
              prop=IN_FIGURE_FONT)
    ax.set_xlabel("Epoch", fontdict=AXIS_LABEL_FONT)
    ax.set_ylabel("Accuracy (%)", fontdict=AXIS_LABEL_FONT)
    ax.set_ylim(72.5, 100.15)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return {
        "train_top1_max_pct": float(train_top1.max()),
        "train_top1_best_epoch": int(train_top1.argmax() + 1),
        "val_top1_max_pct": float(val_top1.max()),
        "val_top1_best_epoch": int(val_top1.argmax() + 1),
        "train_top3_max_pct": float(train_top3.max()),
        "train_top3_best_epoch": int(train_top3.argmax() + 1),
        "val_top3_max_pct": float(val_top3.max()),
        "val_top3_best_epoch": int(val_top3.argmax() + 1),
    }


def plot_gains(
    first_desired: list[dict],
    desired_rows: list[dict],
    first_interfering: list[dict],
    interfering_rows: list[dict],
    output: Path,
) -> dict:
    first_desired_train = values(first_desired, "train_mae_db")
    first_desired_val = values(first_desired, "val_mae_db")
    first_interfering_train = values(first_interfering, "train_mae_db")
    first_interfering_val = values(first_interfering, "val_mae_db")
    desired_train = values(desired_rows, "train_mae_db")
    desired_val = values(desired_rows, "val_mae_db")
    interfering_train = values(interfering_rows, "train_mae_db")
    interfering_val = values(interfering_rows, "val_mae_db")

    fig, ax = plt.subplots(figsize=(3.45, 2.55), dpi=240)
    style_two_phase_axes(ax)
    plot_two_phases(ax, first_desired_train, desired_train,
                    label="Train, desired", color=TRAIN_COLOR, linestyle="-")
    plot_two_phases(ax, first_desired_val, desired_val,
                    label="Validation, desired", color=VALIDATION_COLOR, linestyle="-")
    plot_two_phases(ax, first_interfering_train, interfering_train,
                    label="Train, interfering", color=TRAIN_COLOR, linestyle="--")
    plot_two_phases(ax, first_interfering_val, interfering_val,
                    label="Validation, interfering", color=VALIDATION_COLOR, linestyle="--")

    desired_val_epoch = int(desired_val.argmin() + 1)
    interfering_val_epoch = int(interfering_val.argmin() + 1)
    best_validation_reference(ax, float(desired_val.min()), desired_val_epoch, VALIDATION_COLOR)
    best_validation_reference(ax, float(interfering_val.min()), interfering_val_epoch, VALIDATION_COLOR)
    metric_box(
        ax,
        "Best validation MAE\n"
        f"Desired: {desired_val.min():.2f} dB\nInterfering: {interfering_val.min():.2f} dB",
        position=(0.975, 0.48),
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.94), ncol=2,
              frameon=True, framealpha=0.94, borderpad=0.35,
              handlelength=2.2, columnspacing=0.9, labelspacing=0.25,
              prop=IN_FIGURE_FONT)
    ax.set_xlabel("Epoch", fontdict=AXIS_LABEL_FONT)
    ax.set_ylabel("MAE (dB)", fontdict=AXIS_LABEL_FONT)
    plotted_values = np.concatenate(
        [
            first_desired_train,
            first_desired_val,
            first_interfering_train,
            first_interfering_val,
            desired_train,
            desired_val,
            interfering_train,
            interfering_val,
        ]
    )
    value_span = float(plotted_values.max() - plotted_values.min())
    margin = max(0.25, 0.035 * value_span)
    ax.set_ylim(max(0.0, float(plotted_values.min()) - margin),
                float(plotted_values.max()) + margin)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return {
        "desired_train_min_mae_db": float(desired_train.min()),
        "desired_train_best_epoch": int(desired_train.argmin() + 1),
        "desired_val_min_mae_db": float(desired_val.min()),
        "desired_val_best_epoch": int(desired_val.argmin() + 1),
        "interfering_train_min_mae_db": float(interfering_train.min()),
        "interfering_train_best_epoch": int(interfering_train.argmin() + 1),
        "interfering_val_min_mae_db": float(interfering_val.min()),
        "interfering_val_best_epoch": int(interfering_val.argmin() + 1),
    }


def main() -> None:
    args = parse_args()
    tasks = ("beam", "desired_gain", "interfering_gain")
    roots = {
        "stage1": {task: args.stage1_results for task in tasks},
        "stage2": {task: args.stage2_results for task in tasks},
    }
    if args.interfering_stage1_results is not None:
        roots["stage1"]["interfering_gain"] = args.interfering_stage1_results
        roots["stage2"]["interfering_gain"] = args.interfering_stage2_results
    source_paths = {
        stage: {task: root / task / "history.json" for task, root in stage_roots.items()}
        for stage, stage_roots in roots.items()
    }
    stages = {
        stage: {task: read_history(path) for task, path in paths.items()}
        for stage, paths in source_paths.items()
    }
    first_stage, histories = stages["stage1"], stages["stage2"]
    args.figures.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase_boundary_epoch": PHASE_BOUNDARY,
        "history_sources": {
            stage: {task: {"path": str(path.resolve()),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for task, path in paths.items()}
            for stage, paths in source_paths.items()
        },
        "stage1_best_validation": {
            "beam_top1_pct": float(values(first_stage["beam"], "val_top1_accuracy_pct").max()),
            "beam_top3_pct": float(values(first_stage["beam"], "val_top3_accuracy_pct").max()),
            "desired_mae_db": float(values(first_stage["desired_gain"], "val_mae_db").min()),
            "interfering_mae_db": float(values(first_stage["interfering_gain"], "val_mae_db").min()),
        },
        "beam": plot_beam(first_stage["beam"], histories["beam"], args.figures / "NN_training_curves(a).pdf"),
        "gains": plot_gains(
            first_stage["desired_gain"],
            histories["desired_gain"],
            first_stage["interfering_gain"],
            histories["interfering_gain"],
            args.figures / "NN_training_curves(b).pdf",
        ),
    }
    summary_path = args.summary or args.stage2_results / "fig4_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
