#!/usr/bin/env python3
"""Plot the fixed-decision evidence for Reviewer 2, Comment 1.

Run from any directory with:
    python response_letter/plot_interference_validation.py

Reads the audited full-precision figure manifest and independently checks its
means against the 54 MEET-COBRA run summaries. No simulations are modified.
Outputs a single-panel receive-rate figure as a vector PDF and a PNG preview.
The CSV retains both metrics, including the interference ratios no longer plotted.
The legacy JSON key ``service_capacity_ratio`` is the approximate/directional
total receive-rate ratio; no new service-capacity metric is introduced here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment/results/revision_directional_20260922"
OUTPUT = ROOT / "response_letter/figures"
STEM = "r2c1_interference_validation"


def load_data():
    manifest = json.loads((RESULTS / "paper_figures/figure_manifest.json").read_text())
    protocol_bytes = (RESULTS / "grid/protocol.json").read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest() == manifest["protocol_sha256"]
    protocol = json.loads(protocol_bytes)
    rows = manifest["fixed_decision_comparison"]
    assert [row["rate_mbps"] for row in rows] == list(range(1, 36, 2))
    assert manifest["seeds"] == protocol["seeds"] == [1, 2, 3]
    for row in rows:
        assert row["seeds"] == len(manifest["seeds"])
        runs = []
        for seed in manifest["seeds"]:
            path = RESULTS / "grid/runs" / f"meet_cobra_rate{row['rate_mbps']}_seed{seed}.json"
            run = json.loads(path.read_text())
            assert run["protocol_sha256"] == manifest["protocol_sha256"]
            assert run["method"] == "meet_cobra"
            assert run["rate_mbps"] == row["rate_mbps"] and run["seed"] == seed
            runs.append(run["comparison"])
        np.testing.assert_allclose(
            row["interference_ratio"],
            np.mean([run["interference_ratio"] for run in runs]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            row["service_underestimation_percent"],
            np.mean([100 * (1 - run["service_capacity_ratio"]) for run in runs]),
            rtol=1e-12,
        )
    return rows


def main():
    rows = load_data()
    rates = np.array([row["rate_mbps"] for row in rows])
    underestimation = np.array([row["service_underestimation_percent"] for row in rows])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / f"{STEM}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arrival_rate_mbps", "number_of_seeds", "mean_interference_ratio",
                         "mean_total_receive_rate_underestimation_percent"])
        for row in rows:
            writer.writerow([row["rate_mbps"], row["seeds"], row["interference_ratio"],
                             row["service_underestimation_percent"]])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(4.75, 2.9))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.19, top=0.96)
    ax.set_xlim(0, 36)
    ax.set_xticks([1, 5, 9, 13, 17, 21, 25, 29, 33, 35])
    ax.set_xlabel(r"Mean arrival rate $\lambda$ (Mbps)")
    ax.set_axisbelow(True)
    ax.grid(color="#e2e2e2", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.7)
    ax.plot(rates, underestimation, color="#b75324", marker="s", linewidth=1.55,
            markersize=3.6, markerfacecolor="white", markeredgewidth=0.9, zorder=3)
    ax.set_ylim(0, 21.5)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("Total receive rate underestimation (%)")
    ax.annotate(f"{underestimation[0]:.2f}%", (rates[0], underestimation[0]),
                xytext=(8, -5), textcoords="offset points", ha="left", va="top",
                fontsize=9, color="#873d1b")
    maximum = int(np.argmax(underestimation))
    ax.annotate(f"{underestimation[maximum]:.2f}%",
                (rates[maximum], underestimation[maximum]),
                xytext=(0, -12), textcoords="offset points", ha="center", va="top",
                fontsize=9, color="#873d1b")

    for suffix in ("pdf", "png"):
        path = OUTPUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=240)
        print(path.relative_to(ROOT))
    plt.close(fig)
    print("Validated 18 loads and three seed means against all 54 run summaries.")


if __name__ == "__main__":
    main()
