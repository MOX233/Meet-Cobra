#!/usr/bin/env python3
"""Plot five-seed mean paper curves with 95% confidence bands."""

from __future__ import annotations

import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiment/results_multiseed/aggregate/multiseed_curve_data.csv"
OUTPUT = ROOT / "latexCodes/figures"
RATES = np.arange(1.0, 36.0, 2.0)
METHODS = (
    "MEET-COBRA",
    "Oracle-MC",
    "Oracle-CR-LB",
    "Reactive-OBRA",
    "w/o GAP-HO",
    "w/o PET-BF",
    "w/o OTR-RA",
    "MTS-GS-HBF-adapted",
    "O-MAPPO-adapted",
)
COLORS = ("C3", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8")
MARKERS = ("o", "s", "^", "D", "v", ">", "<", "p", "h")
STYLES = {
    name: {
        "color": COLORS[index],
        "marker": MARKERS[index],
        "linestyle": (
            "--"
            if name == "MTS-GS-HBF-adapted"
            else "-."
            if name == "O-MAPPO-adapted"
            else "-"
        ),
    }
    for index, name in enumerate(METHODS)
}


def load_curves():
    with INPUT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    curves = collections.OrderedDict()
    for method in METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["data_rate_mbps"]),
        )
        rates = np.asarray([float(row["data_rate_mbps"]) for row in selected])
        if not np.array_equal(rates, RATES):
            raise ValueError("incomplete curve for {}".format(method))
        curves[method] = {"rows": selected, "rates_mbps": rates}
    return curves


def values(record, metric, statistic="mean"):
    key = "{}_{}".format(metric, statistic)
    return np.asarray([float(row[key]) for row in record["rows"]])


def plot_metric(curves, metric, filename, ylabel, omit=(), log_y=False, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=240)
    for method, record in curves.items():
        if method in omit:
            continue
        style = STYLES[method]
        mean = values(record, metric)
        ci = values(record, metric, "ci95_halfwidth")
        if metric == "macro_association_ratio":
            mean = 100.0 * mean
            ci = 100.0 * ci
        lower = mean - ci
        upper = mean + ci
        if log_y:
            mean = np.maximum(mean, 1e-4)
            lower = np.maximum(lower, 1e-4)
        ax.fill_between(
            record["rates_mbps"],
            lower,
            upper,
            color=style["color"],
            alpha=0.075,
            linewidth=0,
        )
        ax.plot(
            record["rates_mbps"],
            mean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=6,
            label=method,
        )
    ax.set_xlabel(r"Data arrival rate $\lambda$ (Mbps)")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", ncol=2, fontsize=7.6, columnspacing=0.8, handlelength=2.4)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename)
    plt.close(fig)


def plot_percentiles(curves):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=240)
    plotted = [name for name in curves if name != "Oracle-CR-LB"]
    for method in plotted:
        record = curves[method]
        style = STYLES[method]
        for metric, linestyle in (
            ("queueing_proxy_p90_ms", "-"),
            ("queueing_proxy_p99_ms", "-."),
        ):
            mean = values(record, metric)
            ci = values(record, metric, "ci95_halfwidth")
            ax.fill_between(
                record["rates_mbps"],
                np.maximum(mean - ci, 1e-4),
                mean + ci,
                color=style["color"],
                alpha=0.035,
                linewidth=0,
            )
            ax.plot(
                record["rates_mbps"],
                mean,
                color=style["color"],
                marker=style["marker"],
                linestyle=linestyle,
                linewidth=1.5,
                markersize=5.5,
            )
    ax.axhline(20.0, color="black", linestyle="--", linewidth=1.2)
    ax.text(35.0, 18.0, "20 ms", va="top", ha="right", color="black")
    ax.set_xlabel(r"Data arrival rate $\lambda$ (Mbps)")
    ax.set_ylabel(r"Percentiles of normalized-backlog proxy $d_v^{\mathrm{Q}}$ (ms)")
    ax.set_yscale("log")
    method_handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[name]["color"],
            marker=STYLES[name]["marker"],
            linestyle="-",
            linewidth=1.5,
            markersize=5.5,
            label=name,
        )
        for name in plotted
    ]
    method_legend = ax.legend(
        handles=method_handles,
        loc="upper left",
        ncol=2,
        fontsize=7.2,
        columnspacing=0.7,
        handlelength=2.2,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", label=r"$L_{90}^{\mathrm{Q}}$"),
            Line2D([0], [0], color="black", linestyle="-.", label=r"$L_{99}^{\mathrm{Q}}$"),
        ],
        loc="lower right",
        fontsize=8.2,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "latency_90th_99th_comparison_curves_WBL_MS.pdf")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves = load_curves()
    plot_metric(
        curves,
        "average_system_power_w",
        "power_comparison_curves_WBL_MS.pdf",
        r"Average system transmit power $\bar{P}$ (W)",
    )
    plot_metric(
        curves,
        "queue_violation_percent",
        "violation_prob_comparison_curves_WBL_MS.pdf",
        r"Queue-length violation probability $U$ (\%)",
        omit=("Oracle-CR-LB",),
        log_y=True,
        ylim=(1e-3, 1e2),
    )
    plot_percentiles(curves)
    plot_metric(
        curves,
        "macro_association_ratio",
        "BS0_assoc_ratio_comparison_curves_WBL_MS.pdf",
        r"Vehicles associated with BS-0 $\rho_0$ (\%)",
        omit=("Oracle-CR-LB",),
    )


if __name__ == "__main__":
    main()
