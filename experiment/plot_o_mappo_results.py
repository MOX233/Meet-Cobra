#!/usr/bin/env python
"""Plot the frozen O-MAPPO and DQL-HBT endpoint/high-load results."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean_ci(aggregate, rate, metric):
    item = aggregate["rate_{:g}Mbps".format(rate)]
    return item[metric + "_mean"], item[metric + "_ci95"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--o-mappo",
        default="experiment/results/o_mappo/final_test/loaded_network_results.json",
    )
    parser.add_argument(
        "--o-mappo-high-load",
        default="experiment/results/o_mappo/high_load_test/loaded_network_results.json",
    )
    parser.add_argument(
        "--dql",
        default="experiment/results_dql_hbt/final_test_30s_3seeds/loaded_network_results.json",
    )
    parser.add_argument(
        "--dql-high-load",
        default="experiment/results_dql_hbt/final_original_high_load_control_seed1/loaded_network_results.json",
    )
    parser.add_argument(
        "--output-prefix",
        default="experiment/results/o_mappo/o_mappo_dql_comparison",
    )
    args = parser.parse_args()

    om = _load(args.o_mappo)["aggregate"]
    om_high = _load(args.o_mappo_high_load)["aggregate"]
    dql = _load(args.dql)["aggregate"]
    dql_high = _load(args.dql_high_load)["aggregate"]
    rates = np.asarray([1.0, 19.0, 27.0, 35.0])

    def series(primary, high, metric):
        means = []
        cis = []
        for rate in rates:
            source = primary if rate in (1.0, 19.0) else high
            mean, ci = _mean_ci(source, rate, metric)
            means.append(mean)
            cis.append(ci)
        return np.asarray(means), np.asarray(cis)

    om_power, om_power_ci = series(om, om_high, "average_system_power_w")
    om_vio, om_vio_ci = series(om, om_high, "queue_violation_percent")
    dql_power, dql_power_ci = series(dql, dql_high, "average_system_power_w")
    dql_vio, dql_vio_ci = series(dql, dql_high, "queue_violation_percent")

    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    style = {
        "O-MAPPO-adapted": ("#1f77b4", "o"),
        "DQL-HBT-adapted": ("#d62728", "s"),
    }
    for label, values, ci in (
        ("O-MAPPO-adapted", om_power, om_power_ci),
        ("DQL-HBT-adapted", dql_power, dql_power_ci),
    ):
        color, marker = style[label]
        axes[0].errorbar(
            rates,
            values,
            yerr=ci,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=4.5,
            capsize=2.5,
            label=label,
        )
    for label, values, ci in (
        ("O-MAPPO-adapted", om_vio, om_vio_ci),
        ("DQL-HBT-adapted", dql_vio, dql_vio_ci),
    ):
        color, marker = style[label]
        axes[1].errorbar(
            rates,
            values,
            yerr=ci,
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=4.5,
            capsize=2.5,
            label=label,
        )
    axes[0].set_ylabel("Average system power (W)")
    axes[1].set_ylabel("Queue violation probability (%)")
    for axis in axes:
        axis.set_xlabel("Per-vehicle offered traffic (Mbps)")
        axis.set_xticks(rates)
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="upper left")
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    fig.savefig(args.output_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(args.output_prefix + ".png", bbox_inches="tight", dpi=220)


if __name__ == "__main__":
    main()
