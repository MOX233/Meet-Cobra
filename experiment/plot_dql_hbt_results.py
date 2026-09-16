#!/usr/bin/env python
"""Plot the audited endpoint comparison for DQL-HBT-adapted."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_RESULTS = {
    "MEET-COBRA": {
        "power": [4.010, 44.378],
        "violation": [0.110, 0.340],
    },
    "Reactive-OBRA": {
        "power": [12.283, 78.111],
        "violation": [0.157, 8.090],
    },
    "PQL-BA-adapted": {
        "power": [58.627, 185.126],
        "violation": [7.300, 53.862],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dql-results",
        default=(
            "experiment/results_dql_hbt/final_test_30s_3seeds/"
            "loaded_network_results.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="experiment/results_dql_hbt/dql_hbt_endpoint_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.dql_results, "r", encoding="utf-8") as handle:
        aggregate = json.load(handle)["aggregate"]

    dql = {
        "power": [
            aggregate["rate_1Mbps"]["average_system_power_w_mean"],
            aggregate["rate_19Mbps"]["average_system_power_w_mean"],
        ],
        "violation": [
            aggregate["rate_1Mbps"]["queue_violation_percent_mean"],
            aggregate["rate_19Mbps"]["queue_violation_percent_mean"],
        ],
        "power_ci": [
            aggregate["rate_1Mbps"]["average_system_power_w_ci95"],
            aggregate["rate_19Mbps"]["average_system_power_w_ci95"],
        ],
        "violation_ci": [
            aggregate["rate_1Mbps"]["queue_violation_percent_ci95"],
            aggregate["rate_19Mbps"]["queue_violation_percent_ci95"],
        ],
    }
    methods = dict(REFERENCE_RESULTS)
    methods["DQL-HBT-adapted"] = dql

    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    x = np.arange(2)
    width = 0.19
    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47"]
    for index, (method, result) in enumerate(methods.items()):
        offset = (index - 1.5) * width
        power_error = result.get("power_ci")
        violation_error = result.get("violation_ci")
        axes[0].bar(
            x + offset,
            result["power"],
            width,
            label=method,
            color=colors[index],
            yerr=power_error,
            capsize=2 if power_error is not None else 0,
        )
        axes[1].bar(
            x + offset,
            result["violation"],
            width,
            color=colors[index],
            yerr=violation_error,
            capsize=2 if violation_error is not None else 0,
        )

    axes[0].set_ylabel("Average transmit power (W)")
    axes[1].set_ylabel("Queue violation probability (%)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xticks(x, ["1 Mbps", "19 Mbps"])
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.10, right=0.99, wspace=0.22)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    fig.savefig(args.output_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(args.output_prefix + ".png", dpi=240, bbox_inches="tight")
    print(args.output_prefix + ".pdf")
    print(args.output_prefix + ".png")


if __name__ == "__main__":
    main()
