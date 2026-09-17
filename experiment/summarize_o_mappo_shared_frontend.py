#!/usr/bin/env python3
"""Audit and summarize the paired 36-point shared-frontend comparison."""
import collections
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiment/results/o_mappo_shared_frontend_20260917"
METHODS = ("legacy", "shared", "meet_cobra")
RATES = (1, 13, 27, 35)
SEEDS = (1, 2, 3)
LABELS = dict(legacy="Old O-MAPPO", shared="Shared-frontend O-MAPPO", meet_cobra="MEET-COBRA")
METRICS = ("power_w", "violation_percent", "mean_proxy_ms", "p99_proxy_ms",
           "pilots_per_vehicle_slot", "handovers_per_vehicle_s")


def summarize(output=OUTPUT):
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            group = []
            for method in METHODS:
                suffix = "_batch" if method == "meet_cobra" else ""
                path = output / "runs" / f"{method}_rate{rate}_seed{seed}_ho10_t1_rician{suffix}.json"
                row = json.loads(path.read_text())
                assert row["method"] == method and row["rate_mbps"] == rate and row["seed"] == seed
                assert row["ho_interruption_ms"] == 10 and row["rician_fading"]
                assert row["solver_threads"] == 1
                assert all(np.isfinite(row["metrics"][key]) for key in METRICS)
                group.append(row)
            assert len({row["traffic_sha256"] for row in group}) == 1
            rows.extend(group)
    for method in ("legacy", "shared"):
        assert len({row["checkpoint_sha256"] for row in rows if row["method"] == method}) == 1
    selected = json.loads((output / "selected_policy.json").read_text())
    assert selected["checkpoint_sha256"] == next(row["checkpoint_sha256"] for row in rows if row["method"] == "shared")
    aggregated, differences = [], []
    for rate in RATES:
        for method in METHODS:
            chosen = [row for row in rows if row["rate_mbps"] == rate and row["method"] == method]
            entry = dict(rate_mbps=rate, method=method, seeds=len(chosen))
            for metric in METRICS:
                values = [row["metrics"][metric] for row in chosen]
                entry[metric + "_mean"] = float(np.mean(values))
                entry[metric + "_std"] = float(np.std(values, ddof=1))
            aggregated.append(entry)
        for reference in ("legacy", "meet_cobra"):
            entry = dict(rate_mbps=rate, comparison="shared-minus-" + reference)
            for metric in METRICS:
                values = []
                for seed in SEEDS:
                    pair = {r["method"]: r["metrics"][metric] for r in rows if r["rate_mbps"] == rate and r["seed"] == seed}
                    values.append(pair["shared"] - pair[reference])
                entry[metric + "_mean"] = float(np.mean(values))
                entry[metric + "_std"] = float(np.std(values, ddof=1))
            differences.append(entry)
    for filename, values in (("comparison_summary.csv", aggregated), ("paired_differences.csv", differences)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / "comparison_summary.json").write_text(json.dumps(dict(
        selected_policy=selected, runs=rows, aggregate=aggregated, paired_differences=differences,
        audit="36 complete runs; identical paired traffic hashes; fixed checkpoints; 10-ms HO; paired Rician"
    ), indent=2) + "\n")
    header = "| Rate (Mbit/s) | Method | Power (W) | U (%) | p99 proxy (ms) | BF pilots / vehicle-slot | HO / vehicle-s |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for row in aggregated:
        cells = [str(row["rate_mbps"]), LABELS[row["method"]]]
        for metric in ("power_w", "violation_percent", "p99_proxy_ms", "pilots_per_vehicle_slot", "handovers_per_vehicle_s"):
            cells.append(f"{row[metric+'_mean']:.4f} ± {row[metric+'_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    (output / "result_table.md").write_text("Mean ± sample SD over three paired seeds.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric, ylabel in zip(axes.flat,
            ("power_w", "violation_percent", "p99_proxy_ms", "handovers_per_vehicle_s"),
            ("System transmit power (W)", "Violation probability (%)", "99th-percentile latency proxy (ms)", "HO / vehicle / s")):
        for method, marker in zip(METHODS, ("s", "o", "^")):
            selected_rows = [r for r in aggregated if r["method"] == method]
            ax.errorbar([r["rate_mbps"] for r in selected_rows], [r[metric+"_mean"] for r in selected_rows],
                        yerr=[r[metric+"_std"] for r in selected_rows], marker=marker, capsize=3, label=LABELS[method])
        ax.set(xlabel="Mean arrival rate per vehicle (Mbit/s)", ylabel=ylabel, xticks=RATES)
        ax.grid(alpha=.25)
        if metric == "violation_percent":
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=9)
    fig.savefig(output / "comparison_curves.pdf")
    fig.savefig(output / "comparison_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    summarize(Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT)
