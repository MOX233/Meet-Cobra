"""Summarize all paired full-duration results without selecting on test data."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "experiment/results/o_mappo_shared_frontend_20260917"
OUTPUT = ROOT / "experiment/results/o_mappo_report_input_20260917"
METHODS = ("legacy", "shared", "report", "meet_cobra")
RATES, SEEDS = (1, 13, 27, 35), (1, 2, 3)
LABELS = dict(legacy="Original O-MAPPO", shared="Pilot-input O-MAPPO",
              report="Report-input O-MAPPO", meet_cobra="MEET-COBRA")
METRICS = ("power_w", "violation_percent", "mean_proxy_ms", "p99_proxy_ms",
           "pilots_per_vehicle_slot", "handovers_per_vehicle_s")


def read_run(method, rate, seed):
    directory, gpu = (OUTPUT, 0 if rate == 35 else 3) if method == "report" else (PRIOR, 5)
    suffix = "_batch" if method == "meet_cobra" else ""
    filename = f"{method}_rate{rate}_seed{seed}_ho10_t1_rician{suffix}_cuda{gpu}.json"
    return json.loads((directory / "runs" / filename).read_text())


def summarize():
    # Verify cross-device random draws and one complete reference simulation.
    audit = json.loads((OUTPUT / "gpu_pairing_audit.json").read_text())
    assert all(x["channel_and_pet_bitwise_equal"] for x in audit["checks"])
    reference = read_run("legacy", 1, 1)
    for gpu in (0, 3):
        anchor = json.loads((OUTPUT / f"gpu_crosscheck/runs/legacy_rate1_seed1_ho10_t1_rician_cuda{gpu}.json").read_text())
        assert anchor["traffic_sha256"] == reference["traffic_sha256"]
        assert anchor["checkpoint_sha256"] == reference["checkpoint_sha256"]
        for key in reference["metrics"]:
            np.testing.assert_allclose(anchor["metrics"][key], reference["metrics"][key], rtol=0, atol=1e-10)
    compiled = json.loads((OUTPUT / "gpu_crosscheck/runs/meet_cobra_rate1_seed1_ho10_t1_rician_batch_cuda3.json").read_text())
    reference = read_run("meet_cobra", 1, 1)
    for key in reference["metrics"]:
        np.testing.assert_allclose(compiled["metrics"][key], reference["metrics"][key], rtol=0, atol=1e-10)
    rows = []
    for rate in RATES:
        for seed in SEEDS:
            group = [read_run(m, rate, seed) for m in METHODS]
            assert len({x["traffic_sha256"] for x in group}) == 1
            for row in group:
                assert row["ho_interruption_ms"] == 10 and row["rician_fading"]
                assert row["fading_generator"] == "torch_float64_cuda" and row["solver_threads"] == 1
                assert row["seed"] == seed and row["rate_mbps"] == rate
                assert all(np.isfinite(row["metrics"][key]) for key in METRICS)
            assert group[2]["actor_state_variant"] == "report"
            rows.extend(group)
    for method, directory in (("shared", PRIOR), ("report", OUTPUT)):
        selected = json.loads((directory / "selected_policy.json").read_text())
        assert {r["checkpoint_sha256"] for r in rows if r["method"] == method} == {selected["checkpoint_sha256"]}
    aggregate, differences = [], []
    for rate in RATES:
        for method in METHODS:
            chosen = [r for r in rows if r["rate_mbps"] == rate and r["method"] == method]
            entry = dict(rate_mbps=rate, method=method, seeds=len(chosen))
            for metric in METRICS:
                values = [r["metrics"][metric] for r in chosen]
                entry[metric + "_mean"] = float(np.mean(values))
                entry[metric + "_std"] = float(np.std(values, ddof=1))
            aggregate.append(entry)
        for baseline in ("legacy", "shared", "meet_cobra"):
            entry = dict(rate_mbps=rate, comparison="report-minus-" + baseline)
            for metric in METRICS:
                values = []
                for seed in SEEDS:
                    pair = {r["method"]: r["metrics"][metric] for r in rows
                            if r["rate_mbps"] == rate and r["seed"] == seed}
                    values.append(pair["report"] - pair[baseline])
                entry[metric + "_mean"] = float(np.mean(values))
                entry[metric + "_std"] = float(np.std(values, ddof=1))
            differences.append(entry)
    for filename, values in (("comparison_summary.csv", aggregate), ("paired_differences.csv", differences)):
        with (OUTPUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (OUTPUT / "comparison_summary.json").write_text(json.dumps(dict(runs=rows,
        aggregate=aggregate, paired_differences=differences, cross_gpu_reference_equal=True,
        report_selected_policy=json.loads((OUTPUT / "selected_policy.json").read_text())), indent=2) + "\n")
    lines = ["Mean ± sample SD over three paired seeds, 30 s per run.", "",
             "| Rate (Mbps) | Method | Power (W) | U (%) | p99 proxy (ms) | HO / vehicle-s |",
             "|---|---|---|---|---|---|"]
    for row in aggregate:
        cells = [str(row["rate_mbps"]), LABELS[row["method"]]]
        for key in ("power_w", "violation_percent", "p99_proxy_ms", "handovers_per_vehicle_s"):
            cells.append(f"{row[key+'_mean']:.4f} ± {row[key+'_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    (OUTPUT / "result_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric, label in zip(axes.flat,
            ("power_w", "violation_percent", "p99_proxy_ms", "handovers_per_vehicle_s"),
            ("System transmit power (W)", "Violation probability (%)",
             "99th-percentile latency proxy (ms)", "HO / vehicle / s")):
        for method, marker in zip(METHODS, ("s", "o", "D", "^")):
            selected_rows = [r for r in aggregate if r["method"] == method]
            ax.errorbar(RATES, [r[metric + "_mean"] for r in selected_rows],
                        yerr=[r[metric + "_std"] for r in selected_rows],
                        marker=marker, capsize=3, label=LABELS[method])
        ax.set(xlabel="Mean arrival rate per vehicle (Mbps)", ylabel=label, xticks=RATES)
        ax.grid(alpha=.25)
        if metric == "violation_percent":
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUTPUT / "comparison_curves.pdf")
    fig.savefig(OUTPUT / "comparison_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    summarize()
