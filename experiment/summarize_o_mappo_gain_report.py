"""Audit and summarize the gain-only report ablation and saved references."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiment/results/o_mappo_report_input_20260917"
OUTPUT = ROOT / "experiment/results/o_mappo_gain_report_20260917"
METHODS = ("legacy", "shared", "report", "gain_report", "meet_cobra")
RATES, SEEDS = (1, 13, 27, 35), (1, 2, 3)
LABELS = dict(legacy="Original O-MAPPO", shared="Pilot-input O-MAPPO",
              report="Gain + beam-index O-MAPPO", gain_report="Gain-only O-MAPPO",
              meet_cobra="MEET-COBRA")
METRICS = ("power_w", "violation_percent", "mean_proxy_ms", "p99_proxy_ms",
           "pilots_per_vehicle_slot", "handovers_per_vehicle_s")


def summarize():
    reference = json.loads((REFERENCE / "comparison_summary.json").read_text())
    assert len(reference["runs"]) == 48 and reference["cross_gpu_reference_equal"]
    audit = json.loads((REFERENCE / "gpu_pairing_audit.json").read_text())
    assert {0, 3, 5} <= set(audit["gpu_indices"])
    assert all(x["channel_and_pet_bitwise_equal"] for x in audit["checks"])
    for seed in (20, 21):
        old = json.loads((REFERENCE / f"training_seed{seed}/protocol.json").read_text())
        new = json.loads((OUTPUT / f"training_seed{seed}/protocol.json").read_text())
        assert old["frontend_manifest"] == new["frontend_manifest"]
        assert new["state_variant"] == "gain_report" and new["actor_input_dim"] == 37
        assert old["episodes"] == new["episodes"] == 72
        # Same sampled training intervals and load order, not just same seed labels.
        old_train = json.loads((REFERENCE / f"training_seed{seed}/training.json").read_text())
        new_train = json.loads((OUTPUT / f"training_seed{seed}/training.json").read_text())
        assert len(new_train["history"]) == 72
        for a, b in zip(old_train["history"], new_train["history"]):
            assert (a["episode"], a["start"], a["data_rate_mbps"]) == (b["episode"], b["start"], b["data_rate_mbps"])
        for key in old_train["config"]:
            if key != "state_variant":
                assert old_train["config"][key] == new_train["config"][key], key
        assert old_train["reward"] == new_train["reward"]
    selected = json.loads((OUTPUT / "selected_policy.json").read_text())
    rows = list(reference["runs"])
    for rate in RATES:
        for seed in SEEDS:
            gpu = 3 if rate in (1, 13) else 0
            path = OUTPUT / "runs" / f"gain_report_rate{rate}_seed{seed}_ho10_t1_rician_cuda{gpu}.json"
            row = json.loads(path.read_text())
            assert row["method"] == "gain_report" and row["actor_state_variant"] == "gain_report"
            assert row["rate_mbps"] == rate and row["seed"] == seed and row["physics_gpu"] == gpu
            assert row["ho_interruption_ms"] == 10 and row["rician_fading"]
            assert row["fading_generator"] == "torch_float64_cuda" and row["solver_threads"] == 1
            assert row["checkpoint_sha256"] == selected["checkpoint_sha256"]
            pair = [r for r in reference["runs"] if r["rate_mbps"] == rate and r["seed"] == seed]
            assert {r["traffic_sha256"] for r in pair} == {row["traffic_sha256"]}
            assert all(np.isfinite(row["metrics"][key]) for key in METRICS)
            rows.append(row)
    aggregate, differences = [], []
    for rate in RATES:
        for method in METHODS:
            chosen = [r for r in rows if r["rate_mbps"] == rate and r["method"] == method]
            assert len(chosen) == 3
            entry = dict(rate_mbps=rate, method=method, seeds=3)
            for key in METRICS:
                values = [r["metrics"][key] for r in chosen]
                entry[key + "_mean"] = float(np.mean(values))
                entry[key + "_std"] = float(np.std(values, ddof=1))
            aggregate.append(entry)
        for baseline in ("report", "shared", "legacy", "meet_cobra"):
            entry = dict(rate_mbps=rate, comparison="gain_report-minus-" + baseline)
            for key in METRICS:
                values = []
                for seed in SEEDS:
                    pair = {r["method"]: r["metrics"][key] for r in rows
                            if r["rate_mbps"] == rate and r["seed"] == seed}
                    values.append(pair["gain_report"] - pair[baseline])
                entry[key + "_mean"] = float(np.mean(values))
                entry[key + "_std"] = float(np.std(values, ddof=1))
            differences.append(entry)
    for filename, values in (("comparison_summary.csv", aggregate), ("paired_differences.csv", differences)):
        with (OUTPUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (OUTPUT / "comparison_summary.json").write_text(json.dumps(dict(runs=rows,
        aggregate=aggregate, paired_differences=differences, selected_policy=selected,
        audit="60 complete paired runs; same training segments, frontend, rewards; frozen checkpoints"), indent=2) + "\n")
    lines = ["Mean ± sample SD, three paired seeds and 30 s per run.", "",
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
    for ax, key, label in zip(axes.flat,
            ("power_w", "violation_percent", "p99_proxy_ms", "handovers_per_vehicle_s"),
            ("System transmit power (W)", "Violation probability (%)",
             "99th-percentile latency proxy (ms)", "HO / vehicle / s")):
        for method, marker in zip(("report", "gain_report", "meet_cobra"), ("s", "o", "^")):
            chosen = [r for r in aggregate if r["method"] == method]
            ax.errorbar(RATES, [r[key+"_mean"] for r in chosen],
                        yerr=[r[key+"_std"] for r in chosen], marker=marker,
                        capsize=3, label=LABELS[method])
        ax.set(xlabel="Mean arrival rate per vehicle (Mbps)", ylabel=label, xticks=RATES)
        ax.grid(alpha=.25)
        if key == "violation_percent":
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUTPUT / "comparison_curves.pdf")
    fig.savefig(OUTPUT / "comparison_curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    summarize()
