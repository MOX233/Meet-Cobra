"""Paper figures for violation probability and power; existing gain-error results only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meet-cobra-gain-error-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiment/results/gain_error_sensitivity_20260929"


def plot_metric(summary, metric, label, stem, output):
    fig, axes = plt.subplots(1, 2, figsize=(3.65, 1.80), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.985, bottom=.27, top=.76, wspace=.18)
    styles = [(9, "#3178b5", "o", "-"), (19, "#009c73", "s", "--"),
              (29, "#de8f05", "^", "-."), (35, "#c33b4d", "D", "-")]
    for ax, kind, title in zip(axes, ("desired", "interfering"),
                               ("(a) Desired-link gain", "(b) Interfering-link gain")):
        for rate, color, marker, linestyle in styles:
            rows = sorted((r for r in summary["aggregate"]
                           if r["kind"] == kind and r["rate_mbps"] == rate),
                          key=lambda r: r["sigma_db"])
            assert [r["sigma_db"] for r in rows] == list(range(11))
            assert all(r["seeds"] == 3 for r in rows)
            x = [r["sigma_db"] for r in rows]
            y = [r[f"{metric}_mean"] for r in rows]
            lower = [r[f"{metric}_min"] for r in rows]
            upper = [r[f"{metric}_max"] for r in rows]
            assert all(0 < lo <= mean <= hi for lo, mean, hi in zip(lower, y, upper))
            ax.fill_between(x, lower, upper, color=color, alpha=.16, lw=0)
            ax.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                    lw=1.05, ms=2.8, markeredgewidth=.45, label=f"{rate} Mbps")
        ax.set(xlim=(-.2, 10.2), xticks=[0, 2, 4, 6, 8, 10])
        if metric == "violation_percent":
            ax.set(yscale="log", ylim=(.17, 20))
            ax.yaxis.set_major_locator(FixedLocator([.2, 1, 10]))
            ax.set_yticklabels(["0.2", "1", "10"])
        else:
            ax.set(ylim=(0, 190), yticks=[0, 50, 100, 150])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(color="#d9d9d9", linewidth=.45, alpha=.8)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", length=2.5, width=.6)
        ax.set_title(title, fontsize=9.5, pad=4)
    axes[0].set_ylabel(label, labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .995),
               ncol=4, frameon=False, handlelength=1.35, handletextpad=.35,
               columnspacing=.8, borderaxespad=0)
    fig.supxlabel(r"Additional gain-noise standard deviation $\sigma$ (dB)",
                  x=.54, y=.035, fontsize=9.5)
    for suffix in ("pdf", "png"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300)
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=ROOT / "latexCodes/figures")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    audit = json.loads((args.results / "independent_audit.json").read_text())
    assert summary["complete"] and summary["cases"] == summary["expected"] == 252
    assert audit["all_passed"] and audit["checked"] == 252
    assert audit["protocol_sha256"] == summary["protocol_sha256"]
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 9.5, "axes.labelsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5, "pdf.fonttype": 42, "axes.linewidth": .6,
    })
    args.figures.mkdir(parents=True, exist_ok=True)
    plot_metric(summary, "violation_percent", "Violation\n" + r"probability $U$ (%)",
                "gain_error_sensitivity_revision1", args.figures)
    plot_metric(summary, "power_w", "Average transmit\n" + r"power $\bar{P}$ (W)",
                "gain_error_power_revision1", args.figures)


if __name__ == "__main__":
    main()
