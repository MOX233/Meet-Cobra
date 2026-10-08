#!/usr/bin/env python3
"""Paper figures for the approved R3C4 mobility analysis; existing data only.

Adds two new figure stems without touching any prior manuscript figure.
The CDF is evaluated on the audit's 0.5-km/h grid; no curve smoothing is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/meet-cobra-mobility-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiment/results/mobility_audit_20260926"
RESULTS = ROOT / "experiment/results/mobility_conditioned_20260927"
GROUPS = ["[0,1)", "[1,20)", "[20,40)", ">=40"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes():
    fig, ax = plt.subplots(figsize=(3.65, 2.45))
    fig.subplots_adjust(left=.14, right=.97, bottom=.21, top=.97)
    ax.grid(color="#d9d9d9", linewidth=.4, alpha=.75)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", length=3, width=.6)
    return fig, ax


def save(fig, output, stem):
    result = []
    for suffix in ("pdf", "png"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240)
        result.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=ROOT/"latexCodes/figures")
    args = parser.parse_args()
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "STIXGeneral"],
                         "mathtext.fontset":"stix", "font.size":8, "axes.labelsize":8.5,
                         "xtick.labelsize":7.5, "ytick.labelsize":7.5,
                         "legend.fontsize":7, "pdf.fonttype":42, "axes.linewidth":.6})
    cdf = pd.read_csv(AUDIT/"speed_cdf.csv")
    statistics = pd.read_csv(AUDIT/"speed_statistics.csv").set_index("dataset")
    source = pd.read_csv(RESULTS/"system_by_speed_summary.csv")
    args.figures.mkdir(parents=True, exist_ok=True)
    outputs = []
    fig, ax = axes()
    for name, label, color, linestyle in [
        ("training_targets", "Training", "#3873b3", "-"),
        ("validation_targets", "Validation", "#bc7a00", "--"),
        ("system_test_scored", "System evaluation", "#008477", "-.")]:
        row = cdf[cdf.dataset==name].sort_values("speed_kmh")
        np.testing.assert_allclose(row.iloc[0].cdf,statistics.loc[name,"stopped_percent"]/100,atol=1e-12)
        assert (np.diff(row.cdf)>=0).all() and row.iloc[-1].cdf==1
        # Include the exact upper endpoint rather than rounding it to a grid cell.
        maximum = statistics.loc[name,"maximum_kmh"]
        row = pd.concat([row,pd.DataFrame([dict(speed_kmh=maximum,cdf=1.)])]).sort_values("speed_kmh")
        ax.plot(row.speed_kmh,row.cdf,color=color,ls=linestyle,lw=1.25,label=label)
        ax.plot([0,0],[0,row.iloc[0].cdf],color=color,ls=linestyle,lw=1.25)
    ax.set(xlim=(0,70),ylim=(0,1.02),xticks=np.arange(0,71,10),yticks=np.arange(0,1.01,.2),
           xlabel="Vehicle speed (km/h)",ylabel="CDF")
    ax.legend(loc="upper left",framealpha=.95,edgecolor="#bbbbbb",borderpad=.45)
    outputs += save(fig,args.figures,"vehicle_speed_cdf_revision1")

    fig, ax = axes()
    x = np.arange(4)
    plotted = []
    for method, color, marker in [("meet_cobra","#c32f27","o"),("oracle_mc","#222222","s")]:
        for rate, linestyle in [(19,"-"),(29,"--")]:
            rows = source[(source.method==method)&(source.rate_mbps==rate)].set_index("speed_group").loc[GROUPS]
            assert (rows.seeds==3).all()
            assert (rows.violation_percent_min<=rows.violation_percent_mean).all()
            assert (rows.violation_percent_mean<=rows.violation_percent_max).all()
            ax.fill_between(x,rows.violation_percent_min,rows.violation_percent_max,color=color,alpha=.14)
            ax.plot(x,rows.violation_percent_mean,color=color,ls=linestyle,marker=marker,
                    markerfacecolor="white" if rate==29 else color,markersize=4,lw=1.25,zorder=3)
            plotted.extend(rows.reset_index().to_dict("records"))
    # Separate encodings keep the legend short and clear of the peak near x=2.
    handles=[Line2D([],[],color="#c32f27",marker="o",lw=1.25,label="MEET-COBRA"),
             Line2D([],[],color="#222222",marker="s",lw=1.25,label="Oracle-MC"),
             Line2D([],[],color="#666666",ls="-",lw=1.25,label=r"$\lambda=19$ Mbps"),
             Line2D([],[],color="#666666",ls="--",lw=1.25,label=r"$\lambda=29$ Mbps")]
    ax.legend(handles=handles,loc="upper left",ncol=2,columnspacing=.9,handlelength=1.8,
              framealpha=.96,edgecolor="#bbbbbb",borderpad=.4)
    ax.set(xlim=(-.16,3.16),ylim=(-.015,.78),xticks=x,
           xticklabels=["[0, 1)","[1, 20)","[20, 40)",r"$\geq40$"],
           yticks=np.arange(0,.71,.1),xlabel="Speed group (km/h)",
           ylabel=r"Violation probability $U$ (%)")
    outputs += save(fig,args.figures,"violation_by_speed_revision1")
    manifest=dict(inputs={str(path.relative_to(ROOT)):sha(path) for path in
                          [AUDIT/"speed_cdf.csv",AUDIT/"speed_statistics.csv",RESULTS/"system_by_speed_summary.csv"]},
                  script_sha256=sha(Path(__file__)),outputs=outputs,
                  speed_groups=GROUPS,loads_mbps=[19,29],system_rows=plotted,
                  uncertainty="Per-speed-group three-seed minimum and maximum, not a confidence interval",
                  original_figures_unchanged=True)
    (RESULTS/"paper_figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
