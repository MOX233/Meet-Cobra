#!/usr/bin/env python3
"""Preview speed-conditioned results in their experiment directory only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ["[0,1)", "[1,20)", "[20,40)", ">=40"]
LABELS = ["[0, 1)", "[1, 20)", "[20, 40)", r"$\geq 40$"]


def save(fig, path):
    for suffix in (".pdf", ".png"):
        output = path.with_suffix(suffix)
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.labelsize":10, "legend.fontsize":9,
                         "axes.spines.top":False, "axes.spines.right":False})
    out=args.results/"previews"
    out.mkdir(exist_ok=True)
    pred=pd.read_csv(args.results/"prediction_by_speed.csv")
    x=np.arange(4)
    fig,axes=plt.subplots(2,2,figsize=(8.5,6),layout="constrained")
    definitions=[("top1_percent","Top-1 accuracy (%)",(85,98)),
                 ("top5_percent","Top-5 accuracy (%)",(98.9,100)),
                 ("desired_mae_db","Desired-link gain MAE (dB)",(0,4.1)),
                 ("interfering_mae_db","Interfering-link gain MAE (dB)",(0,4.1))]
    for ax,(metric,label,ylim) in zip(axes.flat,definitions):
        for name,text,color,marker in [("validation","Validation","#1769aa","o"),
                                       ("test","Test","#c65c13","s")]:
            rows=pred[(pred.dataset==name)&(pred.speed_group!="all")].set_index("speed_group").loc[GROUPS]
            ax.plot(x,rows[metric],color=color,marker=marker,lw=1.6,label=text)
        ax.set(xticks=x,xticklabels=LABELS,xlabel="Speed group (km/h)",ylabel=label,ylim=ylim)
        ax.grid(alpha=.2)
    axes[0,0].legend(loc="upper right")
    save(fig,out/"prediction_by_speed")

    data=pd.read_csv(args.results/"system_by_speed_summary.csv")
    fig,axes=plt.subplots(2,2,figsize=(8.5,6),layout="constrained")
    for col,rate in enumerate((19,29)):
        for row,(metric,label) in enumerate([("violation_percent","Violation probability U (%)"),
                                           ("p99_proxy_ms",r"$L_{99}$ (ms)")]):
            ax=axes[row,col]
            for method,text,color,marker in [("meet_cobra","MEET-COBRA","#1769aa","o"),
                                              ("oracle_mc","Oracle-MC","#666666","s")]:
                a=data[(data.method==method)&(data.rate_mbps==rate)&(data.speed_group!="all")].set_index("speed_group").loc[GROUPS]
                ax.plot(x,a[metric+"_mean"],color=color,marker=marker,lw=1.6,label=text)
                ax.fill_between(x,a[metric+"_min"],a[metric+"_max"],color=color,alpha=.18)
            ax.set(xticks=x,xticklabels=LABELS,xlabel="Speed group (km/h)",ylabel=label)
            ax.set_ylim((-0.015,.67) if row==0 else (1.35,2.1))
            ax.grid(alpha=.2)
            if row==0: ax.set_title(f"Arrival rate: {rate} Mbps")
    axes[0,1].legend(loc="upper left")
    save(fig,out/"system_by_speed_19_29")


if __name__=="__main__":
    main()
