#!/usr/bin/env python3
"""Audit and export the frozen Fig.5--8 rerun, without touching old figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiment/results/revision_fig5_8_20260917"
METHODS = ("meet_cobra", "oracle_mc", "reactive_obra", "wo_gap_ho", "wo_pet_bf", "wo_otr_ra", "o_mappo", "mts")
LABELS = ("MEET-COBRA", "Oracle-MC", "Reactive-OBRA", "w/o GAP-HO", "w/o PET-BF", "w/o OTR-RA", "O-MAPPO-adapted", "MTS-GS-HBF-adapted")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_runs(check_raw=False):
    protocol_path = OUTPUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = digest(protocol_path)
    if check_raw:
        for path, sha in protocol["code_sha256"].items():
            assert digest(ROOT/path) == sha, path
        for task in protocol["frontend"]["frontend"].values():
            assert digest(task["checkpoint"]) == task["sha256"]
        legacy = protocol["legacy_policy"]
        assert digest(legacy["path"]) == legacy["sha256"]
        cache = ROOT / "experiment/results/o_mappo_shared_frontend_20260917/test_prepared.pkl"
        assert digest(cache) == protocol["frontend"]["test"]["cache_sha256"]
        source = protocol["frontend"]["test"]
        assert digest(source["source"]) == source["source_sha256"]
    rows = []
    traffic = {}
    for path in sorted((OUTPUT / "runs").glob("*.json")):
        row = json.loads(path.read_text())
        assert row["protocol_sha256"] == protocol_sha, path
        assert row["frames"] == 300 and row["retained_frames"] == 298
        key = row["rate_mbps"], row["seed"]
        if key in traffic:
            assert traffic[key] == row["traffic_sha256"], "Unpaired traffic"
        traffic[key] = row["traffic_sha256"]
        raw_path = OUTPUT / "raw" / (path.stem + ".npz")
        assert raw_path.exists()
        m = row["metrics"]
        assert all(np.isfinite(v) and v >= 0 for v in m.values()), path
        assert m["power_w"] <= protocol["power_ceiling_w"] + 1e-6
        assert m["p99_proxy_ms"] >= m["p90_proxy_ms"]
        assert m["violation_percent"] <= 100 and m["macro_association_percent"] <= 100
        if row["method"] == "reactive_obra":
            assert row["method_configuration_sha256"] == digest(OUTPUT / "reactive_configuration.json")
        if check_raw:
            assert digest(raw_path) == row["raw_sha256"], raw_path
            with np.load(raw_path, allow_pickle=False) as data:
                selected = data["queue_frame"] >= 2
                proxy = data["queue_bits"][selected] / (row["rate_mbps"]*1e6) * 1000
                np.testing.assert_allclose([m["p90_proxy_ms"], m["p99_proxy_ms"]], np.percentile(proxy,[90,99]), rtol=1e-12)
                np.testing.assert_allclose(m["power_w"], data["energy_j"][2:].mean()/.1, rtol=1e-12)
                np.testing.assert_allclose(m["violation_percent"], data["violation_probability"][2:].mean()*100, rtol=1e-12)
                counts = data["association_counts"][2:]
                np.testing.assert_allclose(m["macro_association_percent"], counts[:,0].sum()/counts.sum()*100, rtol=1e-12)
        rows.append(row)
    assert len({(r["method"], r["rate_mbps"], r["seed"]) for r in rows}) == len(rows)
    return protocol, rows


def summarize(methods, rates, seeds, check_raw=False):
    protocol, rows = read_runs(check_raw)
    lookup = {(r["method"],r["rate_mbps"],r["seed"]): r for r in rows}
    missing = [(m,r,s) for m in methods for r in rates for s in seeds if (m,r,s) not in lookup]
    status = {m: sum(r["method"] == m for r in rows) for m in METHODS}
    print(json.dumps(dict(completed=len(rows), by_method=status, missing_requested=len(missing)),indent=2))
    if missing:
        print("First missing cases:", missing[:12])
        return None
    aggregate = []
    for method in methods:
        for rate in rates:
            selected = [lookup[method,rate,seed] for seed in seeds]
            record = dict(method=method, label=selected[0]["label"], rate_mbps=rate, seeds=seeds, n=len(seeds), metrics={})
            for metric in selected[0]["metrics"]:
                values = np.array([r["metrics"][metric] for r in selected])
                sd = float(values.std(ddof=1)) if len(seeds)>1 else None
                half = float(t.ppf(.975,len(seeds)-1)*sd/np.sqrt(len(seeds))) if sd is not None else None
                record["metrics"][metric] = dict(mean=float(values.mean()), sd=sd, ci95_halfwidth=half, seed_values=values.tolist())
            aggregate.append(record)
    destination = OUTPUT / "aggregate"
    destination.mkdir(exist_ok=True)
    suffix = "full" if rates == protocol["rates"] and seeds == protocol["seeds"] else "preflight"
    metadata = dict(protocol_sha256=digest(OUTPUT/"protocol.json"), methods=methods, rates=rates, seeds=seeds,
        confidence_interval="Student t interval for the mean across independent evaluation seeds, conditional on the fixed trace and checkpoints; no smoothing",
        percentile_aggregation="P90 and P99 computed within each seed, then averaged across seeds",
        oracle_cr_lb_included=False, raw_checked=check_raw, rows=aggregate)
    (destination/f"summary_{suffix}.json").write_text(json.dumps(metadata,indent=2)+"\n")
    flat = []
    for row in aggregate:
        record = {k: row[k] for k in ("method","label","rate_mbps","n")}
        for name, metric in row["metrics"].items():
            record.update({f"{name}_{stat}":metric[stat] for stat in ("mean","sd","ci95_halfwidth")})
        flat.append(record)
    with (destination/f"curves_{suffix}.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=flat[0].keys())
        writer.writeheader(); writer.writerows(flat)
    return metadata


def plot_curves(metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sys.path.insert(0,str(ROOT))
    from experiment.plot_paper_results_multiseed import STYLES
    plt.rcParams.update({"font.family":"Times New Roman", "font.size":10, "mathtext.fontset":"stix", "pdf.fonttype":42})
    destination = OUTPUT / "preview_figures"
    destination.mkdir(exist_ok=True)
    methods = metadata["methods"]
    records = {method: sorted((r for r in metadata["rows"] if r["method"]==method),key=lambda r:r["rate_mbps"]) for method in methods}

    def curve(ax, method, metric, linestyle=None, alpha=.055):
        selected = records[method]
        label = selected[0]["label"]
        style = STYLES[label]
        x = np.array([r["rate_mbps"] for r in selected])
        y = np.array([r["metrics"][metric]["mean"] for r in selected])
        ci = np.array([r["metrics"][metric]["ci95_halfwidth"] or 0 for r in selected])
        ax.fill_between(x,np.maximum(y-ci,0),y+ci,color=style["color"],alpha=alpha,linewidth=0)
        ax.plot(x,y,color=style["color"],marker=style["marker"],linestyle=linestyle or style["linestyle"],linewidth=1.4,markersize=5,label=label)

    def finish(fig,ax,filename):
        ax.set_xlabel(r"Data arrival rate $\lambda$ (Mbps)")
        ax.set_xlim(0,36)
        ax.grid(True,which="major",color=".85",linewidth=.5)
        fig.tight_layout()
        fig.savefig(destination/f"{filename}_WBL_MS_R1.pdf")
        fig.savefig(destination/f"{filename}_WBL_MS_R1.png",dpi=180)
        plt.close(fig)

    for metric,filename,ylabel in (("power_w","power_comparison_curves",r"Average system transmit power $\bar{P}$ (W)"),
        ("violation_percent","violation_prob_comparison_curves",r"Latency violation probability $U$ (%)"),
        ("macro_association_percent","BS0_assoc_ratio_comparison_curves",r"Vehicles associated with BS-0 $\rho_0$ (%)")):
        fig,ax = plt.subplots(figsize=(6,4))
        for method in methods:
            curve(ax,method,metric)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best",ncol=2,fontsize=7.4,columnspacing=.8,handlelength=2.4)
        if metric == "violation_percent":
            # Keep actual zero probabilities; never replace them with positive data.
            ax.set_yscale("symlog",linthresh=.01,linscale=.4)
            ax.set_ylim(bottom=0)
            ax.set_yticks([0,.01,.1,1,10,100],labels=["0","0.01","0.1","1","10","100"])
        else:
            ax.set_ylim(bottom=0)
        finish(fig,ax,filename)

    fig,ax = plt.subplots(figsize=(6,4.5))
    for method in methods:
        curve(ax,method,"p90_proxy_ms",linestyle="-",alpha=.035)
        curve(ax,method,"p99_proxy_ms",linestyle="-.",alpha=.035)
    ax.axhline(20,color="black",linestyle="--",linewidth=1)
    ax.text(35.5,20,"20 ms",ha="right",va="bottom",fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"90th- and 99th-percentile latencies $L_{90}, L_{99}$ (ms)")
    handles = [Line2D([0],[0],color=STYLES[records[m][0]["label"]]["color"],marker=STYLES[records[m][0]["label"]]["marker"],label=records[m][0]["label"],markersize=5,linewidth=1.4) for m in methods]
    legend = ax.legend(handles=handles,loc="upper left",ncol=2,fontsize=7.2,columnspacing=.7,handlelength=2.2)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([0],[0],color="black",linestyle="-",label=r"$L_{90}$"),Line2D([0],[0],color="black",linestyle="-.",label=r"$L_{99}$")],loc="lower right",fontsize=8)
    finish(fig,ax,"latency_90th_99th_comparison_curves")
    print("Preview figures:",destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",default=",".join(METHODS))
    parser.add_argument("--rates",default=",".join(map(str,range(1,36,2))))
    parser.add_argument("--seeds",default="1,2,3,4,5")
    parser.add_argument("--check-raw",action="store_true")
    parser.add_argument("--plot",action="store_true")
    args = parser.parse_args()
    result = summarize(args.methods.split(","),list(map(int,args.rates.split(","))),list(map(int,args.seeds.split(","))),args.check_raw)
    if result is None:
        sys.exit(2)
    if args.plot:
        plot_curves(result)
