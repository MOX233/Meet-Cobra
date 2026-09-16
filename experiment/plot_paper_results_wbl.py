#!/usr/bin/env python3
"""Generate paper load-sweep figures with the two adapted HO--BF baselines.

The original manuscript figures are left untouched.  The four generated PDF
files use the ``_WBL`` suffix and are written to ``latexCodes/figures``.

For the normalized-backlog percentiles, every original curve is recomputed as
q_v/lambda_v with the traffic rate of its own load point.  This avoids the
stale ``args.data_rate`` value in the historical notebook, which equals the
last rate (35 Mbps) after the simulation sweep.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RESULT = (
    ROOT
    / "experiment/results_paper_exp1"
    / "lbd1.00_800_830.0_2025-12-22 06:26:16"
    / "sim_result_dict.npy"
)
BASELINE_RESULTS = (
    ROOT
    / "experiment/results_joint_baselines/full_grid_30s_seed1"
    / "joint_baselines_curve_data.csv",
    ROOT
    / "experiment/results_mts_gs_hbf/pressure_early/exact_800_830_seed1"
    / "curve_data.csv",
)
FIGURE_DIR = ROOT / "latexCodes/figures"

EXPECTED_RATES_MBPS = np.arange(1.0, 36.0, 2.0)
ORIGINAL_NAME_MAP = {
    "Proposed": "MEET-COBRA",
    "Oracle-LP-LB": "Oracle-CR-LB",
}
ORIGINAL_METHODS = (
    "MEET-COBRA",
    "Oracle-MC",
    "Oracle-CR-LB",
    "Reactive-OBRA",
    "w/o GAP-HO",
    "w/o PET-BF",
    "w/o OTR-RA",
)
BASELINE_METHODS = ("MTS-GS-HBF-adapted", "O-MAPPO-adapted")

# The first seven entries reproduce utils.plot_utils; the last two continue
# the same Matplotlib color/marker cycles for the new baselines.
COLORS = ("C3", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8")
MARKERS = ("o", "s", "^", "D", "v", ">", "<", "p", "h")
METHOD_STYLES = {
    name: {
        "color": COLORS[index],
        "marker": MARKERS[index],
        "linestyle": (
            "--" if name == "MTS-GS-HBF-adapted"
            else "-." if name == "O-MAPPO-adapted"
            else "-"
        ),
    }
    for index, name in enumerate(ORIGINAL_METHODS + BASELINE_METHODS)
}


def _check_rates(rates_mbps: np.ndarray, source: str) -> None:
    if rates_mbps.shape != EXPECTED_RATES_MBPS.shape or not np.allclose(
        rates_mbps, EXPECTED_RATES_MBPS, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            f"{source} has traffic rates {rates_mbps.tolist()}, expected "
            f"{EXPECTED_RATES_MBPS.tolist()}"
        )


def _flatten_queue_record(frame_records: Mapping) -> np.ndarray:
    queue_values = []
    for vehicle_records in frame_records.values():
        queue_values.extend(vehicle_records.values())
    return np.asarray(queue_values, dtype=np.float64)


def load_original_results() -> OrderedDict[str, Dict[str, np.ndarray]]:
    payload = np.load(ORIGINAL_RESULT, allow_pickle=True).item()
    rates_mbps = np.asarray(payload["data_rate_list"], dtype=float) / 1e6
    _check_rates(rates_mbps, str(ORIGINAL_RESULT))

    results: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
    for stored_name, record in payload.items():
        if stored_name in {"args", "data_rate_list"}:
            continue
        name = ORIGINAL_NAME_MAP.get(stored_name, stored_name)
        power = np.asarray(record["avg_system_power_list"], dtype=float).copy()
        violation = np.asarray(record["vio_prob_list"], dtype=float)
        if len(power) != len(rates_mbps) or len(violation) != len(rates_mbps):
            raise ValueError(f"Incomplete original curve for {name}")

        # Match the correction applied in paper_plot_results.ipynb.
        if name == "Oracle-CR-LB":
            power[-2:] = 185.43

        # Oracle-CR-LB is a power-only, non-implementable lower bound and has
        # no association or queue records.  Its placeholders are never plotted.
        if name == "Oracle-CR-LB":
            macro_association_percent = np.zeros_like(rates_mbps)
            p90_ms = np.zeros_like(rates_mbps)
            p99_ms = np.zeros_like(rates_mbps)
        else:
            association_counts = np.asarray(
                record["carnum_under_BS_list"], dtype=float
            ).mean(axis=-2)
            macro_association_percent = (
                100.0
                * association_counts[:, 0]
                / association_counts.sum(axis=-1)
            )

            p90_ms = []
            p99_ms = []
            for rate_mbps, frame_records in zip(
                rates_mbps, record["queuelen_4eachVeh_record_list"]
            ):
                queue_bits = _flatten_queue_record(frame_records)
                if queue_bits.size == 0 or not np.isfinite(queue_bits).all():
                    raise ValueError(
                        f"Invalid queue record for {name} at {rate_mbps:g} Mbps"
                    )
                denominator_bps = rate_mbps * 1e6
                p90_ms.append(
                    float(np.percentile(queue_bits, 90) / denominator_bps * 1e3)
                )
                p99_ms.append(
                    float(np.percentile(queue_bits, 99) / denominator_bps * 1e3)
                )

        results[name] = {
            "rates_mbps": rates_mbps.copy(),
            "power_w": power,
            "violation_percent": violation,
            "p90_ms": np.asarray(p90_ms),
            "p99_ms": np.asarray(p99_ms),
            "macro_association_percent": macro_association_percent,
        }

    if tuple(results) != ORIGINAL_METHODS:
        raise ValueError(
            f"Unexpected original method order {tuple(results)}; expected {ORIGINAL_METHODS}"
        )
    return results


def load_baseline_results() -> OrderedDict[str, Dict[str, np.ndarray]]:
    grouped = {name: [] for name in BASELINE_METHODS}
    for result_path in BASELINE_RESULTS:
        with result_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["method"] in grouped:
                    grouped[row["method"]].append(row)

    results: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()
    for name in BASELINE_METHODS:
        rows = sorted(grouped[name], key=lambda row: float(row["data_rate_mbps"]))
        rates_mbps = np.asarray([float(row["data_rate_mbps"]) for row in rows])
        _check_rates(rates_mbps, name)
        results[name] = {
            "rates_mbps": rates_mbps,
            "power_w": np.asarray(
                [float(row["average_system_power_w"]) for row in rows]
            ),
            "violation_percent": np.asarray(
                [float(row["queue_violation_percent"]) for row in rows]
            ),
            "p90_ms": np.asarray(
                [float(row["queueing_proxy_p90_ms"]) for row in rows]
            ),
            "p99_ms": np.asarray(
                [float(row["queueing_proxy_p99_ms"]) for row in rows]
            ),
            "macro_association_percent": 100.0
            * np.asarray([float(row["macro_association_ratio"]) for row in rows]),
        }
    return results


def _validate_curves(curves: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    expected_methods = ORIGINAL_METHODS + BASELINE_METHODS
    if tuple(curves) != expected_methods:
        raise ValueError(f"Unexpected method order: {tuple(curves)}")
    for name, record in curves.items():
        _check_rates(record["rates_mbps"], name)
        for metric in (
            "power_w",
            "violation_percent",
            "p90_ms",
            "p99_ms",
            "macro_association_percent",
        ):
            values = np.asarray(record[metric])
            if values.shape != EXPECTED_RATES_MBPS.shape:
                raise ValueError(f"{name}:{metric} has shape {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name}:{metric} contains non-finite values")
            if (values < 0).any():
                raise ValueError(f"{name}:{metric} contains negative values")


def _plot_single_metric(
    curves: Mapping[str, Mapping[str, np.ndarray]],
    metric: str,
    output_name: str,
    ylabel: str,
    *,
    omit: Iterable[str] = (),
    log_y: bool = False,
    ylim=None,
) -> None:
    omitted = set(omit)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=240)
    for name, record in curves.items():
        if name in omitted:
            continue
        style = METHOD_STYLES[name]
        ax.plot(
            record["rates_mbps"],
            record[metric],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=6,
            label=name,
        )
    ax.set_xlabel(r"Data arrival rate $\lambda$ (Mbps)")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", ncol=2, fontsize=7.6, columnspacing=0.8, handlelength=2.4)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name)
    plt.close(fig)


def _plot_percentiles(curves: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=240)
    plotted_methods = [name for name in curves if name != "Oracle-CR-LB"]
    for name in plotted_methods:
        record = curves[name]
        style = METHOD_STYLES[name]
        for metric, linestyle in (("p90_ms", "-"), ("p99_ms", "-.")):
            ax.plot(
                record["rates_mbps"],
                record[metric],
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
            color=METHOD_STYLES[name]["color"],
            marker=METHOD_STYLES[name]["marker"],
            linestyle="-",
            linewidth=1.5,
            markersize=5.5,
            label=name,
        )
        for name in plotted_methods
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
    percentile_handles = [
        Line2D([0], [0], color="black", linestyle="-", label=r"$L_{90}^{\mathrm{Q}}$"),
        Line2D([0], [0], color="black", linestyle="-.", label=r"$L_{99}^{\mathrm{Q}}$"),
    ]
    ax.legend(handles=percentile_handles, loc="lower right", fontsize=8.2)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "latency_90th_99th_comparison_curves_WBL.pdf")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    curves = OrderedDict()
    curves.update(load_original_results())
    curves.update(load_baseline_results())
    _validate_curves(curves)

    _plot_single_metric(
        curves,
        "power_w",
        "power_comparison_curves_WBL.pdf",
        r"Average system transmit power $\bar{P}$ (W)",
    )
    _plot_single_metric(
        curves,
        "violation_percent",
        "violation_prob_comparison_curves_WBL.pdf",
        r"Queue-length violation probability $U$ (\%)",
        omit=("Oracle-CR-LB",),
        log_y=True,
        ylim=(1e-3, 1e2),
    )
    _plot_percentiles(curves)
    _plot_single_metric(
        curves,
        "macro_association_percent",
        "BS0_assoc_ratio_comparison_curves_WBL.pdf",
        r"Vehicles associated with BS-0 $\rho_0$ (\%)",
        omit=("Oracle-CR-LB",),
    )

    for output_name in (
        "power_comparison_curves_WBL.pdf",
        "violation_prob_comparison_curves_WBL.pdf",
        "latency_90th_99th_comparison_curves_WBL.pdf",
        "BS0_assoc_ratio_comparison_curves_WBL.pdf",
    ):
        output = FIGURE_DIR / output_name
        print(f"generated {output.relative_to(ROOT)} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
