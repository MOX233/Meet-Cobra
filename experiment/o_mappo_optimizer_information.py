#!/usr/bin/env python3
"""Single-seed, frozen-actor diagnostic of target optimizer information.

Oracle variants are diagnostic controls, never deployable baselines. Only
the post-actor optimizer interface is changed. No checkpoint is retrained
or selected using these test results.
"""
import argparse
import collections
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.o_mappo_shared_frontend import (
    OUTPUT as CACHE, LEGACY, read_pickle, temporal_slice, paper_args,
    MICRO_BS_LOCATIONS, OMAPPPolicy, torch, np, single_thread_solvers,
    make_paired_traffic, run_sim_o_mappo, metric_arrays, FrameProgress,
    write_json, digest,
)
from utils.alg_utils import estimate_num_RB_allocated_perBS
from utils.beam_utils import generate_dft_codebook
from utils.o_mappo import _candidate_links
from utils.pql_ba import best_beam_pair, no_bf_gain_db, macro_gain_db
from utils.pql_ba_adapted import _capacity_per_rb_bps, _interference_db

OUTPUT = ROOT / "experiment/results/o_mappo_optimizer_information_20260917"
CHECKPOINTS = {
    "gain_report": ROOT / "experiment/results/o_mappo_gain_report_20260917/training_seed20/best_policy.pt",
    "report": ROOT / "experiment/results/o_mappo_report_input_20260917/training_seed20/best_policy.pt",
    "legacy": LEGACY,
}
VARIANTS = (
    "baseline", "true_desired", "true_interference", "true_gains",
    "legacy_load", "legacy_fixed", "legacy_all", "next_true_gains",
    "report_load", "report_fixed", "report_both",
)


def fixed_allocation(args, learners, backlog, serving, no_bf, load, rb_total, config):
    """The original fixed-user occupancy calculation, with explicit inputs."""
    capacities = np.array([args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs)
    duration = args.slots_per_frame * args.slot_len
    allocated = {}
    for v in backlog:
        bs = int(learners[v].action)
        interference = _interference_db(args, bs, no_bf[v], load)
        capacity = _capacity_per_rb_bps(args, bs, serving[v], interference,
                                       config.tracking_pilots if bs else 0.0)
        allocated[v] = min(backlog[v] / max(capacity * duration, 1e-12), capacities[bs])
    for bs in range(config.num_bs):
        ids = [v for v in backlog if learners[v].action == bs]
        total = sum(allocated[v] for v in ids)
        scale = min(float(rb_total[bs]), capacities[bs]) / max(total, 1e-12)
        for v in ids:
            allocated[v] *= min(scale, 1.0)
    return allocated


def report_estimates(args, records, learners, backlog, config, macro_loc):
    """Reestimate load/occupancy using only reports, position and public state.

    Desired-gain reports estimate best-pair gain, not gain of the currently
    tracked pair. That approximation is kept explicit; no hidden beam sweep
    or true-channel access is used to repair it.
    """
    gains, no_bf, serving = {}, {}, {}
    connection = {v: int(learners[v].action) for v in records}
    rates = {v: args.data_rate for v in records}
    for v, record in records.items():
        p = record["shared_prediction"]
        macro = macro_gain_db(args, record["pos"], macro_loc)
        no_bf[v] = np.concatenate(([macro], p["interference"]))
        gains[v] = no_bf[v].copy()
        bs = connection[v]
        serving[v] = macro if bs == 0 else float(p["gain"][bs - 1])
        gains[v][bs] = serving[v]
    bs_locations = np.zeros((config.num_bs, 2))  # estimator uses only its length
    rb = estimate_num_RB_allocated_perBS(args, connection, bs_locations,
        list(records), gains, rates, infer_g_dict=no_bf)
    capacities = np.array([args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs)
    load = np.clip(rb / capacities, 0, 1.5)
    fixed = fixed_allocation(args, learners, backlog, serving, no_bf, load, rb, config)
    return load, fixed


def oracle_labels(timeline, config):
    tx, rx = generate_dft_codebook(config.num_tx_beams), generate_dft_codebook(config.num_rx_beams)
    labels = {}
    for frame, records in timeline.items():
        labels[frame] = {
            v: dict(gain=np.array([best_beam_pair(r["h"], m, tx, rx)[2]
                                  for m in range(config.num_micro_bs)]),
                    interference=no_bf_gain_db(r["h"])) for v, r in records.items()
        }
    return labels


class InputAblation:
    def __init__(self, variant, labels=None):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.labels = variant, labels
        self.frames = list(labels) if labels else []
        self.next_frames = dict(zip(self.frames[:-1], self.frames[1:]))
        self.diagnostics = []

    def __call__(self, **c):
        variant = self.variant
        records, load, allocated = c["records"], c["load"], c["allocated_rb"]
        config, args = c["config"], c["args"]
        capacities = np.array([args.num_RB_macro] + [args.num_RB_micro] * config.num_micro_bs)
        if variant in ("true_desired", "true_interference", "true_gains", "legacy_all", "next_true_gains"):
            frame = c["frame"]
            labels = self.labels[self.next_frames.get(frame, frame) if variant == "next_true_gains" else frame]
            replaced = {}
            for v, record in records.items():
                truth = labels.get(v, self.labels[frame][v])
                prediction = dict(record["shared_prediction"])
                if variant != "true_interference":
                    prediction["gain"] = truth["gain"]
                if variant != "true_desired":
                    prediction["interference"] = truth["interference"]
                replaced[v] = dict(record, shared_prediction=prediction)
            records = replaced
        if variant in ("legacy_load", "legacy_fixed", "legacy_all"):
            legacy_load = np.clip(c["legacy_rb"] / capacities, 0.0, 1.5)
            if variant in ("legacy_load", "legacy_all"):
                load = legacy_load
            if variant in ("legacy_fixed", "legacy_all"):
                allocated = fixed_allocation(args, c["learners"], c["backlog"],
                    c["serving_gain"], c["no_bf_gain"], legacy_load, c["legacy_rb"], config)
        if variant.startswith("report_"):
            # Deliberately drop environment fields before the deployable estimator.
            public = {v: dict(pos=r["pos"], shared_prediction={
                k: r["shared_prediction"][k] for k in ("gain", "interference")})
                for v, r in records.items()}
            report_load, report_fixed = report_estimates(args, public, c["learners"],
                c["backlog"], config, c["macro_loc"])
            if variant in ("report_load", "report_both"):
                load = report_load
            if variant in ("report_fixed", "report_both"):
                allocated = report_fixed
        self.diagnostics.append(dict(frame=float(c["frame"]), input_load=np.asarray(load).tolist(),
            original_load=np.asarray(c["load"]).tolist(),
            allocated_sum=float(sum(allocated.values())),
            original_allocated_sum=float(sum(c["allocated_rb"].values()))))
        return dict(records=records, load=load, allocated_rb=allocated, config=config)


def evaluate(variants, gpu, actor, end):
    torch.set_num_threads(1)
    single_thread_solvers()
    timeline = temporal_slice(read_pickle(CACHE / "test_prepared.pkl"), 800, end)
    args = paper_args(13e6)
    args.device = torch.device("cpu")
    if args.random_factor_range4data_rate != 0:
        raise ValueError("The public-state estimator here requires the fixed-rate diagnostic protocol")
    traffic = make_paired_traffic(args, timeline, 1)
    checkpoint = CHECKPOINTS[actor]
    policy = OMAPPPolicy.load(str(checkpoint))
    labels = oracle_labels(timeline, policy.config) if any("true" in v or v == "legacy_all" for v in variants) else None
    for variant in variants:
        if actor == "legacy" and variant != "baseline":
            raise ValueError("Keep the original actor only as an external reference")
        name = f"{actor}_{variant}_rate13_seed1_end{end:g}_cuda{gpu}"
        path = OUTPUT / "runs" / f"{name}.json"
        if path.exists():
            print("REUSE", name, flush=True)
            continue
        print("START", name, flush=True)
        started = time.monotonic()
        hook = InputAblation(variant, labels) if actor != "legacy" else None
        progress = FrameProgress(name, len(timeline) - 1)
        result = run_sim_o_mappo(args, MICRO_BS_LOCATIONS, timeline, policy,
            seed=1, prt=False, rician_fading=True, optimizer_solver="milp",
            traffic_trace=traffic, ho_interruption_ms=10, paired_fading_seed=1,
            physics_device=f"cuda:{gpu}", progress_callback=progress.tick,
            optimizer_input_hook=hook)
        metrics, raw = metric_arrays(args, timeline, result)
        metrics["optimizer_failures"] = int(np.sum(result.optimizer_failure_record))
        metrics["mean_optimizer_overflow_rb"] = float(np.mean(result.optimizer_overflow_record[2:]))
        (OUTPUT / "raw").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUTPUT / "raw" / f"{name}.npz", **raw)
        row = dict(actor=actor, variant=variant, rate_mbps=13, seed=1,
            interval=[800, end], ho_interruption_ms=10, physics_gpu=gpu,
            traffic_sha256=traffic["sha256"], checkpoint_sha256=digest(checkpoint),
            checkpoint_path=str(checkpoint), metrics=metrics,
            elapsed_s=time.monotonic()-started,
            intervention="optimizer inputs only; actor weights and observation mapping unchanged",
            information_class="reported_or_public" if variant.startswith("report_") or variant == "baseline" and actor != "legacy" else "privileged_diagnostic",
            next_truth_fallback="current truth for departing vehicles and last frame" if variant == "next_true_gains" else None)
        write_json(path, row)
        if hook:
            write_json(OUTPUT / "diagnostics" / f"{name}.json", hook.diagnostics)
        print("DONE", name, json.dumps(metrics), flush=True)


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted((OUTPUT / "runs").glob("*.json"))]
    rows = [r for r in rows if r["interval"] == [800, 830]]
    if not rows:
        raise RuntimeError("No complete results")
    assert len({r["traffic_sha256"] for r in rows}) == 1
    for actor in CHECKPOINTS:
        assert len({r["checkpoint_sha256"] for r in rows if r["actor"] == actor}) <= 1
    reference = json.loads((ROOT / "experiment/results/o_mappo_gain_report_20260917/comparison_summary.json").read_text())
    ref = [r for r in reference["runs"] if r["rate_mbps"] == 13 and r["seed"] == 1]
    assert {r["traffic_sha256"] for r in ref} == {rows[0]["traffic_sha256"]}
    checks = []
    for row in rows:
        if row["variant"] != "baseline":
            continue
        old = next(r for r in ref if r["method"] == row["actor"])
        for key, value in old["metrics"].items():
            np.testing.assert_allclose(row["metrics"][key], value, rtol=1e-12, atol=1e-12)
        checks.append(row["actor"])
    write_json(OUTPUT / "summary.json", dict(runs=rows, references=ref,
        baseline_reproduction_passed=checks, scope="13 Mbps, seed 1 only; frozen checkpoints; exploratory diagnosis"))
    lines = ["13 Mbps, seed 1, 800--830 s, 10 ms HO interruption.", "",
        "| Actor | Optimizer intervention | P (W) | U (%) | mean proxy (ms) | p99 proxy (ms) | macro (%) | HO/vehicle/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["metrics"]
        lines.append("| " + " | ".join([row["actor"], row["variant"]] + [f"{m[k]:.5f}" for k in (
            "power_w", "violation_percent", "mean_proxy_ms", "p99_proxy_ms", "macro_association_percent", "handovers_per_vehicle_s")]) + " |")
    (OUTPUT / "result_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="baseline")
    p.add_argument("--gpu", type=int, default=4)
    p.add_argument("--actor", choices=CHECKPOINTS, default="gain_report")
    p.add_argument("--end", type=float, default=830)
    p.add_argument("--summarize", action="store_true")
    cli = p.parse_args()
    if cli.summarize:
        summarize()
    else:
        variants = cli.variants.split(",")
        if not set(variants) <= set(VARIANTS):
            p.error("Unknown intervention")
        evaluate(variants, cli.gpu, cli.actor, cli.end)


if __name__ == "__main__":
    main()
