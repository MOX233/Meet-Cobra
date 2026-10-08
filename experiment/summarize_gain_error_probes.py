"""Count probes per actual beam search from saved results; no simulation changes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize_case(root, name, warmup):
    paths = [root / "runs" / f"{name}.json",
             root / "diagnostics" / f"{name}.json", root / "raw" / f"{name}.npz"]
    run = json.loads(paths[0].read_text())
    frames = json.loads(paths[1].read_text())["ho"]
    with np.load(paths[2], allow_pickle=False) as raw:
        mean_probes = raw["pilots"].copy()
    assert len(frames) == len(mean_probes) == run["frames"]
    total_probes = total_searches = 0
    for frame, mean in zip(frames[warmup:], mean_probes[warmup:]):
        association = frame["association"]
        vehicle_slots = frame["active_vehicle_slots"]
        slots, remainder = divmod(vehicle_slots, len(association))
        assert remainder == 0
        switched = frame["switched"]
        interrupted_slots = 0
        if switched:
            interrupted_slots, remainder = divmod(frame["blocked_vehicle_slots"], len(switched))
            assert remainder == 0
        else:
            assert frame["blocked_vehicle_slots"] == 0
        micro_vehicles = sum(bs > 0 for bs in association.values())
        micro_handovers = sum(association[str(v)] > 0 for v in switched)
        searches = slots * micro_vehicles - interrupted_slots * micro_handovers
        probes = int(round(mean * vehicle_slots))
        np.testing.assert_allclose(probes, mean * vehicle_slots, rtol=0, atol=1e-6)
        assert searches <= probes <= 5 * searches
        total_probes += probes
        total_searches += searches
    assert total_searches > 0
    return dict(case=name, seed=run["seed"], sigma_db=run["sigma_db"],
                probes=total_probes, searches=total_searches,
                mean_probes_per_search=total_probes / total_searches,
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "experiment/results/gain_error_sensitivity_20260929")
    parser.add_argument("--rate", type=int, default=29)
    parser.add_argument("--kind", choices=["desired", "interfering"], default="desired")
    parser.add_argument("--sigmas", type=int, nargs="+", default=[0, 5])
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    rows, aggregate = [], []
    for sigma in args.sigmas:
        prefix = "control" if sigma == 0 else f"{args.kind}_sigma{sigma}"
        cases = [summarize_case(args.root, f"{prefix}_rate{args.rate}_seed{seed}",
                                protocol["warmup_frames"]) for seed in protocol["seeds"]]
        values = [c["mean_probes_per_search"] for c in cases]
        rows.extend(cases)
        aggregate.append(dict(sigma_db=sigma, seeds=len(cases),
                              mean=float(np.mean(values)), minimum=min(values), maximum=max(values)))
    result = dict(kind=args.kind, rate_mbps=args.rate, warmup_frames=protocol["warmup_frames"],
                  metric="Total probes / actual beam searches within each run, then mean across seeds.",
                  denominator="One search per micro-BS-associated vehicle per slot outside HO interruption.",
                  original_metrics_unchanged=True, per_seed=rows, aggregate=aggregate,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = args.root / f"probes_per_search_{args.kind}_rate{args.rate}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), aggregate=aggregate), indent=2))


if __name__ == "__main__":
    main()
