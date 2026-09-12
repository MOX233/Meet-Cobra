#!/usr/bin/env python3
"""Direct paired comparison: paper window model vs stateful-TBPTT model."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.benchmark_nn_overhead import digest, json_write
from experiment.compare_stateful_prediction import save_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--tbptt-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old = np.load(args.paper_results / "predictions.npz")
    new = np.load(args.tbptt_results / "predictions.npz")
    identity = ("vehicle", "frame", "target_frame", "age", "segment", "true_beam",
                "true_desired_gain", "true_interfering_gain", "nonzero_link")
    for key in identity:
        np.testing.assert_array_equal(old[key], new[key])
    raw = {key: old[key] for key in identity}
    for target in ("beam", "desired_gain", "interfering_gain"):
        raw[f"window_{target}"] = old[f"window_{target}"]
        raw[f"stateful_{target}"] = new[f"stateful_{target}"]
    summary = save_summaries(args.output, raw, args.bootstrap_replicates)
    metadata = {
        "comparison_names": {"window": "paper_checkpoint_with_10_frame_sliding_window",
                             "stateful": "stateful_TBPTT_checkpoint_with_persistent_streaming_state"},
        "paper_results": str(args.paper_results.resolve()),
        "paper_predictions_sha256": digest(args.paper_results / "predictions.npz"),
        "tbptt_results": str(args.tbptt_results.resolve()),
        "tbptt_predictions_sha256": digest(args.tbptt_results / "predictions.npz"),
        "exact_sample_and_label_identity": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "primary_difference_direction": "stateful_TBPTT minus paper_sliding_window",
        "counts": summary["counts"],
    }
    json_write(args.output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
