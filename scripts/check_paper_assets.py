#!/usr/bin/env python3
"""Read-only checks of current paper dependencies and frozen source/model hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def check():
    assets = json.loads((ROOT/"configs/paper_r1_assets.json").read_text())
    paths = assets["retained_datasets"] + assets["entrypoints"]
    paths += list(assets["formal_results"].values()) + assets["upstream_results_to_preserve"]
    paths += list(assets["models"].values())
    for name in paths:
        if not (ROOT/name).exists():
            raise FileNotFoundError(name)
    grid = ROOT/assets["formal_results"]["main_system_evaluation"]/"grid/protocol.json"
    protocol = json.loads(grid.read_text())
    hashes = {}
    frozen_sources_in_git = {}
    for name, expected in protocol["code_sha256"].items():
        actual = digest(ROOT/name)
        submitted = subprocess.check_output(["git", "show", assets["submission_tag"]+":"+name], cwd=ROOT)
        if actual != hashlib.sha256(submitted).hexdigest():
            raise ValueError(f"Scientific source differs from submitted snapshot: {name}")
        if actual != expected:
            # Some old grid modules were subsequently extended for the final RL
            # baseline. Verify their historical source rather than bypass hashes.
            historical = subprocess.check_output(["git", "show", protocol["git_revision"]+":"+name], cwd=ROOT)
            if hashlib.sha256(historical).hexdigest() != expected:
                raise ValueError(f"Frozen source unavailable from recorded Git revision: {name}")
            frozen_sources_in_git[name] = dict(commit=protocol["git_revision"], sha256=expected)
        hashes[name] = actual
    for row in protocol["cache_manifest"]["models"].values():
        path = Path(row["checkpoint"])
        actual = digest(path)
        if actual != row["sha256"]:
            raise ValueError(f"Checkpoint changed: {path}")
        hashes[str(path.relative_to(ROOT))] = actual
    for task in ("beam", "desired_gain", "interfering_gain"):
        a = ROOT/assets["models"]["nn_bundle"]/task/"best.pth"
        b = ROOT/assets["models"]["nn_timing_bundle"]/task/"best.pth"
        if digest(a) != digest(b):
            raise ValueError(f"Inference/timing model mismatch: {task}")
    om = ROOT/assets["formal_results"]["o_mappo"]
    selected = json.loads((om/"selection.json").read_text())["selected"]
    analysis = json.loads((om/"analysis.json").read_text())
    actual = digest(selected["policy"])
    if actual != analysis["selected_sha256"]:
        raise ValueError("Selected O-MAPPO checkpoint changed")
    hashes[str(Path(selected["policy"]).relative_to(ROOT))] = actual
    for name in ("o_mappo_initial",):
        hashes[assets["models"][name]] = digest(ROOT/assets["models"][name])
    subprocess.run(["sha256sum", "-c", "SUBMISSION_SHA256SUMS"],
                   cwd=ROOT/"latexCodes/revision1", check=True)
    return dict(passed=True, checked_paths=len(paths), verified_sha256=hashes,
                historical_frozen_sources_verified_in_git=frozen_sources_in_git,
                submission_checksums_verified=True,
                note="Large retained datasets are checked for existence; cleanup separately verifies their file identities. This is not a new simulation or full numeric re-audit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(f"PASS: {report['checked_paths']} asset paths; {len(report['verified_sha256'])} source/model hashes; {len(report['historical_frozen_sources_verified_in_git'])} older source versions verified in Git; submitted files unchanged")


if __name__ == "__main__":
    main()
