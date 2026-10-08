#!/usr/bin/env python3
"""Audited, explicit round-2 cleanup; never modify scientific source or results.

plan records existing provenance hashes and snapshots all other experiment files.
Only purported duplicate caches require full binary hashing; old raw-result
hashes are copied from their retained metadata, not represented as revalidated.
apply requires the reviewed plan hash, validates all targets, then unlinks files
individually. No recursive removal, model deletion or automatic Git staging.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.workspace_maintenance import (
    assert_no_jobs, checked_path, digest, identity, tracked_files,
)

POLICY = ROOT / "configs/experiment_cleanup_round2.json"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


def under(name, directories):
    return any(name == d or name.startswith(d + "/") for d in directories)


def protected_paths(policy):
    assets = read(ROOT / policy["asset_manifest"])
    trees = list(assets["formal_results"].values()) + assets["upstream_results_to_preserve"]
    trees += policy["additional_protected_trees"]
    references = set(assets["retained_datasets"])
    pattern = re.compile(r'"([^"\n]*experiment/results[^"\n]*)"')
    evidence = {}
    # Scan paths in formal/upstream metadata without traversing large numeric arrays.
    for directory in trees:
        print("Inspecting dependency metadata:", directory, flush=True)
        for path in (ROOT / directory).rglob("*.json"):
            if path.is_symlink() or path.stat().st_size > 10 * 2**20:
                continue
            for encoded in pattern.findall(path.read_text()):
                if len(encoded) > 2000:
                    continue
                candidate = json.loads('"' + encoded + '"')
                candidate = candidate[candidate.index("experiment/results"):]
                if (ROOT / candidate).exists():
                    checked_path(ROOT, candidate)
                    references.add(candidate)
                    evidence.setdefault(candidate, str(path.relative_to(ROOT)))
    return trees, references, evidence


def raw_evidence(root, name, policy):
    """Only named superseded trees, raw/sample NPZ, with saved case metrics."""
    p = Path(name)
    if not under(name, policy["superseded_raw_trees"]) or p.suffix != ".npz":
        return None
    if p.parent.name not in ("raw", "samples"):
        return None
    evidence = p.parent.parent / "runs" / (p.stem + ".json")
    if not (root / evidence).is_file():
        return None
    row = read(root / evidence)
    metrics = row.get("metrics", row.get("manuscript_metrics", {}))
    required = ({"power_w", "violation_percent"},
                {"average_system_power_w", "queue_violation_percent"})
    if not any(keys <= metrics.keys() for keys in required):
        return None
    field = "samples_sha256" if p.parent.name == "samples" else "raw_sha256"
    return dict(evidence=str(evidence), expected_sha256=row.get(field),
                category="superseded_raw_results")


def eligible(root, name, policy, trees, references, tracked):
    if name in tracked or under(name, trees) or under(name, references):
        return None
    checked_path(root, name)
    if Path(name).suffix in (".pt", ".pth") or "policy" in Path(name).name:
        return None
    result = raw_evidence(root, name, policy)
    if result:
        return result
    for row in policy["obsolete_caches"]:
        if name == row["path"]:
            value = read(root / row["evidence"])
            for key in row["hash_keys"]:
                value = value[key]
            return dict(evidence=row["evidence"], expected_sha256=value,
                        category="obsolete_prediction_or_training_cache")
    dup = policy["duplicate_validation_caches"]
    if name in dup["candidates"]:
        return dict(evidence=dup["retained"], replacement=dup["retained"],
                    category="byte_identical_validation_cache")
    return None


def reclaimed_bytes(rows):
    """Count an inode once, and only if all its hard links will be removed."""
    groups = collections.defaultdict(list)
    for r in rows:
        s = r["identity"]
        groups[(s["device"], s["inode"])].append(r)
    return sum(g[0]["identity"]["allocated_bytes"] for g in groups.values()
               if len(g) == g[0]["links"])


def snapshot_remaining(experiment_files, targets):
    rows = {}
    for path in experiment_files:
        name = str(path.relative_to(ROOT))
        if name in targets:
            continue
        if path.is_symlink():
            rows[name] = dict(symlink=os.readlink(path))
        else:
            row = dict(identity=identity(path))
            if path.suffix in (".json", ".csv", ".md", ".py", ".pt", ".pth"):
                row["sha256"] = digest(path)
            rows[name] = row
    return rows


def verify_snapshot(snapshot):
    for name, row in snapshot.items():
        path = ROOT / name
        if "symlink" in row:
            if not path.is_symlink() or os.readlink(path) != row["symlink"]:
                raise ValueError(f"Retained symlink changed: {name}")
        else:
            checked_path(ROOT, name)
            if identity(path) != row["identity"]:
                raise ValueError(f"Retained file changed: {name}")
            if "sha256" in row and digest(path) != row["sha256"]:
                raise ValueError(f"Retained contents changed: {name}")


def plan(args, policy):
    trees, references, evidence = protected_paths(policy)
    tracked = tracked_files(ROOT)
    files = sorted(p for p in (ROOT / "experiment").rglob("*")
                   if p.is_file() or p.is_symlink())
    candidates = []
    sha_by_inode = {}
    for path in files:
        if path.is_symlink():
            continue
        name = str(path.relative_to(ROOT))
        record = eligible(ROOT, name, policy, trees, references, tracked)
        if record is None:
            continue
        before = identity(path)
        recorded_sha = record.pop("expected_sha256", None)
        if "replacement" in record:
            print("Checking duplicate cache byte hashes:", name, flush=True)
            for item in (path, ROOT / record["replacement"]):
                st = item.stat()
                key = (st.st_dev, st.st_ino)
                if key not in sha_by_inode:
                    sha_by_inode[key] = digest(item)
            recorded_sha = sha_by_inode[(before["device"], before["inode"])]
            rep = (ROOT / record["replacement"]).stat()
            evidence_sha = sha_by_inode[(rep.st_dev, rep.st_ino)]
            if recorded_sha != evidence_sha:
                raise ValueError(f"Purported duplicate differs: {name}")
            verification = "byte_sha256_verified"
        else:
            evidence_sha = digest(ROOT / record["evidence"])
            verification = "metadata_provenance_and_file_identity"
        if identity(path) != before:
            raise ValueError(f"File changed while hashing: {name}")
        candidates.append(dict(path=name, identity=before, links=path.stat().st_nlink,
                               recorded_sha256=recorded_sha, verification=verification,
                               evidence_sha256=evidence_sha, **record))
        if len(candidates) % 100 == 0:
            print(f"Checked metadata and file identities for {len(candidates)} candidates", flush=True)
    names = {r["path"] for r in candidates}
    if not candidates or len(names) != len(candidates):
        raise ValueError("Empty or duplicate candidate list")
    # A target may not remove the evidence or replacement for another target.
    if any(r["evidence"] in names for r in candidates):
        raise ValueError("Evidence/replacement included in deletion targets")
    snap = ROOT / "archive/maintenance" / args.directory.name / "retained_snapshot.json"
    snapshot = snapshot_remaining(files, names)
    write_new(snap, snapshot)
    parts = {}
    for start in range(0, len(candidates), 400):
        path = args.directory / f"targets_{start//400+1:02d}.json"
        write_new(path, candidates[start:start+400])
        parts[path.name] = digest(path)
    protections = args.directory / "dependency_references.json"
    write_new(protections, dict(trees=trees, existing_references=evidence))
    totals = collections.Counter()
    for row in candidates:
        totals[row["category"]] += 1
    index = dict(root=str(ROOT), created_unix=time.time(),
                 git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                 policy_sha256=digest(POLICY), asset_manifest_sha256=digest(ROOT/policy["asset_manifest"]),
                 parts=parts, snapshot=str(snap.relative_to(ROOT)), snapshot_sha256=digest(snap),
                 dependencies_sha256=digest(protections),
                 files=len(candidates), counts=dict(totals),
                 retained_files=len(snapshot), retained_content_hashes=sum("sha256" in r for r in snapshot.values()),
                 uniquely_reclaimable_allocated_bytes=reclaimed_bytes(candidates),
                 warning="Untracked old raw arrays will be deleted, not backed up. Summaries and all models remain. Old per-slot analyses require regeneration; current paper experiments are excluded.")
    write_new(args.directory / "index.json", index)
    print(json.dumps(dict(files=index["files"], counts=index["counts"],
                          GiB=index["uniquely_reclaimable_allocated_bytes"]/2**30,
                          retained_files=len(snapshot), index_sha256=digest(args.directory/"index.json")), indent=2))


def load_plan(args, policy):
    index_path = args.directory / "index.json"
    index = read(index_path)
    if index["root"] != str(ROOT) or index["policy_sha256"] != digest(POLICY):
        raise ValueError("Plan root/policy changed")
    if index["asset_manifest_sha256"] != digest(ROOT / policy["asset_manifest"]):
        raise ValueError("Asset manifest changed")
    snap = checked_path(ROOT, index["snapshot"])
    if digest(snap) != index["snapshot_sha256"]:
        raise ValueError("Retained-file snapshot changed")
    dep = args.directory / "dependency_references.json"
    if digest(dep) != index["dependencies_sha256"]:
        raise ValueError("Dependency record changed")
    rows = []
    for name, expected in index["parts"].items():
        if Path(name).name != name or digest(args.directory / name) != expected:
            raise ValueError("Target manifest changed")
        rows.extend(read(args.directory / name))
    if len(rows) != index["files"] or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Wrong/duplicate targets")
    verify_snapshot(read(snap))
    return index, rows


def apply(args, policy, index, rows):
    if args.confirm_sha256 != digest(args.directory / "index.json"):
        raise ValueError("Supply the exact reviewed index SHA-256")
    trees, references, _ = protected_paths(policy)
    tracked = tracked_files(ROOT)
    targets = []
    for row in rows:
        name = row["path"]
        spec = eligible(ROOT, name, policy, trees, references, tracked)
        if spec is None or spec["category"] != row["category"] or spec["evidence"] != row["evidence"]:
            raise ValueError(f"No longer eligible: {name}")
        path = checked_path(ROOT, name)
        if identity(path) != row["identity"] or path.stat().st_nlink != row["links"]:
            raise ValueError(f"Deletion target changed: {name}")
        if digest(ROOT / row["evidence"]) != row["evidence_sha256"]:
            raise ValueError(f"Evidence/replacement changed: {name}")
        targets.append(path)
    assert_no_jobs(ROOT, {str(p) for p in targets})
    receipt = args.directory / "deleted.jsonl"
    with receipt.open("x", buffering=1) as stream:
        for number, (path, row) in enumerate(zip(targets, rows), 1):
            if identity(path) != row["identity"]:
                raise ValueError(f"Deletion target changed during cleanup: {path}")
            path.unlink()
            stream.write(json.dumps(dict(path=row["path"], recorded_sha256=row["recorded_sha256"], removed_unix=time.time()))+"\n")
            if number % 100 == 0:
                stream.flush()
                os.fsync(stream.fileno())
                print(f"Deleted {number}/{len(rows)} explicit targets", flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"Removed {len(rows)} files; all models and metadata retained", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "apply", "verify"))
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--confirm-sha256")
    args = p.parse_args()
    policy = read(POLICY)
    if args.command == "plan":
        plan(args, policy)
        return
    index, rows = load_plan(args, policy)
    if args.command == "apply":
        apply(args, policy, index, rows)
    else:
        if any((ROOT / row["path"]).exists() for row in rows):
            raise ValueError("Some cleanup targets remain")
        print(f"PASS: {len(rows)} targets absent; {index['retained_files']} retained experiment files unchanged; {index['retained_content_hashes']} content hashes verified")


if __name__ == "__main__":
    main()
