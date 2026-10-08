#!/usr/bin/env python3
"""Plan-first, file-by-file cleanup with explicit paths and protected assets.

No recursive deletion and no automatic Git staging. A plan is a record of file
identities, not a wildcard to be expanded at deletion time. apply requires the
exact SHA-256 of a reviewed plan; all targets are validated before any unlink.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/paper_r1_assets.json"
BUILD_SUFFIXES = (".aux", ".blg", ".fls", ".fdb_latexmk", ".log", ".out", ".synctex.gz", ".xdv")
DATA_ROOTS = ("prepared_dataset", "data4sim", "sionna_result", "sumo_data")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def tracked_files(root):
    result = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return set(result.decode().rstrip("\0").split("\0"))


def checked_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(p in ("..", ".") for p in rel.parts):
        raise ValueError(f"Not an explicit repository-relative path: {relative}")
    path = root.joinpath(*rel.parts)
    for node in (path, *path.parents):
        if node == root:
            break
        if node.is_symlink():
            raise ValueError(f"Symlink target is not eligible: {relative}")
    if not path.is_relative_to(root):
        raise ValueError(relative)
    return path


def identity(path):
    s = path.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f"Only regular files may be deleted: {path}")
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, allocated_bytes=s.st_blocks * 512)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def protections(root, manifest):
    """Honor dataset references in formal and upstream experiment metadata."""
    result = set(manifest["retained_datasets"])
    evidence = list(manifest["formal_results"].values()) + manifest["upstream_results_to_preserve"]
    pattern = re.compile(r'"([^"\n]*(?:prepared_dataset|data4sim|sionna_result|sumo_data)/[^"\n]*)"')
    for directory in evidence:
        print(f"Checking metadata references: {directory}", flush=True)
        for path in (root / directory).rglob("*.json"):
            if path.is_symlink() or path.stat().st_size > 10 * 2**20:
                continue
            # Read only serialized path strings, not millions of numeric list items.
            # Dataset filenames are ASCII; JSON decoding handles escaped prefixes.
            for encoded in pattern.findall(path.read_text()):
                candidate = json.loads('"' + encoded + '"')
                prefix = str(root) + "/"
                rel = candidate[len(prefix):] if candidate.startswith(prefix) else candidate.removeprefix("./")
                if len(PurePosixPath(rel).parts) == 2 and PurePosixPath(rel).parts[0] in DATA_ROOTS:
                    if (root / rel).is_file():
                        result.add(rel)
    return result


def category(relative, protected, tracked, manifest):
    if relative in protected or relative in tracked:
        return None
    parts = PurePosixPath(relative).parts
    # No deletion anywhere in an experimental result or frozen submission tree.
    if any(relative == p or relative.startswith(p + "/") for p in manifest["protected_trees"]):
        return None
    if len(parts) == 2:
        directory, name = parts
        if directory == "prepared_dataset" and name.endswith(".pkl"):
            return "obsolete_window_datasets"
        if directory == "data4sim" and name.startswith("lbd") and name.endswith(".pkl"):
            return "obsolete_simulation_inputs"
        if directory == "sionna_result" and name.startswith("trajectoryInfo_lbd") and name.endswith(".pkl"):
            return "obsolete_channel_configurations"
        if directory == "sumo_data" and re.fullmatch(r"trajectory_Lbd\d+\.\d+\.csv", name):
            return "obsolete_mobility_configurations"
    if "__pycache__" in parts and relative.endswith(".pyc"):
        return "python_bytecode"
    if parts[0] in ("latexCodes", "response_letter") and relative.endswith(BUILD_SUFFIXES):
        return "latex_build_files"
    return None


def assert_no_jobs(root, targets):
    """Refuse relevant running programs or open target files, for current UID."""
    problems = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            # /proc may expose host PIDs while getpid() returns a namespace PID.
            if proc.samefile("/proc/self"):
                continue
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            args = (proc / "cmdline").read_bytes().split(b"\0")
            cwd = (proc / "cwd").resolve()
            program = comm.startswith("python") or comm in ("sumo", "pdflatex", "xelatex", "latexmk")
            local = cwd == root or root in cwd.parents
            local |= any(str(root).encode() in arg for arg in args)
            if program and local:
                problems.append(f"running project process {proc.name}: {comm}")
            for fd in (proc / "fd").iterdir():
                try:
                    if os.readlink(fd) in targets:
                        problems.append(f"open target in process {proc.name}")
                except (OSError, PermissionError):
                    pass
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if problems:
        raise RuntimeError("Cleanup refused: " + "; ".join(problems))


def validate_targets(root, plan, manifest, protected, tracked):
    targets = []
    for row in plan["files"]:
        path = checked_path(root, row["path"])
        if category(row["path"], protected, tracked, manifest) != row["category"]:
            raise ValueError(f"Target is not eligible: {row['path']}")
        if identity(path) != row["identity"]:
            raise ValueError(f"Target changed after planning: {row['path']}")
        targets.append(path)
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate cleanup targets")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "verify"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--confirm-sha256")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    for name in manifest["retained_datasets"] + manifest["entrypoints"]:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Required retained asset missing: {name}")
    if args.command == "plan":
        protected = protections(ROOT, manifest)
        tracked = tracked_files(ROOT)
        candidates = []
        for directory in DATA_ROOTS:
            candidates.extend(p for p in (ROOT / directory).iterdir() if p.is_file())
        for directory in ("latexCodes", "response_letter", "utils", "__pycache__"):
            candidates.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
        rows = []
        for path in sorted(set(candidates)):
            relative = str(path.relative_to(ROOT))
            kind = category(relative, protected, tracked, manifest)
            if kind:
                checked_path(ROOT, relative)
                rows.append(dict(path=relative, category=kind, identity=identity(path)))
        counts = collections.Counter(r["category"] for r in rows)
        totals = {kind: sum(r["identity"]["allocated_bytes"] for r in rows if r["category"] == kind)
                  for kind in counts}
        plan = dict(version=1, root=str(ROOT), created_unix=time.time(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                    manifest_sha256=digest(MANIFEST), protected_datasets=sorted(protected),
                    retained_identities={p: identity(ROOT/p) for p in sorted(protected)},
                    counts=dict(counts), allocated_bytes_by_category=totals, files=rows,
                    warning="Deleted untracked datasets are not recoverable through Git. Existing current data, all experiment results and all models are excluded.")
        write_new(args.plan, plan)
        print(json.dumps(dict(files=len(rows), counts=dict(counts),
                              estimated_GiB={k:round(v/2**30,3) for k,v in totals.items()},
                              plan_sha256=digest(args.plan)), indent=2))
        return
    plan = json.loads(args.plan.read_text())
    if plan["root"] != str(ROOT) or plan["manifest_sha256"] != digest(MANIFEST):
        raise ValueError("Root or protection manifest changed")
    for name, expected in plan["retained_identities"].items():
        if identity(ROOT / name) != expected:
            raise ValueError(f"Protected dataset changed: {name}")
    if args.command == "verify":
        remaining = [r["path"] for r in plan["files"] if (ROOT/r["path"]).exists()]
        if remaining:
            raise ValueError(f"Cleanup incomplete: {len(remaining)} targets remain")
        print(f"PASS: {len(plan['files'])} targets absent; {len(plan['retained_identities'])} protected dataset identities unchanged")
        return
    if args.confirm_sha256 != digest(args.plan):
        raise ValueError("Supply the exact SHA-256 of the reviewed plan")
    protected = protections(ROOT, manifest)
    targets = validate_targets(ROOT, plan, manifest, protected, tracked_files(ROOT))
    assert_no_jobs(ROOT, {str(p) for p in targets})
    receipt = args.plan.with_suffix(".deleted.jsonl")
    with receipt.open("x", buffering=1) as stream:
        for path, row in zip(targets, plan["files"]):
            if identity(path) != row["identity"]:
                raise ValueError(f"Target changed during cleanup: {path}")
            path.unlink()
            stream.write(json.dumps(dict(path=row["path"], category=row["category"],
                                         removed_unix=time.time(), identity=row["identity"]))+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"Deleted {len(targets)} individually verified files; receipt: {receipt}")


if __name__ == "__main__":
    main()
