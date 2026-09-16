#!/usr/bin/env python3
"""Launch the bounded multi-seed queue as a detached server process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parallel", type=int, default=80)
    args = parser.parse_args()
    results = ROOT / "experiment/results_multiseed"
    results.mkdir(parents=True, exist_ok=True)
    log_path = results / "queue_manager.log"
    log_handle = log_path.open("ab", buffering=0)
    command = [
        sys.executable,
        str(ROOT / "experiment/run_multiseed_queue.py"),
        "--max-parallel",
        str(args.max_parallel),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_handle.close()
    manifest = {
        "launched_unix_s": time.time(),
        "pid": process.pid,
        "command": command,
        "log": str(log_path),
    }
    temporary = results / "queue_launch_manifest.json.{}.tmp".format(os.getpid())
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, results / "queue_launch_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
