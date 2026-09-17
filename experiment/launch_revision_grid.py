#!/usr/bin/env python3
"""Bounded detached runner: preflight gate, full grid, audit, preview exports."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiment/results/revision_fig5_8_20260917"
DEFAULT_METHODS = "meet_cobra,oracle_mc,wo_gap_ho,wo_pet_bf,wo_otr_ra,o_mappo,mts"


def write_json(path, data):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data,indent=2)+"\n")
    os.replace(tmp,path)


def status(phase, **kwargs):
    row = dict(phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
    write_json(OUTPUT/"pipeline_status.json",row)
    print(json.dumps(row),flush=True)


def run(args):
    methods = args.methods.split(",")
    with (OUTPUT/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.wait_preflight:
                expected = [OUTPUT/"runs"/f"{m}_rate{r}_seed1.json" for m in methods for r in (1,19,35)]
                start = time.monotonic()
                while True:
                    remaining = sum(not p.exists() for p in expected)
                    if not remaining:
                        break
                    result = OUTPUT/"queue_last_result.json"
                    if result.exists():
                        report = json.loads(result.read_text())
                        if report["rates"] == [1,19,35] and report["seeds"] == [1] and report["failures"]:
                            raise RuntimeError("Preflight queue reported failures; inspect individual logs")
                    if time.monotonic()-start > 7200:
                        raise RuntimeError("Preflight did not finish within two hours")
                    status("waiting_for_preflight",remaining=remaining,expected=len(expected))
                    time.sleep(30)
                status("auditing_preflight")
                subprocess.run([sys.executable,str(ROOT/"experiment/summarize_revision_grid.py"),"--methods",args.methods,"--rates","1,19,35","--seeds","1","--check-raw"],cwd=ROOT,check=True)
                summary = json.loads((OUTPUT/"aggregate/summary_preflight.json").read_text())
                assert len(summary["rows"]) == 3*len(methods) and summary["raw_checked"]
                assert summary["methods"] == methods and summary["rates"] == [1,19,35]
            status("running_full_grid",methods=methods,expected=90*len(methods),gpus=args.gpus)
            cmd = [sys.executable,"-u",str(ROOT/"experiment/revision_grid.py"),"queue","--methods",args.methods,
                "--rates",",".join(map(str,range(1,36,2))),"--seeds","1,2,3,4,5","--gpus",args.gpus]
            with (OUTPUT/"full_queue.log").open("a") as log:
                process = subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED="0"))
                while process.poll() is None:
                    count = sum((OUTPUT/"runs"/f"{m}_rate{r}_seed{s}.json").exists() for m in methods for r in range(1,36,2) for s in (1,2,3,4,5))
                    status("running_full_grid",completed=count,expected=90*len(methods),queue_pid=process.pid,methods=methods)
                    time.sleep(30)
                if process.returncode:
                    raise RuntimeError(f"Full queue exited with code {process.returncode}; successful cases retained")
            status("auditing_full_grid")
            subprocess.run([sys.executable,str(ROOT/"experiment/summarize_revision_grid.py"),"--methods",args.methods,"--check-raw","--plot"],cwd=ROOT,check=True)
            final = json.loads((OUTPUT/"aggregate/summary_full.json").read_text())
            assert len(final["rows"]) == 18*len(methods) and final["raw_checked"]
            assert final["methods"] == methods and final["rates"] == list(range(1,36,2))
            assert final["seeds"] == [1,2,3,4,5]
            status("completed_selected_methods",completed=90*len(methods),methods=methods,
                reactive_included="reactive_obra" in methods,oracle_cr_lb_included=False,
                note="Preview outputs only; manuscript and old figures unchanged")
        except Exception:
            status("failed",error=traceback.format_exc())
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--wait-preflight",action="store_true")
    parser.add_argument("--methods",default=DEFAULT_METHODS)
    parser.add_argument("--gpus",default="0,1,2,3,4,5,6")
    args = parser.parse_args()
    if args.worker:
        run(args)
    else:
        # Refuse a duplicate manager rather than dispatching duplicate cases.
        with (OUTPUT/"pipeline.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        cmd = [sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--methods",args.methods,"--gpus",args.gpus]
        if args.wait_preflight:
            cmd.append("--wait-preflight")
        with (OUTPUT/"pipeline.log").open("a") as log:
            child = subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,env=dict(os.environ,PYTHONHASHSEED="0"))
        manifest = dict(pid=child.pid,command=cmd,launched_utc=datetime.now(timezone.utc).isoformat())
        write_json(OUTPUT/"pipeline_launch.json",manifest)
        print(json.dumps(manifest,indent=2),flush=True)
