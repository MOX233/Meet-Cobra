#!/usr/bin/env python3
"""Frozen prediction evaluation and speed-conditioned queue postprocessing.

No training, system simulation, data reassignment, or manuscript edits.
Validation reuses the existing stateful evaluator and captures its outputs.
"""
import argparse
import collections
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from experiment import train_stateful_tbptt as stateful
from experiment.benchmark_stateful_nn_overhead import load_selected_models
from experiment.vehicle_split import trajectory_indices
from utils.directional_service import beam_average_gain_db

REV = ROOT / "experiment/results/revision_directional_20260922"
MODELS = REV / "selected_models"
SPLIT = ROOT / "experiment/results/stateful_tbptt_unified_split_20260913/vehicle_split_seed20.npz"
AUDIT = ROOT / "experiment/results/mobility_audit_20260926"
GROUPS = ("[0,1)", "[1,20)", "[20,40)", ">=40")
TASKS = ("beam", "desired_gain", "interfering_gain")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2)+"\n")


def read(path):
    return json.loads(path.read_text())


def groups(speed):
    speed = np.asarray(speed)
    assert np.isfinite(speed).all() and np.min(speed) >= 0
    return np.digitize(speed, [1., 20., 40.], right=False)


def prediction_summary(raw, dataset):
    n = len(raw["speed_kmh"])
    rows = []
    assignment = groups(raw["speed_kmh"])
    true = raw["true_beam"]
    correct = raw["pred_beam"] == true[..., None]
    errors = {task: np.abs(raw["pred_"+task].astype(np.float64)-raw["true_"+task])
              for task in TASKS[1:]}
    for group in (-1, 0, 1, 2, 3):
        mask = np.ones(n, dtype=bool) if group == -1 else assignment == group
        assert mask.any()
        row = dict(dataset=dataset, speed_group="all" if group == -1 else GROUPS[group],
                   vehicle_frames=int(mask.sum()), links=int(mask.sum()*4),
                   top1_percent=float(correct[mask, :, 0].mean()*100),
                   top5_percent=float(correct[mask].any(-1).mean()*100),
                   desired_mae_db=float(errors["desired_gain"][mask].mean()),
                   interfering_mae_db=float(errors["interfering_gain"][mask].mean()))
        rows.append(row)
    # Group results must reconstitute each overall metric with link weighting.
    for key in ("top1_percent", "top5_percent", "desired_mae_db", "interfering_mae_db"):
        combined = sum(r[key]*r["links"] for r in rows[1:])/rows[0]["links"]
        np.testing.assert_allclose(combined, rows[0][key], atol=1e-10)
    return rows


def prepare(args):
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    _, inventory = load_selected_models(MODELS)
    inputs = [REV/"training_data.npz", SPLIT, REV/"test_predictions.pkl",
              AUDIT/"test_speed_alignment.csv", ROOT/"sumo_data/trajectory_Lbd1.00.csv"]
    inputs += [MODELS/t/"best.pth" for t in TASKS]
    checksums = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    assert checksums[str((REV/"training_data.npz").relative_to(ROOT))] == read(REV/"training_data.json")["output_sha256"]
    assert checksums[str((REV/"test_predictions.pkl").relative_to(ROOT))] == read(REV/"test_predictions.json")["cache_sha256"]
    assert checksums[str(SPLIT.relative_to(ROOT))] == read(AUDIT/"audit.json")["split_sha256"]
    source = [Path(__file__), ROOT/"experiment/train_stateful_tbptt.py",
              ROOT/"experiment/vehicle_split.py", ROOT/"utils/directional_service.py"]
    protocol = dict(
        scope="Frozen inference and existing-result postprocessing only", input_sha256=checksums,
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in source},
        models=inventory, speed_boundaries_kmh=[0, 1, 20, 40], speed_groups=list(GROUPS),
        grouping="Target-frame speed; group after full-trajectory stateful inference",
        validation=dict(samples=195915, batch_size=128, chunk_length=10, seed=20,
                        noise_seed=100020, device=args.device, threads=args.threads,
                        noise_power=1e-14, dtype="FP32",
                        reset="Trajectory start only; state carried across chunks",
                        note="CPU pilot RNG is not bit-identical to the earlier CUDA validation realization"),
        system=dict(methods=["meet_cobra", "oracle_mc"], rates=list(range(1,36,2)), seeds=[1,2,3],
                    warmup_frames=2, presentation_loads=[19,29],
                    u="Within-frame group vehicle-slot violation fraction, then mean across 298 frames",
                    percentiles="Within each run and speed group over vehicle-slot q/lambda, then mean across seeds",
                    uncertainty="Seed min/max; conditional on the same fixed mobility trace"),
        protected_files={str(p.relative_to(ROOT)):digest(p) for p in
                         [ROOT/"latexCodes/main_revision1.tex", ROOT/"response_letter/response_letter.tex"]},
        torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
    args.output.mkdir(parents=True)
    dump(args.output/"protocol.json", protocol)
    print("Prepared",args.output,flush=True)


def validate_protocol(args):
    p = read(args.output/"protocol.json")
    for path, h in p["code_sha256"].items():
        assert digest(ROOT/path) == h, path
    return p


def validation(args):
    p = validate_protocol(args)
    dest = args.output/"validation_predictions.npz"
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(p["validation"]["threads"])
    device = torch.device(p["validation"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no silent fallback")
    data = stateful.load_data(REV/"training_data.npz")
    _, indices, _, _ = trajectory_indices(data, SPLIT)
    offsets = data["offsets"]
    selected = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in indices])
    assert len(selected) == 195915
    inverse = np.full(len(data["beam"]), -1, dtype=np.int64)
    inverse[selected] = np.arange(len(selected))
    ids = np.repeat(data["vehicle_ids"], np.diff(offsets))[selected]
    target = np.rint(data["target_frame"][selected]*10).astype(np.int64)
    # Use the already audited legacy mapping, without changing identifiers.
    csv = pd.read_csv(ROOT/"sumo_data/trajectory_Lbd1.00.csv", header=None,
                      usecols=[0,1,5], names=["step","name","speed"])
    csv["vehicle"] = csv.name.str.replace("flow", "", regex=False).astype(float).astype(str)
    csv = csv.drop_duplicates(["step","vehicle"], keep="last").set_index(["step","vehicle"])
    speed = csv.reindex(pd.MultiIndex.from_arrays([target,ids])).speed.to_numpy()*3.6
    assert np.isfinite(speed).all()
    raw = dict(vehicle=ids, target_step=target, sample_index=selected, speed_kmh=speed,
               true_beam=data["beam"][selected], true_desired_gain=data["desired_gain"][selected],
               true_interfering_gain=data["interfering_gain"][selected])
    np.testing.assert_array_equal(np.bincount(groups(speed), minlength=4),[35311,17412,26780,116412])
    # Reconstruct the unchanged evaluator's mask order solely to attach IDs.
    schedule = []
    for group in stateful.trajectory_groups(indices, offsets, 128, False, np.random.default_rng(20)):
        starts, lengths = offsets[group], offsets[group+1]-offsets[group]
        for begin in range(0,int(lengths.max()),10):
            counts = np.clip(lengths-begin,0,10)
            flat = np.concatenate([np.arange(s+begin,s+begin+c) for s,c in zip(starts,counts)])
            if len(flat):
                schedule.append(inverse[flat])
    np.testing.assert_array_equal(np.sort(np.concatenate(schedule)),np.arange(195915))
    models, _ = load_selected_models(MODELS)
    evaluator = stateful.forward_valid
    metrics, timings = {}, {}
    for task in TASKS:
        started = time.monotonic()
        model = models[task].to(device)
        before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        output = np.empty((195915,4,5),dtype=np.int16) if task=="beam" else np.empty((195915,4),dtype=np.float32)
        pending = collections.deque(schedule)
        def capture(current_model, lstm_output, mask, current_task):
            result = evaluator(current_model,lstm_output,mask,current_task)
            positions = pending.popleft()
            assert result.shape[0] == len(positions)
            value = result.topk(5,-1,sorted=True).indices if task=="beam" else 20*(result-7)
            output[positions] = value.detach().cpu().numpy()
            if len(pending)%100 == 0:
                print(f"validation {task}: {len(schedule)-len(pending)}/{len(schedule)} chunks",flush=True)
            return result
        stateful.forward_valid = capture
        try:
            metrics[task] = stateful.run_epoch(model,task,data,indices,device,10,128,None,False,20,1.0,True,beam_topk_max=5)
        finally:
            stateful.forward_valid = evaluator
        assert not pending and metrics[task]["targets"]==195915*4
        assert metrics[task]["optimizer_steps"]==0
        for key, value in model.state_dict().items():
            assert torch.equal(value.cpu(),before[key]), (task,key)
        raw["pred_"+task] = output
        timings[task] = time.monotonic()-started
        print(task,metrics[task],"seconds",timings[task],flush=True)
    summary = prediction_summary(raw,"validation")
    np.testing.assert_allclose(summary[0]["top1_percent"],metrics["beam"]["top1_accuracy_pct"],atol=1e-10)
    np.testing.assert_allclose(summary[0]["top5_percent"],metrics["beam"]["topk_accuracy_pct"]["5"],atol=1e-10)
    # Stored dB denormalization and original normalized-loss accumulation can
    # differ by FP32 rounding; both must agree to much less than 0.001 dB.
    for t,key in [("desired_gain","desired_mae_db"),("interfering_gain","interfering_mae_db")]:
        np.testing.assert_allclose(summary[0][key],metrics[t]["mae_db"],atol=3e-5,rtol=0)
    np.savez_compressed(dest,**raw)
    pd.DataFrame(summary).to_csv(args.output/"validation_by_speed.csv",index=False)
    dump(args.output/"validation_checks.json",dict(metrics=metrics,timings_s=timings,
        models_and_buffers_unchanged=True, samples=195915, capture_reaggregates=True,
        predictions_sha256=digest(dest), original_evaluator_reused=True))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def cached_test(args):
    validate_protocol(args)
    dest=args.output/"test_predictions_by_sample.npz"
    if dest.exists(): raise FileExistsError(dest)
    with (REV/"test_predictions.pkl").open("rb") as f:
        cache=pickle.load(f)
    frames=sorted(cache)
    records=collections.defaultdict(list)
    skipped=0
    for fi, frame in enumerate(frames[1:]):
        if fi<2: continue
        previous=cache[frames[fi]]
        for v,r in cache[frame].items():
            if v not in previous:
                skipped+=1
                continue
            report=previous[v]["shared_prediction"]
            assert abs(report["target_frame"]-frame)<1e-7
            records["vehicle"].append(str(v))
            records["target_step"].append(int(round(frame*10)))
            records["speed_kmh"].append(r["v"]*3.6)
            records["pred_beam"].append(report["beam"])
            records["pred_desired_gain"].append(report["gain"])
            records["pred_interfering_gain"].append(report["interference"])
            records["true_beam"].append(r["best_beam_pair_idx"])
            records["true_desired_gain"].append(r["g_opt_beam"])
            records["true_interfering_gain"].append(beam_average_gain_db(r["h"]))
    raw={k:np.asarray(v) for k,v in records.items()}
    assert len(raw["vehicle"])==40591 and skipped==28
    summary=prediction_summary(raw,"test")
    np.savez_compressed(dest,**raw)
    pd.DataFrame(summary).to_csv(args.output/"test_by_speed.csv",index=False)
    dump(args.output/"test_checks.json",dict(samples=40591,excluded_new_vehicle_frames=28,
         target_alignment_checked=True,interference_label="beam-average with original -180 dB zero convention",
         predictions_sha256=digest(dest),nn_inference_runs=0))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


def queue_summary(q, frames, speed_group, rate):
    assert q.shape[1]==100 and np.isfinite(q).all() and (q>=0).all()
    keep=frames>=2
    result=[]
    violating=q>rate*1e6*.020
    for g in (-1,0,1,2,3):
        mask=keep & ((speed_group==g) if g>=0 else True)
        f=frames[mask]
        counts=np.bincount(f,minlength=300)
        assert np.all(counts[2:]>0)
        successes=np.bincount(f,weights=violating[mask].sum(axis=1),minlength=300)
        u=100*successes[2:]/(counts[2:]*100)
        delay=q[mask].ravel()*1000/(rate*1e6)
        p90,p99=np.percentile(delay,[90,99])
        result.append(dict(speed_group="all" if g<0 else GROUPS[g], vehicle_frames=int(mask.sum()),
            vehicle_slots=int(mask.sum()*100),violation_percent=float(u.mean()),
            pooled_violation_percent=float(violating[mask].mean()*100),
            p90_proxy_ms=float(p90),p99_proxy_ms=float(p99),mean_proxy_ms=float(delay.mean()),
            frames_present=int(np.count_nonzero(counts))))
    return result


def system(args):
    p=validate_protocol(args)
    dest=args.output/"system_by_speed_per_seed.csv"
    if dest.exists(): raise FileExistsError(dest)
    ref=REV/"grid/raw/meet_cobra_rate29_seed1.npz"
    with np.load(ref) as r: frames,vehicles=r["queue_frame"],r["queue_vehicle"]
    aligned=pd.read_csv(AUDIT/"test_speed_alignment.csv",dtype={"vehicle":str})
    mask=frames>=2
    np.testing.assert_array_equal(aligned.service_frame,frames[mask])
    np.testing.assert_array_equal(aligned.vehicle,vehicles[mask])
    group=np.full(len(frames),-1,dtype=int)
    group[mask]=groups(aligned.speed_mps.to_numpy()*3.6)
    rows,checks=[],[]
    for method in p["system"]["methods"]:
        for rate in p["system"]["rates"]:
            for seed in p["system"]["seeds"]:
                name=f"{method}_rate{rate}_seed{seed}"
                path=REV/"grid/raw"/(name+".npz")
                meta=read(REV/"grid/runs"/(name+".json"))
                assert digest(path)==meta["raw_sha256"],name
                with np.load(path) as r:
                    np.testing.assert_array_equal(r["queue_frame"],frames)
                    np.testing.assert_array_equal(r["queue_vehicle"],vehicles)
                    q=r["queue_bits"]
                    summary=queue_summary(q,frames,group,rate)
                    frame_u=np.bincount(frames,weights=(q>rate*1e6*.020).sum(axis=1),minlength=300)/(np.bincount(frames)*100)
                    np.testing.assert_allclose(frame_u,r["violation_probability"],atol=1e-12)
                for key in ("violation_percent","p90_proxy_ms","p99_proxy_ms","mean_proxy_ms"):
                    np.testing.assert_allclose(summary[0][key],meta["metrics"][key],rtol=1e-10,atol=1e-10)
                rows.extend(dict(method=method,rate_mbps=rate,seed=seed,**row) for row in summary)
                checks.append(dict(case=name,raw_sha256=meta["raw_sha256"],traffic_sha256=meta["traffic_sha256"]))
            print(f"system {method} {rate} Mbps: 3 seeds done",flush=True)
    assert len(checks)==108
    for rate in p["system"]["rates"]:
        for seed in p["system"]["seeds"]:
            matched=[c["traffic_sha256"] for c in checks if c["case"].endswith(f"rate{rate}_seed{seed}")]
            assert len(matched)==2 and len(set(matched))==1
    df=pd.DataFrame(rows)
    df.to_csv(dest,index=False)
    aggregate=[]
    for keys,g in df.groupby(["method","rate_mbps","speed_group"],sort=False):
        record=dict(zip(["method","rate_mbps","speed_group"],keys))
        record.update(seeds=len(g),vehicle_frames_per_seed=int(g.vehicle_frames.iloc[0]))
        for metric in ("violation_percent","pooled_violation_percent","p90_proxy_ms","p99_proxy_ms","mean_proxy_ms"):
            record.update({metric+"_mean":float(g[metric].mean()),metric+"_min":float(g[metric].min()),
                           metric+"_max":float(g[metric].max()),metric+"_sd":float(g[metric].std(ddof=1))})
        aggregate.append(record)
    adf=pd.DataFrame(aggregate)
    adf.to_csv(args.output/"system_by_speed_summary.csv",index=False)
    adf[adf.rate_mbps.isin([19,29])].to_csv(args.output/"system_selected_loads.csv",index=False)
    dump(args.output/"system_checks.json",dict(cases=checks,all_original_metrics_reproduce=True,
         all_raw_hashes_match=True,all_join_keys_match=True,paired_traffic=True,new_simulations=0))
    print("Completed all 108 existing system cases",flush=True)


def finalize(args):
    p=validate_protocol(args)
    for path,h in p["input_sha256"].items(): assert digest(ROOT/path)==h,path
    for path,h in p["protected_files"].items(): assert digest(ROOT/path)==h,path
    for name in ("validation_checks.json","test_checks.json","system_checks.json"):
        assert (args.output/name).exists()
    df=pd.concat([pd.read_csv(args.output/"validation_by_speed.csv"),pd.read_csv(args.output/"test_by_speed.csv")])
    df.to_csv(args.output/"prediction_by_speed.csv",index=False)
    dump(args.output/"completion.json",dict(status="complete",inputs_and_checkpoints_unchanged=True,
        manuscript_and_response_unchanged=True,validation_samples=195915,test_samples=40591,
        system_cases=108,trained_epochs=0,new_system_runs=0,
        outputs={x.name:digest(x) for x in args.output.iterdir() if x.is_file() and x.suffix in (".csv",".npz",".json")}))
    print(df.to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=["prepare","validation","test","system","finalize"])
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--threads",type=int,default=1)
    args=ap.parse_args()
    {"prepare":prepare,"validation":validation,"test":cached_test,"system":system,"finalize":finalize}[args.phase](args)


if __name__=="__main__":
    main()
