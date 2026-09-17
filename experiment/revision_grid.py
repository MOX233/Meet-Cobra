#!/usr/bin/env python3
"""Frozen-model, paired five-seed Fig.5--8 rerun. No legacy output overwritten."""
import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import o_mappo_shared_frontend as shared
from experiment.ho_interruption_experiment import oracle_cache
from experiment.benchmark_nn_overhead import digest
from utils.gap_refinement import GAPRefinementConfig
from utils.ho_utils import make_paired_traffic
from utils.mts_gs_hbf import candidate_configs
from utils.mts_gs_hbf_sim import run_sim_mts_gs_hbf
from utils.o_mappo import OMAPPPolicy
from utils.o_mappo_sim import run_sim_o_mappo
from utils.sim_utils import run_sim_withUMa
import utils.alg_utils as alg
import numpy as np
import torch

OUTPUT = ROOT / "experiment/results/revision_fig5_8_20260917"
CACHE = shared.OUTPUT / "test_prepared.pkl"
RATES = tuple(range(1,36,2))
SEEDS = (1,2,3,4,5)
METHODS = ("meet_cobra", "oracle_mc", "reactive_obra", "wo_gap_ho", "wo_pet_bf", "wo_otr_ra", "o_mappo", "mts")
LABELS = ("MEET-COBRA", "Oracle-MC", "Reactive-OBRA", "w/o GAP-HO", "w/o PET-BF", "w/o OTR-RA", "O-MAPPO-adapted", "MTS-GS-HBF-adapted")
CODE = ("experiment/revision_grid.py", "utils/sim_utils.py", "utils/alg_utils.py", "utils/gpu_phy.py",
    "utils/mts_gs_hbf.py", "utils/mts_gs_hbf_sim.py", "utils/o_mappo.py", "utils/o_mappo_sim.py",
    "utils/ho_utils.py", "utils/gap_refinement.py", "utils/fast_pet_measurement.py", "utils/compiled_matching.py", "utils/queue_utils.py")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def protocol():
    manifest = json.loads((shared.OUTPUT / "frontend_manifest.json").read_text())
    models, inventory = shared.load_selected_models(shared.DEFAULT_CHECKPOINT_ROOT)
    del models
    for task in inventory:
        assert inventory[task]["sha256"] == manifest["frontend"][task]["sha256"]
    assert digest(CACHE) == manifest["test"]["cache_sha256"]
    args = shared.paper_args()
    return dict(version=1, rollback_git="c1b0cd3", methods=list(METHODS), rates=list(RATES), seeds=list(SEEDS),
        test_interval=[800,830], service_frames=300, warmup_frames=2, slot_s=args.slot_len,
        slots_per_frame=100, latency_ms=20, ho_ms=10, gap_iterations=2, zeta=1.1, candidate_beams=5,
        random_bf_index_fix=True, reactive_configuration="reactive_configuration.json (separately confirmed)", oracle_hold_macro_position=True,
        oracle_cr_lb="pending separate mathematical definition; never fabricate queue metrics",
        frontend=manifest, legacy_policy=dict(path=str(shared.LEGACY), sha256=digest(shared.LEGACY)),
        mts_config=dataclasses.asdict(dataclasses.replace(candidate_configs()["pressure_early"], ho_interruption_ms=10)),
        physical_parameters={k: getattr(args,k) for k in ("num_RB_macro", "num_RB_micro", "p_macro", "p_micro", "RB_intervel_macro", "RB_intervel_micro", "NF_macro_dB", "NF_micro_dB", "N0", "K_rician", "pilot_overhead_factor")},
        power_ceiling_w=float(args.num_RB_macro*args.p_macro+4*args.num_RB_micro*args.p_micro),
        fading="torch FP64, frame/seed keyed, same draws irrespective of actions",
        code_sha256={name: digest(ROOT/name) for name in CODE})


def prepare(reactive_input):
    new = protocol()
    path = OUTPUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != new:
        raise RuntimeError("Protocol changed: preserve existing results and explicitly version the run")
    write_json(path, new)
    if reactive_input != "pending":
        config_path = OUTPUT / "reactive_configuration.json"
        config = dict(input=reactive_input, protocol_sha256=digest(path))
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError("Reactive configuration is already frozen")
        write_json(config_path, config)
    print("PROTOCOL READY", new["power_ceiling_w"], flush=True)


def validate_protocol():
    ppath = OUTPUT / "protocol.json"
    p = json.loads(ppath.read_text())
    for path, sha in p["code_sha256"].items():
        assert digest(ROOT/path) == sha, f"Code changed mid-experiment: {path}"
    return p, digest(ppath)


def method_configuration(method):
    if method != "reactive_obra":
        return None, None
    path = OUTPUT / "reactive_configuration.json"
    if not path.exists():
        raise RuntimeError("Reactive information source awaits user decision")
    config = json.loads(path.read_text())
    assert config["protocol_sha256"] == digest(OUTPUT / "protocol.json")
    return config["input"], digest(path)


class Diagnostics(list):
    def append(self, value):
        super().append(value)
        if len(self) % 50 == 0:
            print("FRAME", len(self), flush=True)


def extract(args, result, diagnostics, traffic):
    if isinstance(result, tuple):
        energy, ho, _, violation, mean_q, pilots, rb, queues = result
        association = {i: d["association"] for i,d in enumerate(diagnostics)}
        extra = {}
    else:
        energy, ho, violation, mean_q, pilots, rb, queues = (result.energy_record,
            result.handover_record, result.violation_probability_record, result.average_queue_record,
            result.pilot_record, result.rb_allocated_record, result.queue_per_vehicle_record)
        association = result.association_record
        extra = {k: getattr(result,k) for k in ("decision_record", "trigger_record", "optimizer_failure_record", "optimizer_overflow_record")}
    frames, vehicles, data = [], [], []
    for fi, rows in queues.items():
        for v in sorted(rows, key=str):
            frames.append(fi); vehicles.append(str(v)); data.append(rows[v])
    data = np.asarray(data)
    frames = np.asarray(frames)
    assert data.shape[1] == args.slots_per_frame
    assert len(energy) == 300
    assert len(association) == len(energy)
    counts = np.array([np.bincount(list(association[i].values()), minlength=5) for i in range(len(energy))])
    selected = slice(2,None)
    sampled = data[frames >= 2] / args.data_rate * 1000
    # Recompute every frame from precisely the stored post-service queues.
    recomputed_u = np.array([np.mean(data[frames == i] > args.data_rate*.02) for i in range(len(energy))])
    np.testing.assert_allclose(recomputed_u, violation, rtol=1e-12, atol=1e-12)
    recomputed_q = np.array([np.mean(data[frames == i]) for i in range(len(energy))])
    np.testing.assert_allclose(recomputed_q, mean_q, rtol=1e-12, atol=1e-8)
    caps = np.array([args.num_RB_macro]+[args.num_RB_micro]*4)
    assert np.isfinite(data).all() and (data >= 0).all()
    assert np.all(rb <= caps[None] + 1e-8) and np.all(rb >= 0)
    powers = np.array([args.p_macro]+[args.p_micro]*4)
    np.testing.assert_allclose(np.asarray(rb) @ powers * .1, energy, rtol=1e-10, atol=1e-10)
    metrics = dict(power_w=float(np.mean(energy[selected])/.1), violation_percent=float(np.mean(violation[selected])*100),
        mean_proxy_ms=float(np.mean(mean_q[selected])/args.data_rate*1000),
        p90_proxy_ms=float(np.percentile(sampled,90)), p99_proxy_ms=float(np.percentile(sampled,99)),
        macro_association_percent=float(100*counts[selected,0].sum()/counts[selected].sum()),
        handovers_per_vehicle_s=float(np.sum(ho[selected])/(counts[selected].sum()*.1)),
        pilots_per_vehicle_slot=float(np.mean(pilots[selected])),
        blocked_vehicle_time_percent=float(100*np.sum(ho[selected])*10/(counts[selected].sum()*100)))
    raw = dict(energy_j=np.asarray(energy), handover_count=np.asarray(ho), violation_probability=np.asarray(violation),
        mean_queue_bits=np.asarray(mean_q), pilots=np.asarray(pilots), rb_per_bs=np.asarray(rb),
        association_counts=counts, queue_bits=data, queue_frame=frames, queue_vehicle=np.array(vehicles), **extra)
    return metrics, raw


def run_case(method, rate, seed, gpu):
    key = f"{method}_rate{rate}_seed{seed}"
    saved = OUTPUT/"runs"/f"{key}.json"
    p, psha = validate_protocol()
    reactive_input, method_sha = method_configuration(method)
    if saved.exists():
        old = json.loads(saved.read_text())
        assert old["protocol_sha256"] == psha
        assert old["method_configuration_sha256"] == method_sha
        assert (OUTPUT/"raw"/f"{key}.npz").exists()
        return old
    torch.set_num_threads(1)
    shared.single_thread_solvers()
    from utils.compiled_matching import km_algorithm_compiled
    alg.km_algorithm = km_algorithm_compiled
    km_algorithm_compiled(np.zeros((2,2)))
    timeline = shared.temporal_slice(shared.read_pickle(CACHE), 800, 830)
    assert len(timeline) == 301
    args = shared.paper_args(rate*1e6)
    args.device = torch.device("cpu")
    traffic = make_paired_traffic(args, timeline, seed)
    # Separate method-side draws from precomputed arrivals and GPU fading.
    np.random.seed(seed)
    diag = Diagnostics()
    started = time.monotonic()
    common = dict(prt=False, rician_fading=True, traffic_trace=traffic,
        ho_interruption_ms=10, paired_fading_seed=seed, physics_device=f"cuda:{gpu}")
    print("START", key, "gpu", gpu, flush=True)
    if method == "mts":
        result = run_sim_mts_gs_hbf(args, shared.MICRO_BS_LOCATIONS, timeline,
            candidate_configs()["pressure_early"], ho_diagnostics=diag, seed=seed, **common)
    elif method == "o_mappo":
        policy = OMAPPPolicy.load(str(shared.LEGACY))
        result = run_sim_o_mappo(args, shared.MICRO_BS_LOCATIONS, timeline, policy,
            seed=seed, optimizer_solver="milp", progress_callback=lambda n,t: print("FRAME",n,flush=True) if n%50==0 else None, **common)
    else:
        cache = {f: {v:r["shared_prediction"] for v,r in records.items()} for f,records in timeline.items()}
        oracle = method == "oracle_mc"
        reactive = method == "reactive_obra"
        random_bf = method in ("reactive_obra", "wo_pet_bf")
        ho = alg.HO_EE_Greedy_offload if method in ("reactive_obra", "wo_gap_ho") else alg.HO_EE_GAP_APX_SINR_conservative_adaptive
        ra = alg.RA_OTR3_SINR if method in ("reactive_obra", "wo_otr_ra") else alg.RA_OTR_SINR
        use_nn = not oracle and not (reactive and reactive_input == "current")
        extra = dict(oracle_ho_cache=oracle_cache(args, shared.MICRO_BS_LOCATIONS,timeline),
                     oracle_hold_macro_position=True) if oracle else {}
        result = run_sim_withUMa(args, shared.MICRO_BS_LOCATIONS, timeline, None,
            True if use_nn else None, True if use_nn else None, True if use_nn else None,
            HO_func=ho, RA_func=ra, save_pilot=not random_bf,
            BF_func="topKbeam_NoPred" if random_bf else "topKbeam_savePilot", K_BF=5,
            prediction_cache=cache if use_nn else None, ho_capacity_correction=True,
            gap_refinement_config=GAPRefinementConfig(max_iterations=2,tolerance_rb=None,relaxation_factor=1.1),
            vectorized_pet_measurement=not random_bf, correct_random_beam_index=True,
            reactive_current_measurements=reactive and reactive_input == "current",
            ho_diagnostics=diag, **common, **extra)
    metrics, raw = extract(args, result, diag, traffic)
    (OUTPUT/"raw").mkdir(exist_ok=True)
    tmp = OUTPUT/"raw"/f"{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **raw)
    destination = OUTPUT/"raw"/f"{key}.npz"
    os.replace(tmp,destination)
    row = dict(method=method, label=LABELS[METHODS.index(method)], rate_mbps=rate, seed=seed,
        gpu=gpu, frames=300, retained_frames=298, metrics=metrics, traffic_sha256=traffic["sha256"],
        protocol_sha256=psha, method_configuration_sha256=method_sha,
        raw_sha256=digest(destination), elapsed_s=time.monotonic()-started)
    write_json(saved,row)
    print("DONE",key,json.dumps(metrics),flush=True)
    return row


def queue(methods, rates, seeds, gpus):
    """One worker per specified GPU, isolated child per point, resumable files."""
    import queue as queue_module
    p, psha = validate_protocol()
    assert set(methods) <= set(METHODS) and set(rates) <= set(RATES) and set(seeds) <= set(SEEDS)
    pending = queue_module.Queue()
    configs = {m: method_configuration(m)[1] for m in methods}
    for rate in rates:
        for seed in seeds:
            for method in methods:
                saved = OUTPUT/"runs"/f"{method}_rate{rate}_seed{seed}.json"
                if not saved.exists():
                    pending.put((method,rate,seed))
                else:
                    row = json.loads(saved.read_text())
                    assert row["protocol_sha256"] == psha and row["method_configuration_sha256"] == configs[method]
                    assert (OUTPUT/"raw"/f"{method}_rate{rate}_seed{seed}.npz").exists()
    (OUTPUT/"logs").mkdir(exist_ok=True)
    failures = []
    def worker(gpu):
        while True:
            try:
                method,rate,seed = pending.get_nowait()
            except queue_module.Empty:
                return
            log = OUTPUT/"logs"/f"{method}_rate{rate}_seed{seed}.log"
            cmd = [sys.executable,"-u",str(Path(__file__).resolve()),"case","--method",method,"--rate",str(rate),"--seed",str(seed),"--gpu",str(gpu)]
            with log.open("a") as out:
                code = subprocess.call(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONHASHSEED="0"))
            if code:
                failures.append(dict(method=method,rate=rate,seed=seed,code=code,log=str(log)))
            print("QUEUE",method,rate,seed,"exit",code,"remaining",pending.qsize(),flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(worker,gpus))
    write_json(OUTPUT/"queue_last_result.json",dict(failures=failures,methods=methods,rates=rates,seeds=seeds))
    if failures:
        raise RuntimeError(f"{len(failures)} cases failed; inspect logs before retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare","case","queue"))
    parser.add_argument("--reactive-input", choices=("pending","current","legacy"), default="pending")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--rate", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--methods", default=",".join(m for m in METHODS if m != "reactive_obra"))
    parser.add_argument("--rates", default="1,19,35")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--gpus", default="0,2,3,4,5,6")
    cli = parser.parse_args()
    if cli.phase == "prepare":
        prepare(cli.reactive_input)
    elif cli.phase == "case":
        run_case(cli.method, cli.rate, cli.seed, cli.gpu)
    else:
        queue(cli.methods.split(","),list(map(int,cli.rates.split(","))),list(map(int,cli.seeds.split(","))),list(map(int,cli.gpus.split(","))))
