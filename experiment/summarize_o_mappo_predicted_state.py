"""Audit/reproduce the original-layout, predicted-CSI actor experiment."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import o_mappo_shared_frontend as driver
from utils.o_mappo import OMAPPOConfig, OMAPPPolicy, state_feature_names
import numpy as np
import torch

OUTPUT = ROOT / "experiment/results/o_mappo_predicted_state_20260917"
REFERENCE = ROOT / "experiment/results/o_mappo_gain_report_20260917"
PREVIOUS = ROOT / "experiment/results/o_mappo_actor_derived_20260917"


def architecture_audit():
    config = OMAPPOConfig(state_variant="predicted_adapted", information_mode="shared_prediction",
                          ho_interruption_ms=10, torch_threads=1)
    policies = {
        "original": OMAPPPolicy.load(str(driver.LEGACY)),
        "gain_report": OMAPPPolicy.load(str(REFERENCE / "training_seed20/best_policy.pt")),
        "predicted_adapted_initial": OMAPPPolicy(config, 20),
    }
    rows = {name: dict(actor=str(p.actor), critic=str(p.critic),
        actor_parameters=sum(x.numel() for x in p.actor.parameters()),
        critic_parameters=sum(x.numel() for x in p.critic.parameters()),
        actor_input=p.local_dim, critic_input=p.global_dim,
        feature_names=state_feature_names(p.config), config=dataclasses.asdict(p.config))
        for name, p in policies.items()}
    for seed in (20, 21):
        original = OMAPPPolicy(dataclasses.replace(config, state_variant="adapted", information_mode="legacy"), seed)
        predicted = OMAPPPolicy(config, seed)
        assert original.feature_names == predicted.feature_names
        for a, b in zip(list(original.actor.parameters()) + list(original.critic.parameters()),
                        list(predicted.actor.parameters()) + list(predicted.critic.parameters())):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    driver.write_json(OUTPUT / "architecture_audit.json", dict(models=rows,
        original_layout_and_random_initialization_equal_for_seeds=[20, 21],
        initialization="random from scratch; no trained weights transferred",
        rollback_git="106e141", implementation_git="3ebbd84",
        predictor_timing="CSI at x predicts x+1, no label input",
        demand_estimator="mean-rate demand, original 10-round/atol-1-RB refinement, interference occupancy capped at one",
        actor_demand_range=[0, 1.5], critic_restored_to_original_94_inputs=True,
        target_optimizer_changed=False))


def reproduce_baseline(gpu):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    timeline = driver.temporal_slice(driver.read_pickle(driver.OUTPUT / "test_prepared.pkl"), 800, 830)
    driver.CONTEXT = (OUTPUT, timeline, REFERENCE / "training_seed20/best_policy.pt", True, False, gpu, False)
    driver.evaluate_one(("gain_report", 13, 1, 10))


def summarize(gpu):
    architecture_audit()
    for seed in (20, 21):
        old = json.loads((REFERENCE / f"training_seed{seed}/training.json").read_text())
        new = json.loads((OUTPUT / f"training_seed{seed}/training.json").read_text())
        op = json.loads((REFERENCE / f"training_seed{seed}/protocol.json").read_text())
        np_ = json.loads((OUTPUT / f"training_seed{seed}/protocol.json").read_text())
        assert op["frontend_manifest"] == np_["frontend_manifest"]
        assert len(old["history"]) == len(new["history"]) == 72
        for a, b in zip(old["history"], new["history"]):
            assert (a["episode"], a["start"], a["data_rate_mbps"]) == (b["episode"], b["start"], b["data_rate_mbps"])
        assert old["reward"] == new["reward"]
        assert {k: v for k, v in old["config"].items() if k != "state_variant"} == {
            k: v for k, v in new["config"].items() if k != "state_variant"}
        assert np_["actor_input_dim"] == 31 and np_["critic_input_dim"] == 94
        assert not np_["test_used_for_selection"]
    rows = [json.loads(p.read_text()) for p in (OUTPUT / "runs").glob("*.json")]
    new = next(r for r in rows if r["method"] == "predicted_adapted" and r["seed"] == 1 and r["rate_mbps"] == 13)
    selected = json.loads((OUTPUT / "selected_policy.json").read_text())
    choices = [json.loads(p.read_text()) for p in OUTPUT.glob("training_seed*/selection.json")]
    assert selected["score"] == min(c["score"] for c in choices)
    assert new["checkpoint_sha256"] == selected["checkpoint_sha256"]
    reference = json.loads((PREVIOUS / "summary.json").read_text())["runs"]
    base = next(r for r in reference if r["method"] == "gain_report")
    reproduced = next(r for r in rows if r["method"] == "gain_report" and r["physics_gpu"] == gpu)
    for key, value in base["metrics"].items():
        np.testing.assert_allclose(reproduced["metrics"][key], value, rtol=1e-12, atol=1e-12)
    assert {r["traffic_sha256"] for r in reference + [new, reproduced]} == {new["traffic_sha256"]}
    final = reference + [new]
    delta = {key: new["metrics"][key] - value for key, value in base["metrics"].items()}
    driver.write_json(OUTPUT / "summary.json", dict(runs=final, baseline_reproduction=reproduced,
        baseline_reproduction_equal=True, selection=selected, difference_from_gain_report=delta,
        protocol="same two training seeds, 72 segments, reward and target optimizer; original actor/critic dimensions restored; one test seed at 13 Mbps"))
    lines = ["13 Mbps, seed 1, 800--830 s, 10 ms HO interruption.", "",
        "| Method | P (W) | U (%) | mean proxy (ms) | p99 proxy (ms) | macro (%) | HO/vehicle/s |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in final:
        m = r["metrics"]
        lines.append("| " + " | ".join([r["method"]] + [f"{m[k]:.5f}" if k in m else "—" for k in (
            "power_w", "violation_percent", "mean_proxy_ms", "p99_proxy_ms", "macro_association_percent", "handovers_per_vehicle_s")]) + " |")
    (OUTPUT / "result_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", action="store_true")
    p.add_argument("--reproduce-baseline", action="store_true")
    p.add_argument("--gpu", type=int, default=1)
    cli = p.parse_args()
    if cli.audit:
        architecture_audit()
    elif cli.reproduce_baseline:
        reproduce_baseline(cli.gpu)
    else:
        summarize(cli.gpu)
