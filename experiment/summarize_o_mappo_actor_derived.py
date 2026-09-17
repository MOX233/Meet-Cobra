"""Audit the actor-only feature experiment against its matched gain-report run."""
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

OUTPUT = ROOT / "experiment/results/o_mappo_actor_derived_20260917"
REFERENCE = ROOT / "experiment/results/o_mappo_gain_report_20260917"


def architecture_audit():
    policies = {
        "original": driver.OMAPPPolicy.load(str(driver.LEGACY)),
        "full_report": driver.OMAPPPolicy.load(str(ROOT / "experiment/results/o_mappo_report_input_20260917/training_seed20/best_policy.pt")),
        "gain_report": driver.OMAPPPolicy.load(str(REFERENCE / "training_seed20/best_policy.pt")),
        "gain_derived_initial": OMAPPPolicy(OMAPPOConfig(state_variant="gain_derived",
            information_mode="shared_prediction", ho_interruption_ms=10, torch_threads=1), 20),
    }
    rows = {name: dict(actor=str(p.actor), critic=str(p.critic),
        actor_parameters=sum(x.numel() for x in p.actor.parameters()),
        critic_parameters=sum(x.numel() for x in p.critic.parameters()),
        actor_input=p.local_dim, critic_input=p.global_dim,
        feature_names=state_feature_names(p.config), config=dataclasses.asdict(p.config))
        for name, p in policies.items()}
    checks = []
    for seed in (20, 21):
        config = policies["gain_report"].config
        old = OMAPPPolicy(config, seed)
        new = OMAPPPolicy(dataclasses.replace(config, state_variant="gain_derived"), seed)
        torch.testing.assert_close(old.actor.model[0].weight, new.actor.model[0].weight[:, :37], rtol=0, atol=0)
        for a, b in zip(old.critic.parameters(), new.critic.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.count_nonzero(new.actor.model[0].weight[:, 37:]).item() == 0
        torch.testing.assert_close(old.actor.model[2].weight, new.actor.model[2].weight, rtol=0, atol=0)
        checks.append(dict(seed=seed, common_weights_and_critic_equal=True, derived_columns_zero=True))
    driver.write_json(OUTPUT / "architecture_audit.json", dict(models=rows, paired_initialization=checks,
        initialization="random from scratch; no trained checkpoint transferred"))


def reproduce_baseline(gpu):
    timeline = driver.temporal_slice(driver.read_pickle(driver.OUTPUT / "test_prepared.pkl"), 800, 830)
    driver.CONTEXT = (OUTPUT, timeline, REFERENCE / "training_seed20/best_policy.pt", True, False, gpu, False)
    driver.evaluate_one(("gain_report", 13, 1, 10))


def summarize(gpu):
    architecture_audit()
    for seed in (20, 21):
        old = json.loads((REFERENCE / f"training_seed{seed}/training.json").read_text())
        new = json.loads((OUTPUT / f"training_seed{seed}/training.json").read_text())
        old_protocol = json.loads((REFERENCE / f"training_seed{seed}/protocol.json").read_text())
        new_protocol = json.loads((OUTPUT / f"training_seed{seed}/protocol.json").read_text())
        assert old_protocol["frontend_manifest"] == new_protocol["frontend_manifest"]
        assert len(old["history"]) == len(new["history"]) == 72
        for key in ("average_system_power_w", "queue_violation_percent", "trigger_ratio",
                    "handover_per_vehicle_per_s", "average_queueing_proxy_ms"):
            assert old["history"][0][key] == new["history"][0][key], key
        for a, b in zip(old["history"], new["history"]):
            assert (a["episode"], a["start"], a["data_rate_mbps"]) == (b["episode"], b["start"], b["data_rate_mbps"])
        assert old["reward"] == new["reward"]
        assert {k:v for k,v in old["config"].items() if k != "state_variant"} == {
            k:v for k,v in new["config"].items() if k != "state_variant"}
        assert new_protocol["actor_input_dim"] == 62 and new_protocol["critic_input_dim"] == 112
    rows = [json.loads(p.read_text()) for p in (OUTPUT / "runs").glob("*.json")]
    new = next(r for r in rows if r["method"] == "gain_derived" and r["seed"] == 1 and r["rate_mbps"] == 13)
    selected = json.loads((OUTPUT / "selected_policy.json").read_text())
    assert new["checkpoint_sha256"] == selected["checkpoint_sha256"]
    reference = json.loads((REFERENCE / "comparison_summary.json").read_text())
    reference = [r for r in reference["runs"] if r["seed"] == 1 and r["rate_mbps"] == 13]
    base = next(r for r in reference if r["method"] == "gain_report")
    reproduced = next(r for r in rows if r["method"] == "gain_report" and r["physics_gpu"] == gpu)
    for key, value in base["metrics"].items():
        np.testing.assert_allclose(reproduced["metrics"][key], value, rtol=1e-12, atol=1e-12)
    assert {r["traffic_sha256"] for r in reference + [new, reproduced]} == {new["traffic_sha256"]}
    final = [r for r in reference if r["method"] in ("legacy", "report", "gain_report", "meet_cobra")] + [new]
    delta = {key: new["metrics"][key] - value for key, value in base["metrics"].items()}
    driver.write_json(OUTPUT / "summary.json", dict(runs=final, baseline_reproduction=reproduced,
        baseline_reproduction_equal=True, selection=selected, difference_from_gain_report=delta,
        first_rollout_identical_for_both_training_seeds=True,
        protocol="same two training seeds, 72 segments, rewards, critic input and optimizer; single test seed at 13 Mbps"))
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
    p.add_argument("--gpu", type=int, default=4)
    cli = p.parse_args()
    if cli.audit:
        architecture_audit()
    elif cli.reproduce_baseline:
        reproduce_baseline(cli.gpu)
    else:
        summarize(cli.gpu)
