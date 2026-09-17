"""Original actor feature layout reconstructed exclusively from predicted gains."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import test_o_mappo_report_input as report_tests
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.alg_utils import estimate_num_RB_allocated_perBS
from utils.ho_utils import make_paired_traffic
from utils.o_mappo import (OMAPPPolicy, predicted_actor_link_states, make_local_state,
    make_global_state, state_feature_names, o_mappo_reward_presets,
    run_fluid_o_mappo_episode)
from utils.o_mappo_sim import run_sim_o_mappo
from utils.pql_ba import macro_gain_db


class PredictedStateTests(unittest.TestCase):
    def setUp(self):
        report_tests.ReportInputTests.setUp(self)
        self.config = dataclasses.replace(self.config, state_variant="predicted_adapted")
        self.base = dataclasses.replace(self.config, state_variant="adapted", information_mode="legacy")
        self.args = paper_args(13e6)
        self.report.pop("beam")
        self.record = dict(pos=np.array([20., 30.]), shared_prediction=self.report)
        self.kw["config"] = self.config

    def context(self, records=None, connection=None, rate=13e6):
        records = records or {"v": self.record}
        connection = connection or {"v": 1}
        return predicted_actor_link_states(self.args, self.config, records, connection,
                                           {v: rate for v in connection})

    def test_original_layout_architecture_and_initialization(self):
        old, new = OMAPPPolicy(self.base, 20), OMAPPPolicy(self.config, 20)
        self.assertEqual(state_feature_names(self.base), state_feature_names(self.config))
        self.assertEqual((new.local_dim, new.global_dim), (31, 94))
        self.assertEqual(sum(p.numel() for p in new.actor.parameters()), 2178)
        self.assertEqual(sum(p.numel() for p in new.critic.parameters()), 6145)
        for a, b in zip(list(old.actor.parameters()) + list(old.critic.parameters()),
                        list(new.actor.parameters()) + list(new.critic.parameters())):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_privileged_arguments_cannot_override_predicted_slots(self):
        context = self.context()["v"]
        state = make_local_state(**(self.kw | {"predicted_link_state": context}))
        poisoned = make_local_state(**(self.kw | dict(predicted_link_state=context,
            serving_sinr_db=np.nan, interference_db=np.nan, rb_load=np.full(5, np.nan))))
        np.testing.assert_array_equal(state, poisoned)
        expected = make_local_state(**(self.kw | dict(config=self.base,
            serving_sinr_db=context[0], interference_db=context[1], rb_load=context[2])))
        np.testing.assert_array_equal(state, expected)
        with self.assertRaises(ValueError):
            make_local_state(**self.kw)
        with self.assertRaises(ValueError):
            dataclasses.replace(self.config, information_mode="legacy").validate()

    def test_public_only_context_and_no_queue_dependence(self):
        class PublicOnly(dict):
            def __getitem__(self, key):
                if key not in ("pos", "shared_prediction", "gain", "interference"):
                    raise AssertionError(f"Nonpublic access: {key}")
                return super().__getitem__(key)
        record = PublicOnly(pos=self.record["pos"], shared_prediction=PublicOnly(self.report))
        with patch("utils.o_mappo.best_beam_pair", side_effect=AssertionError), \
             patch("utils.o_mappo.no_bf_gain_db", side_effect=AssertionError):
            value = self.context({"v": record})["v"]
        np.testing.assert_equal(value, self.context()["v"])
        enriched = self.record | {"h": np.nan, "queue": 1e99, "CSI_preprocessed": np.nan}
        np.testing.assert_equal(value, self.context({"v": enriched})["v"])
        changed = self.record | {"shared_prediction": self.report | {"gain": self.report["gain"] - 10}}
        self.assertLess(self.context({"v": changed})["v"][0], value[0])

    def test_mean_rate_load_matches_legacy_estimator_without_overload(self):
        records = {str(i): self.record for i in range(5)}
        connections = {str(i): i for i in range(5)}
        gains = {v: np.r_[macro_gain_db(self.args, r["pos"], np.zeros(2)), r["shared_prediction"]["gain"]]
                 for v, r in records.items()}
        interference = {v: np.r_[-180., self.report["interference"]] for v in records}
        rate = {v: 1e5 for v in records}
        expected = estimate_num_RB_allocated_perBS(self.args, connections,
            np.zeros((5, 2)), list(records), gains, rate, infer_g_dict=interference)
        caps = np.r_[self.args.num_RB_macro, np.full(4, self.args.num_RB_micro)]
        actual = predicted_actor_link_states(self.args, self.config, records, connections, rate)
        # Legacy concatenation can round the macro gain to FP32.
        np.testing.assert_allclose(actual["1"][2], expected / caps, rtol=1e-7, atol=1e-12)
        self.assertEqual(actual["0"][1], -np.inf)

    def test_demand_can_exceed_capacity_but_interference_is_bounded(self):
        records = {str(i): self.record for i in range(1, 5)}
        connections = {str(i): i for i in range(1, 5)}
        high = self.context(records, connections, 1e14)
        for bs in range(1, 5):
            sinr, inr, load = high[str(bs)]
            np.testing.assert_array_equal(load, [0, 1.5, 1.5, 1.5, 1.5])
            noise = self.args.N0 * self.args.RB_intervel_micro * 10**(self.args.NF_micro_dB / 10)
            expected_inr = 3 * self.args.p_micro * 10**(-135 / 10) / noise
            self.assertAlmostEqual(inr, 10 * np.log10(expected_inr), places=9)
        low = self.context(records, connections, 1e5)
        self.assertLess(low["1"][1], high["1"][1])
        self.assertGreater(low["1"][0], high["1"][0])

    def timeline(self):
        rng = np.random.default_rng(8)
        record = self.record | dict(angle=0., v=0.,
            h=(rng.normal(size=(8, 4, 32)) + 1j*rng.normal(size=(8, 4, 32))) * 1e-5)
        return {800 + .1*i: {"v": dict(record)} for i in range(5)}

    def test_roundtrip_and_training_path(self):
        policy = OMAPPPolicy(self.config, 20)
        state = make_local_state(**(self.kw | {"predicted_link_state": self.context()["v"]}))[None]
        global_state = make_global_state(state, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "policy.pt")
            policy.save(path)
            other = OMAPPPolicy.load(path)
            for a, b in zip(policy.act(state, global_state, False), other.act(state, global_state, False)):
                np.testing.assert_array_equal(a, b)
        with patch("utils.o_mappo.make_local_state", wraps=make_local_state) as spy:
            result = run_fluid_o_mappo_episode(self.args, self.timeline(), policy,
                o_mappo_reward_presets()["qos_energy020_load1"], 13, seed=7, learn=True)
        self.assertTrue(np.isfinite(result["average_system_power_w"]))
        self.assertGreater(spy.call_count, 0)
        self.assertTrue(all("predicted_link_state" in c.kwargs for c in spy.call_args_list))

    def test_exact_path_same_actions_preserve_optimizer_and_physics(self):
        # With scripted equal actions, only policy observations may differ.
        # This exercises the optimizer, BF, RA and HO paths after micro association.
        class Policy:
            def __init__(self, config):
                self.config = config
            def act(self, local, global_state, explore):
                return np.ones(len(local), dtype=int), np.zeros(len(local)), np.zeros(len(local))
        timeline = self.timeline()
        kwargs = dict(prt=False, rician_fading=False, ho_interruption_ms=10,
            traffic_trace=make_paired_traffic(self.args, timeline, 7))
        base = dataclasses.replace(self.config, state_variant="gain_report")
        old = run_sim_o_mappo(self.args, MICRO_BS_LOCATIONS, timeline, Policy(base), **kwargs)
        with patch("utils.o_mappo_sim.make_local_state", wraps=make_local_state) as spy:
            new = run_sim_o_mappo(self.args, MICRO_BS_LOCATIONS, timeline, Policy(self.config), **kwargs)
        for field in dataclasses.fields(old):
            if not field.name.endswith("time_record"):
                np.testing.assert_equal(getattr(old, field.name), getattr(new, field.name))
        self.assertTrue(all("predicted_link_state" in c.kwargs for c in spy.call_args_list))


if __name__ == "__main__":
    unittest.main()
