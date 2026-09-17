"""Actor-only derived-report feature and paired-initialization regression tests."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import test_o_mappo_report_input as report_tests
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.beam_utils import generate_dft_codebook
from utils.ho_utils import make_paired_traffic
from utils.o_mappo import (OMAPPPolicy, report_decision_features, shared_actor_inputs,
    make_local_state, make_global_state, state_feature_names, critic_local_feature_count,
    _candidate_links, o_mappo_reward_presets, run_fluid_o_mappo_episode)
from utils.o_mappo_sim import run_sim_o_mappo


class DerivedActorTests(unittest.TestCase):
    def setUp(self):
        report_tests.ReportInputTests.setUp(self)
        self.config = dataclasses.replace(self.config, state_variant="gain_derived")
        self.base = dataclasses.replace(self.config, state_variant="gain_report")
        self.args = paper_args(13e6)
        self.report.pop("beam")
        self.context = dict(args=self.args, serving_bs=0, backlog_bits=1.3e6,
                            load=np.full(5, .2), own_rb_fraction=.1, macro_loc=np.zeros(2))
        self.kw["config"] = self.config

    def inputs(self):
        return shared_actor_inputs(self.config,
            dict(pos=np.array([20., 30.]), shared_prediction=self.report), **self.context)

    def test_only_actor_receives_additional_features(self):
        state = make_local_state(**(self.kw | self.inputs()))
        base = make_local_state(**(self.kw | {"config": self.base}))
        np.testing.assert_array_equal(state[:37], base)
        self.assertEqual(state.shape, (62,))
        self.assertEqual(state_feature_names(self.config)[:37], state_feature_names(self.base))
        count = critic_local_feature_count(self.config)
        self.assertEqual(count, 37)
        np.testing.assert_array_equal(make_global_state(state[None], 1, count),
                                       make_global_state(base[None], 1))

    def test_no_privileged_input_and_no_beam_indices(self):
        class PublicOnly(dict):
            def __getitem__(self, key):
                if key not in ("pos", "shared_prediction"):
                    raise AssertionError("Nonpublic record access")
                return super().__getitem__(key)
        record = PublicOnly(pos=np.array([20., 30.]), shared_prediction=self.report)
        with patch("utils.o_mappo.best_beam_pair", side_effect=AssertionError), \
             patch("utils.o_mappo.no_bf_gain_db", side_effect=AssertionError):
            out = shared_actor_inputs(self.config, record, **self.context)
        features = out["derived_features"].reshape(5, 5)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[3, 0], 0)
        self.assertEqual(features[4, 0], 0)
        with self.assertRaises(ValueError):
            shared_actor_inputs(self.config, record)

    def test_paired_common_weights_and_unchanged_critic(self):
        old, new = OMAPPPolicy(self.base, 20), OMAPPPolicy(self.config, 20)
        np.testing.assert_array_equal(old.actor.model[0].weight.detach(),
                                       new.actor.model[0].weight[:, :37].detach())
        self.assertEqual(torch.count_nonzero(new.actor.model[0].weight[:, 37:]).item(), 0)
        for a, b in zip(old.critic.parameters(), new.critic.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        state = torch.tensor(make_local_state(**(self.kw | self.inputs()))[None])
        torch.testing.assert_close(old.actor(state[:, :37]), new.actor(state), rtol=1e-6, atol=1e-7)
        self.assertEqual(sum(x.numel() for x in new.actor.parameters()), 4162)
        self.assertEqual(sum(x.numel() for x in new.critic.parameters()), 7297)
        new.actor(state).square().mean().backward()
        self.assertGreater(new.actor.model[0].weight.grad[:, 37:].abs().sum().item(), 0)

    def test_optimizer_candidates_unchanged(self):
        record = dict(pos=np.array([20., 30.]), shared_prediction=self.report)
        args = (self.args, "v", record, 0, 1.3e6, np.full(5, .2))
        tx, rx = generate_dft_codebook(32), generate_dft_codebook(8)
        a = _candidate_links(*args, self.base, tx, rx, np.zeros(2))
        b = _candidate_links(*args, self.config, tx, rx, np.zeros(2))
        self.assertEqual(a, b)

    def test_roundtrip_and_both_simulation_paths(self):
        policy = OMAPPPolicy(self.config, 20)
        state = make_local_state(**(self.kw | self.inputs()))[None]
        global_state = make_global_state(state, 1, 37)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "policy.pt")
            policy.save(path)
            other = OMAPPPolicy.load(path)
            for a, b in zip(policy.act(state, global_state, False), other.act(state, global_state, False)):
                np.testing.assert_array_equal(a, b)
        rng = np.random.default_rng(8)
        record = dict(pos=np.array([20., 30.]), angle=0., v=0., shared_prediction=self.report,
            h=(rng.normal(size=(8, 4, 32)) + 1j*rng.normal(size=(8, 4, 32))) * 1e-5)
        timeline = {800 + .1*i: {"v": dict(record)} for i in range(5)}
        result = run_fluid_o_mappo_episode(self.args, timeline, policy,
            o_mappo_reward_presets()["qos_energy020_load1"], 13, seed=7, learn=True)
        self.assertTrue(np.isfinite(result["average_system_power_w"]))
        result = run_sim_o_mappo(self.args, MICRO_BS_LOCATIONS, timeline, policy,
            prt=False, rician_fading=False, traffic_trace=make_paired_traffic(self.args, timeline, 7),
            ho_interruption_ms=10)
        self.assertTrue(np.isfinite(result.energy_record).all())


if __name__ == "__main__":
    unittest.main()
