"""Edge actor receives the reported predictions, not a vehicle's raw CSI."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.ho_utils import make_paired_traffic
from utils.o_mappo import (OMAPPOConfig, OMAPPPolicy, encode_prediction_report,
    shared_actor_inputs, make_local_state, make_global_state, state_feature_names,
    o_mappo_reward_presets, run_fluid_o_mappo_episode)
from utils.o_mappo_sim import run_sim_o_mappo


class ReportInputTests(unittest.TestCase):
    def setUp(self):
        self.config = OMAPPOConfig(state_variant="report", information_mode="shared_prediction",
                                   ho_interruption_ms=10, torch_threads=1)
        self.report = dict(gain=np.array([-70., -90., -100., -110.], dtype=np.float32),
                           interference=np.full(4, -135., dtype=np.float32),
                           beam=np.array([[255, 17, 0, 9, 40]] * 4, dtype=np.uint8))
        self.kw = dict(config=self.config, position=[20, 30], heading_deg=0, speed_mps=5,
                  serving_bs=0, serving_sinr_db=-20, queue_ratio=.5, traffic_mbps=19,
                  rb_load=np.zeros(5), user_load=np.ones(5), interference_db=-100,
                  previous_handover=False, system_throughput_ratio=1, own_rb_fraction=.1,
                  tx_beam=None, rx_beam=None, prediction_report=self.report)

    def test_report_size_order_and_scaling(self):
        feature = encode_prediction_report(self.config, self.report).reshape(4, 7)
        np.testing.assert_allclose(feature[:, 0] * 40 - 100, self.report["gain"])
        np.testing.assert_allclose(feature[:, 1] * 40 - 100, self.report["interference"])
        np.testing.assert_array_equal(np.rint(feature[:, 2:] * 255), self.report["beam"])
        self.assertEqual(sum(x.nbytes for x in self.report.values()) * 8, 416)
        state = make_local_state(**self.kw)
        self.assertEqual(state.shape, (57,))
        self.assertEqual(len(state_feature_names(self.config)), 57)
        self.assertFalse(any("pilot" in x or "sinr" in x for x in state_feature_names(self.config)))
        reversed_report = self.report | {"beam": self.report["beam"][:, ::-1]}
        self.assertFalse(np.array_equal(state, make_local_state(**(self.kw | {"prediction_report": reversed_report}))))

    def test_information_boundary(self):
        # Missing raw observations/channel is intentional; this is a wire report.
        inputs = shared_actor_inputs(self.config, {"shared_prediction": self.report})
        self.assertEqual(set(inputs), {"prediction_report"})
        a = make_local_state(**self.kw)
        b = make_local_state(**(self.kw | dict(serving_sinr_db=999, interference_db=999,
                                               pilot_observation=np.full(128, np.nan))))
        np.testing.assert_array_equal(a, b)
        changed = self.report | {"gain": self.report["gain"] + 5}
        self.assertFalse(np.array_equal(a, make_local_state(**(self.kw | {"prediction_report": changed}))))

    def test_validation_and_checkpoint_roundtrip(self):
        for beam in (np.full((4, 5), 256), np.full((4, 5), .5), np.zeros((4, 4))):
            with self.assertRaises(ValueError):
                encode_prediction_report(self.config, self.report | {"beam": beam})
        with self.assertRaises(ValueError):
            dataclasses.replace(self.config, information_mode="legacy").validate()
        state = make_local_state(**self.kw)[None]
        global_state = make_global_state(state, 1)
        policy = OMAPPPolicy(self.config, seed=7)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "policy.pt")
            policy.save(path)
            restored = OMAPPPolicy.load(path)
            self.assertEqual(restored.config.state_variant, "report")
            for a, b in zip(policy.act(state, global_state, False), restored.act(state, global_state, False)):
                np.testing.assert_array_equal(a, b)

    def test_training_and_exact_simulation_without_pilot_field(self):
        rng = np.random.default_rng(8)
        record = dict(pos=np.array([20., 30.]), angle=0., v=0., shared_prediction=self.report,
            h=(rng.normal(size=(8, 4, 32)) + 1j*rng.normal(size=(8, 4, 32))) * 1e-5)
        timeline = {800 + .1*i: {"v": dict(record)} for i in range(5)}
        args = paper_args(1e6)
        policy = OMAPPPolicy(self.config, seed=7)
        result = run_fluid_o_mappo_episode(args, timeline, policy,
            o_mappo_reward_presets()["qos_energy020_load1"], 1, seed=7, learn=True)
        self.assertTrue(np.isfinite(result["average_system_power_w"]))
        result = run_sim_o_mappo(args, MICRO_BS_LOCATIONS, timeline, policy, prt=False,
            rician_fading=False, traffic_trace=make_paired_traffic(args, timeline, 7),
            ho_interruption_ms=10)
        self.assertTrue(np.isfinite(result.energy_record).all())


if __name__ == "__main__":
    unittest.main()
