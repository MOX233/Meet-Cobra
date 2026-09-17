"""Regression and information-boundary tests for optimizer-only diagnostics."""
import dataclasses
import unittest
from unittest.mock import patch
import numpy as np
from experiment.o_mappo_optimizer_information import (
    InputAblation, report_estimates, oracle_labels, fixed_allocation,
)
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.beam_utils import generate_dft_codebook
from utils.ho_utils import make_paired_traffic
from utils.o_mappo import OMAPPOConfig, OMAPPOLearnerState, _candidate_links
from utils.o_mappo_sim import run_sim_o_mappo


class OptimizerInformationTests(unittest.TestCase):
    def setUp(self):
        self.args = paper_args(13e6)
        self.config = OMAPPOConfig(state_variant="gain_report", information_mode="shared_prediction")
        rng = np.random.default_rng(41)
        self.record = dict(pos=np.array([20., 30.]), angle=0., v=0.,
            shared_prediction=dict(gain=np.array([-80., -90., -100., -110.]),
                                   interference=np.full(4, -125.)),
            h=(rng.normal(size=(8, 4, 32)) + 1j * rng.normal(size=(8, 4, 32))) * 1e-5)
        self.timeline = {800 + .1 * i: {"v": self.record.copy()} for i in range(4)}
        self.learner = OMAPPOLearnerState(action=1, rx_beam=0, tx_beam=0,
            pending_action=None, last_position=np.zeros(2), distance_since_event=10)

    def test_report_estimator_does_not_require_channel_or_predicted_beams(self):
        public = {"v": {k: self.record[k] for k in ("pos", "shared_prediction")}}
        with patch("experiment.o_mappo_optimizer_information.best_beam_pair", side_effect=AssertionError), \
             patch("experiment.o_mappo_optimizer_information.no_bf_gain_db", side_effect=AssertionError):
            load, allocation = report_estimates(self.args, public, {"v": self.learner},
                {"v": 1.3e6}, self.config, np.zeros(2))
        self.assertEqual(load.shape, (5,))
        self.assertTrue(np.isfinite(load).all())
        self.assertGreater(allocation["v"], 0)
        self.assertLessEqual(allocation["v"], self.args.num_RB_micro)

    def test_true_gains_exactly_reconstruct_legacy_candidates(self):
        labels = oracle_labels(self.timeline, self.config)
        record = dict(self.record, shared_prediction=labels[800]["v"])
        tx, rx = generate_dft_codebook(32), generate_dft_codebook(8)
        kwargs = dict(args=self.args, vehicle="v", record=record, current_bs=1,
            backlog_bits=1.3e6, load=np.array([.2, .3, .4, .5, .6]),
            dft_tx=tx, dft_rx=rx, macro_bs_loc=np.zeros(2))
        new = _candidate_links(config=self.config, **kwargs)
        old = _candidate_links(config=dataclasses.replace(self.config, information_mode="legacy"), **kwargs)
        for a, b in zip(new, old):
            self.assertEqual(a.bs, b.bs)
            self.assertEqual(a.gain_db, b.gain_db)
            self.assertEqual(a.required_rb, b.required_rb)
            self.assertEqual(a.base_cost, b.base_cost)

    def test_noop_hook_is_bitwise_identical(self):
        config = self.config
        class Policy:
            def __init__(self):
                self.config = config
            def act(self, local, global_state, explore):
                return np.ones(len(local), dtype=int), np.zeros(len(local)), np.zeros(len(local))
        common = dict(prt=False, rician_fading=False, ho_interruption_ms=10,
            traffic_trace=make_paired_traffic(self.args, self.timeline, 1))
        old = run_sim_o_mappo(self.args, MICRO_BS_LOCATIONS, self.timeline, Policy(), **common)
        hook = InputAblation("baseline")
        new = run_sim_o_mappo(self.args, MICRO_BS_LOCATIONS, self.timeline, Policy(),
                              optimizer_input_hook=hook, **common)
        self.assertTrue(hook.diagnostics)
        for field in dataclasses.fields(old):
            if field.name.endswith("time_record"):
                continue
            np.testing.assert_equal(getattr(old, field.name), getattr(new, field.name))

    def test_oracle_override_does_not_mutate_actor_report(self):
        labels = oracle_labels(self.timeline, self.config)
        load, fixed = np.zeros(5), {"v": 2.}
        records = self.timeline[800]
        before = records["v"]["shared_prediction"]["gain"].copy()
        out = InputAblation("true_desired", labels)(args=self.args, config=self.config,
            frame=800, records=records, load=load, allocated_rb=fixed)
        np.testing.assert_array_equal(records["v"]["shared_prediction"]["gain"], before)
        np.testing.assert_array_equal(out["records"]["v"]["shared_prediction"]["gain"], labels[800]["v"]["gain"])
        self.assertIs(out["load"], load)
        self.assertIs(out["allocated_rb"], fixed)
        self.assertIs(out["records"]["v"]["shared_prediction"]["interference"], records["v"]["shared_prediction"]["interference"])

    def test_inverse_control_changes_optimizer_not_actor_configuration(self):
        config = dataclasses.replace(self.config, state_variant="adapted", information_mode="legacy")
        feedback_load = np.full(5, .2)
        feedback_fixed = {"v": 3.}
        out = InputAblation("reported_all")(args=self.args, config=config,
            frame=800, records=self.timeline[800], load=np.zeros(5), allocated_rb={"v": 1.},
            feedback_load=feedback_load, feedback_allocated=feedback_fixed)
        self.assertEqual(config.information_mode, "legacy")
        self.assertEqual(out["config"].information_mode, "shared_prediction")
        self.assertIs(out["load"], feedback_load)
        self.assertIs(out["allocated_rb"], feedback_fixed)

    def test_inverse_single_gain_factor_preserves_other_true_gain(self):
        labels = oracle_labels(self.timeline, self.config)
        config = dataclasses.replace(self.config, state_variant="adapted", information_mode="legacy")
        original = self.record["shared_prediction"]
        for variant, replaced, kept in (("predicted_desired", "gain", "interference"),
                                        ("predicted_interference", "interference", "gain")):
            out = InputAblation(variant, labels)(args=self.args, config=config,
                frame=800, records=self.timeline[800], load=np.zeros(5), allocated_rb={"v": 1.})
            pred = out["records"]["v"]["shared_prediction"]
            np.testing.assert_array_equal(pred[replaced], original[replaced])
            np.testing.assert_array_equal(pred[kept], labels[800]["v"][kept])
            self.assertEqual(config.information_mode, "legacy")
        self.assertIs(self.record["shared_prediction"], original)


if __name__ == "__main__":
    unittest.main()
