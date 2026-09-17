"""Information-boundary and outage tests for the shared-prediction baseline."""
import dataclasses
import unittest
from unittest.mock import patch
import numpy as np
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.beam_utils import generate_dft_codebook
from utils.ho_utils import make_paired_traffic
from utils.o_mappo import (OMAPPOConfig, OMAPPOLearnerState, _candidate_links,
                           make_local_state, state_feature_names, optimize_triggered_targets)
from utils.o_mappo_sim import run_sim_o_mappo
from utils.channel_utils import rician_channel_gain
from utils.sim_utils import run_sim_withUMa
from utils.alg_utils import measure_gain
from utils.fast_pet_measurement import measure_pet_batch


class SharedFrontendTests(unittest.TestCase):
    def setUp(self):
        self.args = paper_args(1e6)
        self.config = OMAPPOConfig(state_variant="pilot", information_mode="shared_prediction",
                                   ho_interruption_ms=10)
        self.tx, self.rx = generate_dft_codebook(32), generate_dft_codebook(8)
        self.pred = dict(gain=np.array([-65., -90., -100., -110.]),
                         interference=np.full(4, -135.))

    def test_actor_drops_true_sinr_and_interference(self):
        kw = dict(config=self.config, position=[20, 30], heading_deg=0, speed_mps=5,
                  serving_bs=0, serving_sinr_db=-20, queue_ratio=.5, traffic_mbps=19,
                  rb_load=np.zeros(5), user_load=np.ones(5), interference_db=-100,
                  previous_handover=False, system_throughput_ratio=1, own_rb_fraction=.1,
                  tx_beam=None, rx_beam=None, pilot_observation=np.arange(128)/100)
        a = make_local_state(**kw)
        b = make_local_state(**(kw | dict(serving_sinr_db=100, interference_db=100)))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (157,))
        self.assertNotIn("serving_sinr", state_feature_names(self.config))

    def test_optimizer_needs_no_channel_or_beam_labels(self):
        # Deliberately omit H and all Oracle labels. A physical sweep here
        # would raise, regardless of whether its output is eventually used.
        records = {"v": dict(pos=np.array([10., 20.]), shared_prediction=self.pred)}
        learner = OMAPPOLearnerState(action=0, rx_beam=None, pending_action=None,
                                     last_position=np.zeros(2), distance_since_event=10)
        with patch("utils.o_mappo.best_beam_pair", side_effect=AssertionError("privileged sweep")), \
             patch("utils.o_mappo.no_bf_gain_db", side_effect=AssertionError("privileged gain")):
            for solver in ("greedy", "milp"):
                result = optimize_triggered_targets(self.args, records, {"v": learner}, ["v"],
                    {"v": 1e5}, {"v": 0}, np.zeros(5), self.config, self.tx, self.rx, solver=solver)
                self.assertEqual(result.targets["v"], 1)
                self.assertTrue(result.solver_success)

    def test_ho_changes_capacity_not_target_cost(self):
        record = dict(pos=np.zeros(2), shared_prediction=self.pred)
        a = _candidate_links(self.args, "v", record, 0, 1e5, np.zeros(5),
                             self.config, self.tx, self.rx, np.zeros(2))
        b = _candidate_links(self.args, "v", record, 0, 1e5, np.zeros(5),
                             dataclasses.replace(self.config, ho_interruption_ms=0), self.tx, self.rx, np.zeros(2))
        for x, y in zip(a, b):
            self.assertEqual(x.base_cost, y.base_cost)
            self.assertAlmostEqual(x.required_rb, y.required_rb / .9)

    def test_exact_outage_preserves_arrivals_and_suppresses_pilots(self):
        config = self.config
        class TriggerPolicy:
            def __init__(self):
                self.config = config
            def act(self, local, global_state, explore):
                n = len(local)
                return np.ones(n, dtype=int), np.zeros(n), np.zeros(n)
        record = dict(pos=np.array([20., 30.]), angle=0., v=0.,
                      h=np.full((8, 4, 32), 1e-5, dtype=complex),
                      CSI_preprocessed=np.zeros((1, 128)), shared_prediction=self.pred)
        timeline = {800 + .1*i: {"v": dict(record)} for i in range(4)}
        trace = make_paired_traffic(self.args, timeline, 3)
        result = run_sim_o_mappo(self.args, MICRO_BS_LOCATIONS, timeline, TriggerPolicy(),
            prt=False, rician_fading=False, traffic_trace=trace, ho_interruption_ms=10)
        self.assertEqual(result.handover_record[1], 1)
        # The command from frame 800.1 first takes effect at 800.2.
        start = result.queue_per_vehicle_record[0]["v"][-1]
        expected = start + np.cumsum(trace["arrivals"][800.2]["v"][:10])
        np.testing.assert_allclose(result.queue_per_vehicle_record[1]["v"][:10], expected)

    def test_common_rician_realizations_between_simulators(self):
        class StayPolicy:
            config = self.config
            def act(self, local, global_state, explore):
                return np.zeros(len(local), dtype=int), np.zeros(len(local)), np.zeros(len(local))
        pred = self.pred | dict(beam=np.zeros((4, 5), dtype=int))
        record = dict(pos=np.array([20., 30.]), angle=0., v=0.,
                      h=np.full((8, 4, 32), 1e-5, dtype=complex),
                      CSI_preprocessed=np.zeros((1, 128)), shared_prediction=pred)
        timeline = {800 + .1*i: {"b": dict(record), "a": dict(record)} for i in range(3)}
        cache = {f: {v: pred for v in r} for f, r in timeline.items()}
        trace = make_paired_traffic(self.args, timeline, 3)
        traces = [[], []]
        def capture(index):
            def draw(*a, **kw):
                value = rician_channel_gain(*a, **kw)
                traces[index].append(value.ravel()[:5].copy())
                return value
            return draw
        with patch("utils.o_mappo_sim.rician_channel_gain", side_effect=capture(0)):
            run_sim_o_mappo(self.args, MICRO_BS_LOCATIONS, timeline, StayPolicy(),
                prt=False, rician_fading=True, traffic_trace=trace, paired_fading_seed=3)
        self.args.device = "cpu"
        def stay(args, vehicles, *a, **kw):
            return {v: 0 for v in vehicles}, np.zeros(5)
        with patch("utils.alg_utils.rician_channel_gain", side_effect=capture(1)):
            run_sim_withUMa(self.args, MICRO_BS_LOCATIONS, timeline, None, True, True, True,
                HO_func=stay, prt=False, save_pilot=True, K_BF=5, prediction_cache=cache,
                traffic_trace=trace, rician_fading=True, paired_fading_seed=3)
        self.assertEqual(len(traces[0]), 400)
        np.testing.assert_array_equal(traces[0], traces[1])

    def test_batched_pet_arithmetic_and_pilot_parity(self):
        rng = np.random.default_rng(8)
        records = {str(i): {"h": (rng.normal(size=(8,4,32)) + 1j*rng.normal(size=(8,4,32))) * 1e-6}
                   for i in range(7)}
        pairs = {v: rng.integers(0, 256, (4,5)) for v in records}
        pred = {v: rng.uniform(-125, -95, 4) for v in records}
        positional = (self.args, 0, list(records), {0: records}, MICRO_BS_LOCATIONS,
                      pairs, pred, self.tx, self.rx, "topKbeam_savePilot", {})
        for fading in (False, True):
            np.random.seed(33)
            old = measure_gain(*positional, rician_fading=fading, K_BF=5)
            np.random.seed(33)
            new = measure_pet_batch(*positional, rician_fading=fading, K_BF=5)
            for i, (a, b) in enumerate(zip(old, new)):
                for v in records:
                    if i < 2:
                        np.testing.assert_allclose(a[v], b[v], atol=1e-10, rtol=1e-12)
                    else:
                        np.testing.assert_array_equal(a[v], b[v])


if __name__ == "__main__":
    unittest.main()
