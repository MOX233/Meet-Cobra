import os
import tempfile
import unittest

import numpy as np
import torch

from experiment.dql_hbt_experiment import _aggregate_seeds
from utils.beam_utils import generate_dft_codebook
from utils.dql_hbt import (
    DQLHBTConfig,
    DQLHBTPolicy,
    HBTLearnerState,
    apply_hbt_action,
    hbt_action_target_bs,
    local_track_beam_pair,
    make_state_vector,
    state_feature_names,
    valid_action_mask,
)
from utils.pql_ba import fixed_pair_gain_db


class DQLHBTTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9)
        self.channel = self.rng.normal(size=(8, 4, 32)) + 1j * self.rng.normal(
            size=(8, 4, 32)
        )
        self.dft_tx = generate_dft_codebook(32)
        self.dft_rx = generate_dft_codebook(8)

    def test_action_semantics_and_mask(self):
        config = DQLHBTConfig()
        self.assertIsNone(hbt_action_target_bs(0, config))
        self.assertEqual(hbt_action_target_bs(1, config), 0)
        self.assertEqual(hbt_action_target_bs(5, config), 4)
        mask = valid_action_mask(3, config)
        self.assertTrue(mask[0])
        self.assertFalse(mask[4])
        self.assertEqual(int(mask.sum()), 5)

    def test_local_tracking_is_exhaustive_inside_neighbourhood(self):
        tx, rx, gain, pilots = local_track_beam_pair(
            self.channel,
            micro_index=2,
            tx_beam=0,
            rx_beam=0,
            dft_tx=self.dft_tx,
            dft_rx=self.dft_rx,
        )
        candidates_tx = [31, 0, 1]
        candidates_rx = [7, 0, 1]
        exhaustive = {
            (a, b): fixed_pair_gain_db(
                self.channel, 2, a, b, self.dft_tx, self.dft_rx
            )
            for a in candidates_tx
            for b in candidates_rx
        }
        expected = max(exhaustive, key=exhaustive.get)
        self.assertEqual((tx, rx), expected)
        self.assertAlmostEqual(gain, exhaustive[expected], places=9)
        self.assertEqual(pilots, 9)

    def test_handover_full_sweep_then_local_tracking(self):
        config = DQLHBTConfig()
        learner = HBTLearnerState(
            action=0,
            rx_beam=None,
            pending_action=None,
            last_position=np.zeros(2),
            distance_since_event=0.0,
        )
        outcome = apply_hbt_action(
            learner,
            dql_action=2,  # target micro BS 1
            vehicle_record={"h": self.channel},
            config=config,
            dft_tx=self.dft_tx,
            dft_rx=self.dft_rx,
        )
        self.assertTrue(outcome.handover)
        self.assertEqual(outcome.sweep_pilots, 256)
        self.assertEqual(learner.action, 1)
        old_pair = (learner.tx_beam, learner.rx_beam)
        outcome = apply_hbt_action(
            learner,
            dql_action=0,
            vehicle_record={"h": self.channel},
            config=config,
            dft_tx=self.dft_tx,
            dft_rx=self.dft_rx,
        )
        self.assertFalse(outcome.handover)
        self.assertEqual(outcome.sweep_pilots, 9)
        self.assertIsNotNone(old_pair[0])

    def test_source_and_adapted_state_dimensions(self):
        common = dict(
            position=(100.0, -250.0),
            heading_deg=90.0,
            speed_mps=15.0,
            serving_bs=2,
            sinr_db=5.0,
            tracking_indicator=True,
            queue_ratio=1.5,
            load_ratio=np.linspace(0.1, 0.5, 5),
            interference_db=10.0,
            traffic_mbps=19.0,
            tx_beam=7,
            rx_beam=3,
        )
        source = DQLHBTConfig(state_variant="source")
        adapted = DQLHBTConfig(state_variant="adapted")
        source_state = make_state_vector(source, **common)
        adapted_state = make_state_vector(adapted, **common)
        self.assertEqual(source_state.size, len(state_feature_names(source)))
        self.assertEqual(adapted_state.size, len(state_feature_names(adapted)))
        self.assertGreater(adapted_state.size, source_state.size)
        self.assertTrue(np.all(np.isfinite(adapted_state)))

    def test_replay_gradient_and_masked_action(self):
        config = DQLHBTConfig(
            hidden_sizes=(16,),
            batch_size=8,
            replay_capacity=64,
            replay_warmup=8,
            train_every_transitions=1,
            target_update_steps=2,
            epsilon_start=0.0,
            epsilon_end=0.0,
            torch_threads=1,
        )
        policy = DQLHBTPolicy(config, seed=3)
        state = np.zeros(policy.state_dim, dtype=np.float32)
        next_state = np.ones(policy.state_dim, dtype=np.float32) * 0.1
        for i in range(12):
            serving = i % 5
            action = 0
            policy.observe(state, action, 1.0, next_state, serving)
        losses = policy.learn_available()
        self.assertTrue(losses)
        self.assertTrue(np.all(np.isfinite(losses)))
        selected = policy.select_action(state, serving_bs=2, explore=False)
        self.assertNotEqual(selected, 3)  # action 1+serving_bs is invalid

    def test_checkpoint_round_trip(self):
        config = DQLHBTConfig(hidden_sizes=(16,), torch_threads=1)
        policy = DQLHBTPolicy(config, seed=4)
        state = np.zeros(policy.state_dim, dtype=np.float32)
        before = policy.select_action(state, serving_bs=0, explore=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "policy.pt")
            policy.save(path)
            restored = DQLHBTPolicy.load(path, seed=4)
        after = restored.select_action(state, serving_bs=0, explore=False)
        self.assertEqual(before, after)

    def test_three_seed_ci_uses_student_t(self):
        metric_names = (
            "average_system_power_w",
            "queue_violation_percent",
            "average_queueing_proxy_ms",
            "handover_per_vehicle_per_s",
            "beam_switch_per_vehicle_per_s",
            "average_pilots_per_micro_link_slot",
            "macro_association_ratio",
        )
        raw = {}
        for seed, value in enumerate((0.0, 1.0, 2.0), start=1):
            item = {name: value for name in metric_names}
            item["data_rate_mbps"] = 1.0
            raw["seed_{}".format(seed)] = item
        summary = _aggregate_seeds(raw)["rate_1Mbps"]
        expected = 4.302652730 / np.sqrt(3.0)  # sample standard deviation is 1
        self.assertAlmostEqual(
            summary["average_system_power_w_ci95"], expected, places=10
        )


if __name__ == "__main__":
    unittest.main()
