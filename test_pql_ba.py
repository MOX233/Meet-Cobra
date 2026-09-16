import unittest
import pickle

import numpy as np

from utils.beam_utils import generate_dft_codebook
from utils.pql_ba_adapted import contextual_config, reward_presets
from utils.pql_ba import (
    PQLBAConfig,
    PQLBAPolicy,
    _LearnerState,
    _apply_pending_action,
    action_to_link,
    best_beam_pair,
    best_rx_for_tx,
    fixed_pair_gain_db,
    link_to_action,
    sweep_pilots_for_slot,
)


class PQLBATest(unittest.TestCase):
    def test_action_encoding_is_bijective(self):
        config = PQLBAConfig()
        for action in range(config.num_actions):
            self.assertEqual(
                link_to_action(*action_to_link(action, config), config), action
            )

    def test_receive_sweep_matches_best_fixed_pair(self):
        rng = np.random.default_rng(7)
        channel = rng.normal(size=(8, 4, 32)) + 1j * rng.normal(size=(8, 4, 32))
        dft_tx = generate_dft_codebook(32)
        dft_rx = generate_dft_codebook(8)
        rx_beam, swept_gain = best_rx_for_tx(channel, 2, 11, dft_tx, dft_rx)
        fixed_gains = [
            fixed_pair_gain_db(channel, 2, 11, rx, dft_tx, dft_rx)
            for rx in range(8)
        ]
        self.assertEqual(rx_beam, int(np.argmax(fixed_gains)))
        self.assertAlmostEqual(swept_gain, max(fixed_gains), places=10)

    def test_hierarchical_action_and_full_pair_sweep(self):
        rng = np.random.default_rng(13)
        channel = rng.normal(size=(8, 4, 32)) + 1j * rng.normal(size=(8, 4, 32))
        dft_tx = generate_dft_codebook(32)
        dft_rx = generate_dft_codebook(8)
        config = PQLBAConfig(hierarchical_bs_action=True)
        self.assertEqual(config.num_actions, 5)
        self.assertEqual(action_to_link(3, config), (3, None))
        tx_beam, rx_beam, gain = best_beam_pair(
            channel, 2, dft_tx, dft_rx
        )
        exhaustive = np.asarray(
            [
                fixed_pair_gain_db(channel, 2, tx, rx, dft_tx, dft_rx)
                for tx in range(32)
                for rx in range(8)
            ]
        ).reshape(32, 8)
        expected = np.unravel_index(int(np.argmax(exhaustive)), exhaustive.shape)
        self.assertEqual((tx_beam, rx_beam), expected)
        self.assertAlmostEqual(gain, float(exhaustive[expected]), places=10)

    def test_full_sweep_is_spread_below_zero_capacity(self):
        factor = 2.0 / 112.0
        pilots = [sweep_pilots_for_slot(256, 1, slot, factor) for slot in range(100)]
        self.assertEqual(pilots[:5], [55, 55, 55, 55, 36])
        self.assertTrue(all(count * factor < 1.0 for count in pilots))
        self.assertEqual(sum(pilots), 256 + 95)

    def test_old_config_pickle_is_backfilled(self):
        config = PQLBAConfig()
        del config.__dict__["hierarchical_bs_action"]
        restored = pickle.loads(pickle.dumps(config))
        self.assertFalse(restored.hierarchical_bs_action)
        self.assertFalse(restored.include_queue_state)

    def test_same_tx_action_refreshes_receive_beam(self):
        rng = np.random.default_rng(11)
        channel = rng.normal(size=(8, 4, 32)) + 1j * rng.normal(size=(8, 4, 32))
        config = PQLBAConfig()
        dft_tx = generate_dft_codebook(32)
        dft_rx = generate_dft_codebook(8)
        expected_rx, _ = best_rx_for_tx(channel, 0, 0, dft_tx, dft_rx)
        learner = _LearnerState(
            action=1,
            rx_beam=(expected_rx + 1) % 8,
            pending_action=1,
            last_position=np.zeros(2),
            distance_since_event=0.0,
        )
        changed = _apply_pending_action(
            learner, {"h": channel}, config, dft_tx, dft_rx
        )
        self.assertFalse(changed)
        self.assertEqual(learner.rx_beam, expected_rx)

    def test_location_and_heading_state(self):
        config = PQLBAConfig(
            include_heading=True,
            num_heading_bins=4,
            location_bin_size_m=100.0,
        )
        policy = PQLBAPolicy(config)
        state = policy.make_state(-90.0, 1, heading_deg=90.0, position=(0.0, 0.0))
        self.assertEqual(state, (5, 1, 1, 5, 5))

    def test_q_update(self):
        config = PQLBAConfig(learning_rate=1.0, discount_factor=0.0)
        policy = PQLBAPolicy(config)
        state = (2, 0)
        next_state = (3, 1)
        td_error = policy.update(state, 4, reward=3.5, next_state=next_state)
        self.assertAlmostEqual(td_error, 3.5)
        self.assertAlmostEqual(float(policy.q_table[state][4]), 3.5)
        self.assertEqual(policy.visited_state_action_pairs, 1)

    def test_frozen_policy_ignores_unvisited_zero_actions(self):
        config = PQLBAConfig(learning_rate=1.0, discount_factor=0.0)
        policy = PQLBAPolicy(config)
        state = (2, 0)
        policy.update(state, 7, reward=-3.0, next_state=(3, 1))
        selected = policy.select_action(
            state, np.random.default_rng(1), explore=False
        )
        self.assertEqual(selected, 7)

    def test_contextual_state_bins(self):
        policy = PQLBAPolicy(contextual_config(include_traffic_state=True))
        state = policy.make_state(
            -90.0,
            1,
            heading_deg=90.0,
            position=(0.0, 0.0),
            queue_ratio=1.0,
            load_ratio=0.6,
            interference_db=5.0,
            traffic_mbps=19.0,
        )
        self.assertEqual(state, (5, 1, 1, 5, 5, 3, 2, 2, 3))

    def test_energy_reward_presets_are_ordered(self):
        names = ["qos", "qos_energy_005", "qos_energy_020", "qos_energy_050"]
        weights = [reward_presets()[name].energy_weight for name in names]
        self.assertEqual(weights, [0.0, 0.05, 0.2, 0.5])


if __name__ == "__main__":
    unittest.main()
