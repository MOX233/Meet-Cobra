"""Checks for 16 coarse + 16 fine acquisition, with unchanged local tracking."""

import dataclasses
import collections
import unittest

import numpy as np

from experiment.pql_ba_experiment import paper_args
from utils.beam_utils import generate_dft_codebook
from utils.hierarchical_beam import coarse_codebook, hierarchical_beam_pair
from utils.o_mappo import (OMAPPOConfig, OMAPPOCommand, OMAPPOLearnerState,
                          apply_o_mappo_command, average_sweep_pilots, _candidate_links,
                          OMAPPPolicy, run_fluid_o_mappo_episode, o_mappo_reward_presets)
from utils.pql_ba import best_beam_pair, fixed_pair_gain_db, sweep_pilots_for_slot


class HierarchicalBeamTest(unittest.TestCase):
    def setUp(self):
        self.tx = generate_dft_codebook(32)
        self.rx = generate_dft_codebook(8)
        rng = np.random.default_rng(42)
        self.h = rng.normal(size=(8, 4, 32)) + 1j * rng.normal(size=(8, 4, 32))

    def test_unit_norm_and_broad_sector_coverage(self):
        for n, sectors in ((32, 8), (8, 2)):
            wide = coarse_codebook(n, sectors)
            np.testing.assert_allclose(np.linalg.norm(wide, axis=0), 1)
            self.assertTrue(np.all(np.count_nonzero(wide, axis=0) == sectors))
            fine = generate_dft_codebook(n) / np.sqrt(n)
            gains = np.abs(wide.conj().T @ fine) ** 2
            np.testing.assert_array_equal(gains.argmax(axis=0), np.arange(n) // 4)

    def test_all_256_single_path_directions(self):
        for tx in range(32):
            for rx in range(8):
                h = np.outer(self.rx[:, rx], self.tx[:, tx].conj())[:, None, :]
                chosen = hierarchical_beam_pair(h, 0, self.tx, self.rx)
                self.assertEqual(chosen[:2], (tx, rx))
                self.assertAlmostEqual(chosen[2], best_beam_pair(h, 0, self.tx, self.rx)[2])

    def test_fine_gain_is_not_oracle_gain(self):
        rng = np.random.default_rng(2)
        missed = 0
        for _ in range(50):
            h = rng.normal(size=(8, 1, 32)) + 1j * rng.normal(size=(8, 1, 32))
            tx, rx, gain = hierarchical_beam_pair(h, 0, self.tx, self.rx)
            ref = best_beam_pair(h, 0, self.tx, self.rx)
            self.assertAlmostEqual(gain, fixed_pair_gain_db(h, 0, tx, rx, self.tx, self.rx))
            self.assertLessEqual(gain, ref[2] + 1e-10)
            missed += (tx, rx) != ref[:2]
        self.assertGreater(missed, 0)  # Cannot hide a full search behind a 32-probe charge.

    def test_default_and_command_costs(self):
        for variant, probes in (("exhaustive", 256), ("hierarchical32", 32)):
            config = OMAPPOConfig(beam_search_variant=variant)
            config.validate()
            self.assertEqual(config.full_sweep_pilots, probes)
            learner = OMAPPOLearnerState(action=0, rx_beam=None, pending_action=None,
                                        last_position=np.zeros(2), distance_since_event=10)
            record = dict(h=self.h, pos=np.zeros(2))
            outcome = apply_o_mappo_command(learner, OMAPPOCommand(1, 1), record, config, self.tx, self.rx)
            self.assertTrue(outcome.handover)
            self.assertEqual(outcome.sweep_pilots, probes)
            outcome = apply_o_mappo_command(learner, OMAPPOCommand(0, 1), record, config, self.tx, self.rx)
            self.assertFalse(outcome.handover)
            self.assertEqual(outcome.sweep_pilots, 9)
            self.assertEqual(apply_o_mappo_command(learner, None, record, config, self.tx, self.rx).sweep_pilots, 0)
            self.assertEqual(apply_o_mappo_command(learner, OMAPPOCommand(1, 0), record, config, self.tx, self.rx).sweep_pilots, 0)

    def test_pilot_schedule_and_optimizer_cost(self):
        args = paper_args()
        schedule = [sweep_pilots_for_slot(32, 1, i, args.pilot_overhead_factor) for i in range(100)]
        self.assertEqual(schedule, [32] + [1] * 99)
        self.assertAlmostEqual(average_sweep_pilots(args, 32, 1), 1.31)
        self.assertAlmostEqual(average_sweep_pilots(args, 256, 1), 3.51)
        config = OMAPPOConfig(candidate_count=4)
        kwargs = dict(args=args, vehicle="v", record=dict(h=self.h, pos=np.ones(2)),
                      current_bs=0, backlog_bits=1e6, load=np.ones(5) / 2,
                      dft_tx=self.tx, dft_rx=self.rx, macro_bs_loc=np.zeros(2))
        old = {c.bs: c for c in _candidate_links(config=config, **kwargs)}
        new = _candidate_links(config=dataclasses.replace(config, beam_search_variant="hierarchical32"), **kwargs)
        for c in new:
            self.assertEqual(c.gain_db, old[c.bs].gain_db)
            self.assertLess(c.required_rb, old[c.bs].required_rb)

    def test_search_consistent_target_gain(self):
        config = OMAPPOConfig(beam_search_variant='hierarchical32', candidate_gain_mode='search',
                             candidate_count=4)
        record = dict(h=self.h, pos=np.ones(2))
        candidates = _candidate_links(paper_args(), 'v', record, 0, 1e6,
                                      np.ones(5)/2, config, self.tx, self.rx, np.zeros(2))
        for c in candidates:
            expected = hierarchical_beam_pair(self.h, c.bs-1, self.tx, self.rx)
            self.assertEqual((c.tx_beam, c.rx_beam), expected[:2])
            self.assertAlmostEqual(c.gain_db, expected[2])

    def test_collect_only_does_not_update_actor(self):
        import torch
        policy = OMAPPPolicy(OMAPPOConfig(beam_search_variant='hierarchical32',
            candidate_gain_mode='search', ho_interruption_ms=10, torch_threads=1))
        timeline = collections.OrderedDict((i/10, {'v':dict(h=self.h, pos=np.array([i*11.,20.]),
            v=11., angle=0.)}) for i in range(8))
        before = {k:v.clone() for k,v in policy.actor.state_dict().items()}
        result, memory = run_fluid_o_mappo_episode(paper_args(), timeline, policy,
            o_mappo_reward_presets()['qos_energy020_load1'], 5, learn=True, collect_only=True)
        self.assertGreater(len(memory), 0)
        for k,v in policy.actor.state_dict().items():
            self.assertTrue(torch.equal(v,before[k]))
        self.assertEqual(result['actor_loss'], 0)


if __name__ == "__main__":
    unittest.main()
