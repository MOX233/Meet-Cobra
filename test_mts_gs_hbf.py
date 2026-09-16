import collections
import unittest

import numpy as np

from experiment.pql_ba_experiment import MICRO_BS_LOCATIONS, paper_args
from utils.beam_utils import generate_dft_codebook
from utils.mts_gs_hbf import (
    MTSCommand,
    MTSGSHBFConfig,
    MTSLinkCandidate,
    MTSLinkState,
    apply_mts_command,
    build_link_candidates,
    candidate_configs,
    capacity_aware_gale_shapley,
)
from utils.mts_gs_hbf_sim import run_sim_mts_gs_hbf


def candidate(vehicle, bs, vehicle_score, bs_score, demand=1.0):
    return MTSLinkCandidate(
        vehicle=vehicle,
        bs=bs,
        tx_beam=None if bs == 0 else 0,
        rx_beam=None if bs == 0 else 0,
        gain_db=-80.0,
        sinr_db=10.0,
        capacity_per_rb_bps=1e6,
        demand_rb=demand,
        vehicle_score=vehicle_score,
        bs_score=bs_score,
    )


class MTSGSHBFTest(unittest.TestCase):
    def test_all_candidate_configs_validate(self):
        for config in candidate_configs().values():
            config.validate()

    def test_capacity_aware_deferred_acceptance(self):
        candidates = {
            "a": [candidate("a", 0, 3.0, 1.0), candidate("a", 1, 2.0, 2.0)],
            "b": [candidate("b", 0, 3.0, 3.0), candidate("b", 1, 2.0, 1.0)],
            "c": [candidate("c", 0, 3.0, 2.0), candidate("c", 1, 2.0, 3.0)],
        }
        result = capacity_aware_gale_shapley(candidates, [2.0, 2.0])
        self.assertEqual(set(result.assignments), {"a", "b", "c"})
        self.assertEqual(result.assignments["b"].bs, 0)
        self.assertEqual(result.assignments["c"].bs, 0)
        self.assertEqual(result.assignments["a"].bs, 1)
        self.assertTrue(np.all(result.used_capacity <= 2.0 + 1e-9))
        self.assertGreaterEqual(result.proposal_count, 4)

    def test_overload_falls_back_to_current_link(self):
        candidates = {
            "a": [candidate("a", 0, 1.0, 2.0, demand=2.0)],
            "b": [candidate("b", 0, 1.0, 1.0, demand=2.0)],
        }
        result = capacity_aware_gale_shapley(
            candidates, [2.0], current_bs={"a": 0, "b": 0}
        )
        self.assertEqual(set(result.assignments), {"a", "b"})
        self.assertEqual(result.unassigned, ("b",))
        self.assertAlmostEqual(result.used_capacity[0], 4.0)

    def test_causal_command_application(self):
        config = MTSGSHBFConfig()
        state = MTSLinkState()
        command = MTSCommand(1, 3, 2, config.full_sweep_pilots, "association")
        outcome = apply_mts_command(state, command, config)
        self.assertTrue(outcome.handover)
        self.assertTrue(outcome.beam_switch)
        self.assertEqual(state.action, 1)
        self.assertEqual((state.tx_beam, state.rx_beam), (3, 2))
        outcome = apply_mts_command(state, None, config)
        self.assertFalse(outcome.handover)
        self.assertEqual(state.current_sweep_pilots, 0)

    def test_candidate_dimensions_and_finiteness(self):
        args = paper_args(7e6)
        config = MTSGSHBFConfig()
        dft_tx = generate_dft_codebook(config.num_tx_beams)
        dft_rx = generate_dft_codebook(config.num_rx_beams)
        channel = (
            np.ones((config.num_rx_beams, config.num_micro_bs, config.num_tx_beams))
            + 1j
        ) * 1e-5
        records = {"v": {"pos": np.asarray([10.0, 0.0]), "h": channel}}
        states = {"v": MTSLinkState()}
        links = build_link_candidates(
            args,
            records,
            states,
            {"v": 1e5},
            {"v": args.lat_slot_ub * args.data_rate * args.slot_len},
            {"v": args.data_rate},
            np.zeros(config.num_bs),
            config,
            dft_tx,
            dft_rx,
        )["v"]
        self.assertGreaterEqual(len(links), 1)
        self.assertEqual(len({link.bs for link in links}), len(links))
        for link in links:
            self.assertTrue(np.isfinite(link.vehicle_score))
            self.assertTrue(np.isfinite(link.bs_score))
            self.assertGreater(link.capacity_per_rb_bps, 0.0)

    def test_minimal_exact_simulation(self):
        args = paper_args(1e6)
        config = MTSGSHBFConfig(
            association_interval_frames=2,
            full_sweep_interval_frames=1,
            local_tracking_interval_frames=1,
        )
        channel = (
            np.ones((config.num_rx_beams, config.num_micro_bs, config.num_tx_beams))
            + 1j
        ) * 1e-5
        timeline = collections.OrderedDict()
        for index in range(4):
            timeline[index * 0.1] = {
                "v": {
                    "pos": np.asarray([10.0 + index, 0.0]),
                    "angle": 0.0,
                    "v": 10.0,
                    "h": channel,
                }
            }
        result = run_sim_mts_gs_hbf(
            args,
            MICRO_BS_LOCATIONS,
            timeline,
            config,
            seed=1,
            prt=False,
            rician_fading=False,
        )
        self.assertEqual(result.energy_record.shape, (3,))
        self.assertEqual(result.rb_allocated_record.shape, (3, 5))
        self.assertTrue(np.isfinite(result.energy_record).all())
        self.assertEqual(result.association_epoch_record.sum(), 2)


if __name__ == "__main__":
    unittest.main()
