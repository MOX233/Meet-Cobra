"""Regression checks for the unified Fig.5--8 experiment."""
import dataclasses
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.alg_utils import measure_gain
from utils.beam_utils import generate_dft_codebook
from utils.gpu_phy import GPUFramePHY
from utils.ho_utils import make_paired_traffic
from utils.mts_gs_hbf import MTSCommand, MTSLinkState, candidate_configs, build_link_candidates
from utils.mts_gs_hbf_sim import run_sim_mts_gs_hbf


class RevisionGridTests(unittest.TestCase):
    def setUp(self):
        self.args = paper_args(13e6)
        rng = np.random.default_rng(7)
        self.records = {v: dict(pos=np.array([40., 30.]),
            h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5)
            for v in ("a", "b")}
        self.tx, self.rx = generate_dft_codebook(32), generate_dft_codebook(8)
        self.config = candidate_configs()["pressure_early"]

    def test_gpu_random_probes_match_corrected_scalar_rule(self):
        self.args.slots_per_frame = 4
        phy = GPUFramePHY(self.args, self.records, 801, 1, os.environ.get("TEST_PHY_DEVICE", "cpu"))
        batched = phy.random_tracking({}, 51)
        channels = phy.h.cpu().numpy()
        np.random.seed(51)
        previous = {}
        # Dummy predictions deliberately cannot identify selected random probes.
        beams = {v: np.zeros((4,5), dtype=int) for v in phy.ids}
        gains = {v: np.full(4, -80.) for v in phy.ids}
        for slot in range(4):
            records = {v: {"h": channels[slot,j]} for j,v in enumerate(phy.ids)}
            scalar = measure_gain(self.args, 0, phy.ids, {0: records}, MICRO_BS_LOCATIONS,
                beams, gains, self.tx, self.rx, "topKbeam_NoPred", previous,
                correct_random_beam_index=True, K_BF=5)
            for index in range(4):
                np.testing.assert_allclose(batched[index][slot], np.stack([scalar[index][v] for v in phy.ids]),
                                           rtol=1e-12, atol=1e-10)
            previous = scalar[2]
        blocked = np.ones((4, len(phy.ids)), dtype=bool)
        previous = {v: np.full(4, 7) for v in phy.ids}
        frozen = phy.random_tracking(previous, 51, blocked=blocked)
        self.assertTrue((frozen[2] == 7).all() and (frozen[3] == 0).all())

    def test_mts_capacity_factor_does_not_change_preference_scores(self):
        states = {v: MTSLinkState() for v in self.records}
        kwargs = dict(args=self.args, records=self.records, states=states,
            queue_bits={v: 1e4 for v in states}, queue_upper_bound={v: 2.6e5 for v in states},
            vehicle_rate={v: 13e6 for v in states}, estimated_load=np.zeros(5),
            dft_tx=self.tx, dft_rx=self.rx)
        a = build_link_candidates(config=self.config, **kwargs)
        b = build_link_candidates(config=dataclasses.replace(self.config, ho_interruption_ms=10), **kwargs)
        for v in a:
            for old, new in zip(a[v], b[v]):
                self.assertEqual(old.vehicle_score, new.vehicle_score)
                self.assertEqual(old.bs_score, new.bs_score)
                self.assertAlmostEqual(new.demand_rb, old.demand_rb / (.9 if new.bs else 1))

    def test_mts_interruption_and_pairing(self):
        timeline = {800 + .1*i: self.records for i in range(5)}
        def force(*args):
            states = args[2]
            target = 1 if next(iter(states.values())).action == 0 else 0
            commands = {v: MTSCommand(target, 0 if target else None, 0 if target else None,
                256 if target else 0, "association") for v in states}
            return commands, SimpleNamespace(proposal_count=len(states), unassigned=(), used_capacity=np.zeros(5))
        config = dataclasses.replace(self.config, association_interval_frames=1, full_sweep_interval_frames=1)
        traffic = make_paired_traffic(self.args, timeline, 3)
        diagnostics = []
        with patch("utils.mts_gs_hbf_sim.association_commands", side_effect=force):
            result = run_sim_mts_gs_hbf(self.args, MICRO_BS_LOCATIONS, timeline, config,
                prt=False, rician_fading=False, traffic_trace=traffic, ho_interruption_ms=10,
                ho_diagnostics=diagnostics)
        self.assertEqual([d["blocked_vehicle_slots"] for d in diagnostics], [0,20,20,20])
        for fi, d in enumerate(diagnostics):
            for v in d["switched"]:
                queue = result.queue_per_vehicle_record[fi][v]
                np.testing.assert_allclose(np.diff(queue[:10]), traffic["arrivals"][d["frame"]][v][1:10], rtol=0, atol=1e-9)
        # Exact default equivalence when opt-in features are disabled.
        a = run_sim_mts_gs_hbf(self.args, MICRO_BS_LOCATIONS, timeline, config, prt=False, rician_fading=False)
        b = run_sim_mts_gs_hbf(self.args, MICRO_BS_LOCATIONS, timeline, config, prt=False, rician_fading=False,
                              ho_interruption_ms=0, traffic_trace=None, paired_fading_seed=None)
        for field in dataclasses.fields(a):
            if not field.name.endswith("time_record"):
                np.testing.assert_equal(getattr(a, field.name), getattr(b, field.name))


if __name__ == "__main__":
    unittest.main()
