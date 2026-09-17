"""Check batched PHY arithmetic against scalar code on identical channels."""
import os
import unittest
import numpy as np
import torch
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.gpu_phy import GPUFramePHY
from utils.alg_utils import measure_gain
from utils.beam_utils import generate_dft_codebook
from utils.pql_ba import fixed_pair_gain_db
from utils.o_mappo import OMAPPOLearnerState


class GPUPhyTests(unittest.TestCase):
    def test_measurement_parity_pairing_and_distribution(self):
        args = paper_args()
        args.slots_per_frame = 20
        rng = np.random.default_rng(13)
        records = {v: dict(h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-6)
                   for v in ("a", "b", "c")}
        device = os.environ.get("TEST_PHY_DEVICE", "cpu")
        physical = GPUFramePHY(args, records, 801.2, 3, device)
        repeated = GPUFramePHY(args, records, 801.2, 3, device)
        self.assertTrue(torch.equal(physical.h, repeated.h))
        h = physical.h.cpu().numpy()
        original = np.stack([records[v]["h"] for v in physical.ids])
        factor = np.abs(h / original[None]) ** 2
        self.assertAlmostEqual(float(factor.mean()), 1., delta=.02)
        pairs = {v: rng.integers(0,256,(4,5)) for v in records}
        predicted = {v: rng.uniform(-130,-95,4) for v in records}
        batched = physical.pet(pairs, predicted)
        learners = {v: OMAPPOLearnerState(action=j+1, tx_beam=2*j, rx_beam=j,
                        pending_action=None, last_position=np.zeros(2), distance_since_event=0)
                    for j, v in enumerate(physical.ids)}
        connection = {v: learners[v].action for v in records}
        fixed = physical.fixed_pairs(connection, learners)
        tx, rx = generate_dft_codebook(32), generate_dft_codebook(8)
        for slot in range(3):
            current = {v: {"h": h[slot,j]} for j,v in enumerate(physical.ids)}
            scalar = measure_gain(args, 0, physical.ids, {0:current}, MICRO_BS_LOCATIONS,
                pairs, predicted, tx, rx, "topKbeam_savePilot", {}, K_BF=5)
            for item in range(4):
                for j,v in enumerate(physical.ids):
                    np.testing.assert_allclose(batched[item][slot,j], scalar[item][v], atol=1e-10, rtol=1e-12)
            for j,v in enumerate(physical.ids):
                value = fixed_pair_gain_db(h[slot,j], connection[v]-1,
                                           learners[v].tx_beam, learners[v].rx_beam, tx, rx)
                self.assertAlmostEqual(fixed[slot,j], value, places=10)


if __name__ == "__main__":
    unittest.main()
