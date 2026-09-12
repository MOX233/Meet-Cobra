"""Fast checks for the revision-only NN overhead audit (no simulations)."""

import unittest

import torch

from experiment.benchmark_nn_overhead import (
    BeamPredictionLSTMModel,
    BestGainPredictionLSTMModel,
    matrix_arithmetic,
    report_bits,
)


class OverheadAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.beam = BeamPredictionLSTMModel(128, 4, 256).eval()
        cls.gain = BestGainPredictionLSTMModel(128, 4).eval()

    def test_parameter_inventory(self):
        beam = sum(p.numel() for p in self.beam.parameters())
        gain = sum(p.numel() for p in self.gain.parameters())
        self.assertEqual(beam, 1060864)
        self.assertEqual(gain, 533524)
        self.assertEqual(beam + 2 * gain, 2127912)

    def test_history_length_scaling_and_shortcuts(self):
        for model in (self.beam, self.gain):
            one = matrix_arithmetic(model, torch.zeros(1, 1, 128))
            ten = matrix_arithmetic(model, torch.zeros(1, 10, 128))
            self.assertEqual(ten["macs"] - one["macs"], 9 * 4 * 128 * (128 + 128))
            self.assertEqual(ten["matrix_flops"], 2 * ten["macs"])
            self.assertEqual(sum("downsample" in row["module"] for row in ten["modules"]), 4)

    def test_batch_scaling(self):
        one = matrix_arithmetic(self.beam, torch.zeros(1, 10, 128))
        two = matrix_arithmetic(self.beam, torch.zeros(2, 10, 128))
        self.assertEqual(two["matrix_flops"], 2 * one["matrix_flops"])

    def test_forward_hooks_are_removed(self):
        matrix_arithmetic(self.beam, torch.zeros(1, 10, 128))
        self.assertTrue(all(not m._forward_hooks for m in self.beam.modules()))

    def test_payloads_and_common_period(self):
        budget = report_bits()
        bits = budget["bits_per_vehicle_per_frame"]
        self.assertEqual(bits["full_prediction_output"], 33024)
        self.assertEqual(bits["full_complex_microcell_CSI"], 65536)
        self.assertEqual(bits["new_superposed_CSI_input_for_central_NN"], 4096)
        self.assertEqual(budget["primary_report"], "ranked_candidates_and_two_gains")
        self.assertEqual(bits[budget["primary_report"]], 416)
        self.assertEqual(budget["primary_report_components_bit"], {"ranked_indices": 160, "two_gains_per_BS": 256})
        self.assertAlmostEqual(budget["kbit_per_vehicle_per_s"][budget["primary_report"]], 4.16)
        self.assertEqual(budget["prediction_to_full_CSI_ratio"], 416 / 65536)
        self.assertEqual(budget["prediction_to_superposed_input_ratio"], 416 / 4096)
        self.assertEqual(report_bits(scalar_bits=16)["bits_per_vehicle_per_frame"][budget["primary_report"]], 288)

    def test_ranked_candidate_budget_scaling(self):
        one, five = report_bits(top_k=1), report_bits(top_k=5)
        key = five["primary_report"]
        self.assertEqual(five["bits_per_vehicle_per_frame"][key] - one["bits_per_vehicle_per_frame"][key], 4 * 4 * 8)
        self.assertEqual(five["bits_per_beam_pair_index"], 8)


if __name__ == "__main__":
    unittest.main()
