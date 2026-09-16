"""Checks for the batch-one stateful inference timing implementation."""
import unittest

import numpy as np
import torch

from experiment.benchmark_stateful_nn_overhead import (
    BeamPredictionLSTMModel, BestGainPredictionLSTMModel, StatefulStepModule,
    VehicleStatefulPipeline, extract_trajectories, forward_with_state,
    matrix_arithmetic, measure_streams, validate_pipeline,
)


class StatefulOverheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(314)
        cls.models = {"beam": BeamPredictionLSTMModel(128, 4, 256).eval(),
                      "desired_gain": BestGainPredictionLSTMModel(128, 4).eval(),
                      "interfering_gain": BestGainPredictionLSTMModel(128, 4).eval()}

    def test_stream_outputs_match_native_full_prefix_and_reset(self):
        trajectory = {"csi": np.random.default_rng(17).normal(size=(22, 128)).astype(np.float32)}
        result = validate_pipeline(self.models, trajectory)
        self.assertTrue(result["ranked_indices_match"])
        self.assertIn(21, result["prefix_lengths_checked"])
        self.assertTrue(result["trajectory_reset_matches_fresh_state"])

    @torch.inference_mode()
    def test_persistent_state_is_not_reset_at_ten_frames(self):
        x = torch.randn(1, 21, 128)
        pipeline = VehicleStatefulPipeline(self.models)
        for i in range(21):
            pipeline.step(x[0, i].numpy())
        for name, model in self.models.items():
            _, expected = model.lstm_layers(x)
            _, window = model.lstm_layers(x[:, -10:])
            for actual, target in zip(pipeline.states[name], expected):
                torch.testing.assert_close(actual, target, rtol=2e-5, atol=2e-6)
            self.assertGreater((pipeline.states[name][0] - window[0]).abs().max().item(), 1e-7)

    @torch.inference_mode()
    def test_single_step_matrix_count_with_carried_state(self):
        x = torch.randn(1, 1, 128)
        expected = {"beam": 2097152, "desired_gain": 1050632, "interfering_gain": 1050632}
        for name, model in self.models.items():
            _, state = forward_with_state(model, x)
            result = matrix_arithmetic(StatefulStepModule(model, state), x)
            self.assertEqual(result["matrix_flops"], expected[name])
            self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def test_trajectory_extraction_splits_gaps_and_takes_latest_csi(self):
        def record(value):
            return {"CSI_preprocessed": np.full((10, 128), value, dtype=np.float32)}
        timeline = {0.0: {"a": record(1), "b": record(3)},
                    0.1: {"a": record(2)}, 0.2: {"b": record(4)}}
        trajectories = extract_trajectories(timeline)
        self.assertEqual([t["vehicle"] for t in trajectories], ["a", "b", "b"])
        self.assertEqual([len(t["csi"]) for t in trajectories], [2, 1, 1])
        np.testing.assert_array_equal(trajectories[0]["csi"][:, 0], [1, 2])

    def test_timing_counts_all_frames_and_rounds(self):
        trajectories = [{"csi": np.zeros((3, 128), dtype=np.float32)} for _ in range(2)]
        result, raw = measure_streams(VehicleStatefulPipeline(self.models), trajectories, 2, 1, 7)
        self.assertEqual(result["samples"], 12)
        self.assertEqual(result["trajectory_starts"]["samples"], 4)
        self.assertEqual(raw.shape, (12, 4))
        self.assertTrue((raw[:, 3] > 0).all())


if __name__ == "__main__":
    unittest.main()
