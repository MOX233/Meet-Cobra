"""Checks for the standalone frozen-NN comparison; no data or model mutation."""
import unittest
import numpy as np
from experiment.compare_stateful_prediction import (
    forward_with_state, StatefulPredictor, sliding_predictions, targets, next_frame_ids,
)
from experiment.benchmark_nn_overhead import BeamPredictionLSTMModel, BestGainPredictionLSTMModel
import torch


class StatefulPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        cls.models = [BeamPredictionLSTMModel(128, 4, 256).eval(), BestGainPredictionLSTMModel(128, 4).eval()]

    @torch.inference_mode()
    def test_wrapper_matches_original_forward(self):
        for model in self.models:
            x = torch.randn(3, 10, 128)
            output, hc = forward_with_state(model, x)
            torch.testing.assert_close(output, model(x), rtol=0, atol=0)
            torch.testing.assert_close(forward_with_state(model, x, hc)[0], model(x, hc), rtol=0, atol=0)

    @torch.inference_mode()
    def test_stream_equals_full_prefix_not_rolling_window(self):
        for model in self.models:
            x = torch.randn(3, 21, 128)
            stream = StatefulPredictor(model)
            for t in range(21):
                out = stream.step(["a", "b", "c"], x[:, t:t+1])
                torch.testing.assert_close(out, model(x[:, :t+1]), atol=2e-6, rtol=2e-5)
            self.assertGreater((out - model(x[:, -10:])).abs().max().item(), 1e-7)

    @torch.inference_mode()
    def test_state_identity_reordering_departure_and_reset(self):
        model = self.models[1]
        stream = StatefulPredictor(model)
        a, b, c, d = (torch.randn(1, 1, 128) for _ in range(4))
        stream.step(["a", "b"], torch.cat([a, b]))
        out = stream.step(["b", "a"], torch.cat([c, d]))
        torch.testing.assert_close(out[0], model(torch.cat([b, c], 1))[0])
        torch.testing.assert_close(out[1], model(torch.cat([a, d], 1))[0])
        stream.step(["a"], d)
        out = stream.step(["b"], c)  # b re-enters: state must be zero.
        torch.testing.assert_close(out, model(c))
        stream.reset()
        torch.testing.assert_close(stream.step(["b"], d), model(d))

    @torch.inference_mode()
    def test_grouped_sliding_matches_single(self):
        histories = [np.random.default_rng(n).normal(size=(n, 128)).astype(np.float32) for n in (1, 10, 3, 10)]
        for model in self.models:
            grouped = sliding_predictions(model, histories, torch.device("cpu"))
            for i, history in enumerate(histories):
                torch.testing.assert_close(grouped[i], model(torch.from_numpy(history[None]))[0])

    def test_target_interference_is_max_not_average(self):
        h = np.zeros((8, 4, 32), dtype=np.complex64)
        h[0, :, 0] = [1e-5, 2e-5, 3e-5, 0]
        record = {"h": h, "g_opt_beam": np.arange(4), "best_beam_pair_idx": np.arange(4)}
        label = targets([record])
        np.testing.assert_allclose(label["true_interfering_gain"][0], 20*np.log10(np.array([1e-5, 2e-5, 3e-5, 0])+1e-9), atol=2e-5)
        np.testing.assert_array_equal(label["nonzero_link"], [[True, True, True, False]])

    def test_score_only_next_frame_common_vehicles(self):
        current, following = {"a": {}, "b": {}}, {"c": {}, "a": {}}
        self.assertEqual(next_frame_ids(current, following), ["a"])


if __name__ == "__main__":
    unittest.main()
