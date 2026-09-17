"""Ablate only predicted beam indices; keep the eight gain entries unchanged."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
import numpy as np
import test_o_mappo_report_input as report_tests
from utils.o_mappo import (OMAPPPolicy, encode_prediction_report, shared_actor_inputs,
                           make_local_state, make_global_state, state_feature_names)


class GainReportTests(unittest.TestCase):
    def setUp(self):
        report_tests.ReportInputTests.setUp(self)
        self.full_config = self.config
        self.config = dataclasses.replace(self.config, state_variant="gain_report")
        self.kw["config"] = self.config

    def test_exact_subset_and_dimensions(self):
        full_names = state_feature_names(self.full_config)
        full = make_local_state(**(self.kw | {"config": self.full_config}))
        keep = [i for i, name in enumerate(full_names) if not name.startswith("reported_beam_")]
        reduced = make_local_state(**self.kw)
        np.testing.assert_array_equal(reduced, full[keep])
        self.assertEqual(len(reduced), 37)
        self.assertEqual(state_feature_names(self.config), [full_names[i] for i in keep])
        self.assertEqual(make_global_state(reduced[None], 1).shape, (112,))
        self.assertEqual(len(encode_prediction_report(self.config, self.report)), 8)
        self.assertIn("tx_beam_sin", state_feature_names(self.config))

    def test_missing_or_changed_predicted_beams_have_no_effect(self):
        class GainsOnly(dict):
            def __getitem__(self, key):
                if key == "beam":
                    raise AssertionError("Gain-only actor must never read predicted beam indices")
                return super().__getitem__(key)
        gains = GainsOnly({k: self.report[k] for k in ("gain", "interference")})
        inputs = shared_actor_inputs(self.config, {"shared_prediction": gains})
        self.assertEqual(set(inputs["prediction_report"]), {"gain", "interference"})
        a = make_local_state(**self.kw)
        np.testing.assert_array_equal(a, make_local_state(**(self.kw | inputs)))
        changed = self.report | {"beam": np.full((4, 5), np.nan)}
        np.testing.assert_array_equal(a, make_local_state(**(self.kw | {"prediction_report": changed})))
        np.testing.assert_array_equal(a, make_local_state(**(self.kw | dict(
            serving_sinr_db=999, interference_db=999, pilot_observation=np.full(128, np.nan)))))
        with self.assertRaises(ValueError):
            encode_prediction_report(self.config, gains | {"gain": np.full(4, np.nan)})

    def test_checkpoint_roundtrip(self):
        state = make_local_state(**self.kw)[None]
        global_state = make_global_state(state, 1)
        policy = OMAPPPolicy(self.config, seed=7)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "gain_only.pt")
            policy.save(path)
            restored = OMAPPPolicy.load(path)
            self.assertEqual(restored.local_dim, 37)
            self.assertEqual(restored.config.state_variant, "gain_report")
            for a, b in zip(policy.act(state, global_state, False), restored.act(state, global_state, False)):
                np.testing.assert_array_equal(a, b)

    def test_training_and_simulation_without_predicted_beams(self):
        self.report.pop("beam")
        report_tests.ReportInputTests.test_training_and_exact_simulation_without_pilot_field(self)


if __name__ == "__main__":
    unittest.main()
