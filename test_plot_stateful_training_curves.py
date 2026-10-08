"""Regression checks for mixing unchanged models with a retrained gain model."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiment.plot_stateful_training_curves import (
    best_validation_reference, parse_args, style_two_phase_axes, values,
)
import matplotlib.pyplot as plt


class TrainingCurveTests(unittest.TestCase):
    def test_override_requires_both_stages_and_separate_summary(self):
        for flags in (
            ["--interfering-stage1-results", "first"],
            ["--interfering-stage1-results", "first",
             "--interfering-stage2-results", "second"],
        ):
            with patch.object(sys, "argv", ["plot", *flags]):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_override_preserves_explicit_provenance_destination(self):
        with patch.object(sys, "argv", [
            "plot", "--interfering-stage1-results", "first",
            "--interfering-stage2-results", "second", "--summary", "new.json",
        ]):
            args = parse_args()
        self.assertEqual(args.interfering_stage1_results, Path("first"))
        self.assertEqual(args.interfering_stage2_results, Path("second"))
        self.assertEqual(args.summary, Path("new.json"))

    def test_final_epoch_star_has_room_inside_axes(self):
        fig, ax = plt.subplots()
        try:
            style_two_phase_axes(ax)
            best_validation_reference(ax, 2.51, 100, "orange")
            self.assertEqual(float(ax.lines[-1].get_xdata()[0]), 200.0)
            self.assertGreaterEqual(ax.get_xlim()[1], 203.0)
        finally:
            plt.close(fig)

    def test_nonfinite_metric_is_rejected(self):
        rows = [{"mae": 2.0} for _ in range(100)]
        rows[50]["mae"] = float("nan")
        with self.assertRaises(ValueError):
            values(rows, "mae")


if __name__ == "__main__":
    unittest.main()
