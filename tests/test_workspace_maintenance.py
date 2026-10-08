import json
from pathlib import Path
import tempfile
import unittest

from scripts.workspace_maintenance import category, checked_path, identity, validate_targets


class CleanupSafetyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"protected_trees": ["experiment", "latexCodes/revision1", ".git"]}

    def test_protect_current_and_tracked_datasets(self):
        name = "prepared_dataset/current.pkl"
        self.assertIsNone(category(name, {name}, set(), self.manifest))
        self.assertIsNone(category(name, set(), {name}, self.manifest))
        self.assertEqual(category(name, set(), set(), self.manifest), "obsolete_window_datasets")

    def test_no_models_results_sources_or_frozen_submission(self):
        for name in ["experiment/results/old/data.pkl", "experiment/__pycache__/a.pyc",
                     "utils/pql_ba.py", "latexCodes/revision1/Manuscript_LaTeX/main.aux",
                     "latexCodes/main.tex", "latexCodes/main.bbl", "sumo_data/road.net.xml",
                     ".ipynb_checkpoints/unsaved.ipynb"]:
            self.assertIsNone(category(name, set(), set(), self.manifest), name)

    def test_reject_escaping_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "file").write_text("x")
            (root / "link").symlink_to(root / "file")
            for name in ["../file", "/tmp/file", "link"]:
                with self.assertRaises(ValueError):
                    checked_path(root, name)

    def test_detect_changed_or_duplicate_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "prepared_dataset/old.pkl"
            path.parent.mkdir()
            path.write_bytes(b"old")
            row = dict(path="prepared_dataset/old.pkl", category="obsolete_window_datasets", identity=identity(path))
            self.assertEqual(validate_targets(root, {"files": [row]}, self.manifest, set(), set()), [path])
            with self.assertRaises(ValueError):
                validate_targets(root, {"files": [row, row]}, self.manifest, set(), set())
            path.write_bytes(b"updated")
            with self.assertRaises(ValueError):
                validate_targets(root, {"files": [row]}, self.manifest, set(), set())


if __name__ == "__main__":
    unittest.main()
