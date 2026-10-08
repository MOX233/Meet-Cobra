import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import experiment_cleanup as cleanup


class ExperimentCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy = {
            "superseded_raw_trees": {"experiment/results/old": "superseded"},
            "obsolete_caches": [],
            "duplicate_validation_caches": {"retained": "keep.pkl", "candidates": []},
        }
        self.name = "experiment/results/old/raw/case.npz"
        self.path = self.root / self.name
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"test fixture, not a scientific result")
        self.evidence = self.path.parent.parent / "runs/case.json"
        self.evidence.parent.mkdir()
        self.evidence.write_text(json.dumps({"metrics": {"power_w": 1, "violation_percent": 0}}))

    def eligible(self, **overrides):
        args = dict(root=self.root, name=self.name, policy=self.policy,
                    trees=[], references=set(), tracked=set())
        args.update(overrides)
        return cleanup.eligible(**args)

    def test_current_referenced_and_tracked_files_are_excluded(self):
        self.assertIsNotNone(self.eligible())
        self.assertIsNone(self.eligible(trees=["experiment/results/old"]))
        self.assertIsNone(self.eligible(references={self.name}))
        self.assertIsNone(self.eligible(tracked={self.name}))

    def test_missing_or_insufficient_case_evidence_is_excluded(self):
        self.evidence.write_text(json.dumps({"elapsed_s": 10}))
        self.assertIsNone(self.eligible())
        self.evidence.unlink()
        self.assertIsNone(self.eligible())

    def test_source_models_and_other_directories_are_excluded(self):
        for name in ("experiment/results/old/best.pth", "experiment/results/old/policy.pkl",
                     "experiment/tool.py", "experiment/results/other/raw/case.npz",
                     "experiment/results/old/config.json"):
            self.assertIsNone(self.eligible(name=name), name)

    def test_path_boundaries_and_symlinks(self):
        self.assertFalse(cleanup.under("experiment/results/older/raw/a.npz",
                                       ["experiment/results/old"]))
        self.path.unlink()
        self.path.symlink_to(self.evidence)
        with self.assertRaises(ValueError):
            self.eligible()

    def test_disk_accounting_preserves_external_hard_links(self):
        row = dict(identity=dict(device=1, inode=2, allocated_bytes=4096), links=2)
        self.assertEqual(cleanup.reclaimed_bytes([row]), 0)
        self.assertEqual(cleanup.reclaimed_bytes([row, row]), 4096)

    def test_actual_apply_handles_two_hard_links_and_keeps_metadata(self):
        other = self.path.with_name("second.npz")
        os.link(self.path, other)
        other_evidence = self.evidence.with_name("second.json")
        other_evidence.write_bytes(self.evidence.read_bytes())
        plan_dir = self.root / "plan"
        plan_dir.mkdir()
        (plan_dir / "index.json").write_text("{}")
        rows = []
        for path in (self.path, other):
            name = str(path.relative_to(self.root))
            spec = self.eligible(name=name)
            rows.append(dict(path=name, identity=cleanup.identity(path), links=2,
                             recorded_sha256=cleanup.digest(path),
                             evidence_sha256=cleanup.digest(self.root/spec["evidence"]), **spec))
        args = argparse.Namespace(directory=plan_dir,
                                  confirm_sha256=cleanup.digest(plan_dir/"index.json"))
        with patch.object(cleanup, "ROOT", self.root), \
             patch.object(cleanup, "protected_paths", return_value=([], set(), {})), \
             patch.object(cleanup, "tracked_files", return_value=set()), \
             patch.object(cleanup, "assert_no_jobs"):
            cleanup.apply(args, self.policy, {}, rows)
        self.assertFalse(self.path.exists())
        self.assertFalse(other.exists())
        self.assertTrue(self.evidence.exists())
        self.assertTrue(other_evidence.exists())
        self.assertEqual(len((plan_dir/"deleted.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
