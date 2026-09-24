"""The approved baseline must replace only O-MAPPO, on the same traffic."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiment import plot_revision_system_results as plots


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/test/actor2')
        self.protocol = dict(rates=list(range(1, 36, 2)), seeds=[1, 2, 3],
                             warmup_frames=2, cache_sha256='same-channel-cache')
        self.rows, self.data, provenance = [], {}, []
        self.curves = {m: {k: np.zeros((18, 3)) for k in plots.METRICS} for m in plots.METHODS}
        for method in plots.METHODS:
            for rate in self.protocol['rates']:
                for seed in self.protocol['seeds']:
                    row = dict(method=method, rate_mbps=rate, seed=seed, frames=300,
                               traffic_sha256=f'{rate}-{seed}', metrics={k: -1 for k in plots.METRICS})
                    self.rows.append(row)
                    if method == 'o_mappo':
                        path = self.root / f'case{rate}_{seed}.json'
                        self.data[path] = dict(row, checkpoint_sha256='fixed-policy',
                                              metrics={k: rate + seed / 10 for k in plots.METRICS})
                        provenance.append(dict(file=str(path), sha256=str(path),
                                               raw_sha256=str(path.with_suffix('.npz'))))
        self.data[self.root / 'summary.json'] = dict(cases=54, raw_metrics_recomputed=True,
            rates=self.protocol['rates'], seeds=[1, 2, 3], seconds=30, warmup_frames=2,
            protocol_sha256=str(self.root / 'protocol.json'), policy_sha256='fixed-policy',
            provenance=provenance)
        self.data[self.root / 'protocol.json'] = dict(timeline_sha256='same-channel-cache')

    def replace(self):
        with patch.object(plots, 'read', side_effect=lambda p: self.data[p]), \
             patch.object(plots, 'digest', side_effect=str):
            return plots.replace_o_mappo(self.root, self.protocol, self.rows, self.curves)

    def test_only_approved_baseline_changes(self):
        original = copy.deepcopy(self.curves)
        rows, provenance = self.replace()
        self.assertEqual(len(rows), 432)
        self.assertEqual(provenance['cases'], 54)
        for old, new in zip(self.rows, rows):
            if old['method'] != 'o_mappo':
                self.assertIs(old, new)
        for method in plots.METHODS:
            for metric in plots.METRICS:
                if method == 'o_mappo':
                    expected = np.array([[r + s / 10 for s in [1, 2, 3]]
                                         for r in range(1, 36, 2)])
                    np.testing.assert_array_equal(self.curves[method][metric], expected)
                else:
                    np.testing.assert_array_equal(self.curves[method][metric], original[method][metric])

    def test_reject_unpaired_traffic(self):
        self.data[self.root / 'case1_1.json']['traffic_sha256'] = 'different-traffic'
        with self.assertRaises(AssertionError):
            self.replace()

    def test_reject_changed_cache(self):
        self.data[self.root / 'protocol.json']['timeline_sha256'] = 'different-cache'
        with self.assertRaises(AssertionError):
            self.replace()

    def test_reject_load_specific_checkpoint(self):
        self.data[self.root / 'case1_1.json']['checkpoint_sha256'] = 'different-policy'
        with self.assertRaises(AssertionError):
            self.replace()


if __name__ == '__main__':
    unittest.main()
