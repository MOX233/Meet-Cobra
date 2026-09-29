"""Information boundaries and paired-noise invariants for the gain study."""
import copy
import unittest
import numpy as np
from experiment.gain_error_sensitivity import (standard_noise, perturb_reports,
    report_rows, prediction_audit, all_tasks, key)


def sample_timeline():
    result = {}
    for i in range(5):
        frame = round(800 + .1*i, 1)
        result[frame] = {}
        for v in ('v2', 'v1'):
            result[frame][v] = dict(h=np.ones((8, 4, 32), complex)*1e-5,
                pos=np.array([100., 200.]), g_opt_beam=np.full(4, -80.),
                shared_prediction=dict(source_frame=frame, target_frame=frame+.1,
                    gain=np.full(4, -81., dtype=np.float32),
                    interference=np.full(4, -102., dtype=np.float32),
                    beam=np.tile(np.arange(5), (4, 1))))
    return result


class GainErrorTests(unittest.TestCase):
    def setUp(self):
        self.timeline = sample_timeline()
        self.count = len(report_rows(self.timeline))

    def test_zero_preserves_original_object_and_values(self):
        for kind in ('desired', 'interfering'):
            self.assertIs(perturb_reports(self.timeline, kind, 0, standard_noise(self.count, 1, kind)), self.timeline)

    def test_one_field_only_and_no_mutation_of_physical_inputs(self):
        original = copy.deepcopy(self.timeline)
        for kind, field, other in (('desired', 'gain', 'interference'), ('interfering', 'interference', 'gain')):
            z = standard_noise(self.count, 1, kind)
            out = perturb_reports(self.timeline, kind, 3, z)
            for i, (frame, v) in enumerate(report_rows(self.timeline)):
                before, after = self.timeline[frame][v], out[frame][v]
                p, q = before['shared_prediction'], after['shared_prediction']
                for name in ('h', 'pos', 'g_opt_beam'):
                    self.assertIs(before[name], after[name])
                    np.testing.assert_array_equal(before[name], original[frame][v][name])
                self.assertIs(q[other], p[other]); self.assertIs(q['beam'], p['beam'])
                self.assertEqual(q['source_frame'], p['source_frame'])
                self.assertEqual(q['target_frame'], p['target_frame'])
                np.testing.assert_allclose(q[field]-p[field], 3*z[i], atol=1e-13)
                np.testing.assert_array_equal(p[field], original[frame][v]['shared_prediction'][field])

    def test_rng_independence_reproducibility_and_scaling(self):
        np.random.seed(123)
        before = np.random.get_state()
        z = standard_noise(100000, 1, 'desired')
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(z, standard_noise(100000, 1, 'desired'))
        self.assertFalse(np.array_equal(z, standard_noise(100000, 2, 'desired')))
        self.assertFalse(np.array_equal(z, standard_noise(100000, 1, 'interfering')))
        self.assertLess(abs(z.mean()), .01); self.assertLess(abs(z.std()-1), .01)
        z = z[:self.count]
        one = perturb_reports(self.timeline, 'desired', 1, z)
        ten = perturb_reports(self.timeline, 'desired', 10, z)
        for frame, v in report_rows(self.timeline):
            p = self.timeline[frame][v]['shared_prediction']['gain']
            np.testing.assert_allclose(ten[frame][v]['shared_prediction']['gain']-p,
                10*(one[frame][v]['shared_prediction']['gain']-p), atol=1e-12)

    def test_labels_use_target_frames_and_exclude_unlabeled_final(self):
        frames = sorted(self.timeline)
        for i, frame in enumerate(frames):
            for record in self.timeline[frame].values():
                record['g_opt_beam'] = np.full(4, -80.+i)
        result = prediction_audit(self.timeline, [1], [0, 10])
        self.assertEqual(result['original_mae_db']['desired'], 3.5)
        self.assertTrue(all(r['labeled_reports'] == 8 and r['reports'] == 10 for r in result['rows']))

    def test_invalid_inputs_fail(self):
        z = standard_noise(self.count, 1, 'desired')
        for sigma in (-1, np.inf, np.nan):
            with self.assertRaises(ValueError): perturb_reports(self.timeline, 'desired', sigma, z)
        with self.assertRaises(ValueError): perturb_reports(self.timeline, 'bad', 1, z)
        with self.assertRaises(ValueError): perturb_reports(self.timeline, 'desired', 1, z[:-1])

    def test_grid_contains_252_unique_cases_and_shared_zero(self):
        p = dict(rates=[9,19,29,35], seeds=[1,2,3])
        control, other = all_tasks(p)
        self.assertEqual(len(control), 12); self.assertEqual(len(other), 240)
        self.assertEqual(len({key(*t) for t in control+other}), 252)
        self.assertEqual(key('desired',0,9,1), key('interfering',0,9,1))


if __name__ == '__main__':
    unittest.main()
