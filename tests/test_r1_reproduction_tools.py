"""Safety and parity checks for the new wrappers, not new scientific algorithms."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import os
import sys

import numpy as np
from scripts import r1_data


class ReproductionToolsTest(unittest.TestCase):
    def test_fresh_output_is_narrow_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r1_data, 'ROOT', Path(tmp)):
            root = Path(tmp)
            (root/'experiment/results').mkdir(parents=True)
            with self.assertRaises(ValueError):
                r1_data.fresh_output(root/'latexCodes/revision1')
            target = root/'experiment/results/data_rebuild_test'
            self.assertEqual(r1_data.fresh_output(target), target)
            with self.assertRaises(FileExistsError):
                r1_data.fresh_output(target)

    def test_system_preprocessing_matches_existing_implementation(self):
        # Seeded random H, including a zero channel, and a vehicle entering late.
        rng = np.random.default_rng(4)
        def record(zero=False):
            h = (rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5
            if zero:
                h[:] = 0
            return dict(h=h.astype(np.complex64), pos=np.zeros(2), v=0., angle=0.)
        raw = {800.+i*.1: {'v1': record(), 'v2': record(zero=True)} for i in range(3)}
        raw[800.2]['new'] = record()
        actual = r1_data.prepare_system(copy.deepcopy(raw), seed=1)
        previous_cwd, previous_argv = Path.cwd(), sys.argv.copy()
        try:
            from utils.sim_utils import get_default_sim_params
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                (work/'sionna_result').mkdir()
                file = work/'sionna_result/trajectoryInfo_lbd1.00_800_950_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10.pkl'
                with file.open('wb') as stream:
                    pickle.dump(raw, stream)
                os.chdir(work)
                sys.argv = ['test']
                with redirect_stdout(io.StringIO()):
                    result = get_default_sim_params(str(work/'logs'), load_predictors=False)
                expected = result[2]
                for frame in raw:
                    for vehicle in raw[frame]:
                        for key in ('best_beam_pair_idx','best_beam_idx_pair','g_opt_beam','g_avg','CSI_preprocessed'):
                            np.testing.assert_array_equal(actual[frame][vehicle][key], expected[frame][vehicle][key])
        finally:
            os.chdir(previous_cwd)
            sys.argv = previous_argv


if __name__ == '__main__':
    unittest.main()
