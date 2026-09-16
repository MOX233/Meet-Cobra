"""Regression and stopping-rule tests for the opt-in GAP-HO path."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from utils import alg_utils
from utils.gap_refinement import GAPRefinementConfig, iterate_assignment
from utils.ho_utils import capacity_factors


def example_inputs():
    args = SimpleNamespace(slots_per_frame=100, vio_prob_threshold=.01,
                           N0=4e-21, RB_intervel_micro=1.44e6,
                           RB_intervel_macro=.36e6, p_micro=.2, p_macro=1.,
                           NF_micro_dB=10., NF_macro_dB=5.,
                           num_RB_micro=66, num_RB_macro=133,
                           pilot_overhead_factor=2/112)
    vehicles = ['a', 'b', 'c', 'd']
    gains = {v: np.array([-106-i, -98-i, -105+i], dtype=float)
             for i, v in enumerate(vehicles)}
    interference = {v: np.array([-120, -125-i, -126+i], dtype=float)
                    for i, v in enumerate(vehicles)}
    inputs = (args, vehicles, {}, {v: 5e6 for v in vehicles},
              {v: np.zeros(2) for v in vehicles}, gains, np.zeros((3, 2)))
    kwargs = dict(infer_g_dict=interference,
                  num_pilot_dict={v: np.array([1., 3.]) for v in vehicles},
                  current_connection=dict(a=0, b=1, c=2, d=1),
                  ho_interruption_slots=10)
    return inputs, kwargs


class GAPRefinementTest(unittest.TestCase):
    def test_config_validation(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                GAPRefinementConfig(max_iterations=value)
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                GAPRefinementConfig(tolerance_rb=value)
        for value in (1, .9, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                GAPRefinementConfig(relaxation_factor=value)

    def test_bounded_update_and_inclusive_absolute_tolerance(self):
        demand = lambda load, iteration: (load/2)[:, None]
        assign = lambda weights: np.ones_like(weights)
        _, _, traces, reason = iterate_assignment(
            np.array([8.]), demand, assign, GAPRefinementConfig(10, .5))
        self.assertEqual([x['residual_rb'] for x in traces], [4., 2., 1., .5])
        self.assertEqual(reason, 'tolerance')
        _, _, traces, reason = iterate_assignment(
            np.array([8.]), demand, assign, GAPRefinementConfig(3, .01))
        self.assertEqual(len(traces), 3)
        self.assertEqual(reason, 'iteration_limit')

    def test_zero_tolerance_is_not_disabled_stopping(self):
        for eps, expected in ((None, 3), (0, 1)):
            _, _, traces, _ = iterate_assignment(
                np.array([8.]), lambda load, iteration: load[:, None],
                lambda weights: np.ones_like(weights), GAPRefinementConfig(3, eps))
            self.assertEqual(len(traces), expected)

    def test_oscillation_is_not_mislabeled_as_convergence(self):
        _, _, traces, reason = iterate_assignment(
            np.array([1.]), lambda load, iteration: (3-load)[:, None],
            lambda weights: np.ones_like(weights), GAPRefinementConfig(5, .1))
        self.assertEqual(reason, 'iteration_limit')
        self.assertEqual(len(traces), 5)
        self.assertTrue(traces[1]['repeated_load'])
        self.assertTrue(all(t['residual_rb'] == 1 for t in traces))

    def test_nonfinite_iterate_is_rejected(self):
        with self.assertRaises(ValueError):
            iterate_assignment(np.array([1.]), lambda load, iteration: np.array([[np.nan]]),
                               lambda weights: np.ones_like(weights), GAPRefinementConfig())

    def test_two_pass_real_solvers_are_bitwise_legacy_compatible(self):
        inputs, base = example_inputs()
        for corrected in (False, True):
            for reserve in (False, True):
                for fallback in (False, True):
                    with self.subTest(corrected=corrected, reserve=reserve, fallback=fallback):
                        kwargs = dict(base, ho_capacity_correction=corrected,
                                      vio_prob_history=np.array([.03, 0, .04]) if reserve else [])
                        if fallback:
                            kwargs['infer_g_dict'] = None
                        old = alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(*inputs, **kwargs)
                        traces = []
                        new = alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(
                            *inputs, **kwargs, gap_refinement_config=GAPRefinementConfig(2, None),
                            gap_refinement_diagnostics=traces)
                        self.assertEqual(old[0], new[0])
                        np.testing.assert_array_equal(old[1], new[1])
                        self.assertEqual(traces[0]['iterations'], 2)

    def test_interruption_changes_capacity_but_not_power_or_feedback(self):
        inputs, kwargs = example_inputs()
        chosen = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
        diagnostics = []
        with patch.object(alg_utils, 'alg_GAP_APX_adap', return_value=chosen.copy()) as solve, \
             patch.object(alg_utils, '_ITERATIVE_OFFLOAD', return_value=(chosen.copy(), True)) as repair:
            _, loads = alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(
                *inputs, **kwargs, ho_capacity_correction=True,
                gap_refinement_config=GAPRefinementConfig(3, None, 1.05),
                gap_refinement_diagnostics=diagnostics)
        factors = capacity_factors(inputs[1], 3, kwargs['current_connection'], 10, 100)
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(repair.call_count, 1)
        for call in solve.call_args_list:
            average = call.kwargs['c']/np.array([1., .2, .2])[:, None]
            np.testing.assert_allclose(call.kwargs['a'], average*factors)
            self.assertEqual(call.kwargs['adap_mtp'], 1.05)
        last = solve.call_args.kwargs['c']/np.array([1., .2, .2])[:, None]
        np.testing.assert_allclose(loads, (last*chosen).sum(axis=1))
        self.assertEqual(diagnostics[0]['iterations'], 3)

    def test_default_dispatch_does_not_enter_new_path(self):
        inputs, kwargs = example_inputs()
        with patch('utils.gap_refinement.refined_gap_handover', side_effect=AssertionError):
            alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(*inputs, **kwargs)

    def test_empty_vehicle_set(self):
        inputs, kwargs = example_inputs()
        inputs = (inputs[0], [], {}, {}, {}, {}, inputs[-1])
        commands, loads = alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(
            *inputs, gap_refinement_config=GAPRefinementConfig())
        self.assertFalse(commands)
        np.testing.assert_array_equal(loads, np.zeros(3))


if __name__ == '__main__':
    unittest.main()
