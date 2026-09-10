import collections
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from utils import alg_utils
from utils.ho_utils import capacity_factors, interruption_slots, make_paired_traffic


class HOInterruptionTest(unittest.TestCase):
    def test_slot_alignment_and_subframe_limit(self):
        for ms in (0, 1, 5, 10, 99):
            self.assertEqual(interruption_slots(ms, .001, 100), ms)
        for ms in (-1, .5, 100, 101, float('nan')):
            with self.assertRaises(ValueError):
                interruption_slots(ms, .001, 100)

    def test_stay_switch_and_initial_access(self):
        factors = capacity_factors(['a', 'b', 'new'], 3, {'a': 0, 'b': 2}, 10, 100)
        np.testing.assert_allclose(factors[:, 0], [1, 1/.9, 1/.9])
        np.testing.assert_allclose(factors[:, 1], [1/.9, 1/.9, 1])
        np.testing.assert_array_equal(factors[:, 2], np.ones(3))
        np.testing.assert_array_equal(capacity_factors(['a'], 2, None, 0, 100),
                                      np.ones((2, 1)))
        with self.assertRaises(ValueError):
            capacity_factors(['a'], 2, None, 1, 100)

    def test_gap_receives_separate_capacity_and_power(self):
        demand = np.array([[2., 3.], [4., 5.]])
        power = np.array([1., .2])
        capacities = demand * np.array([[1., 1/.9], [1/.9, 1.]])
        costs = demand * power[:, None]
        with patch.object(alg_utils, 'alg_GAP_APX_adap', return_value=np.eye(2)) as solve:
            alg_utils._HO_GAP_APX(capacities, np.array([10., 10.]), power, T_COST=costs)
        np.testing.assert_array_equal(solve.call_args.kwargs['a'], capacities)
        np.testing.assert_array_equal(solve.call_args.kwargs['c'], costs)

    def test_repair_uses_uninflated_power_cost(self):
        # BS0 is overloaded. Among feasible destinations, the supplied true
        # power cost prefers BS2 even though capacity * per-RB power prefers BS1.
        demand = np.array([[2.], [1.], [2.]])
        init = np.array([[1], [0], [0]])
        result, feasible = alg_utils._ITERATIVE_OFFLOAD(
            init, demand, np.array([1., 4., 4.]), np.ones(3),
            T_COST=np.array([[1.], [3.], [.5]]))
        self.assertTrue(feasible)
        self.assertEqual(result[:, 0].argmax(), 2)

    def test_both_gap_iterations_correct_capacity_not_average_load(self):
        args = SimpleNamespace(slots_per_frame=100, vio_prob_threshold=.01,
                               N0=4e-21, RB_intervel_micro=1.44e6,
                               RB_intervel_macro=.36e6, p_micro=.2, p_macro=1.,
                               NF_micro_dB=10., NF_macro_dB=5.,
                               num_RB_micro=66, num_RB_macro=133,
                               pilot_overhead_factor=2/112)
        vehicles = ['a', 'b']
        gains = {v: np.array([-90., -80.]) for v in vehicles}
        with patch.object(alg_utils, '_HO_GAP_APX', return_value=(np.eye(2), True)) as first, \
             patch.object(alg_utils, '_HO_GAP_APX_with_offload', return_value=(np.eye(2), True)) as second:
            commands, loads = alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(
                args, vehicles, {}, dict(a=1e6, b=1e6),
                {v: np.array([0., 0.]) for v in vehicles}, gains, np.zeros((2, 2)),
                infer_g_dict=gains, current_connection=dict(a=1, b=0),
                ho_capacity_correction=True, ho_interruption_slots=10)
        factors = capacity_factors(vehicles, 2, dict(a=1, b=0), 10, 100)
        for call in (first.call_args, second.call_args):
            average = call.kwargs['T_COST'] / np.array([1., .2])[:, None]
            np.testing.assert_allclose(call.kwargs['T_KR'], average * factors)
        average = second.call_args.kwargs['T_COST'] / np.array([1., .2])[:, None]
        np.testing.assert_allclose(loads, (average * np.eye(2)).sum(axis=1))

    def test_paired_traffic_is_independent_of_policy_rng(self):
        args = SimpleNamespace(data_rate=5e6, random_factor_range4data_rate=0.,
                               lat_slot_ub=20, slot_len=.001, slots_per_frame=100)
        timeline = collections.OrderedDict([(800., {'a': {}}),
                                             (800.1, {'a': {}, 'b': {}}),
                                             (800.2, {'b': {}})])
        first = make_paired_traffic(args, timeline, 1)
        np.random.seed(999)
        np.random.normal(size=10000)
        second = make_paired_traffic(args, timeline, 1)
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertNotEqual(first['sha256'], make_paired_traffic(args, timeline, 2)['sha256'])
        self.assertEqual(set(first['initial_queues'][800.1]), {'b'})
        self.assertEqual(first['initial_queues'][800.2], {})


if __name__ == '__main__':
    unittest.main()
