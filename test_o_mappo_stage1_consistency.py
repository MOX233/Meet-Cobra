import collections
from contextlib import ExitStack
import dataclasses
import unittest
from unittest.mock import patch
import numpy as np

from experiment.o_mappo_stage1_consistency import (
    ModuleAdapter, VARIANTS, paper_args, MICRO_BS_LOCATIONS, sim, om,
    Observer, make_paired_traffic, beam_average_gain_db)
from utils.pql_ba import no_bf_gain_db, macro_gain_db


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.args = paper_args(13e6)
        self.cfg = om.OMAPPOConfig(beam_search_variant='hierarchical32',
            candidate_gain_mode='search', actor_hidden_sizes=(64,64),
            ho_interruption_ms=10, trigger_gate='periodic')
        rng = np.random.default_rng(31)
        self.record = dict(pos=np.array([80.,60.]), angle=0., v=20.,
            h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5)
        self.timeline = collections.OrderedDict((710+.1*i, {'v':dict(self.record,
            pos=np.array([80.+11*i,60.]))}) for i in range(5))

    def context(self, variant):
        a = ModuleAdapter(variant, self.timeline)
        rec = self.timeline[710.1]['v']
        macro = macro_gain_db(self.args, rec['pos'], np.zeros(2))
        interference = {'v':np.r_[macro, no_bf_gain_db(rec['h'])]}
        gains = {'v':interference['v'].copy()}
        gains['v'][1] = -95.
        connection = {'v':1}
        output = a.estimate(self.args, connection, np.zeros((5,2)), {'v'}, gains,
                           {'v':13e6}, infer_g_dict=interference)
        args = [self.cfg, rec['pos'], 0.,20.,1,10.,.2,13.,
                np.clip(output/np.array([133,66,66,66,66]),0,1.5),
                [0,1,0,0,0],-80.,False,.6,.1,0,0]
        return a, args, interference, output

    def test_unchanged_actor_scopes_are_bit_identical(self):
        for variant in ('baseline','C_optimizer','E_optimizer'):
            a, args, _, _ = self.context(variant)
            np.testing.assert_array_equal(a.state(*args), a.old_state(*args))
            self.assertEqual(a.state_changed, 0)

    def test_actor_only_changes_declared_features(self):
        for variant in ('C_actor_optimizer','C_all','E_actor_optimizer','E_all'):
            a, args, _, _ = self.context(variant)
            before = a.old_state(*args)
            after = a.state(*args)
            names = om.state_feature_names(self.cfg)
            allowed = {'serving_sinr','interference_to_noise',*(f'bs_rb_load_{i}' for i in range(5))}
            ix = [i for i,n in enumerate(names) if n not in allowed]
            np.testing.assert_array_equal(after[ix],before[ix])
            self.assertEqual(len(after),31)
            self.assertTrue(np.any(before != after))

    def test_estimator_returns_legacy_and_preserves_original_gain(self):
        for variant in VARIANTS:
            a, args, original, legacy = self.context(variant)
            np.testing.assert_array_equal(args[8],np.clip(legacy/a.caps,0,1.5))
            np.testing.assert_array_equal(original['v'][1:],no_bf_gain_db(self.record['h']))
            if variant.startswith('E'):
                np.testing.assert_array_equal(a.corrected_gain['v'][1:],beam_average_gain_db(self.record['h']))
                self.assertIsNot(a.corrected_gain,original)
            if variant != 'baseline':
                self.assertTrue(np.all((a.corrected_rb >= 0)&(a.corrected_rb <= a.caps)))

    def test_ra_override_is_limited_to_gain_and_occupancy(self):
        for variant in VARIANTS:
            a, _, original, legacy = self.context(variant)
            a.old_ra = lambda pars, **kw: kw
            other = {'v':np.arange(5)}
            kw = dict(slot_idx=1, BS_id=1, veh_set=['v'], g_dict=other,
                      infer_g_dict=original, est_num_RB_allocated_perBS=legacy)
            out = a.ra(self.args,**kw)
            self.assertIs(out['g_dict'],other)
            if variant.endswith('_all'):
                self.assertIs(out['infer_g_dict'],a.corrected_gain)
                self.assertIs(out['est_num_RB_allocated_perBS'],a.corrected_rb)
            else:
                self.assertIs(out['infer_g_dict'],original)
                self.assertIs(out['est_num_RB_allocated_perBS'],legacy)
            self.assertIs(kw['infer_g_dict'],original)
            self.assertIs(kw['est_num_RB_allocated_perBS'],legacy)

    def test_actor_and_all_scopes_supply_same_pre_feedback_states(self):
        for route in ('C','E'):
            a, aa, _, _ = self.context(route+'_actor_optimizer')
            b, bb, _, _ = self.context(route+'_all')
            np.testing.assert_array_equal(a.state(*aa),b.state(*bb))

    def test_full_baseline_wrapper_matches_unwrapped_simulator(self):
        cfg = self.cfg
        class Policy:
            config = cfg
            def act(self, local, global_state, explore):
                return np.ones(len(local),dtype=int),np.zeros(len(local)),np.zeros(len(local))
        common = dict(prt=False, rician_fading=False, ho_interruption_ms=10,
            traffic_trace=make_paired_traffic(self.args,self.timeline,101))
        old = sim.run_sim_o_mappo(self.args,MICRO_BS_LOCATIONS,self.timeline,Policy(),**common)
        a = ModuleAdapter('baseline',self.timeline)
        observer = Observer(a.hook)
        original_function = sim.make_local_state
        with ExitStack() as stack:
            stack.enter_context(patch.object(sim,'estimate_num_RB_allocated_perBS',a.estimate))
            stack.enter_context(patch.object(sim,'make_local_state',a.state))
            stack.enter_context(patch.object(om,'_candidate_links',observer.candidate))
            stack.enter_context(patch.object(sim,'optimize_triggered_targets',observer.optimize))
            new = sim.run_sim_o_mappo(self.args,MICRO_BS_LOCATIONS,self.timeline,Policy(),
                ra_func=a.ra,optimizer_input_hook=a.optimizer,**common)
        self.assertIs(sim.make_local_state,original_function)
        for field in dataclasses.fields(old):
            if not field.name.endswith('time_record'):
                np.testing.assert_equal(getattr(old,field.name),getattr(new,field.name))
        self.assertEqual(a.summary()['changed_states'],0)

    def test_all_variants_run_with_consistent_optimizer_and_physical_bounds(self):
        cfg = self.cfg
        class Policy:
            config = cfg
            def act(self, local, global_state, explore):
                return np.ones(len(local),dtype=int),np.zeros(len(local)),np.zeros(len(local))
        for variant in VARIANTS:
            a = ModuleAdapter(variant,self.timeline)
            observer = Observer(a.hook)
            with ExitStack() as stack:
                stack.enter_context(patch.object(sim,'estimate_num_RB_allocated_perBS',a.estimate))
                stack.enter_context(patch.object(sim,'make_local_state',a.state))
                stack.enter_context(patch.object(om,'_candidate_links',observer.candidate))
                stack.enter_context(patch.object(sim,'optimize_triggered_targets',observer.optimize))
                result = sim.run_sim_o_mappo(self.args,MICRO_BS_LOCATIONS,self.timeline,Policy(),
                    prt=False,rician_fading=False,ho_interruption_ms=10,
                    traffic_trace=make_paired_traffic(self.args,self.timeline,101),
                    ra_func=a.ra,optimizer_input_hook=a.optimizer)
            self.assertTrue(np.all(result.rb_allocated_record <= a.caps))
            a.summary()


if __name__ == '__main__':
    unittest.main()
