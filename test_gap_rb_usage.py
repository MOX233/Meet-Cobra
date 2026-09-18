"""GAP-HO usage saturation must not mask demand or capacity infeasibility."""
import unittest
from unittest.mock import patch
import numpy as np

from test_gap_refinement import example_inputs
from utils import alg_utils
from utils.gap_refinement import GAPRefinementConfig, iterate_assignment
from utils.ho_utils import capacity_factors


class GAPRBUsageTest(unittest.TestCase):
    def test_iteration_bounds_usage_without_clipping_link_demands(self):
        caps=np.array([133.,66.])
        matrix=np.array([[260.,400.],[200.,180.]])
        observed=[]
        def demand(load,iteration):
            observed.append(load.copy())
            return matrix.copy()
        def solve(weights):
            np.testing.assert_array_equal(weights,matrix)
            return np.eye(2)
        result,_,traces,_=iterate_assignment(caps*3,demand,solve,
            GAPRefinementConfig(3,None),rb_capacity=caps)
        np.testing.assert_array_equal(result,matrix)
        for values in observed:
            np.testing.assert_array_equal(values,caps)
        for step in traces:
            np.testing.assert_array_equal(step['implied_load'],caps)
            np.testing.assert_array_equal(step['implied_demand'],[260.,180.])

    def test_capacity_validation(self):
        for caps in ([1,2],[-1],[np.nan],[np.inf]):
            with self.subTest(caps=caps),self.assertRaises(ValueError):
                iterate_assignment([1.],lambda x,i:x[:,None],lambda d:d,
                    GAPRefinementConfig(),rb_capacity=caps)

    def test_inactive_cap_leaves_all_iterates_unchanged(self):
        functions=(lambda x,i:(x/2)[:,None],lambda d:np.ones_like(d))
        args=(np.array([8.,16.]),*functions,GAPRefinementConfig(4,None))
        old=iterate_assignment(*args)
        new=iterate_assignment(*args,rb_capacity=np.array([133.,66.]))
        np.testing.assert_array_equal(old[0],new[0])
        np.testing.assert_array_equal(old[1],new[1])
        self.assertEqual(old[2:],new[2:])

    def run_forced_overload(self,refined,capped):
        inputs,kwargs=example_inputs()
        inputs=list(inputs)
        inputs[3]={v:500e6 for v in inputs[1]}
        chosen=np.zeros((3,len(inputs[1])))
        chosen[1]=1
        diagnostics=[]
        kwargs.update(ho_capacity_correction=True,gap_cap_rb_usage=capped,
            vio_prob_history=np.array([.03,.04]),gap_refinement_diagnostics=diagnostics)
        if refined:
            kwargs['gap_refinement_config']=GAPRefinementConfig(2,None)
        with patch.object(alg_utils,'alg_GAP_APX_adap',return_value=chosen.copy()) as solver, \
             patch.object(alg_utils,'_ITERATIVE_OFFLOAD',return_value=(chosen.copy(),False)) as repair:
            result=alg_utils.HO_EE_GAP_APX_SINR_conservative_adaptive(*inputs,**kwargs)
        return inputs,kwargs,chosen,result,diagnostics,solver,repair

    def test_both_paths_preserve_costs_repair_and_physical_capacity(self):
        for refined in (False,True):
            with self.subTest(refined=refined):
                inputs,kwargs,chosen,result,diags,solve,repair=self.run_forced_overload(refined,True)
                _,_,_,legacy,_,uncapped_solve,_=self.run_forced_overload(refined,False)
                caps=np.array([133.,66.,66.])
                powers=np.array([1.,.2,.2])
                factors=capacity_factors(inputs[1],3,kwargs['current_connection'],10,100)
                self.assertEqual(solve.call_count,2)
                self.assertEqual(repair.call_count,1)
                # The initial optimization problem is unchanged by this fix.
                for field in ('a','b','c'):
                    np.testing.assert_array_equal(solve.call_args_list[0].kwargs[field],
                                                  uncapped_solve.call_args_list[0].kwargs[field])
                for call in solve.call_args_list:
                    demand=call.kwargs['c']/powers[:,None]
                    np.testing.assert_allclose(call.kwargs['a'],demand*factors)
                    self.assertGreater(demand[1].sum(),caps[1])
                    # Resource reserve remains distinct from physical capacity.
                    np.testing.assert_array_equal(call.kwargs['b'],[119.,59.,59.])
                last=solve.call_args.kwargs['c']/powers[:,None]
                raw=(last*chosen).sum(1)
                self.assertGreater(raw[1],caps[1])
                np.testing.assert_allclose(result[1],np.minimum(raw,caps))
                self.assertEqual(result[1][1],66.)  # Not reserved capacity 59.
                self.assertGreater(legacy[1][1],66.)
                # The next iteration sees no more than full micro-BS occupancy.
                self.assertLess(solve.call_args.kwargs['c'][2,0],uncapped_solve.call_args.kwargs['c'][2,0])
                np.testing.assert_allclose(repair.call_args.args[1],last*factors)
                if refined:
                    d=diags[0]
                    self.assertFalse(d['capacity_feasible_after_repair'])
                    self.assertGreater(d['post_repair_capacity_load'][1],66.)
                    self.assertGreater(d['post_repair_frame_average_demand'][1],66.)
                    self.assertEqual(d['post_repair_frame_average_load'][1],66.)
                    self.assertEqual(d['traces'][1]['input_load'][1],66.)

    def test_two_paths_match_also_when_the_cap_binds(self):
        for capped in (False,True):
            explicit=self.run_forced_overload(False,capped)[3]
            refined=self.run_forced_overload(True,capped)[3]
            self.assertEqual(explicit[0],refined[0])
            np.testing.assert_array_equal(explicit[1],refined[1])

    def test_simulator_forwards_the_rollback_switch(self):
        from experiment.pql_ba_experiment import paper_args,MICRO_BS_LOCATIONS
        from utils.ho_utils import make_paired_traffic
        from utils.sim_utils import run_sim_withUMa
        args=paper_args(13e6)
        args.device='cpu'
        pred=dict(gain=np.full(4,-80.),interference=np.full(4,-140.),
                  beam=np.tile(np.arange(5),(4,1)))
        record=dict(pos=np.array([20.,30.]),angle=0.,v=0.,
                    h=np.full((8,4,32),1e-5,dtype=complex),
                    CSI_preprocessed=np.zeros((1,128)))
        timeline={800+.1*i:{'a':dict(record),'b':dict(record)} for i in range(3)}
        cache={f:{v:pred for v in records} for f,records in timeline.items()}
        traffic=make_paired_traffic(args,timeline,1)
        for capped in (False,True):
            observed=[]
            def stay(args,vehicles,*unused,**kwargs):
                observed.append(kwargs['gap_cap_rb_usage'])
                return {v:0 for v in vehicles},np.zeros(5)
            run_sim_withUMa(args,MICRO_BS_LOCATIONS,timeline,None,True,True,True,
                HO_func=stay,prt=False,save_pilot=True,K_BF=5,prediction_cache=cache,
                traffic_trace=traffic,rician_fading=False,gap_cap_rb_usage=capped)
            self.assertEqual(observed,[capped,capped])


if __name__=='__main__':
    unittest.main()
