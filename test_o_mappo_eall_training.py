import collections
import dataclasses
import unittest
import numpy as np
import torch

from experiment.o_mappo_eall_training import (EAllFluidAdapter,training_environment,
    initial_policy,schedule,om,old,beam_average_gain_db,estimate_bounded)
from experiment.o_mappo_target_check import OptimizerHook
from utils.beam_utils import generate_dft_codebook


class EAllTrainingTests(unittest.TestCase):
    def setUp(self):
        self.args=old.paper_args(13e6)
        self.cfg=om.OMAPPOConfig(beam_search_variant='hierarchical32',candidate_gain_mode='search',
            actor_hidden_sizes=(64,64),ho_interruption_ms=10,optimizer_solver='milp')
        self.tx,self.rx=generate_dft_codebook(32),generate_dft_codebook(8)
        rng=np.random.default_rng(4)
        self.records={v:dict(pos=np.array([80.+i*50,60.]),v=20.,angle=0.,
            h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5)
            for i,v in enumerate(('a','b'))}
        self.learners={v:om.OMAPPOLearnerState(action=i+1,rx_beam=0,tx_beam=0,
            pending_action=None,last_position=np.zeros(2),distance_since_event=10)
            for i,v in enumerate(self.records)}

    def test_warm_start_same_weights_and_fresh_optimizers(self):
        policies=[initial_policy(s) for s in (11,22,33)]
        for p in policies:
            self.assertEqual(p.config.batch_size,256)
            self.assertEqual(p.config.ppo_epochs,4)
            self.assertEqual(len(p.actor_optimizer.state),0)
            self.assertEqual(len(p.critic_optimizer.state),0)
            for net in ('actor','critic'):
                for k,v in getattr(p,net).state_dict().items():
                    self.assertTrue(torch.equal(v,getattr(policies[0],net).state_dict()[k]))

    def test_nonzero_identical_learning_rate_schedule(self):
        self.assertAlmostEqual(schedule(0),1e-4)
        self.assertAlmostEqual(schedule(120),1e-5)
        self.assertAlmostEqual(schedule(160),1e-5)
        self.assertTrue(all(schedule(i)>0 for i in range(161)))

    def test_shared_estimate_and_actual_fluid_occupancy_are_distinct(self):
        a=EAllFluidAdapter()
        channels={v:r['h'].copy() for v,r in self.records.items()}
        originals=(om.no_bf_gain_db,om.fluid_o_mappo_step,om._fluid_allocation,om.make_local_state,om.optimize_triggered_targets)
        with a.applied():
            result=om.fluid_o_mappo_step(self.args,self.records,self.learners,
                {'a':1e7,'b':1e7},{'a':13e6,'b':13e6},np.zeros(5),np.zeros(2),self.cfg,self.tx,self.rx)
            for v in self.records:
                np.testing.assert_array_equal(a.interference[v][1:],beam_average_gain_db(self.records[v]['h']))
            expected=np.array([sum(result.allocated_rb[v] for v in self.records if self.learners[v].action==bs)
                               for bs in range(5)])/a.caps
            np.testing.assert_allclose(result.load_ratio,expected)
            self.assertFalse(np.allclose(a.load,result.load_ratio))
            self.assertTrue(np.all(a.rb<=a.caps))
            self.assertTrue(np.all(result.load_ratio<=1))
            for v in self.records:
                self.assertAlmostEqual(result.user_power_w[v],result.allocated_rb[v]*.2)
        self.assertEqual(originals,(om.no_bf_gain_db,om.fluid_o_mappo_step,om._fluid_allocation,om.make_local_state,om.optimize_triggered_targets))
        for v in self.records: np.testing.assert_array_equal(self.records[v]['h'],channels[v])

    def test_actor_changes_only_selected_features(self):
        a=EAllFluidAdapter()
        with a.applied():
            result=om.fluid_o_mappo_step(self.args,self.records,self.learners,
                {'a':1e7,'b':1e7},{'a':13e6,'b':13e6},np.zeros(5),np.zeros(2),self.cfg,self.tx,self.rx)
            for v in ('a','b'):
                args=[self.cfg,self.records[v]['pos'],0.,20.,self.learners[v].action,10.,.2,13.,
                    np.ones(5)*1.5,[0,1,1,0,0],-80.,False,.6,.1,0,0]
                original=a.old_state(*args)
                modified=om.make_local_state(*args)
                allowed={18,21,22,23,24,25,26}
                same=[i for i in range(31) if i not in allowed]
                np.testing.assert_array_equal(original[same],modified[same])
                np.testing.assert_allclose(modified[21:26],a.load)

    def test_optimizer_matches_stage1_on_identical_frame_context(self):
        a=EAllFluidAdapter()
        queues={'a':1e6,'b':2e6}
        backlog={v:q+1.3e6 for v,q in queues.items()}
        with a.applied():
            result=om.fluid_o_mappo_step(self.args,self.records,self.learners,queues,
                {'a':13e6,'b':13e6},np.zeros(5),np.zeros(2),self.cfg,self.tx,self.rx)
            a.state_index=2
            new=om.optimize_triggered_targets(self.args,self.records,self.learners,['a'],backlog,
                result.allocated_rb,result.load_ratio,self.cfg,self.tx,self.rx,np.zeros(2))
            hook=OptimizerHook('bounded_average')
            inputs=hook(args=self.args,frame=710.,records=self.records,learners=self.learners,
                backlog=backlog,allocated_rb=result.allocated_rb,load=result.load_ratio,
                config=self.cfg,dft_tx=self.tx,dft_rx=self.rx,macro_loc=np.zeros(2),
                no_bf_gain=a.interference,serving_gain=a.serving,feedback_load=np.zeros(5))
            expected=a.old_optimize(self.args,inputs['records'],self.learners,['a'],backlog,
                inputs['allocated_rb'],inputs['load'],inputs['config'],self.tx,self.rx,np.zeros(2))
            self.assertEqual(new.targets,expected.targets)
            np.testing.assert_allclose(new.overflow,expected.overflow)
            self.assertAlmostEqual(new.objective,expected.objective)

    def timeline(self):
        return collections.OrderedDict((710+.1*i,{v:dict(r,pos=r['pos']+np.array([11.*i,0]))
            for v,r in self.records.items()}) for i in range(8))

    def test_legacy_route_reproduces_original_rollout(self):
        reward=om.o_mappo_reward_presets()['qos_energy020_load1']
        first=om.OMAPPPolicy(self.cfg,seed=10)
        second=om.OMAPPPolicy(self.cfg,seed=10)
        expected,em=om.run_fluid_o_mappo_episode(self.args,self.timeline(),first,reward,13,seed=55,learn=True,collect_only=True)
        with training_environment('legacy'):
            got,gm=om.run_fluid_o_mappo_episode(self.args,self.timeline(),second,reward,13,seed=55,learn=True,collect_only=True)
        for k in expected:
            if k != 'optimizer_ms_per_call': self.assertEqual(expected[k],got[k],k)
        self.assertEqual(len(em),len(gm))
        for a,b in zip(em.transitions,gm.transitions):
            for field in dataclasses.fields(a): np.testing.assert_equal(getattr(a,field.name),getattr(b,field.name))

    def test_eall_collection_and_ppo_update(self):
        policy=om.OMAPPPolicy(self.cfg,seed=10)
        before={k:v.clone() for k,v in policy.actor.state_dict().items()}
        with training_environment('E_all'):
            row,memory=om.run_fluid_o_mappo_episode(self.args,self.timeline(),policy,
                om.o_mappo_reward_presets()['qos_energy020_load1'],13,seed=55,learn=True,collect_only=True)
        self.assertGreater(len(memory),0)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in policy.actor.state_dict().items()))
        update=policy.update(memory)
        self.assertTrue(all(np.isfinite(v) for v in update.values()))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in policy.actor.state_dict().items()))


if __name__=='__main__': unittest.main()
