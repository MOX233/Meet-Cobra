import dataclasses
import unittest
from unittest.mock import patch
import numpy as np

from experiment.o_mappo_target_check import (OptimizerHook, Observer, estimate_bounded,
    optimizer_records, power_candidates, paper_args, MICRO_BS_LOCATIONS)
from utils import o_mappo as om, o_mappo_sim as sim
from utils.beam_utils import generate_dft_codebook
from utils.ho_utils import make_paired_traffic
from utils.pql_ba import no_bf_gain_db, macro_gain_db


class TargetCheckTests(unittest.TestCase):
    def setUp(self):
        self.args = paper_args(13e6)
        self.cfg = om.OMAPPOConfig(beam_search_variant='hierarchical32',
            candidate_gain_mode='search', candidate_count=3, ho_interruption_ms=10,
            optimizer_solver='milp', actor_hidden_sizes=(64,64))
        rng = np.random.default_rng(12)
        self.record = dict(pos=np.array([80., 60.]), angle=0., v=20.,
            h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5)
        self.tx, self.rx = generate_dft_codebook(32), generate_dft_codebook(8)
        macro = macro_gain_db(self.args, self.record['pos'], np.zeros(2))
        no_bf = np.r_[macro, no_bf_gain_db(self.record['h'])]
        learner = om.OMAPPOLearnerState(action=1,rx_beam=0,tx_beam=0,pending_action=None,
            last_position=np.zeros(2),distance_since_event=10)
        self.c = dict(args=self.args, config=self.cfg, frame=800., records={'v':self.record},
            learners={'v':learner}, backlog={'v':1.3e6}, allocated_rb={'v':20.},
            load=np.array([.2,1.5,.7,.3,.4]), dft_tx=self.tx,dft_rx=self.rx,macro_loc=np.zeros(2),
            no_bf_gain={'v':no_bf}, serving_gain={'v':-90.}, feedback_load=np.ones(5)*.2)

    def test_clip_only_preserves_reservations_and_records(self):
        out = OptimizerHook('clip_only')(**self.c)
        np.testing.assert_array_equal(out['load'],[.2,1.,.7,.3,.4])
        self.assertIs(out['allocated_rb'],self.c['allocated_rb'])
        self.assertIs(out['records'],self.c['records'])
        self.assertEqual(self.c['load'][1],1.5)

    def test_bounded_feedback_and_fixed_reservations(self):
        for name in ['bounded','bounded_average']:
            out = OptimizerHook(name)(**self.c)
            self.assertTrue(np.isfinite(out['load']).all())
            self.assertTrue(((out['load']>=0)&(out['load']<=1)).all())
            self.assertLessEqual(out['allocated_rb']['v'],66)
        connection={'v':1}
        gain={'v':np.full(5,-150.)}
        rb=estimate_bounded(self.args,connection,gain,gain,{'v':1e12})
        self.assertEqual(rb[1],66)
        np.testing.assert_array_equal(estimate_bounded(self.args,{}, {}, {}, {}),np.zeros(5))

    def test_scalar_transport_preserves_current_search_gains_and_costs(self):
        records = optimizer_records(self.c,self.c['no_bf_gain'])
        args=[self.args,'v',self.record,1,1.3e6,self.c['load'],self.cfg,self.tx,self.rx,np.zeros(2)]
        old=om._candidate_links(*args)
        args[2]=records['v']; args[6]=dataclasses.replace(self.cfg,information_mode='shared_prediction')
        new=om._candidate_links(*args)
        for a,b in zip(old,new):
            self.assertEqual(a.bs,b.bs)
            for k in ['gain_db','required_rb','base_cost']:
                self.assertEqual(getattr(a,k),getattr(b,k))
        self.assertNotIn('shared_prediction',self.record)

    def test_mean_override_does_not_mutate_actor_config_or_channel(self):
        before=self.record['h'].copy()
        out=OptimizerHook('bounded_average_all')(**self.c)
        self.assertEqual(self.cfg.information_mode,'legacy')
        self.assertEqual(self.cfg.candidate_count,3)
        self.assertEqual(out['config'].candidate_count,4)
        np.testing.assert_array_equal(self.record['h'],before)
        self.assertNotIn('shared_prediction',self.record)
        self.assertIs(out['records']['v']['h'],self.record['h'])

    def test_power_objective_reranks_before_filtering(self):
        seen=[]
        def original(*args):
            seen.append(args[6].candidate_count)
            return [om.TargetCandidate(bs=i,tx_beam=None,rx_beam=None,gain_db=-90.,
                required_rb=d,base_cost=i) for i,d in [(0,10),(2,20),(3,15),(4,1)]]
        got=power_candidates(original,self.args,'v',self.record,1,1.3e6,self.c['load'],
                             self.cfg,self.tx,self.rx,np.zeros(2))
        self.assertEqual(seen,[4])
        self.assertEqual([c.bs for c in got],[4,3,2])
        self.assertAlmostEqual(got[0].base_cost,.2*.9*.2)

    def test_noop_observation_preserves_small_simulation(self):
        cfg=self.cfg
        class Policy:
            config=cfg
            def act(self,local,global_state,explore):
                return np.ones(len(local),dtype=int),np.zeros(len(local)),np.zeros(len(local))
        timeline={800+.1*i:{'v':dict(self.record,pos=np.array([80.+11*i,60.]))} for i in range(5)}
        common=dict(prt=False,rician_fading=False,ho_interruption_ms=10,
                    traffic_trace=make_paired_traffic(self.args,timeline,1))
        old=sim.run_sim_o_mappo(self.args,MICRO_BS_LOCATIONS,timeline,Policy(),**common)
        hook=OptimizerHook('baseline'); obs=Observer(hook)
        with patch.object(om,'_candidate_links',obs.candidate),patch.object(sim,'optimize_triggered_targets',obs.optimize):
            new=sim.run_sim_o_mappo(self.args,MICRO_BS_LOCATIONS,timeline,Policy(),optimizer_input_hook=hook,**common)
        for field in dataclasses.fields(old):
            if not field.name.endswith('time_record'):
                np.testing.assert_equal(getattr(old,field.name),getattr(new,field.name))


if __name__=='__main__':
    unittest.main()
