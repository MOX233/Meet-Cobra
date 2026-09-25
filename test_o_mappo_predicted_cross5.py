"""Information-boundary, causality and exact physical-kernel regression tests."""
import copy
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from experiment import o_mappo_predicted_cross5 as new
from experiment.o_mappo_slot_tracking import track_frame
from experiment.o_mappo_target_check import estimate_bounded
from experiment.pql_ba_experiment import paper_args
from utils import o_mappo as om, alg_utils as alg
from utils.gpu_phy import GPUFramePHY
from utils.directional_service import DirectionalService


class PredictionCrossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args=paper_args(15e6)
        rng=np.random.default_rng(16)
        self.records={v:dict(h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5,
            pos=np.array([30.+j*5,10.]),angle=90.,v=15.,g_opt_beam=np.ones(4)*100,
            shared_prediction=dict(gain=np.array([-79.,-83.,-85.,-84.]),
                interference=np.array([-105.,-111.,-112.,-110.]),source_frame=710.1,target_frame=710.2))
            for j,v in enumerate(['a','b','c','d','e'])}
        self.connection=dict(a=1,b=1,c=2,d=3,e=0)
        self.queues={v:float(250000+10000*j) for j,v in enumerate(self.records)}
        self.rates={v:15e6 for v in self.records}
        self.cfg=new.configuration(om.OMAPPOConfig(actor_hidden_sizes=(64,64),
            beam_search_variant='hierarchical32',candidate_gain_mode='search',optimizer_solver='milp'))

    def states(self):
        return {v:om.OMAPPOLearnerState(action=b,tx_beam=12 if b else None,rx_beam=3 if b else None,
            pending_action=None,last_position=self.records[v]['pos'].copy(),distance_since_event=10,
            current_sweep_pilots=32 if v=='a' else 0,last_handover=v in ('a','e'))
            for v,b in self.connection.items()}

    def test_public_whitelist_and_prediction_time(self):
        public=new.public_records(self.records,710.1)
        self.assertEqual(set(public['a']),{'pos','angle','v','shared_prediction'})
        self.assertEqual(set(public['a']['shared_prediction']),{'gain','interference'})
        with self.assertRaises(ValueError):new.public_records(self.records,710.2)
        self.records['a']['shared_prediction']['gain'][0]=np.nan
        with self.assertRaises(ValueError):new.public_records(self.records,710.1)

    def test_unobserved_csi_cannot_change_decisions(self):
        public=new.public_records(self.records,710.1)
        policy=om.OMAPPPolicy(self.cfg,seed=10)
        outputs=new.decide(self.args,public,self.states(),self.queues,self.rates,self.cfg,policy,1.)
        for r in self.records.values():
            r['h']*=1e12; r['g_opt_beam']*=10
            r['best_beam_pair_idx']=np.full(4,255)
            r['shared_prediction']['beam']=np.full((4,5),250)
        changed=new.decide(self.args,new.public_records(self.records,710.1),self.states(),
            self.queues,self.rates,self.cfg,policy,1.)
        np.testing.assert_array_equal(outputs[0].rb,changed[0].rb)
        for v in outputs[1]:
            a,b=outputs[1][v],changed[1][v]
            self.assertEqual(a['command'],b['command'])
            np.testing.assert_array_equal(a['local'],b['local'])
            np.testing.assert_array_equal(a['global_state'],b['global_state'])

    def test_prediction_context_matches_bounded_estimator(self):
        p=new.public_records(self.records,710.1)
        c=new.context(self.args,p,self.connection,self.rates)
        np.testing.assert_allclose(c.rb,estimate_bounded(self.args,self.connection,c.gains,c.interference,self.rates),rtol=1e-12,atol=1e-10)
        self.assertTrue(np.all(c.rb<=c.caps))
        for v in p:p[v]['shared_prediction']['gain']-=60
        overloaded=new.context(self.args,p,self.connection,self.rates)
        self.assertTrue(np.all((overloaded.load>=0)&(overloaded.load<=1)))

    def test_unpaid_acquisition_is_not_performed(self):
        s=self.states()['a']
        with patch.object(om,'candidate_beam_pair',side_effect=AssertionError('free search')):
            out=new.apply_command(s,om.OMAPPOCommand(1,2))
        self.assertEqual(s.action,2)
        self.assertEqual(out.sweep_pilots,32)
        self.assertEqual((s.tx_beam,s.rx_beam),(0,0))
        out=new.apply_command(s,om.OMAPPOCommand(0,2))
        self.assertFalse(out.handover);self.assertEqual(out.sweep_pilots,0)

    def test_architecture_and_configuration_preserved(self):
        p=om.OMAPPPolicy(self.cfg,seed=1)
        self.assertEqual(p.local_dim,31)
        self.assertEqual(p.global_dim,94)
        self.assertEqual(self.cfg.actor_hidden_sizes,(64,64))
        self.assertEqual(self.cfg.batch_size,256)
        self.assertEqual(self.cfg.ppo_epochs,4)
        self.assertEqual(self.cfg.tracking_pilots,5)

    def test_exact_kernel_matches_existing_otr_and_directional_service(self):
        phy=GPUFramePHY(self.args,self.records,710.1,101,'cpu');phy.records=self.records
        states=self.states()
        gains,pilots,pairs,_=track_frame(phy,self.connection,states)
        inter={v:np.r_[-80.,self.records[v]['shared_prediction']['interference']] for v in self.records}
        rb=np.array([100.,20.,30.,25.,10.])
        rng=np.random.default_rng(13)
        arrivals={v:rng.poisson(15000,size=100) for v in self.records}
        fast,fast_k,served,_,_=new.serve_frame(self.args,phy,copy.deepcopy(states),self.connection,
                                             self.queues,arrivals,inter,rb,101,710.1)
        evaluator=DirectionalService(phy,pairs,self.connection,101,710.1)
        ids=phy.ids
        q={v:np.r_[self.queues[v],np.zeros(100)] for v in ids}
        upper={v:300000. for v in ids}
        for slot in range(100):
            blocked={v for v in ids if states[v].last_handover and slot<10}
            gain={v:np.r_[om.macro_gain_db(self.args,self.records[v]['pos'],np.zeros(2)),np.full(4,-180.)] for v in ids}
            ps={v:np.full(4,pilots[slot,j]) for j,v in enumerate(ids)}
            for j,v in enumerate(ids):
                if self.connection[v]>0: gain[v][self.connection[v]]=gains[slot,j]
            alloc={v:0 for v in blocked}
            for bs in range(5):
                alloc.update(alg.RA_OTR_SINR(self.args,slot,bs,[v for v in ids if self.connection[v]==bs and v not in blocked],
                    self.rates,upper,q,arrivals,gain,ps,infer_g_dict=inter,est_num_RB_allocated_perBS=rb))
            np.testing.assert_array_equal([alloc[v] for v in ids],fast_k[slot])
            evaluator.update(self.args,slot_idx=slot,connection_dict=self.connection,RA_dict=alloc,
                backlog_queue_dict=q,a_dict=arrivals,g_dict=gain,num_pilot_dict=ps)
            np.testing.assert_allclose([q[v][slot+1] for v in ids],fast[:,slot+1],rtol=1e-10,atol=1e-6)
        np.testing.assert_allclose(served,[self.queues[v]+arrivals[v].sum()-q[v][-1] for v in ids],rtol=1e-10)

    def test_batched_beams_match_original_probe_by_probe(self):
        phy=GPUFramePHY(self.args,self.records,710.1,101,'cpu')
        original=track_frame(phy,self.connection,self.states())
        batched=new.track_frame_batched(phy,self.connection,self.states())
        np.testing.assert_allclose(original[0],batched[0],atol=1e-10,rtol=1e-12)
        np.testing.assert_array_equal(original[1],batched[1])
        np.testing.assert_array_equal(original[2],batched[2])
        self.assertEqual(original[3],batched[3])

    def test_batched_future_samples_do_not_change_past_beams(self):
        phy=GPUFramePHY(self.args,self.records,710.1,101,'cpu')
        original=new.track_frame_batched(phy,self.connection,self.states())
        phy.h=phy.h.clone()
        phy.h[51:]*=torch.linspace(.1,5,32)[None,None,None,None,:]
        changed=new.track_frame_batched(phy,self.connection,self.states())
        np.testing.assert_array_equal(original[2][:51],changed[2][:51])
        np.testing.assert_allclose(original[0][:51],changed[0][:51])


if __name__=='__main__':unittest.main()
