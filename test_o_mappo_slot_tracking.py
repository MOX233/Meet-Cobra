"""Causal HO-only acquisition, cross-frame beams and exact paid probes."""
import dataclasses
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from experiment.o_mappo_slot_tracking import SlotTracking,track_frame
from experiment.pql_ba_experiment import paper_args
from utils.gpu_phy import GPUFramePHY
from utils import o_mappo as om
from utils.directional_service import DirectionalService
from utils.hierarchical_tracking import cross_neighbors
from utils.pql_ba import fixed_pair_gain_db


class SlotTrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args=paper_args(15e6)
        rng=np.random.default_rng(64)
        self.records={v:dict(h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5)
                      for v in ['a','b','c']}
        self.phy=GPUFramePHY(self.args,self.records,800.1,1,'cpu')
        self.connection={'a':1,'b':2,'c':0}

    def states(self):
        return {v:SimpleNamespace(action=b,tx_beam=12 if b else None,rx_beam=3 if b else None,
            current_sweep_pilots=32 if v=='a' else 0,last_handover=v=='a')
            for v,b in self.connection.items()}

    def test_ho_only_acquisition_and_no_mutation(self):
        states=self.states(); before={v:vars(s).copy() for v,s in states.items()}
        gains,pilots,pairs,final=track_frame(self.phy,self.connection,states)
        self.assertEqual(pilots[:,0].sum(),32+89*5)
        self.assertEqual(pilots[:,1].sum(),500)
        self.assertTrue((pilots[:10,0]==0).all())
        self.assertEqual(pilots[:,2].sum(),0)
        self.assertIn(pairs[0,1,1],cross_neighbors(torch.tensor(12*8+3)).tolist())
        for v,s in states.items(): self.assertEqual(vars(s),before[v])
        ds=DirectionalService(self.phy,pairs,self.connection,1,800.1)
        h=self.phy.h.numpy(); tx=self.phy.tx.numpy(); rx=self.phy.rx.numpy()
        for j,first in [(0,10),(1,0)]:
            bs=self.connection[self.phy.ids[j]]-1
            for slot in [first,50,99]:
                pair=pairs[slot,j,bs]
                expected=fixed_pair_gain_db(h[slot,j],bs,pair//8,pair%8,tx,rx)
                self.assertAlmostEqual(gains[slot,j],expected,places=10)
                self.assertAlmostEqual(20*np.log10(np.sqrt(ds.own[slot,j])+1e-9),expected,places=10)
            for slot in range(first+1,100):
                self.assertIn(pairs[slot,j,bs],cross_neighbors(torch.tensor(pairs[slot-1,j,bs])).tolist())
        for slot in [10,50,99]:
            a,b=pairs[slot,0,0],pairs[slot,1,1]
            expected=abs(rx[:,a%8].conj() @ h[slot,0,:,1,:] @ tx[:,b//8])**2/256
            np.testing.assert_allclose(ds.cross[slot,0,1],expected,rtol=1e-12,atol=1e-22)
        for v,p in final.items():
            states[v].tx_beam,states[v].rx_beam=divmod(p,8)
            states[v].current_sweep_pilots=0; states[v].last_handover=False
        _,next_pilots,next_pairs,_=track_frame(self.phy,self.connection,states)
        self.assertTrue((next_pilots[:,:2]==5).all())
        self.assertIn(next_pairs[0,0,0],cross_neighbors(torch.tensor(final['a'])).tolist())

    def test_future_samples_cannot_change_past_beams(self):
        *_,pairs,final=track_frame(self.phy,self.connection,self.states())
        other=SimpleNamespace(**vars(self.phy)); other.h=self.phy.h.clone()
        other.db=self.phy.db
        other.h[51:]=torch.flip(other.h[51:],dims=[-1])
        _,_,altered,_=track_frame(other,self.connection,self.states())
        np.testing.assert_array_equal(altered[:51],pairs[:51])

    def test_stay_command_does_not_search_nine_pairs(self):
        adapter=SlotTracking()
        cfg=om.OMAPPOConfig(beam_search_variant='hierarchical32',tracking_pilots=5)
        state=om.OMAPPOLearnerState(action=1,tx_beam=12,rx_beam=3,
            pending_action=None,last_position=np.zeros(2),distance_since_event=0.)
        result=adapter.apply(state,om.OMAPPOCommand(0,1),{},cfg,None,None)
        self.assertEqual((state.tx_beam,state.rx_beam),(12,3))
        self.assertEqual(result.sweep_pilots,0)
        self.assertFalse(result.handover)

    def test_end_state_committed_only_after_service(self):
        adapter=SlotTracking(); states=self.states()
        before={v:(s.tx_beam,s.rx_beam) for v,s in states.items()}
        adapter.physical(self.phy,self.connection,states)
        adapter.state_pairs(self.phy,self.connection,states)
        self.assertEqual({v:(s.tx_beam,s.rx_beam) for v,s in states.items()},before)
        with self.assertRaises(AssertionError): adapter.finish()
        adapter.checked=100; adapter.finish()
        for v,p in adapter.final.items(): self.assertEqual((states[v].tx_beam,states[v].rx_beam),divmod(p,8))

    def test_existing_cost_formula_accounts_for_five_tracking_probes(self):
        self.assertAlmostEqual(om.average_sweep_pilots(self.args,32,5),(32+99*5)/100)
        self.assertGreater(om.average_sweep_pilots(self.args,32,5),om.average_sweep_pilots(self.args,32,1))

    def test_policy_configuration_changes_only_tracking_count(self):
        adapter=SlotTracking()
        config=om.OMAPPOConfig(beam_search_variant='hierarchical32',actor_hidden_sizes=(64,64))
        actor=object(); policy=SimpleNamespace(config=config,actor=actor)
        adapter.original_load=lambda:policy
        result=adapter.load()
        expected=dataclasses.asdict(config); expected['tracking_pilots']=5
        self.assertEqual(dataclasses.asdict(result.config),expected)
        self.assertIs(result.actor,actor)


if __name__=='__main__': unittest.main()
