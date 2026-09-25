"""Probe accounting, scalar parity, causality and dynamic physical beams."""
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from experiment.pql_ba_experiment import paper_args
from utils.beam_utils import generate_dft_codebook
from utils.gpu_phy import GPUFramePHY
from utils.mts_gs_hbf import MTSLinkState
from utils.hierarchical_beam import hierarchical_beam_pair
from utils.hierarchical_tracking import acquire32,cross_neighbors,probe_candidates,search_frame
from utils.directional_service import DirectionalService
from utils.pql_ba import fixed_pair_gain_db


class TrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args=paper_args(15e6)
        rng=np.random.default_rng(83)
        self.records={v:dict(h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5)
                      for v in ['a','b','c']}
        self.tx,self.rx=generate_dft_codebook(32),generate_dft_codebook(8)
        self.connection={'a':1,'b':2,'c':0}

    def states(self): return {v:MTSLinkState(action=b) for v,b in self.connection.items()}

    def test_cross_has_no_diagonals_and_wraps(self):
        pairs=cross_neighbors(torch.tensor([0,255,13*8+3])).numpy()
        np.testing.assert_array_equal(pairs[0],[0,248,8,7,1])
        for center, row in zip([0,255,107],pairs):
            self.assertEqual(len(set(row)),5)
            self.assertTrue(all(p//8==center//8 or p%8==center%8 for p in row))

    def test_acquisition_scalar_parity(self):
        h=np.stack([r['h'][:,0,:] for r in self.records.values()])
        pair,amplitude=acquire32(torch.tensor(h),torch.tensor(self.tx),torch.tensor(self.rx))
        for j in range(len(h)):
            t,r,g=hierarchical_beam_pair(h[j,:,None,:],0,self.tx,self.rx)
            self.assertEqual(int(pair[j]),t*8+r)
            self.assertAlmostEqual(20*np.log10(float(amplitude[j])/np.sqrt(256)+1e-9),g,places=10)

    def test_accounting_causality_and_physical_pairs(self):
        phy=GPUFramePHY(self.args,self.records,800.1,1,'cpu')
        for track,total in [(False,131),(True,527)]:
            states=self.states()
            gains,pilots,pairs,_=search_frame(phy,self.connection,states,{'a'},10,track)
            self.assertEqual(pilots[:,1].sum(),total)
            self.assertEqual(pilots[:,0].sum(),32+89*(5 if track else 1))
            self.assertTrue((pilots[:10,0]==0).all())
            self.assertTrue((pilots[:,2]==0).all())
            if track:
                for j,first in [(0,10),(1,0)]:
                    for slot in range(first+1,100):
                        prev=pairs[slot-1,j,self.connection[phy.ids[j]]-1]
                        chosen=pairs[slot,j,self.connection[phy.ids[j]]-1]
                        self.assertIn(chosen,cross_neighbors(torch.tensor(prev)).tolist())
            ds=DirectionalService(phy,pairs,self.connection,1,800.1)
            h=phy.h.numpy()
            for j,first in [(0,10),(1,0)]:
                b=self.connection[phy.ids[j]]-1
                for slot in [first,50,99]:
                    p=pairs[slot,j,b]
                    expected=fixed_pair_gain_db(h[slot,j],b,p//8,p%8,self.tx,self.rx)
                    self.assertAlmostEqual(gains[slot,j],expected,places=9)
                    self.assertAlmostEqual(20*np.log10(np.sqrt(ds.own[slot,j])+1e-9),expected,places=9)
            for slot in [10,50,99]:
                p0,p1=pairs[slot,0,0],pairs[slot,1,1]
                cross=abs(self.rx[:,p0%8].conj() @ h[slot,0,:,1,:] @ self.tx[:,p1//8])**2/256
                np.testing.assert_allclose(ds.cross[slot,0,1],cross,rtol=1e-12,atol=1e-22)
            # Corrupt future slots; earlier choices must be exactly unchanged.
            altered=SimpleNamespace(args=phy.args,ids=phy.ids,device=phy.device,h=phy.h.clone(),
                                    tx=phy.tx,rx=phy.rx,db=phy.db)
            altered.h[51:]=torch.flip(altered.h[51:],dims=[-1])
            _,_,new_pairs,_=search_frame(altered,self.connection,self.states(),{'a'},10,track)
            np.testing.assert_array_equal(new_pairs[:51],pairs[:51])

    def test_five_measured_gain_no_worse_than_current_same_slot(self):
        h=torch.tensor(np.stack([r['h'][:,0,:] for r in self.records.values()]))
        candidates=cross_neighbors(torch.tensor([0,255,100]))
        values=probe_candidates(h,candidates,torch.tensor(self.tx),torch.tensor(self.rx))
        self.assertTrue(torch.all(values.max(1).values>=values[:,0]))

    def test_load_correction_handles_future_vehicles(self):
        from experiment.mts_hierarchical_tracking import BeamAdapter
        adapter=BeamAdapter('hier32_cross5')
        rates={'a':15e6,'b':15e6,'future':15e6}
        with patch('experiment.mts_hierarchical_tracking.estimate_bounded_report_load',return_value=(None,None)) as spy:
            adapter.load(self.args,{},dict(a=1,b=0),rates,None,None,{})
        used=spy.call_args.args[3]
        self.assertEqual(used['future'],15e6)
        self.assertEqual(used['b'],15e6)
        self.assertAlmostEqual(used['a'],15e6/(1-(32+99*5)/100*self.args.pilot_overhead_factor))


if __name__=='__main__': unittest.main()
