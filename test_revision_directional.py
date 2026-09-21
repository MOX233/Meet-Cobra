"""Independent label, information timing and actual-service checks."""
import copy
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiment.prepare_stateful_trajectories import frame_values
from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.directional_service import DirectionalService, beam_average_gain_db
from utils.gpu_phy import GPUFramePHY
from utils.ho_utils import make_paired_traffic
from utils.revision_meet_sim import predicted_records, run_revised_meet
from experiment.interference_validation import directional_gains,rb_owners,explicit_interference


class RevisionDirectionalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args=paper_args(13e6)
        self.device=os.environ.get('TEST_PHY_DEVICE','cpu')
        rng=np.random.default_rng(91)
        self.records={v:dict(h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5,
            pos=np.array([200.,250.]),CSI_preprocessed=np.zeros((1,128),np.float32)) for v in range(3)}

    def test_new_labels_and_legacy_unchanged(self):
        records=list(self.records.values())
        old=frame_values(records)
        new=frame_values(records,interference_label='beam-average')
        for a,b in zip(old[:3],new[:3]): np.testing.assert_array_equal(a,b)
        h=np.stack([r['h'] for r in records]).astype(np.complex64)
        expected=10*np.log10(np.mean(np.abs(h).astype(np.float64)**2,axis=(1,3)))
        np.testing.assert_allclose(new[3],expected,atol=1e-5)
        np.testing.assert_array_equal(old[3],(20*np.log10(abs(h).max(axis=(1,3))+1e-9)).astype(np.float32))
        self.assertGreater(np.max(abs(new[3]-old[3])),1.)

    def test_uniform_codebook_mean(self):
        h=self.records[0]['h']
        rx=np.exp(-2j*np.pi*np.outer(np.arange(8),np.arange(8))/8)/np.sqrt(8)
        tx=np.exp(-2j*np.pi*np.outer(np.arange(32),np.arange(32))/32)/np.sqrt(32)
        expected=[10*np.log10(np.mean(abs(rx.conj().T@h[:,b,:]@tx)**2)) for b in range(4)]
        np.testing.assert_allclose(beam_average_gain_db(h),expected,rtol=1e-12)

    def test_service_parity_and_no_proxy_read(self):
        phy=GPUFramePHY(self.args,self.records,800.1,2,self.device)
        pairs=np.zeros((100,3,4),int); pairs[:,1,:]=19
        conn={0:1,1:2,2:0}; bs=np.array([1,2,0])
        evaluator=DirectionalService(phy,pairs,conn,2,800.1)
        cross,mean,own=directional_gains(phy,pairs,bs)
        np.testing.assert_allclose(evaluator.cross,cross)
        k=np.array([66,66,0]); owners=rb_owners(bs,k,self.args.num_RB_micro,
            np.random.default_rng(np.random.SeedSequence([2,8001,24681357])))
        interference,assigned=explicit_interference(cross[0],bs,owners,self.args.p_micro)
        noise=self.args.N0*self.args.RB_intervel_micro*10**(self.args.NF_micro_dB/10)
        expected=.98*self.args.RB_intervel_micro*self.args.slot_len*(np.log2(
            1+self.args.p_micro*own[0,:,None]/(noise+interference))*assigned).sum(1)
        q={v:np.full(101,1e8) for v in conn}; arrivals={v:np.full(100,100.) for v in conn}
        class Forbidden(dict):
            def __getitem__(self,key): raise AssertionError('Read proxy in physical service')
        pilots={v:np.full(4,2/self.args.pilot_overhead_factor*.01) for v in conn}
        out=evaluator.update(self.args,slot_idx=0,connection_dict=conn,RA_dict=dict(enumerate(k)),
            backlog_queue_dict=q,a_dict=arrivals,num_pilot_dict=pilots,g_dict={v:np.full(5,-100.) for v in conn},
            infer_g_dict=Forbidden())
        np.testing.assert_allclose([out[v][1] for v in conn],1e8-expected+100,atol=1e-7)
        self.assertGreater(evaluator.stats['directional_interference_sum'],0.)

    def test_capacity_and_zero_service(self):
        phy=GPUFramePHY(self.args,self.records,800.1,2,self.device)
        ev=DirectionalService(phy,np.zeros((100,3,4),int),{v:1 for v in self.records},2,800.1)
        kw=dict(slot_idx=0,connection_dict={v:1 for v in self.records},RA_dict={0:67,1:0,2:0})
        with self.assertRaises(ValueError): ev.update(self.args,**kw)
        kw.update(RA_dict={v:0 for v in self.records},backlog_queue_dict={v:np.ones(101)*12 for v in self.records},
            a_dict={v:np.ones(100)*3 for v in self.records},num_pilot_dict={v:np.zeros(4) for v in self.records})
        out=ev.update(self.args,**kw)
        self.assertTrue(all(out[v][1]==15 for v in self.records))

    def test_frame_alignment_and_causal_bf_ra(self):
        timeline={}
        for j in range(4):
            frame=round(800+j*.1,1)
            timeline[frame]=copy.deepcopy(self.records)
            for v,r in timeline[frame].items():
                r['shared_prediction']=dict(source_frame=frame,target_frame=round(frame+.1,1),
                    gain=np.full(4,-80.+j),interference=np.full(4,-130.+j),
                    beam=np.tile(np.arange(j*5,j*5+5),(4,1)))
        broken=copy.deepcopy(timeline[800.1]); broken[0]['shared_prediction']['source_frame']=800.
        with self.assertRaises(ValueError): predicted_records(broken,800.1)
        traffic=make_paired_traffic(self.args,timeline,1)
        seen=[]; real_pet=GPUFramePHY.pet
        def pet(obj,beams,gains,k):
            seen.append((copy.deepcopy(beams),copy.deepcopy(gains)))
            return real_pet(obj,beams,gains,k)
        ra_seen=[]
        from utils import alg_utils
        ra_original=alg_utils.RA_OTR_SINR
        def ra(*a,**kw):
            if kw['slot_idx']==10 and kw['BS_id']==1: ra_seen.append(copy.deepcopy(kw['infer_g_dict']))
            return ra_original(*a,**kw)
        with patch('utils.revision_meet_sim.alg.HO_EE_GAP_APX_SINR_conservative_adaptive',
            return_value=({v:1 for v in self.records},np.array([0,20,0,0,0]))),\
            patch.object(GPUFramePHY,'pet',pet),patch('utils.revision_meet_sim.alg.RA_OTR_SINR',ra):
            result=run_revised_meet(self.args,MICRO_BS_LOCATIONS,timeline,'meet_cobra',traffic,device=self.device)
        for j,(beams,gains) in enumerate(seen):
            np.testing.assert_array_equal(beams[0],timeline[round(800+j*.1,1)][0]['shared_prediction']['beam'])
            np.testing.assert_allclose(ra_seen[j][0][1:],-130+j)
        self.assertEqual(result.handover_record[1],3)
        self.assertTrue(np.all(result.rb_allocated_record <= [133,66,66,66,66]))


if __name__=='__main__': unittest.main()
