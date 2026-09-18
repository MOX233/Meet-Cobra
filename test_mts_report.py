"""Information boundary, causal probes, overhead and legacy-equation tests."""
import dataclasses
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from experiment.pql_ba_experiment import paper_args, MICRO_BS_LOCATIONS
from utils.beam_utils import generate_dft_codebook
from utils.gpu_phy import GPUFramePHY
from utils.ho_utils import make_paired_traffic
from utils.mts_gs_hbf import MTSLinkState, candidate_configs, build_link_candidates
from utils.mts_report import reports_from_records,build_report_candidates,report_matching,ReportCommand
from utils.mts_report_sim import probe_and_hold,run_sim_mts_report
from utils.pql_ba import best_beam_pair,no_bf_gain_db,fixed_pair_gain_db


class MTSReportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = paper_args(13e6)
        self.config = dataclasses.replace(candidate_configs()['pressure_early'],ho_interruption_ms=10)
        rng = np.random.default_rng(11)
        self.records = {v:dict(pos=np.array([40.,30.]),
            h=(rng.normal(size=(8,4,32))+1j*rng.normal(size=(8,4,32)))*1e-5,
            shared_prediction=dict(gain=np.array([-75.,-85.,-95.,-100.]),
                interference=np.full(4,-130.),beam=np.array([rng.choice(256,5,replace=False) for _ in range(4)])))
            for v in ['a','b']}
        self.reports = reports_from_records(self.records,800.,self.config)
        self.tx,self.rx = generate_dft_codebook(32),generate_dft_codebook(8)
        self.device = os.environ.get('TEST_PHY_DEVICE','cpu')

    def test_controller_accepts_no_oracle_fields(self):
        class Guarded(dict):
            def __getitem__(self,key):
                if key not in ('pos','shared_prediction'):
                    raise AssertionError('Unauthorized record read: '+key)
                return super().__getitem__(key)
        records = {v:Guarded(r) for v,r in self.records.items()}
        reports = reports_from_records(records,800,self.config)
        with patch('utils.mts_gs_hbf.best_beam_pair',side_effect=AssertionError('Oracle search')):
            commands,result = report_matching(self.args,reports,{'a':0,'b':0},
                {'a':1e4,'b':2e4},{'a':2.6e5,'b':2.6e5},{'a':13e6,'b':13e6},np.zeros(5),self.config)
        self.assertEqual(set(commands),{'a','b'})
        self.assertTrue(all(c.report.source_frame==800 for c in commands.values()))
        with self.assertRaises(ValueError):
            reports['a'].gain[0] = 0

    def test_identical_information_reproduces_legacy_preference_equations(self):
        records = {}
        for v,r in self.records.items():
            best = [best_beam_pair(r['h'],bs,self.tx,self.rx) for bs in range(4)]
            pairs = np.array([tx*8+rx for tx,rx,gain in best])
            records[v] = dict(r,shared_prediction=dict(gain=np.array([z[2] for z in best]),
                interference=no_bf_gain_db(r['h']),beam=np.array([[(p+i)%256 for i in range(5)] for p in pairs])))
        reports = reports_from_records(records,800,self.config)
        states = {'a':MTSLinkState(action=0),'b':MTSLinkState(action=2)}
        conn = {v:s.action for v,s in states.items()}
        queue,upper,rates = ({'a':1e4,'b':2e5},{'a':2.6e5,'b':2.6e5},{'a':13e6,'b':13e6})
        load = np.array([.1,.2,.3,.4,.5])
        old = build_link_candidates(self.args,records,states,queue,upper,rates,load,self.config,self.tx,self.rx)
        new = build_report_candidates(self.args,reports,conn,queue,upper,rates,load,self.config,
            pilot_average_override=self.config.full_sweep_pilots/self.args.slots_per_frame)
        for v in old:
            self.assertEqual([x.bs for x in old[v]],[x.bs for x in new[v]])
            for a,b in zip(old[v],new[v]):
                for field in dataclasses.fields(a):
                    if field.name=='vehicle': continue
                    av,bv=getattr(a,field.name),getattr(b,field.name)
                    if av is None: self.assertIsNone(bv)
                    else: np.testing.assert_allclose(av,bv,rtol=1e-12,atol=1e-12)

    def test_gain_prediction_changes_candidate_and_ho_changes_only_capacity(self):
        kw=dict(args=self.args,reports=self.reports,connection={'a':0,'b':0},
            queue={'a':1e4,'b':1e4},upper={'a':2.6e5,'b':2.6e5},rates={'a':13e6,'b':13e6},load=np.zeros(5))
        a=build_report_candidates(config=self.config,**kw)
        b=build_report_candidates(config=dataclasses.replace(self.config,ho_interruption_ms=0),**kw)
        for v in a:
            for x,y in zip(a[v],b[v]):
                self.assertEqual(x.vehicle_score,y.vehicle_score)
                self.assertAlmostEqual(x.demand_rb,y.demand_rb/(.9 if x.bs else 1))
        changed=dict(self.reports)
        changed['a']=dataclasses.replace(changed['a'],gain=changed['a'].gain+3)
        new=build_report_candidates(config=self.config,**(kw|dict(reports=changed)))
        self.assertGreater(next(x.capacity_per_rb_bps for x in new['a'] if x.bs==1),
                           next(x.capacity_per_rb_bps for x in a['a'] if x.bs==1))

    def test_paid_probe_scalar_parity_and_causality(self):
        phy=GPUFramePHY(self.args,self.records,800.1,1,self.device)
        conn={'a':1,'b':2};states={v:MTSLinkState(action=bs) for v,bs in conn.items()}
        h=phy.h.cpu().numpy()
        values,pilots,pairs=probe_and_hold(phy,conn,states,self.reports,{'a'},10)
        for j,v in enumerate(phy.ids):
            bs=conn[v];slot=10 if v=='a' else 0
            candidates=self.reports[v].beams[bs-1]
            gains=[fixed_pair_gain_db(h[slot,j],bs-1,int(p)//8,int(p)%8,self.tx,self.rx) for p in candidates]
            self.assertEqual(pairs[v],candidates[np.argmax(gains)])
            self.assertEqual(pilots[slot,j],5)
            self.assertEqual(pilots[:,j].sum(),94 if v=='a' else 104)
            for i in [slot,50,99]:
                expected=fixed_pair_gain_db(h[i,j],bs-1,pairs[v]//8,pairs[v]%8,self.tx,self.rx)
                self.assertAlmostEqual(values[i,j],expected,places=9)
        self.assertTrue((pilots[:10,0]==0).all())
        # Change all unobservable time samples; beam choices cannot change.
        phy.h=phy.h.clone()
        phy.h[:10,0]*=2
        phy.h[11:,0]*=4
        phy.h[1:,1]*=3
        newstates={v:MTSLinkState(action=bs) for v,bs in conn.items()}
        _,_,newpairs=probe_and_hold(phy,conn,newstates,self.reports,{'a'},10)
        self.assertEqual(pairs,newpairs)

    def test_full_simulator_outage_and_report_delay(self):
        timeline={800+.1*i:self.records for i in range(4)}
        trace=make_paired_traffic(self.args,timeline,1)
        config=dataclasses.replace(self.config,association_interval_frames=1,full_sweep_interval_frames=1)
        def force(args,reports,connection,*unused):
            commands={v:ReportCommand(1 if connection[v]==0 else 0,reports[v]) for v in reports}
            return commands,SimpleNamespace(proposal_count=2,unassigned=(),used_capacity=np.zeros(5))
        diag=[]
        with patch('utils.mts_report_sim.report_matching',side_effect=force):
            result=run_sim_mts_report(self.args,MICRO_BS_LOCATIONS,timeline,config,trace,
                physics_device=self.device,diagnostics=diag)
        self.assertEqual([d['blocked_vehicle_slots'] for d in diag],[0,20,20])
        for fi in (1,2):
            for v in self.records:
                q=result.queue_per_vehicle_record[fi][v]
                start=result.queue_per_vehicle_record[fi-1][v][-1]
                np.testing.assert_allclose(q[:10],start+np.cumsum(trace['arrivals'][diag[fi]['frame']][v][:10]),atol=1e-8)
        self.assertAlmostEqual(result.pilot_record[1],.94)
        self.assertAlmostEqual(diag[1]['source_frame'],800.1)
        self.assertEqual(set(diag[1]['probed_pairs']),set(self.records))


if __name__=='__main__':
    unittest.main()
