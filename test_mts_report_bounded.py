import dataclasses
import unittest
import numpy as np
from test_mts_report import MTSReportTests
from experiment.pql_ba_experiment import MICRO_BS_LOCATIONS
from utils.mts_report_bounded import estimate_bounded_report_load


class BoundedLoadTests(unittest.TestCase):
    def setUp(self):
        fixture=MTSReportTests()
        fixture.setUp()
        self.f=fixture

    def evaluate(self,reports,connection,measurements):
        f=self.f
        return estimate_bounded_report_load(f.args,reports,connection,
            {v:13e6 for v in reports},f.config,MICRO_BS_LOCATIONS,measurements)

    def test_deep_fade_and_large_interference_cannot_exceed_capacity(self):
        reports={v:dataclasses.replace(r,gain=np.full(4,-180.),interference=np.full(4,-40.))
                 for v,r in self.f.reports.items()}
        occupied,load=self.evaluate(reports,{'a':1,'b':2},{})
        np.testing.assert_array_equal(occupied,[0,66,66,0,0])
        np.testing.assert_array_equal(load,[0,1.5,1.5,0,0])

    def test_interference_free_load_matches_closed_form(self):
        f=self.f
        reports={v:dataclasses.replace(r,interference=np.full(4,-300.)) for v,r in f.reports.items()}
        occupied,load=self.evaluate(reports,{'a':1,'b':2},{})
        for v,bs in [('a',1),('b',2)]:
            snr=f.args.p_micro*10**(reports[v].gain[bs-1]/10)/(f.args.N0*f.args.RB_intervel_micro*10**(f.args.NF_micro_dB/10))
            expected=13e6/(f.args.RB_intervel_micro*np.log2(1+snr)+2e-10)
            self.assertAlmostEqual(occupied[bs],min(66,expected),places=9)
            self.assertAlmostEqual(load[bs],min(1.5,expected/66),places=9)

    def test_only_same_bs_last_measurement_is_used(self):
        connection={'a':1,'b':2}
        plain=self.evaluate(self.f.reports,connection,{})
        ignored=self.evaluate(self.f.reports,connection,{'a':(2,-180.)})
        np.testing.assert_array_equal(plain,ignored)
        observed=self.evaluate(self.f.reports,connection,{'a':(1,-180.)})
        self.assertEqual(observed[0][1],66)


if __name__=='__main__':
    unittest.main()
