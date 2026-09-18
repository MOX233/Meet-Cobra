"""Protocol partition and per-round cap audit tests for the new full grid."""
import copy
import unittest
import numpy as np
from experiment.revision_cap_grid import GapAudit,RERUN,REUSE,METHODS,RATES,SEEDS,audit_unaffected_sources


class RevisionCapGridTests(unittest.TestCase):
    def test_grid_partition(self):
        self.assertEqual(len(RERUN)*len(RATES)*len(SEEDS),450)
        self.assertEqual(len(REUSE)*len(RATES)*len(SEEDS),180)
        self.assertFalse(set(REUSE)&set(RERUN))
        self.assertEqual(set(REUSE)|set(RERUN),set(METHODS))

    def record(self):
        return dict(iterations=2,cap_rb_usage=True,physical_capacity=[133.,66.,66.],
            post_repair_frame_average_load=[2.,66.,1.],traces=[
                dict(input_load=[133.,66.,66.],implied_demand=[2.,99.,1.],implied_load=[2.,66.,1.]),
                dict(input_load=[2.,66.,1.],implied_demand=[2.,88.,1.],implied_load=[2.,66.,1.])])

    def test_audit_accepts_overloaded_demand_but_bounded_usage(self):
        records=GapAudit()
        records.append(self.record())
        self.assertEqual(len(records),1)

    def test_audit_rejects_unbounded_feedback(self):
        for key in ('input_load','implied_load'):
            record=self.record()
            record['traces'][1][key][1]=67.
            with self.assertRaises(AssertionError): GapAudit().append(record)

    def test_audit_rejects_incorrect_saturation(self):
        record=self.record()
        record['traces'][0]['implied_load'][1]=59.
        with self.assertRaises(AssertionError): GapAudit().append(record)

    def test_unaffected_reference_sources(self):
        self.assertEqual(audit_unaffected_sources()['unchanged_methods'],REUSE)


if __name__=='__main__':
    unittest.main()
