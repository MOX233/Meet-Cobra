import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiment import plot_revision_system_results as plot


class ReplacementTest(unittest.TestCase):
    def fixture(self, root, field='actor_sha256', bad_hash=False, bad_traffic=False):
        def write(p, value): p.write_text(json.dumps(value))
        rates=list(range(1,36,2)); seeds=[1,2,3]
        protocol=dict(rates=rates,seeds=seeds,warmup_frames=2,cache_sha256='timeline')
        write(root/'protocol.json',dict(timeline_sha256='timeline'))
        paths=[]; rows=[]
        for method in plot.METHODS:
            for rate in rates:
                for seed in seeds:
                    row=dict(method=method,rate_mbps=rate,seed=seed,frames=300,
                             traffic_sha256=f'{rate}_{seed}',metrics={k:1 for k in plot.METRICS})
                    rows.append(row)
                    if method=='o_mappo':
                        p=root/f'{rate}_{seed}.json'
                        replacement=dict(row,metrics={k:rate+seed for k in plot.METRICS})
                        replacement[field]='incorrect' if bad_hash else 'approved'
                        if bad_traffic: replacement['traffic_sha256']='incorrect'
                        write(p,replacement); np.savez(p.with_suffix('.npz'),q=np.array([0.]))
                        paths.append(dict(file=str(p),sha256=plot.digest(p),raw_sha256=plot.digest(p.with_suffix('.npz'))))
        write(root/'summary.json',dict(cases=54,raw_metrics_recomputed=True,rates=rates,seeds=seeds,
              seconds=30,warmup_frames=2,protocol_sha256=plot.digest(root/'protocol.json'),
              policy_sha256='approved',provenance=paths))
        curves={m:{k:np.ones((18,3)) for k in plot.METRICS} for m in plot.METHODS}
        return protocol,rows,curves

    def test_only_o_mappo_changes_for_new_and_old_result_formats(self):
        for field in ('actor_sha256','checkpoint_sha256'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                root=Path(folder); p,rows,curves=self.fixture(root,field)
                before=copy.deepcopy(curves)
                after,proof=plot.replace_o_mappo(root,p,rows,curves)
                self.assertEqual(proof['cases'],54)
                self.assertTrue(proof['paired_traffic_verified'])
                for old,new in zip(rows,after):
                    if old['method']!='o_mappo': self.assertEqual(old,new)
                for method in plot.METHODS:
                    for metric in plot.METRICS:
                        if method=='o_mappo':
                            np.testing.assert_array_equal(curves[method][metric],
                                np.array([[r+s for s in p['seeds']] for r in p['rates']]))
                        else: np.testing.assert_array_equal(curves[method][metric],before[method][metric])

    def test_rejects_checkpoint_or_traffic_mismatch(self):
        for bad in ('bad_hash','bad_traffic'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                root=Path(folder); p,rows,curves=self.fixture(root,**{bad:True})
                with self.assertRaises(AssertionError): plot.replace_o_mappo(root,p,rows,curves)


if __name__=='__main__': unittest.main()
