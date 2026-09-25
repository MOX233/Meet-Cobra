import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiment import plot_revision_system_results as plot


class MTSReplacementTest(unittest.TestCase):
    def fixture(self, root, bad_traffic=False, bad_variant=False):
        def write(p,v): p.write_text(json.dumps(v))
        rates=list(range(1,36,2)); seeds=[1,2,3]
        protocol=dict(rates=rates,seeds=seeds,warmup_frames=2,cache_sha256='cache')
        write(root/'protocol.json',dict(cache_sha256='cache'))
        sha=plot.digest(root/'protocol.json')
        for folder in ['runs','raw','diagnostics']: (root/folder).mkdir()
        rows=[]; provenance=[]
        for method in plot.METHODS:
            for rate in rates:
                for seed in seeds:
                    row=dict(method=method,rate_mbps=rate,seed=seed,frames=300,
                        traffic_sha256=f'{rate}_{seed}',metrics={k:1 for k in plot.METRICS})
                    rows.append(row)
                    if method!='mts_report': continue
                    path=root/'runs'/f'{rate}_{seed}.json'
                    new=dict(row,variant='wrong' if bad_variant else 'hier32_cross5',protocol_sha256=sha,
                        metrics={k:rate+seed for k in plot.METRICS})
                    if bad_traffic: new['traffic_sha256']='wrong'
                    write(path,new)
                    raw=root/'raw'/f'{path.stem}.npz'; np.savez(raw,q=np.zeros(1))
                    dp=root/'diagnostics'/f'{path.stem}.json'; write(dp,{})
                    provenance.append(dict(file=str(path),sha256=plot.digest(path),
                        raw_sha256=plot.digest(raw),diagnostics_sha256=plot.digest(dp)))
        write(root/'summary.json',dict(cases=54,raw_metrics_recomputed=True,variant='hier32_cross5',
            rates=rates,seeds=seeds,seconds=30,warmup_frames=2,protocol_sha256=sha,provenance=provenance))
        curves={m:{k:np.ones((18,3)) for k in plot.METRICS} for m in plot.METHODS}
        return protocol,rows,curves

    def test_only_mts_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p,rows,curves=self.fixture(root); before=copy.deepcopy(curves)
            updated,proof=plot.replace_mts(root,p,rows,curves)
            self.assertTrue(proof['paired_traffic_verified'])
            for old,new in zip(rows,updated):
                if old['method']!='mts_report': self.assertEqual(old,new)
            for m in plot.METHODS:
                for k in plot.METRICS:
                    expected=np.array([[r+s for s in p['seeds']] for r in p['rates']]) if m=='mts_report' else before[m][k]
                    np.testing.assert_array_equal(curves[m][k],expected)

    def test_rejects_information_or_traffic_mismatch(self):
        for flag in ['bad_traffic','bad_variant']:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); p,rows,curves=self.fixture(root,**{flag:True})
                with self.assertRaises(AssertionError): plot.replace_mts(root,p,rows,curves)


if __name__=='__main__': unittest.main()
