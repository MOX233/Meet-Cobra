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

    def prediction_fixture(self, root):
        p, rows, curves = self.fixture(root)
        policy = root/'selected.pt'; policy.write_bytes(b'validation-selected policy')
        selected = dict(label='seed11', policy=str(policy), score=1.1965)
        (root/'test').mkdir()
        (root/'test/protocol.json').write_text('{}')
        (root/'protocol.json').write_text(json.dumps(dict(test_sha256='timeline')))
        (root/'selection.json').write_text(json.dumps(dict(selected=selected)))
        (root/'complete.json').write_text(json.dumps(dict(selection=selected)))
        records = []
        for r in plot.read(root/'summary.json')['provenance']:
            path = Path(r['file']); row = plot.read(path)
            row.pop('actor_sha256')
            row.update(rate=row.pop('rate_mbps'), label='predicted_cross5',
                       policy_sha256=plot.digest(policy),
                       protocol_sha256=plot.digest(root/'test/protocol.json'))
            path.write_text(json.dumps(row))
            records.append(dict(label='Prediction-input O-MAPPO', path=str(path),
                                sha256=plot.digest(path), raw_sha256=r['raw_sha256']))
        # A comparison control must not be mistaken for the approved curve.
        records.append(dict(label='Prediction input, no fine-tuning', path='not-a-selected-case',
                            sha256='ignored', raw_sha256='ignored'))
        analysis = dict(selected=selected, selected_sha256=plot.digest(policy),
                        same_traffic_verified=True, raw_metrics_recomputed=True,
                        simulations_per_scheme=54, rates=p['rates'], seeds=p['seeds'],
                        seconds=30, warmup_frames=2, rows=records)
        (root/'analysis.json').write_text(json.dumps(analysis))
        return p, rows, curves

    def test_prediction_input_format_preserves_other_seven_schemes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); p,rows,curves=self.prediction_fixture(root)
            before=copy.deepcopy(curves)
            after,proof=plot.replace_o_mappo(root,p,rows,curves)
            self.assertEqual(proof['cases'],54)
            self.assertTrue(proof['summary_source'].endswith('analysis.json'))
            for old,new in zip(rows,after):
                if old['method']!='o_mappo': self.assertEqual(old,new)
            for method in plot.METHODS:
                for metric in plot.METRICS:
                    expected=np.array([[r+s for s in p['seeds']] for r in p['rates']])
                    np.testing.assert_array_equal(curves[method][metric],
                        expected if method=='o_mappo' else before[method][metric])

    def test_prediction_input_rejects_changed_selection_or_missing_case(self):
        for bad in ('selection','missing'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as folder:
                root=Path(folder); p,rows,curves=self.prediction_fixture(root)
                a=plot.read(root/'analysis.json')
                if bad=='selection': a['selected']['label']='not-validation-selected'
                else: a['rows'].pop(0)
                (root/'analysis.json').write_text(json.dumps(a))
                with self.assertRaises(AssertionError): plot.replace_o_mappo(root,p,rows,curves)


if __name__=='__main__': unittest.main()
