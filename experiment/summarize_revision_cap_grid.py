#!/usr/bin/env python3
"""Raw audit, five-seed curves, and paired changes from the previous grid."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import revision_cap_grid as run
from experiment import summarize_revision_grid as summarize


def export(preflight=False):
    protocol,sha=run.validate(check_inputs=True)
    summarize.OUTPUT=run.OUTPUT
    summarize.METHODS=run.METHODS
    methods=run.RERUN if preflight else run.METHODS
    rates=run.PREFLIGHT_RATES if preflight else run.RATES
    seeds=[1] if preflight else run.SEEDS
    result=summarize.summarize(methods,rates,seeds,check_raw=True)
    if result is None: raise RuntimeError('Requested grid is incomplete')
    expected_new=len(run.RERUN)*len(rates)*len(seeds)
    origin_counts={}
    differences=[]
    for method in methods:
        for rate in rates:
            current=[]
            previous=[]
            for seed in seeds:
                row=json.loads((run.OUTPUT/'runs'/f'{method}_rate{rate}_seed{seed}.json').read_text())
                origin_counts[row['origin']]=origin_counts.get(row['origin'],0)+1
                prior,prior_path,_=run.reference_row('mts' if method=='mts_report' else method,rate,seed)
                assert prior['traffic_sha256']==row['traffic_sha256']
                reference=row['previous_reference']
                assert str(prior_path.relative_to(ROOT))==reference['path']
                assert run.old.digest(prior_path)==reference['sha256']
                if row['origin']=='reused_unaffected':
                    assert row['metrics']==prior['metrics'] and row['raw_sha256']==prior['raw_sha256']
                else:
                    dpath=run.OUTPUT/'diagnostics'/f'{method}_rate{rate}_seed{seed}.json'
                    assert run.old.digest(dpath)==row['diagnostics_sha256']
                    diagnostic=json.loads(dpath.read_text())
                    assert len(diagnostic['ho'])==300
                    if method!='mts_report':
                        assert len(diagnostic['gap'])==300
                        checker=run.GapAudit()
                        for item in diagnostic['gap']: checker.append(item)
                    else:
                        for frame in diagnostic['ho']:
                            assert (np.array(frame['estimated_rb'])<=np.array([133,66,66,66,66])).all()
                            assert abs(frame['frame']-frame['source_frame']-.1)<1e-8
                current.append(row['metrics'])
                previous.append(prior['metrics'])
            record=dict(method=method,previous_method='mts_true_csi' if method=='mts_report' else method,
                        rate_mbps=rate,n=len(seeds),metrics={})
            for metric in current[0]:
                delta=np.array([a[metric]-b[metric] for a,b in zip(current,previous)])
                sd=float(delta.std(ddof=1)) if len(seeds)>1 else None
                record['metrics'][metric]=dict(mean_change=float(delta.mean()),
                    sd_change=sd,ci95_halfwidth=float(t.ppf(.975,len(seeds)-1)*sd/np.sqrt(len(seeds))) if sd is not None else None,
                    seed_changes=delta.tolist(),old_mean=float(np.mean([r[metric] for r in previous])),
                    new_mean=float(np.mean([r[metric] for r in current])))
            differences.append(record)
    assert origin_counts.get('new_run',0)==expected_new
    if not preflight: assert origin_counts.get('reused_unaffected',0)==180
    suffix='preflight' if preflight else 'full'
    metadata=dict(protocol_sha256=sha,origin_counts=origin_counts,raw_and_diagnostics_checked=True,
        original_results_unchanged=True,
        caveats=[protocol['prediction_timing'],protocol['common_ra_estimator'],protocol['reference_status']],
        paired_difference_definition='new minus previous, paired by traffic and fading seed; MTS change includes information/BF/load-estimator adaptation, not GAP clipping alone.',
        rows=differences)
    run.old.write_json(run.OUTPUT/'aggregate'/f'paired_changes_{suffix}.json',metadata)
    flat=[]
    for row in differences:
        for metric,values in row['metrics'].items():
            flat.append(dict(method=row['method'],previous_method=row['previous_method'],
                rate_mbps=row['rate_mbps'],n=row['n'],metric=metric,
                **{k:v for k,v in values.items() if k!='seed_changes'}))
    with (run.OUTPUT/'aggregate'/f'paired_changes_{suffix}.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=flat[0].keys(),lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
    print('AUDIT COMPLETE',suffix,json.dumps(origin_counts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    export(parser.parse_args().preflight)
