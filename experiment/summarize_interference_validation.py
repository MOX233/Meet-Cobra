#!/usr/bin/env python3
"""Read-only result audit followed by separately saved R2C1 summary tables."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.interference_validation import OUTPUT, MODES, digest, write_json


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),min=float(values.min()),max=float(values.max()),
        sd=float(values.std(ddof=1)) if len(values)>1 else None,values=values.tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-raw',action='store_true')
    args=parser.parse_args()
    protocol_sha=digest(OUTPUT/'protocol.json')
    rows=[]
    for path in sorted((OUTPUT/'runs').glob('*.json')):
        row=json.loads(path.read_text())
        assert row['protocol_sha256']==protocol_sha
        if args.check_raw:
            for directory,field,ext in [('raw','raw_sha256','.npz'),('samples','samples_sha256','.npz'),
                                        ('diagnostics','diagnostics_sha256','.json')]:
                assert digest(OUTPUT/directory/(path.stem+ext))==row[field],path
        if row['mode']=='legacy':
            assert row['control_match']['all_arrays_exact']
        rows.append(row)
    groups=[]
    for mode in MODES:
        for rate in sorted({row['rate_mbps'] for row in rows if row['mode']==mode}):
            selected=sorted((r for r in rows if r['mode']==mode and r['rate_mbps']==rate),key=lambda r:r['seed'])
            group=dict(mode=mode,rate_mbps=rate,seeds=[r['seed'] for r in selected],
                metrics={key:stats([r['metrics'][key] for r in selected]) for key in selected[0]['metrics']},
                replay={variant:{key:stats([r['diagnostics']['modes'][variant][key] for r in selected])
                    for key in selected[0]['diagnostics']['modes'][variant]} for variant in MODES},
                comparisons={comparison:{key:stats([r['diagnostics']['comparisons'][comparison][key] for r in selected])
                    for key in selected[0]['diagnostics']['comparisons'][comparison]}
                    for comparison in selected[0]['diagnostics']['comparisons']})
            overlap=np.sum([r['diagnostics']['overlap_expected_realized'] for r in selected],axis=0)
            e,a=overlap[:,:,0].sum(),overlap[:,:,1].sum()
            group['overlap']=dict(expected=float(e),realized=float(a),relative_error=float(a/e-1))
            groups.append(group)
    comparisons=[]
    lookup={(r['mode'],r['rate_mbps'],r['seed']):r for r in rows}
    for before,after in [('legacy','average_expected'),('legacy','directional_explicit'),
                         ('average_expected','directional_explicit')]:
        for rate in sorted({r['rate_mbps'] for r in rows}):
            seeds=sorted({r['seed'] for r in rows if r['rate_mbps']==rate
                          and (before,rate,r['seed']) in lookup and (after,rate,r['seed']) in lookup})
            if not seeds: continue
            paired={key:stats([lookup[after,rate,s]['metrics'][key]-lookup[before,rate,s]['metrics'][key]
                               for s in seeds]) for key in lookup[before,rate,seeds[0]]['metrics']}
            comparisons.append(dict(before=before,after=after,rate_mbps=rate,seeds=seeds,after_minus_before=paired))
    plan=json.loads((OUTPUT/'run_plan.json').read_text())
    expected={('legacy',rate,1) for rate in plan['fixed_schedule_additional_rates']}
    expected|={(mode,rate,seed) for mode in plan['paired_modes'] for rate in plan['paired_closed_loop_rates']
               for seed in plan['paired_closed_loop_seeds']}
    actual=set(lookup)
    assert actual<=expected
    missing=sorted(expected-actual)
    summary=dict(protocol_sha256=protocol_sha,completed_cases=len(rows),expected_cases=len(expected),
                 missing_cases=missing,groups=groups,paired_changes=comparisons,
                 statistics='Per-seed summaries followed by mean/min/max; no vehicle-slot IID confidence intervals.')
    write_json(OUTPUT/'aggregate/summary.json',summary)
    out=OUTPUT/'aggregate/curves.csv'
    with out.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['mode','rate_mbps','seeds','power_w','violation_percent','p99_proxy_ms'])
        for row in groups:
            writer.writerow([row['mode'],row['rate_mbps'],';'.join(map(str,row['seeds'])),
                *[row['metrics'][m]['mean'] for m in ('power_w','violation_percent','p99_proxy_ms')]])
    text=['# R2C1 interference validation: generated result tables','',
          f'Completed cases: {len(rows)}/{len(expected)}. Protocol SHA256: `{protocol_sha}`.','',
          'All legacy cases exactly reproduce every saved array of the capped-GAP full-grid MEET-COBRA run.',
          'The tables average per-seed statistics. U is the existing queue-length-based latency violation metric.',
          'Only micro users with allocated RBs enter the matched-schedule physical error statistics.', '',
          '## Fixed legacy schedules: manuscript surrogate vs explicit directional interference','',
          '| Mbps | Seeds | Mean-I ratio | Median absolute aggregate-SINR error (dB) | P95 absolute error (dB) | Potential-service ratio | RB-overlap mean error (%) |',
          '|---:|---|---:|---:|---:|---:|---:|']
    for row in groups:
        if row['mode']!='legacy': continue
        m=row['replay']['average_expected']
        text.append(f"| {row['rate_mbps']} | {row['seeds']} | {m['interference_sum_ratio']['mean']:.4f} | "
            f"{m['sinr_of_mean_interference_abs_error_db_median']['mean']:.3f} | "
            f"{m['sinr_of_mean_interference_abs_error_db_p95']['mean']:.3f} | "
            f"{m['potential_service_ratio']['mean']:.4f} | {100*row['overlap']['relative_error']:.4f} |")
    text+=['','## Separating the approximations on the same schedules','',
        '| Mbps | Comparison | Mean-I ratio | Median abs. aggregate-SINR error (dB) | P95 abs. error (dB) | Potential-service ratio |',
        '|---:|---|---:|---:|---:|---:|']
    for row in groups:
        if row['mode']!='legacy': continue
        for name in ('gain_only','overlap_only_directional','overlap_only_average','legacy_to_frame_mean','frame_to_slot_mean'):
            m=row['comparisons'][name]
            text.append(f"| {row['rate_mbps']} | {name} | {m['interference_sum_ratio']['mean']:.4f} | "
                f"{m['sinr_abs_error_median_db']['mean']:.3f} | {m['sinr_abs_error_p95_db']['mean']:.3f} | "
                f"{m['potential_service_ratio']['mean']:.4f} |")
    text+=['','## Closed-loop system metrics (optimizer inputs unchanged)','',
        '| Mbps | Service evaluator | Seeds | Power (W) | U (%) | P99 proxy (ms) |',
        '|---:|---|---|---:|---:|---:|']
    for rate in sorted({r['rate_mbps'] for r in groups}):
        for row in groups:
            if row['rate_mbps']!=rate: continue
            m=row['metrics']
            text.append(f"| {rate} | {row['mode']} | {row['seeds']} | {m['power_w']['mean']:.4f} | "
                f"{m['violation_percent']['mean']:.4f} | {m['p99_proxy_ms']['mean']:.4f} |")
    text+=['','## Interpretation boundaries','',
        '- Mean-I ratios are RB-weighted ratios of total interference; SINR errors first average interference over the user\'s allocated RBs. They are not per-RB SINR quantiles.',
        '- Explicit service sums each RB\'s Shannon service with the original pilot efficiency, preserving nonlinear effects of interference variability.',
        '- Gain-only holds expected overlaps fixed; overlap-only holds directional or average gains fixed.',
        '- Closed-loop variants change physical micro-tier queue service, not information supplied to the predictors or optimizers. They diagnose robustness, not a fully harmonized new system implementation.',
        '- An unchanged fixed schedule necessarily has unchanged transmit power; only separately run closed-loop variants support power/U comparisons.',
        '- Original simulator uses a frame-level maximum-element interference gain. The manuscript surrogate uses a slot-level Frobenius mean. The frame-average bridge separates the aggregation and temporal aspects.',
        '- Existing Sionna RT channels, slot fading, beam report timing, independent data streams and frequency-flat links are retained. No new ray tracing or packet-level/waveform simulation is claimed.',
        '- All seeds share the same map, vehicle trajectory collection and model weights. These results do not establish cross-deployment robustness.','']
    (OUTPUT/'aggregate/results.md').write_text('\n'.join(text))
    print('\n'.join(text),flush=True)


if __name__=='__main__': main()
