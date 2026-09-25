#!/usr/bin/env python3
"""Audit saved raw pilot results and export paired metrics without plotting."""
import argparse
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiment.mts_hierarchical_tracking import DEFAULT, RATES, VARIANTS, REFERENCE
from experiment.revision_pipeline import read,digest,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args(); root=args.root
    protocol=read(root/'protocol.json'); sha=digest(root/'protocol.json')
    for name,checksum in protocol['code'].items(): assert digest(ROOT/name)==checksum,name
    assert digest(protocol['cache'])==protocol['cache_sha256']
    rows=[]
    for rate in RATES:
        for variant in VARIANTS:
            name=f'{variant}_rate{rate}_seed1'
            row=read(root/'runs'/f'{name}.json')
            assert row['frames']==300 and row['seed']==1 and row['protocol_sha256']==sha
            reference=read(REFERENCE/'runs'/f'mts_report_rate{rate}_seed1.json')
            assert row['traffic_sha256']==reference['traffic_sha256']
            rawpath=root/'raw'/f'{name}.npz'; dp=root/'diagnostics'/f'{name}.json'
            assert digest(rawpath)==row['raw_sha256'] and digest(dp)==row['diagnostics_sha256']
            with np.load(rawpath) as z:
                q,f,rb=z['queue_bits'],z['queue_frame'],z['rb_per_bs']
                assert np.isfinite(q).all() and np.all(q>=0)
                assert np.all((rb>=0)&(rb<=np.array([133,66,66,66,66])+1e-8))
                np.testing.assert_allclose(rb@np.array([1,.2,.2,.2,.2])*.1,z['energy_j'],rtol=1e-10,atol=1e-8)
                u=np.array([np.mean(q[f==i]>rate*1e6*.02) for i in range(300)])
                np.testing.assert_allclose(u,z['violation_probability'],atol=1e-12)
                delay=q[f>=2]/(rate*1e6)*1000; count=z['association_counts'][2:]
                actual=dict(power_w=z['energy_j'][2:].mean()/.1,violation_percent=100*u[2:].mean(),
                    mean_proxy_ms=delay.mean(),p90_proxy_ms=np.percentile(delay,90),p99_proxy_ms=np.percentile(delay,99),
                    macro_association_percent=100*count[:,0].sum()/count.sum(),
                    pilots_per_vehicle_slot=z['pilots'][2:].mean())
                for k,v in actual.items(): np.testing.assert_allclose(v,row['metrics'][k],rtol=1e-10,atol=1e-8)
            d=read(dp)
            assert len(d['service'])==len(d['beam'])==300
            assert all(s['slots']==100 for s in d['service'])
            if variant=='report_hold':
                for k,v in reference['metrics'].items():
                    np.testing.assert_allclose(row['metrics'][k],v,rtol=1e-10,atol=1e-8)
            flat=dict(rate_mbps=rate,variant=variant,**row['metrics'])
            flat['elapsed_s']=row['elapsed_s']
            flat['micro_probe_overhead_percent']=100*row['metrics']['micro_probes_per_active_slot']*2/112
            rows.append(flat)
        meet=read(REFERENCE/'runs'/f'meet_cobra_rate{rate}_seed1.json')
        assert meet['traffic_sha256']==reference['traffic_sha256']
        rows.append(dict(rate_mbps=rate,variant='meet_cobra_reference',**meet['metrics']))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    atomic_json(root/'audit.json',dict(passed=True,cases=18,protocol_sha256=sha,
        control_reproduces_formal_results=True,raw_metrics_recomputed=True,
        description='Directional service feeds all queues; gain/beam selection causal tests passed separately.',rows=rows))
    lines=['# MTS hierarchical acquisition and five-point tracking pilot','',
        '## Protocol','',
        '- Arrival rates: 1, 5, 15, 21, 25, 29 Mbps; traffic seed 1.',
        '- Same 800--830 s trace, prediction cache and GPU fading as the formal grid. The first two service frames are omitted from aggregate metrics.',
        '- Control: five predicted candidates per frame, then fixed beam with one tracking observation per slot.',
        '- H32-hold: sixteen wide and sixteen fine probes in the first available slot of each frame; one current-beam observation in later slots.',
        '- H32-cross5: the same initialization, followed by current, Tx-left, Tx-right, Rx-left and Rx-right probes in each later slot. Neighbourhoods wrap around the DFT grid.',
        '- HO interruption: 10 ms. No search during interruption; acquisition starts at the first available slot. No extra five-probe search is added to the acquisition slot.',
        '- GS preferences, association period (five frames), prediction reports and OTR-RA are retained. Candidate-link RB demand estimates use the new nominal frame-average BF overhead. For the two new arms, bounded occupancy estimation also includes that overhead. The formal control is reproduced unchanged.',
        '- Expected candidate gains remain predicted optimal gains, not free measurements of candidate BSs. Only paid serving-link probes affect actual beam choice.',
        '- Actual per-slot selected transmit and receive beams and explicit RB assignment determine the interference and service entering every queue update.',
        '- No actor training or O-MAPPO changes are included in this pilot. Formal manuscript figures are untouched.','',
        '## Results','',
        '| Mbps | Variant | Power (W) | U (%) | L99 (ms) | Macro association (%) | Probes per active micro-link slot |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        probes=f"{r['micro_probes_per_active_slot']:.4f}" if 'micro_probes_per_active_slot' in r else '—'
        lines.append(f"| {r['rate_mbps']} | {r['variant']} | {r['power_w']:.4f} | {r['violation_percent']:.5f} | {r['p99_proxy_ms']:.4f} | {r['macro_association_percent']:.3f} | {probes} |")
    lines += ['','## Audit and interpretation limits','',
        'All 18 cases passed raw power, queue, violation-probability, latency-proxy and RB-capacity checks. The six control cases reproduce the saved formal results within numerical tolerance. All traffic hashes match the formal paired cases.',
        '',
        'These are single-seed, closed-loop comparisons: different BF outcomes may alter queues and later associations. Mean serving gain across a variant is descriptive, not a fixed-association gain comparison. U and L99 retain the manuscript\'s queue-based latency-proxy definitions. No multi-seed or scenario-level robustness conclusion follows from this pilot.',
        '',
        'Per-link probing overhead excludes common reporting/prediction observations. For a full, uninterrupted frame the counts are 104, 131 and 527, respectively. PET-BF uses at most 500 probes per such frame and may stop early.',
        '']
    report_path=root/'report.md'
    retained=''
    if report_path.exists():
        previous=report_path.read_text()
        if '\n## Interpretation\n' in previous:
            retained='\n## Interpretation\n'+previous.split('\n## Interpretation\n',1)[1]
    report_path.write_text('\n'.join(lines)+retained)
    print('\n'.join(lines[18:]))


if __name__=='__main__': main()
