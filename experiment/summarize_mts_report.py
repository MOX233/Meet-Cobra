#!/usr/bin/env python3
"""Audit and combine both report-MTS variants with the frozen references."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment import mts_report_experiment as base
from experiment import mts_report_bounded_experiment as bounded


def main():
    _,original_sha=base.validate()
    folders=[('report_unbounded',base.OUTPUT),('report_bounded',bounded.OUTPUT)]
    bounded.activate()
    _,bounded_sha=base.validate()
    rows,audits=[],[]
    for rate in base.RATES:
        paired_hash=None
        for name,path in [('meet_cobra',base.grid.OUTPUT),('mts',base.grid.OUTPUT),
                          ('oracle_mc',base.grid.OUTPUT),*folders]:
            method='mts_report' if name.startswith('report_') else name
            key=f'{method}_rate{rate}_seed1'
            row=json.loads((path/'runs'/f'{key}.json').read_text())
            raw=path/'raw'/f'{key}.npz'
            assert base.grid.digest(raw)==row['raw_sha256']
            paired_hash=paired_hash or row['traffic_sha256']
            assert row['traffic_sha256']==paired_hash
            if name.startswith('report_'):
                assert row['protocol_sha256']==(bounded_sha if name=='report_bounded' else original_sha)
            data=np.load(raw)
            proxy=data['queue_bits'][data['queue_frame']>=2]/(rate*1e6)*1000
            metrics=row['metrics']
            np.testing.assert_allclose(metrics['power_w'],data['energy_j'][2:].mean()/.1,atol=1e-10)
            np.testing.assert_allclose(metrics['violation_percent'],data['violation_probability'][2:].mean()*100,atol=1e-10)
            np.testing.assert_allclose(metrics['p99_proxy_ms'],np.percentile(proxy,99),atol=1e-10)
            rows.append(dict(rate_mbps=rate,seed=1,method=name,**metrics))
            audit=dict(rate_mbps=rate,method=name,raw_sha256=row['raw_sha256'],
                traffic_sha256=paired_hash,
                mean_rb_per_bs=data['rb_per_bs'][2:].mean(0).tolist(),
                max_proxy_ms=float(proxy.max()),
                pooled_proxy_above_100ms_percent=float(np.mean(proxy>100)*100),
                mean_frame_proxy_ms=metrics['mean_proxy_ms'])
            if name.startswith('report_'):
                diagpath=path/'diagnostics'/f'{key}.json'
                assert base.grid.digest(diagpath)==row['diagnostics_sha256']
                diag=json.loads(diagpath.read_text())
                estimate=np.array([d['estimated_rb'] for d in diag])
                caps=np.array([133,66,66,66,66])
                if name=='report_bounded':
                    assert np.isfinite(estimate).all() and (estimate<=caps).all() and (estimate>=0).all()
                for d in diag:
                    assert abs(d['frame']-d['source_frame']-.1)<1e-8
                    assert d['total_probe_count']==5*sum(int(bs)>0 for bs in d['association'].values())
                audit.update(row['diagnostics'])
                audit.update(max_estimated_rb=float(estimate.max()),
                    estimated_rb_above_capacity_frames=int((estimate>caps).any(1).sum()),
                    matching_overflow_epochs=int((data['optimizer_overflow_record']>1e-8).sum()))
            audits.append(audit)
    output=bounded.OUTPUT
    base.grid.write_json(output/'complete_comparison.json',dict(rows=rows))
    base.grid.write_json(output/'audit.json',dict(
        tests='17 unique CPU/GPU tests; detailed commands in experiment report',
        report_protocol_sha256=original_sha,bounded_protocol_sha256=bounded_sha,cases=audits))
    with (output/'complete_comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['rate_mbps'],row['method'],
              ' '.join(f'{row[k]:.5f}' for k in ['power_w','violation_percent','p99_proxy_ms',
                  'macro_association_percent','handovers_per_vehicle_s','pilots_per_vehicle_slot']))
    print('AUDIT OK',len(rows),'paired comparison rows')


if __name__=='__main__':
    main()
