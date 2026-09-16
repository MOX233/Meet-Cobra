#!/usr/bin/env python3
"""Independently validate saved GAP-refinement runs and summarize diagnostics."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    root = options.output
    protocol = json.loads((root/'protocol.json').read_text())
    repository = Path(__file__).resolve().parents[1]
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((repository/name).read_bytes()).hexdigest() == digest, name
    records = [json.loads(p.read_text()) for p in sorted((root/'runs').glob('*.json'))]
    expected = len(protocol['rates_mbps'])*len(protocol['seeds'])*len(protocol['modes'])
    assert len(records) == expected, (len(records), expected)
    hashes = collections.defaultdict(set)
    grouped = collections.defaultdict(list)
    early_stop_fractions = collections.defaultdict(list)
    args = protocol['args']
    powers = np.array([args['p_macro']]+[args['p_micro']]*4)
    caps = np.array([args['num_RB_macro']]+[args['num_RB_micro']]*4)
    dt = args['slot_len']*args['slots_per_frame']
    warmup = protocol['warmup_frames']
    traces_checked = 0
    for record in records:
        rate, seed, mode = (record[k] for k in ('rate_mbps','seed','mode'))
        name = f'rate{rate:g}_seed{seed}_{mode}'
        with np.load(root/'raw'/f'{name}.npz') as saved:
            data = {k:saved[k] for k in saved.files}
        for key,value in data.items():
            if np.issubdtype(value.dtype,np.number):
                assert np.isfinite(value).all(), (name,key)
        assert (data['queue_bits']>=0).all()
        assert (data['rb_per_bs']>=0).all()
        assert (data['rb_per_bs']<=caps+1e-9).all()
        np.testing.assert_allclose(data['energy_j'], data['rb_per_bs']@powers*dt, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(data['blocked_vehicle_slots'],
                                     data['handover_count']*protocol['ho_ms']/(args['slot_len']*1000))
        per_frame = [np.mean(data['queue_bits'][data['queue_frame_index']==i]>
                            rate*1e6*args['lat_slot_ub']*args['slot_len'])
                     for i in range(protocol['frame_count'])]
        np.testing.assert_allclose(per_frame,data['violation_probability'],rtol=0,atol=0)
        np.testing.assert_allclose(np.mean(per_frame[warmup:])*100,
                                   record['metrics']['violation_percent'],rtol=0,atol=0)
        np.testing.assert_allclose(np.mean(data['energy_j'][warmup:])/dt,
                                   record['metrics']['power_w'],rtol=0,atol=0)
        trace = json.loads((root/'diagnostics'/f'{name}.json').read_text())['refinement']
        assert len(trace)==protocol['frame_count']
        if mode != 'legacy':
            early_stop_fractions[(rate,mode)].append(100*np.mean([
                d['stop_reason']=='tolerance' and d['iterations']<record['config']['max_iterations']
                for d in trace[warmup:]]))
        for decision in trace:
            if mode=='legacy':
                assert decision['iterations']==2
                continue
            cfg = record['config']
            steps = decision['traces']
            assert 1<=len(steps)<=cfg['max_iterations']
            assert len(steps)==decision['iterations']
            for i,step in enumerate(steps):
                np.testing.assert_allclose(step['residual_rb'],np.max(np.abs(
                    np.array(step['implied_load'])-step['input_load'])),rtol=0,atol=0)
                if i:
                    assert step['input_load']==steps[i-1]['implied_load']
            eps=cfg['tolerance_rb']
            if decision['stop_reason']=='tolerance':
                assert eps is not None and steps[-1]['residual_rb']<=eps
            else:
                assert decision['stop_reason']=='iteration_limit'
                assert len(steps)==cfg['max_iterations']
                assert eps is None or steps[-1]['residual_rb']>eps
            assert eps is None or all(s['residual_rb']>eps for s in steps[:-1])
            traces_checked+=1
        hashes[(rate,seed)].add(record['traffic_sha256'])
        grouped[(rate,mode)].append(record)
    assert all(len(values)==1 for values in hashes.values())
    summaries=[]
    by_key={(r['rate_mbps'],r['seed'],r['mode']):r for r in records}
    for (rate,mode),group in sorted(grouped.items()):
        row=dict(rate_mbps=rate,mode=mode,n=len(group))
        if early_stop_fractions[(rate,mode)]:
            row['early_stop_before_cap_percent_mean']=float(np.mean(early_stop_fractions[(rate,mode)]))
        for metric in group[0]['metrics']:
            values=np.array([r['metrics'][metric] for r in group])
            row[metric+'_mean']=float(values.mean())
            if 'legacy' in protocol['modes'] and mode!='legacy':
                if metric not in by_key[(rate,group[0]['seed'],'legacy')]['metrics']:
                    continue
                differences=np.array([r['metrics'][metric]-by_key[(rate,r['seed'],'legacy')]['metrics'][metric] for r in group])
                row[metric+'_delta_mean']=float(differences.mean())
                row[metric+'_delta_ci95_half']=(float(student_t.ppf(.975,len(group)-1)*
                    differences.std(ddof=1)/np.sqrt(len(group))) if len(group)>1 else None)
        summaries.append(row)
        print(rate,mode, 'P=',round(row['power_w_mean'],5),
              'U=',round(row['violation_percent_mean'],6),
              'iterations=',round(row['gap_iterations_mean_mean'],3),
              'median_ms=',round(row['ho_cpu_median_ms_mean'],3),
              'stop_pct=',round(row['tolerance_stop_percent_mean'],2))
    output=dict(validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                validation=dict(runs=len(records),source_hashes='matched',
                               traffic_pairing='matched',physical_rb_caps='passed',
                               energy_from_rbs='passed',queue_violation_recomputation='exact',
                               blocked_slots_from_actual_handovers='passed',
                               iteration_and_stopping_traces_checked=traces_checked),
                summaries=summaries)
    (root/'validated_summary.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    fields=sorted(set().union(*(r.keys() for r in summaries)))
    with (root/'validated_summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print('VALIDATED',output['validation'])


if __name__=='__main__':
    main()
