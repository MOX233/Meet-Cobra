#!/usr/bin/env python3
"""Check the final figure replacement and export paired MTS comparisons."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiment.mts_h32_full_grid import DEFAULT, RATES, SEEDS
from experiment import plot_revision_system_results as plot


def main():
    root=DEFAULT; summary=plot.read(root/'summary.json')
    manifest=plot.read(root/'paper_figures/figure_manifest.json')
    assert manifest['mts_replacement']['summary_sha256']==plot.digest(root/'summary.json')
    assert manifest['mts_replacement']['variant']=='hier32_cross5'
    assert manifest['o_mappo_replacement']['root']==str(plot.OMAPPO_RESULTS)
    def data(path):
        with path.open() as f:
            return {(r['method'],r['rate_mbps'],r['metric']):r for r in csv.DictReader(f)}
    old=data(root/'pre_update_figures/figure_data.csv')
    new=data(root/'paper_figures/figure_data.csv')
    assert set(old)==set(new) and len(old)==720
    unchanged=[k for k in old if k[0]!='mts_report']
    assert len(unchanged)==630 and all(old[k]==new[k] for k in unchanged)
    replacement=[k for k in old if k[0]=='mts_report']
    assert len(replacement)==90
    # Each new MTS plotted value must equal the frozen, raw-audited summary.
    for group in summary['groups']:
        for metric in plot.METRICS:
            point=new['mts_report',str(group['rate_mbps']),metric]
            for k,f in [('mean','mean'),('seed_min','minimum'),('seed_max','maximum')]:
                np.testing.assert_allclose(float(point[k]),group['metrics'][metric][f],rtol=1e-12)
    proof=dict(total_figure_rows=720,unchanged_other_scheme_rows=630,replaced_mts_rows=90,
        original_figure_data_sha256=plot.digest(root/'pre_update_figures/figure_data.csv'),
        new_figure_data_sha256=plot.digest(root/'paper_figures/figure_data.csv'),
        raw_audit_summary_sha256=plot.digest(root/'summary.json'),
        paired_seeds=SEEDS,rates=RATES,all_other_schemes_unchanged=True)
    from experiment.revision_pipeline import atomic_json
    atomic_json(root/'figure_replacement_audit.json',proof)
    lines=['# MTS H32-cross5 full-load, multi-seed results','',
        '## Configuration and preservation','',
        '- All 18 arrival rates from 1 to 35 Mbps in steps of 2; paired seeds 1, 2, 3; 30 seconds per case; first two frames excluded.',
        '- 54 validated cases: six unchanged pilot cases reused with source hashes, 48 additional cases run.',
        '- Five-frame GS association and predicted channel gains retained. BF does not use the predicted beam-pair list or PET-BF.',
        '- Every frame: 32 hierarchical probes in the first available slot, then five axial-neighbour probes in each later available slot. No probes during the 10 ms HO interruption.',
        '- Matching demand and bounded occupancy estimates include the new average BF overhead. Common OTR-RA and explicit directional RB service are retained.',
        '- Raw power, U, latency-proxy quantiles, RB capacities, beam-probe accounting and paired traffic hashes were independently checked for all 54 cases.',
        '- Original MTS data and pilot results remain intact. Previous paper figures and CSV are saved in pre_update_figures. The seven other curves, including the approved E_all O-MAPPO, are unchanged (630/630 figure-data rows).',
        '- The experiments do not retrain or change O-MAPPO. Its separately discussed per-slot five-point tracking remains a future experiment.',
        '', '## Three-seed means','',
        '| Mbps | MEET P (W) | MTS P (W) | MEET U (%) | MTS U (%) | MEET L99 (ms) | MTS L99 (ms) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    def val(m,r,k): return float(new[m,str(r),k]['mean'])
    comparisons=[]
    for rate in RATES:
        values=[val(m,rate,k) for k in ['power_w','violation_percent','p99_proxy_ms'] for m in ['meet_cobra','mts_report']]
        lines.append(f'| {rate} | '+' | '.join(f'{v:.5f}' for v in values)+' |')
        comparisons.append(dict(rate_mbps=rate,meet={k:val('meet_cobra',rate,k) for k in plot.METRICS},
            mts={k:val('mts_report',rate,k) for k in plot.METRICS},
            legacy_mts={k:float(old['mts_report',str(rate),k]['mean']) for k in plot.METRICS}))
    atomic_json(root/'paper_comparison.json',dict(comparisons=comparisons,
        first_load_above_20ms=manifest['l99_first_load_above_20ms']))
    lower_power=[r for r in RATES if val('mts_report',r,'power_w')<val('meet_cobra',r,'power_w')]
    lower_u=[r for r in RATES if val('mts_report',r,'violation_percent')<val('meet_cobra',r,'violation_percent')]
    lines+=['','## Boundaries on interpretation','',
        f'- MTS has lower mean power than MEET at these evaluated loads: {lower_power}.',
        f'- MTS has lower mean U than MEET at these evaluated loads: {lower_u}.',
        f"- First evaluated load with mean L99 above 20 ms: {manifest['l99_first_load_above_20ms']}.",
        '- Means and min/max envelopes use the same three paired seeds; envelopes are not confidence intervals. Vehicle trajectories and deployment are fixed.',
        '- L90/L99 are computed within each seed from q/lambda samples and then averaged across seeds; U is the frame-averaged proxy violation probability.',
        '- Relative results describe this explicitly adapted baseline, not a complete reproduction of the original hybrid analog/digital beamformer.',
        '', '## Interpretation','',
        '- The new MTS beamformer removes the predicted beam-candidate list and PET-BF from beam selection, but retains predicted gains for association and the common OTR-RA. It should not be described as independent of all MEET-COBRA modules.',
        '- For a full micro-link frame without interruption, 32 acquisition probes plus 99 times 5 tracking probes give 527 probes, compared with 5 acquisition probes plus 99 single-pair measurements (104 probes) in the previous predicted-candidate implementation. With eta=2/112, their average modeled probing fractions are approximately 9.41% and 1.86%. The smaller remaining data-service fraction is a plausible contributor to the higher power and stronger high-load queue tails; the comparison does not isolate it from changes in selected beams and subsequent associations.',
        '- At 33 Mbps the new MTS has mean power 177.03 W, U=2.332%, and L99=267.54 ms, versus 160.99 W, 1.080%, and 33.54 ms for the previous MTS. This change is reported internally as a consequence of the selected, more independent beam-search design, not as a performance improvement over the previous MTS.',
        '- Low-load U improvements over MEET are small and do not establish statistical significance with only three seeds. Higher-load comparisons and figures retain every evaluated load and seed.',
        '', '## Reproduction','', '```bash',
        '/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/mts_h32_full_grid.py run',
        '/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/mts_h32_full_grid.py audit',
        '/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/plot_revision_system_results.py',
        '/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/report_mts_h32_full_grid.py',
        '```','',
        'The run command verifies and skips completed cases. To recreate the previous MTS curves, use --legacy-mts and separate --figures and --report output directories.','']
    (root/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[14:]),flush=True)


if __name__=='__main__': main()
