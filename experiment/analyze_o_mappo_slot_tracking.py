#!/usr/bin/env python3
"""Export the paired pilot comparison; never touch paper figures or text."""
import argparse
import csv
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/omappo-cross5-matplotlib')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment.o_mappo_slot_tracking import DEFAULT,read,verify,formal
from experiment.revision_pipeline import digest,atomic_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    root=parser.parse_args().root
    protocol=verify(root); summary=read(root/'summary.json')
    assert summary['passed'] and summary['control_raw_parity']
    assert summary['protocol_sha256']==digest(root/'protocol.json')
    refs={p['file']:p['sha256'] for p in read(formal.DEFAULT/'summary.json')['provenance']}
    evidence=[]
    for group in summary['rows']:
        new=Path(group['file']); assert digest(new)==group['sha256']
        old=formal.DEFAULT/'runs'/f'E_all_trained_rate{group["rate_mbps"]}_seed1.json'
        assert digest(old)==refs[str(old)]
        evidence.append(dict(new_file=str(new),new_sha256=digest(new),
                             original_file=str(old),original_sha256=digest(old)))
    metrics=['power_w','violation_percent','p90_proxy_ms','p99_proxy_ms',
             'macro_association_percent','pilots_per_vehicle_slot']
    methods=['current','slot_cross5','meet']
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['rate_mbps','method',*metrics])
        for r in summary['rows']:
            for method in methods:
                writer.writerow([r['rate_mbps'],method,*[r[method][m] for m in metrics]])
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    labels=['Current O-MAPPO','O-MAPPO HO32 + slot cross5','MEET-COBRA']
    styles=[('#7251a5','D','--'),('#008b80','s','-'),('#d42a22','o','-')]
    rates=summary['comparison_loads']
    for ax,metric,ylabel in zip(axes,['power_w','violation_percent'],['Average transmit power (W)','Violation probability U (%)']):
        for method,label,(color,marker,style) in zip(methods,labels,styles):
            ax.plot(rates,[r[method][metric] for r in summary['rows']],label=label,
                color=color,marker=marker,linestyle=style,linewidth=1.7,markersize=5,markerfacecolor='white')
        ax.set_xlabel('Arrival rate (Mbps)'); ax.set_ylabel(ylabel)
        ax.set_xticks(rates); ax.grid(True,alpha=.25)
        if metric=='violation_percent':
            ax.set_yscale('symlog',linthresh=.01)
            ax.set_ylim(0,1.5*max(r[m][metric] for r in summary['rows'] for m in methods))
        else: ax.set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=9)
    fig.subplots_adjust(top=.83,bottom=.17,left=.09,right=.98,wspace=.28)
    fig.savefig(root/'comparison.pdf'); fig.savefig(root/'comparison.png',dpi=180); plt.close(fig)
    lines=['# O-MAPPO HO32 + per-slot cross-five pilot','',
        '## Protocol','',
        '- Six arrival rates: 1, 5, 15, 21, 25, 29 Mbps; traffic/fading seed 1; 30 s; first two frames excluded.',
        '- Frozen approved two-layer E_all actor (training seed 11, round 20). No retraining, checkpoint search or load-dependent policy selection.',
        '- HO to a micro BS: 32 hierarchical probes in the first service slot after 10 ms interruption. Thereafter five measured pairs per slot: current, Tx-minus, Tx-plus, Rx-minus, Rx-plus. No extra current-beam pilot, no diagonal pairs, no additional event-driven nine-pair search.',
        '- Tracking runs even when no 10 m actor decision occurs, and carries the final beam into the next frame. Unlike the selected MTS variant, it does not restart hierarchical search every frame.',
        '- tracking_pilots changes from 1 to 5 in the existing cost formulas; actor dimensions/weights, target optimizer objective/candidate count, E_all interference/load rule and OTR-RA logic are unchanged. Realized states, triggers and associations may change through closed-loop feedback.',
        '- Actual slot-selected beams determine both desired and cross-link gains in the explicit-RB service evaluator. Final beam states are committed after each frame, not before actor decisions.',
        '- The current formal baseline retains ideal current-frame planning CSI, including its nominal post-HO planning beam and candidate-BS gain estimates. Physical service uses the slot-level search above. Extra CSI acquisition for planning remains uncharged, as in the formal baseline; this is not the equal-information/prediction-report actor variant.',
        '- The current O-MAPPO and MEET references use the same seed, cache, 300 frames, warmup, directional service and HO model. The current O-MAPPO at 15 Mbps was rerun and every saved raw array matched its formal reference exactly.',
        '- Twenty-five unit tests plus a 10-frame GPU smoke passed. Every simulated serving-link slot checked paid probe counts and delivered gains. All six full cases passed independent raw-metric recalculation.',
        '', '## Power and violation probability (single paired seed)','',
        '| Mbps | Current P (W) | New P (W) | MEET P (W) | Current U (%) | New U (%) | MEET U (%) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['rows']:
        values=[r[m][k] for k in ['power_w','violation_percent'] for m in methods]
        lines.append(f'| {r["rate_mbps"]} | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    lines+=['','## Tail and association diagnostics','',
        '| Mbps | Current L99 (ms) | New L99 (ms) | MEET L99 (ms) | Current macro (%) | New macro (%) | MEET macro (%) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['rows']:
        values=[r[m][k] for k in ['p99_proxy_ms','macro_association_percent'] for m in methods]
        lines.append(f'| {r["rate_mbps"]} | '+' | '.join(f'{v:.5f}' for v in values)+' |')
    improved=[r['rate_mbps'] for r in summary['rows'] if all(r['slot_cross5'][k]<r['current'][k] for k in ['power_w','violation_percent'])]
    lines+=['','## Interpretation boundaries','',
        f'- Loads with both lower mean power and lower U than the current O-MAPPO: {improved}.',
        '- Five probes every active slot increase the recurring pilot fraction from 2/112 to 10/112 (about 1.79% to 8.93%), excluding acquisition/event slots. The experiment measures the balance between better tracking and this extra cost.',
        '- This pilot evaluates deployment of the same frozen actor under a changed BF mechanism, not the best achievable policy after dedicated retraining.',
        '- One paired seed and six load points are preliminary evidence, not a multi-seed claim or proof of statistical significance.',
        '- U=0 means no violations were observed in the evaluated samples; it is not a guarantee of zero violation probability.',
        '- No paper text, response letter, formal figures, trained checkpoints or production defaults were modified.',
        '', '## Reproduction','', '```bash',
        '/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/o_mappo_slot_tracking.py run',
        '/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/o_mappo_slot_tracking.py audit',
        '/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/analyze_o_mappo_slot_tracking.py',
        '```','',
        'Code checkpoint: 7f88aa2. Original-code tag: pre-o-mappo-slot-cross5-20260925. Raw arrays and detailed diagnostics remain under runs/.','']
    report=root/'report.md'
    # Preserve any later, separately written discussion on regeneration.
    extra=''
    if report.exists() and '\n## Discussion\n' in report.read_text():
        extra='\n## Discussion\n'+report.read_text().split('\n## Discussion\n',1)[1]
    report.write_text('\n'.join(lines)+extra)
    atomic_json(root/'comparison_manifest.json',dict(summary_sha256=digest(root/'summary.json'),
        csv_sha256=digest(root/'comparison.csv'),protocol_sha256=digest(root/'protocol.json'),evidence=evidence))
    print('\n'.join(lines[14:]),flush=True)


if __name__=='__main__': main()
