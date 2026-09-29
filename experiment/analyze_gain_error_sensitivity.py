#!/usr/bin/env python3
"""Read-only audit and study figures; never change manuscript or formal figures."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import sys
import time
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiment import gain_error_sensitivity as study
from experiment import revision_pipeline as pipeline
from experiment.revision_training import atomic_json


def evaluated_prediction_errors(root, protocol):
    """Match the system metric window, excluding two initial service frames."""
    from utils.directional_service import beam_average_gain_db
    with Path(protocol['cache']).open('rb') as f:
        timeline = pickle.load(f)
    frames = sorted(timeline)
    following = dict(zip(frames[:-1], frames[1:]))
    evaluated_targets = set(frames[1+protocol['warmup_frames']:])
    rows = study.report_rows(timeline)
    selected, errors = [], {kind: [] for kind in study.KINDS}
    for i, (frame, v) in enumerate(rows):
        target = following.get(frame)
        if target not in evaluated_targets or v not in timeline[target]:
            continue
        selected.append(i)
        prediction = timeline[frame][v]['shared_prediction']
        truth = timeline[target][v]
        errors['desired'].append(prediction['gain']-np.asarray(truth['g_opt_beam'], dtype=np.float64))
        errors['interfering'].append(prediction['interference']-beam_average_gain_db(truth['h']))
    selected = np.asarray(selected)
    errors = {kind: np.asarray(values) for kind, values in errors.items()}
    output = []
    for kind in study.KINDS:
        for seed in protocol['seeds']:
            z = study.standard_noise(len(rows), seed, kind)[selected]
            for sigma in protocol['sigmas_db']:
                e = errors[kind] + sigma*z
                output.append(dict(kind=kind, seed=seed, sigma_db=sigma,
                    mae_db=float(np.abs(e).mean()), rmse_db=float(np.sqrt(np.square(e).mean())),
                    bias_db=float(e.mean()), injected_mean_db=float(sigma*z.mean()),
                    injected_std_db=float(sigma*z.std())))
    answer = dict(vehicle_frame_samples=len(selected), link_samples=len(selected)*4,
        first_target_frame=min(evaluated_targets), last_target_frame=max(evaluated_targets),
        source='Frozen formal cache; unchanged original numerical gain convention.',
        scope='All available predicted links in the system evaluation window, excluding two initial service frames and newly appearing vehicles without prior reports. Not restricted to actually selected BSs.',
        original_mae_db={kind: float(np.abs(e).mean()) for kind, e in errors.items()}, rows=output)
    atomic_json(root/'prediction_statistics_eval_window.json', answer)
    return answer


def audit_results(root, protocol, rows):
    """Recompute requested metrics from queues/RBs, independently of extractor."""
    checks = []
    traffic = {}
    powers = np.array([1., .2, .2, .2, .2])
    caps = np.array([133, 66, 66, 66, 66])
    for row in rows:
        name = study.key('desired' if row['kind']=='control' else row['kind'],
                         row['sigma_db'], row['rate_mbps'], row['seed'])
        pair = (row['rate_mbps'], row['seed'])
        assert traffic.setdefault(pair, row['traffic_sha256']) == row['traffic_sha256']
        reference = protocol['references'][pipeline.key('meet_cobra', *pair)]
        assert row['traffic_sha256'] == reference['traffic_sha256']
        with np.load(root/'raw'/(name+'.npz'), allow_pickle=False) as raw:
            q = raw['queue_bits']; frame = raw['queue_frame']
            assert q.shape[1]==100 and np.isfinite(q).all() and (q>=0).all()
            assert raw['serving_bs'].shape == frame.shape
            assert (raw['serving_bs']>=0).all() and (raw['serving_bs']<=4).all()
            assert np.isfinite(raw['rb_per_bs']).all() and (raw['rb_per_bs']>=0).all()
            assert (raw['rb_per_bs'] <= caps+1e-8).all()
            assert raw['rb_per_bs'].shape == (300, 5)
            rate = row['rate_mbps'] * 1e6
            power = float((raw['rb_per_bs'][2:] @ powers).mean())
            violation = float(100*np.mean([np.mean(q[frame==i] > rate*.020) for i in range(2,300)]))
            p99 = float(np.percentile(q[frame>=2]/rate*1000,99))
            np.testing.assert_allclose(raw['energy_j'], raw['rb_per_bs']@powers*.1, rtol=1e-10, atol=1e-8)
        actual = dict(power_w=power, violation_percent=violation, p99_proxy_ms=p99)
        for metric, value in actual.items():
            np.testing.assert_allclose(value, row['metrics'][metric], rtol=1e-10, atol=1e-9)
        diagnostic = pipeline.read(root/'diagnostics'/(name+'.json'))
        assert len(diagnostic['service']) == 300
        assert all(d['slots']==100 for d in diagnostic['service'])
        assert all(d['iterations']==2 and d['cap_rb_usage'] for d in diagnostic['gap'])
        if row['sigma_db']==0:
            assert row['zero_noise_regression']
        checks.append(dict(case=name, recomputed_metrics=actual))
    atomic_json(root/'independent_audit.json', dict(checked=len(checks), expected=protocol['cases'],
        all_passed=True, pairing_checked=True, raw_metrics_checked=True,
        protocol_sha256=pipeline.digest(root/'protocol.json'),
        analysis_source_sha256=pipeline.digest(Path(__file__)), cases=checks))


def plot(root, aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif', 'font.size':10, 'axes.labelsize':10,
                         'legend.fontsize':9, 'pdf.fonttype':42})
    colors = {9:'#3178b5', 19:'#009c73', 29:'#de8f05', 35:'#c33b4d'}
    fig, axes = plt.subplots(3,2,figsize=(8.1,8.6),sharex=True,layout='constrained')
    fields = [('violation_percent','Violation probability $U$ (%)'),
              ('power_w','Average transmit power (W)'),('p99_proxy_ms','$L_{99}$ (ms)')]
    for column, kind in enumerate(study.KINDS):
        axes[0,column].set_title('Desired-link gain perturbation' if kind=='desired' else 'Interfering-link gain perturbation')
        for i, (metric, label) in enumerate(fields):
            ax=axes[i,column]
            for rate, color in colors.items():
                data=sorted([r for r in aggregate if r['kind']==kind and r['rate_mbps']==rate],key=lambda r:r['sigma_db'])
                if not data: continue
                x=[r['sigma_db'] for r in data]
                ax.plot(x,[r[metric+'_mean'] for r in data],'-o',color=color,ms=3.5,lw=1.3,label=f'{rate} Mbps')
                ax.fill_between(x,[r[metric+'_min'] for r in data],[r[metric+'_max'] for r in data],color=color,alpha=.13,lw=0)
            ax.grid(True,alpha=.25); ax.set_ylabel(label)
            if metric=='p99_proxy_ms':
                ax.set_yscale('log'); ax.axhline(20,color='.4',ls='--',lw=.9)
            if metric=='violation_percent': ax.set_yscale('symlog',linthresh=.01,linscale=.3)
            ax.set_xlim(0,10); ax.set_xticks(range(11))
        axes[2,column].set_xlabel('Additional gain-noise standard deviation (dB)')
        axes[0,column].legend(loc='best',framealpha=.9)
    fig.savefig(root/'gain_noise_curves.pdf',bbox_inches='tight')
    fig.savefig(root/'gain_noise_curves.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def report(root, summary, errors):
    lines=['# Gain-prediction error sensitivity experiment', '',
        f"Completed cases: **{summary['cases']}/{summary['expected']}**.", '',
        '## Fixed configuration', '',
        '- Four loads (9, 19, 29, 35 Mbps), three paired seeds, 30 s each, first two frames excluded from system metrics.',
        '- Unchanged formal checkpoints, beam candidate lists, actual channel matrices, algorithms and parameters; no retraining.',
        '- Zero-mean Gaussian noise added in dB to either desired-link or beam-averaged interfering-link reports, independently between reports and BS links.',
        '- The same report is reused consistently by all decision stages. Desired-gain noise affects GAP-HO and the PET-BF stopping threshold, not the directly measured desired gain used by OTR-RA. Interfering-gain noise affects GAP-HO and OTR-RA.',
        '- Actual service is always computed from selected directional beams and explicit RB assignments; queues evolve in closed loop.',
        '- Twelve zero-noise cases are common controls for both branches. Each must reproduce every original saved raw array exactly.',
        '- Curves show three-seed means with min–max bands, not confidence intervals. Randomness across seeds does not change the fixed mobility trace.', '',
        '## Prediction-error statistics', '',
        'The x-axis is added noise standard deviation, NOT final prediction MAE. The initial preparation statistics include warmup reports; the following statistics use the same evaluation window as the system metrics.', '',
        f"Vehicle-frame samples with valid target labels: {errors['vehicle_frame_samples']:,}.", '',
        '| Added noise SD (dB) | Desired-link MAE (dB) | Interfering-link MAE (dB) |',
        '|---:|---:|---:|']
    for sigma in range(11):
        mae=[np.mean([r['mae_db'] for r in errors['rows'] if r['kind']==kind and r['sigma_db']==sigma]) for kind in study.KINDS]
        lines.append(f'| {sigma} | {mae[0]:.4f} | {mae[1]:.4f} |')
    for kind in study.KINDS:
        lines += ['',f'## {kind.capitalize()}-gain perturbations','',
            '| Load (Mbps) | Noise SD (dB) | Power (W) | U (%) | L99 (ms) |', '|---:|---:|---:|---:|---:|']
        for row in summary['aggregate']:
            if row['kind']!=kind or row['sigma_db'] not in (0,1,3,5,10): continue
            lines.append(f"| {row['rate_mbps']} | {row['sigma_db']} | {row['power_w_mean']:.3f} | {row['violation_percent_mean']:.4f} | {row['p99_proxy_ms_mean']:.3f} |")
    lines += ['', '## Interpretation boundaries', '',
        '- Injected errors are zero-mean in dB; this is not equivalent to unbiased linear-power gains, nor does it reproduce every temporal correlation of real model errors.',
        '- This is additional-noise sensitivity around the trained model, not a sweep from perfect prediction. Zero noise retains the actual NN errors.',
        '- Candidate indices are fixed, but the finally selected beams and probe counts can change through gain-dependent stopping and changed associations.',
        '- Increased errors need not produce monotonic power or U for every seed because associations, probing and integer scheduling are coupled.',
        '- No paper text, response text, formal figures, model files or historical results were modified by this experiment.', '',
        'Detailed numeric outputs: `curves.csv`, `summary.json`, `prediction_statistics_eval_window.json`, and `independent_audit.json`.', '']
    (root/'experiment_report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=study.OUTPUT)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--wait',action='store_true',help='Wait for the detached simulation queue, then analyze all results')
    args=parser.parse_args()
    if args.wait:
        while True:
            status_path=args.root/'queue_status.json'
            status=pipeline.read(status_path) if status_path.exists() else {}
            if status.get('state')=='failed':
                raise RuntimeError('Simulation queue failed; inspect queue_status.json and case logs')
            if status.get('state')=='complete':
                break
            time.sleep(15)
    study.summarize(args.root,args.allow_partial)
    protocol,_=study.validate(args.root)
    summary=pipeline.read(args.root/'summary.json')
    errors=evaluated_prediction_errors(args.root,protocol)
    if args.audit: audit_results(args.root,protocol,summary['per_seed'])
    if summary['aggregate']: plot(args.root,summary['aggregate'])
    report(args.root,summary,errors)
    print(json.dumps(dict(cases=summary['cases'],complete=summary['complete'],
        window_aligned_mae=errors['original_mae_db'],report=str(args.root/'experiment_report.md')),indent=2))


if __name__=='__main__': main()
