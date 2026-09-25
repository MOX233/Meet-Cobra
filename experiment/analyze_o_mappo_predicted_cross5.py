#!/usr/bin/env python3
"""Audit and summarize the completed experiment; never edits paper assets."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiment.revision_pipeline import atomic_json,digest
from experiment.train_o_mappo_predicted_cross5 import DEFAULT,RATES,SEEDS,TEST,sources,old

PAPER=ROOT/'experiment/results/revision_directional_20260922/grid'
FORMAL=ROOT/'experiment/results/o_mappo_eall_full_grid_20260925'
KEYS=['power_w','violation_percent','p90_proxy_ms','p99_proxy_ms','macro_association_percent','pilots_per_vehicle_slot']


def audit_row(path,raw,rate,seed):
    row=old.read(path)
    assert row['raw_sha256']==digest(raw)
    with np.load(raw) as z:
        q,f,rb=z['queue_bits'],z['queue_frame'],z['rb_per_bs']
        assert len(rb)==300 and np.isfinite(q).all() and (q>=0).all()
        assert np.all((rb>=0)&(rb<=np.array([133,66,66,66,66])+1e-8))
        np.testing.assert_allclose(rb@np.array([1,.2,.2,.2,.2])*.1,z['energy_j'],rtol=1e-10,atol=1e-8)
        u=np.array([np.mean(q[f==i]>rate*1e6*.02) for i in range(300)])
        np.testing.assert_allclose(u,z['violation_probability'],atol=1e-12)
        delay=q[f>=2]/(rate*1e6)*1000
        counts=z['association_counts'][2:]
        metrics=dict(power_w=float(z['energy_j'][2:].mean()/.1),violation_percent=float(100*u[2:].mean()),
            p90_proxy_ms=float(np.percentile(delay,90)),p99_proxy_ms=float(np.percentile(delay,99)),
            macro_association_percent=float(100*counts[:,0].sum()/counts.sum()),
            pilots_per_vehicle_slot=float(z['pilots'][2:].mean()),
            handovers_postwarm=int(z['handover_count'][2:].sum()),
            macro_power_w=float(rb[2:,0].mean()),micro_power_w=float(.2*rb[2:,1:].sum(1).mean()))
        for k in KEYS:np.testing.assert_allclose(metrics[k],row['metrics'][k],rtol=1e-10,atol=1e-9)
    return dict(rate=rate,seed=seed,metrics=metrics,traffic_sha256=row['traffic_sha256'],
                path=str(path),sha256=digest(path),raw_sha256=row['raw_sha256'])


def collect(root):
    assert (root/'complete.json').exists(),'Training/test pipeline is not complete'
    assert (root/'true_control/summary.json').exists(),'Paired true-CSI control grid is not complete'
    protocol=old.read(root/'protocol.json')
    for name,h in protocol['code'].items():assert digest(ROOT/name)==h,name
    assert digest(TEST)==protocol['test_sha256']
    selected=old.read(root/'selection.json')['selected']
    sha=digest(selected['policy'])
    rows=[]
    for rate in RATES:
        for seed in [1,2,3]:
            specs=[('Prediction-input O-MAPPO',root/'test/runs'/f'predicted_cross5_rate{rate}_seed{seed}.json',None),
                   ('True-CSI cross5 O-MAPPO',root/'true_control/runs'/f'true_cross5_rate{rate}_seed{seed}.json',None),
                   ('MEET-COBRA',PAPER/'runs'/f'meet_cobra_rate{rate}_seed{seed}.json',PAPER/'raw'/f'meet_cobra_rate{rate}_seed{seed}.npz'),
                   ('Oracle-MC',PAPER/'runs'/f'oracle_mc_rate{rate}_seed{seed}.json',PAPER/'raw'/f'oracle_mc_rate{rate}_seed{seed}.npz'),
                   ('Formal O-MAPPO',FORMAL/'runs'/f'E_all_trained_rate{rate}_seed{seed}.json',None)]
            paired=[]
            for label,path,raw in specs:
                r=audit_row(path,raw or path.with_suffix('.npz'),rate,seed)
                r['label']=label;rows.append(r);paired.append(r['traffic_sha256'])
                if label=='Prediction-input O-MAPPO':assert old.read(path)['policy_sha256']==sha
            assert len(set(paired))==1,(rate,seed,'Exogenous traffic differs')
    aggregates=[]
    for label in dict.fromkeys(r['label'] for r in rows):
        for rate in RATES:
            group=[r for r in rows if r['label']==label and r['rate']==rate]
            metrics={k:dict(mean=float(np.mean([r['metrics'][k] for r in group])),
                std=float(np.std([r['metrics'][k] for r in group],ddof=1)),
                minimum=min(r['metrics'][k] for r in group),maximum=max(r['metrics'][k] for r in group),
                per_seed=[r['metrics'][k] for r in group]) for k in group[0]['metrics']}
            aggregates.append(dict(label=label,rate=rate,metrics=metrics))
    summary=dict(selected=selected,selected_sha256=sha,training=old.read(root/'training_complete.json'),
        simulations_per_scheme=54,rates=RATES,seeds=[1,2,3],seconds=30,warmup_frames=2,
        same_traffic_verified=True,raw_metrics_recomputed=True,
        code_sha256=digest(Path(__file__)),aggregates=aggregates,rows=rows)
    atomic_json(root/'analysis.json',summary)
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['scheme','Mbps',*[x for k in KEYS for x in (k+'_mean',k+'_sd')]])
        for a in aggregates:writer.writerow([a['label'],a['rate'],*[a['metrics'][k][stat] for k in KEYS for stat in ('mean','std')]])
    return summary


def plot(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    names=['Prediction-input O-MAPPO','True-CSI cross5 O-MAPPO','MEET-COBRA','Oracle-MC']
    colors=['#D55E00','#0072B2','#009E73','#777777']
    for ax,key,title in zip(axes.flat,['power_w','violation_percent','p99_proxy_ms','macro_association_percent'],
        ['System transmit power (W)','Violation probability (%)','99th-percentile latency proxy (ms)','Macro-BS association (%)']):
        for label,color in zip(names,colors):
            group=[r for r in summary['aggregates'] if r['label']==label]
            y=np.array([r['metrics'][key]['mean'] for r in group]);sd=np.array([r['metrics'][key]['std'] for r in group])
            ax.plot(RATES,y,label=label,color=color,linewidth=1.6,marker='o',markersize=3)
            ax.fill_between(RATES,np.maximum(y-sd,0),y+sd,color=color,alpha=.12,linewidth=0)
        ax.set_xlabel('Arrival rate per vehicle (Mbps)');ax.set_ylabel(title)
        ax.grid(alpha=.2);ax.set_xlim(1,35)
        if key=='violation_percent':ax.set_yscale('symlog',linthresh=.01)
        if key=='p99_proxy_ms':ax.set_yscale('log')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    fig.savefig(root/'comparison.pdf');fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    for seed,color in zip(SEEDS,colors):
        folder=root/'training'/f'seed{seed}'
        updates=[old.read(p) for p in sorted((folder/'updates').glob('*.json'))]
        x=np.array([r['round'] for r in updates]);y=np.array([np.mean([r['mean_event_reward'] for r in u['loads']]) for u in updates])
        axes[0].plot(x,y,color=color,alpha=.2,lw=.6)
        if len(y)>=7:axes[0].plot(x[6:],np.convolve(y,np.ones(7)/7,mode='valid'),color=color,label=f'Seed {seed}')
        h=old.read(folder/'validation_history.json')
        axes[1].plot([r['round'] for r in h],[r['score'] for r in h],'-o',ms=3,color=color,label=f'Seed {seed}')
    axes[0].set_ylabel('Training reward (raw and 7-update mean)')
    axes[1].set_ylabel('Balanced validation cost (lower is better)')
    for ax in axes:ax.set_xlabel('PPO update');ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(root/'training_curves.pdf');fig.savefig(root/'training_curves.png',dpi=180);plt.close(fig)


def report(root,s):
    labels=['Prediction-input O-MAPPO','True-CSI cross5 O-MAPPO','MEET-COBRA','Oracle-MC']
    lookup={(r['label'],r['rate']):r['metrics'] for r in s['aggregates']}
    lines=['# Prediction-only O-MAPPO: completed experiment','',
        'All values below are three-seed means. Each run lasts 30 s; the first two frames are omitted.',
        'The baseline uses one common validation-selected checkpoint at every test load. No paper files were changed.','',
        f"Selected: {s['selected']['label']}; checkpoint SHA-256 `{s['selected_sha256']}`.",'',
        '## Power / violation probability','',
        '| Mbps | Prediction-input O-MAPPO | True-CSI cross5 O-MAPPO | MEET-COBRA | Oracle-MC |',
        '|---:|---:|---:|---:|---:|']
    for rate in RATES:
        cells=[f"{lookup[label,rate]['power_w']['mean']:.3f} W / {lookup[label,rate]['violation_percent']['mean']:.4f}%" for label in labels]
        lines.append('| '+str(rate)+' | '+' | '.join(cells)+' |')
    lines+=['','## Training stability','']
    for seed in SEEDS:
        h=old.read(root/'training'/f'seed{seed}'/'validation_history.json')
        best=old.read(root/'training'/f'seed{seed}'/'best_positive.json')
        state=s['training']['seeds'][str(seed)]
        lines.append(f"- Seed {seed}: selected positive update {best['round']}; validation score {h[0]['score']:.4f} -> {best['score']:.4f}; last {h[-1]['score']:.4f}; stability criterion passed: {state['passed']}.")
    lines+=['','## Scope and interpretation','',
        '- Prediction-only inputs do not imply prediction-only physical simulation: serving-link measurements and actual directional service are retained, with their probe costs.',
        '- True-CSI cross5 uses the frozen approved actor. The trained prediction-input version also changes the actor weights and uses the exact rollout environment. Their difference is not a one-variable causal estimate of CSI error.',
        '- Mean ± one sample standard deviation bands summarize three traffic/fading seeds, not uncertainty over deployments or independent channel environments.',
        '- Zero observed violations do not establish zero violation probability.',
        '- The maximum-gain forecast is a proxy for the subsequent hierarchical search; no true-channel calibration is inserted into candidate selection.',
        '- Raw queues, power, RB capacity bounds, hashes and paired arrivals were independently checked for all compared cases.','',
        'Detailed protocol: `experiment/o_mappo_predicted_cross5_report.md`.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--wait',action='store_true',help='Wait in short intervals for both already-running pipelines')
    a=p.parse_args()
    while a.wait and not ((a.root/'complete.json').exists() and (a.root/'true_control/summary.json').exists()):
        print('WAITING for trained and true-CSI grids',flush=True)
        time.sleep(30)
    summary=collect(a.root);plot(a.root,summary);report(a.root,summary)
    print('AUDITED',len(summary['rows']),'paired scheme/rate/seed rows',flush=True)


if __name__=='__main__':main()
