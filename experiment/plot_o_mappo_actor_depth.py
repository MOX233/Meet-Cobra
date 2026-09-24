#!/usr/bin/env python3
"""Diagnostics for the actor-depth ablation; never modify paper figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SINGLE=ROOT/'experiment/results/o_mappo_h32_retrained_20260924_v5'
SEEDS=(11,22,33)


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'experiment/results/o_mappo_actor_depth_20260924')
    args=p.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    fig.suptitle('Mean across three training seeds; shading shows the seed range',fontsize=10)
    for folder,label,color in [(SINGLE,'One hidden layer','#1673b1'),
                               (args.root/'actor2_training','Two hidden layers','#d66b24')]:
        sequences=[];validation=[]
        for seed in SEEDS:
            d=folder/f'train_seed{seed}'
            files=sorted((d/'updates').glob('*.json'))
            values=np.array([np.mean([r['mean_event_reward'] for r in read(f)['loads']]) for f in files])
            sequences.append(values)
            if (d/'validation_history.json').exists():
                validation.append(read(d/'validation_history.json'))
        n=min(map(len,sequences))
        if n>=10:
            y=np.array([np.convolve(v[:n],np.ones(10)/10,mode='valid') for v in sequences])
            x=np.arange(10,n+1)
            axes[0].plot(x,y.mean(0),label=label,color=color)
            axes[0].fill_between(x,y.min(0),y.max(0),color=color,alpha=.14)
        if len(validation)==3:
            common=sorted(set.intersection(*[{r['round'] for r in h} for h in validation]))
            common=[r for r in common if r>0]
            if common:
                y=np.array([[next(row['score'] for row in h if row['round']==r) for r in common] for h in validation])
                axes[1].plot(common,y.mean(0),marker='o',ms=3,label=label,color=color)
                axes[1].fill_between(common,y.min(0),y.max(0),color=color,alpha=.14)
    axes[0].set_ylabel('Training reward (10-round moving mean)')
    axes[1].set_ylabel('Proxy-validation score (lower is better)')
    for ax in axes:
        ax.set_xlabel('Training round');ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(args.root/'training_depth_comparison.pdf')
    fig.savefig(args.root/'training_depth_comparison.png',dpi=160)
    plt.close(fig)
    path=args.root/'paired_test/summary.json'
    if not path.exists():
        return
    rows=read(path)['rows']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    fig.suptitle('Fixed selected policies; mean and range across three paired traffic seeds',fontsize=10)
    for depth,label,color,marker in [('single','One hidden layer','#1673b1','o'),
                                     ('actor2','Two hidden layers','#d66b24','s')]:
        x=[r['rate_mbps'] for r in rows]
        for ax,key in zip(axes,('power_w','violation_percent')):
            y=np.array([[m[key] for m in r['per_seed'][depth]] for r in rows])
            ax.plot(x,y.mean(1),color=color,marker=marker,label=label)
            ax.fill_between(x,y.min(1),y.max(1),color=color,alpha=.14)
    axes[0].set_ylabel('Average system transmit power (W)')
    axes[1].set_ylabel('Latency violation probability (%)')
    axes[1].set_yscale('symlog',linthresh=.01)
    for ax in axes:
        ax.set_xlabel('Arrival rate (Mbps)');ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(args.root/'paired_test_comparison.pdf')
    fig.savefig(args.root/'paired_test_comparison.png',dpi=160)


if __name__=='__main__':
    main()
