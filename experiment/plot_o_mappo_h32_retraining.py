#!/usr/bin/env python3
"""Training diagnostics, separate from manuscript figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    summaries={}
    for folder in sorted(args.root.glob('train_seed*')):
        source=folder/'validation_history.json'
        if not source.exists(): continue
        h=json.loads(source.read_text())
        x=[r['round'] for r in h]
        label=folder.name.replace('train_seed','Training seed ')
        axes[0,0].plot(x,[r['score'] for r in h],'-o',markersize=3,label=label)
        axes[0,1].plot(x,[np.mean([v['mean_event_reward'] for v in r['rows']]) for r in h],'-o',markersize=3,label=label)
        axes[1,0].plot(x[1:],[100*r['probe_flip_fraction'] for r in h[1:]],'-o',markersize=3,label=label)
        axes[1,1].plot(x[1:],[r['probe_probability_change'] for r in h[1:]],'-o',markersize=3,label=label)
        summaries[folder.name]=json.loads((folder/'convergence.json').read_text())
    for ax in axes.ravel():
        ax.set_xlabel('Mixed-load PPO update round')
        ax.grid(alpha=.25)
    axes[0,0].set_ylabel('Balanced validation cost (lower is better)')
    axes[0,0].set_yscale('log')
    axes[0,0].legend(fontsize=9)
    axes[0,1].set_ylabel('Mean validation event reward')
    axes[1,0].set_ylabel('Changed greedy actions on fixed states (%)')
    axes[1,0].axhline(1,color='gray',linestyle='--',linewidth=1)
    axes[1,1].set_ylabel('Mean absolute action-probability change')
    axes[1,1].axhline(.005,color='gray',linestyle='--',linewidth=1)
    fig.savefig(args.root/'training_curves.pdf')
    fig.savefig(args.root/'training_curves.png',dpi=180)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
