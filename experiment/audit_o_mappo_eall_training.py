#!/usr/bin/env python3
"""Audit saved E_all training/evaluation artifacts; draw diagnostic curves."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiment.o_mappo_eall_training import (
    ROOT, OUTPUT, POLICY, CONFIGURATIONS, RATES, old, verify, digest,
    verified_case, atomic_json, om, initial_policy, schedule)


def audit(root):
    protocol = verify(root)
    complete = old.read(root/'training_complete.json')
    assert complete['rounds'] == protocol['rounds']
    initial = initial_policy(11)
    records = []
    updates_by_key = {}
    for route,seed in CONFIGURATIONS:
        folder=root/'training'/route/f'seed{seed}'
        zero=om.OMAPPPolicy.load(str(folder/'round0000.pt'))
        for model in ('actor','critic'):
            for k,v in getattr(initial,model).state_dict().items():
                assert torch.equal(v,getattr(zero,model).state_dict()[k])
        history=old.read(folder/'validation_history.json')
        assert [r['round'] for r in history]==list(range(0,protocol['rounds']+1,10))
        best=min(history[1:],key=lambda x:x['score'])
        assert old.read(folder/'best_positive.json')==best
        selected=om.OMAPPPolicy.load(str(folder/'best_positive.pt'))
        saved=om.OMAPPPolicy.load(str(folder/f'round{best["round"]:04d}.pt'))
        for model in ('actor','critic'):
            for k,v in getattr(selected,model).state_dict().items():
                assert torch.equal(v,getattr(saved,model).state_dict()[k])
        assert any(not torch.equal(v,initial.actor.state_dict()[k]) for k,v in selected.actor.state_dict().items())
        updates=[]
        for n in range(1,protocol['rounds']+1):
            row=old.read(folder/'updates'/f'{n:04d}.json')
            assert row['round']==n and row['route']==route and row['seed']==seed
            assert row['learning_rate']==schedule(n)
            assert len(row['loads'])==len(RATES)
            assert np.isfinite(list(row['update'].values())).all()
            updates.append(row)
        updates_by_key[route,seed]=updates
        records.append(dict(route=route,seed=seed,best_round=best['round'],
            initial_score=history[0]['score'],best_score=best['score'],
            final_score=history[-1]['score'],convergence=old.convergence(history),
            mean_reward_first20=float(np.mean([r['mean_event_reward'] for u in updates[:20] for r in u['loads']])),
            mean_reward_last20=float(np.mean([r['mean_event_reward'] for u in updates[-20:] for r in u['loads']]))))
    for seed in (11,22,33):
        for a,b in zip(updates_by_key['legacy',seed],updates_by_key['E_all',seed]):
            assert a['update']['transitions']==b['update']['transitions']
    grids=[]
    for name,expected in [('zero_validation',36),('trained_validation',108),('paired_test',36)]:
        folder=root/name
        p=old.read(folder/'protocol.json')
        assert p['code']==protocol['code']
        assert digest(Path(p['timeline']))==p['timeline_sha256']
        assert len(p['jobs'])==expected
        hashes={}
        failures=calls=0
        for label,spec in p['jobs'].items():
            assert verified_case(folder,label)
            assert digest(Path(spec['policy']))==spec['policy_sha256']
            row=old.read(folder/'runs'/f'{label}.json')
            assert row['actor_sha256']==spec['policy_sha256']
            key=(spec['rate'],spec['seed'])
            assert hashes.setdefault(key,row['traffic_sha256'])==row['traffic_sha256']
            metrics=row['metrics']
            assert np.isfinite(list(metrics.values())).all()
            with np.load(folder/'runs'/f'{label}.npz') as z:
                rb=z['rb_per_bs']; q=z['queue_bits']; f=z['queue_frame']
                assert np.all((rb>=0)&(rb<=np.array([133,66,66,66,66])+1e-8))
                assert np.isfinite(q).all() and np.all(q>=0)
                np.testing.assert_allclose(rb@np.array([1,.2,.2,.2,.2])*.1,z['energy_j'],rtol=1e-10,atol=1e-8)
                u=np.array([np.mean(q[f==i]>spec['rate']*1e6*.020) for i in range(len(rb))])
                np.testing.assert_allclose(u,z['violation_probability'],atol=1e-12)
                delay=q[f>=2]/(spec['rate']*1e6)*1000
                counts=z['association_counts'][2:]
                computed=dict(power_w=z['energy_j'][2:].mean()/.1,
                    violation_percent=100*u[2:].mean(),mean_proxy_ms=delay.mean(),
                    p90_proxy_ms=np.percentile(delay,90),p99_proxy_ms=np.percentile(delay,99),
                    macro_association_percent=100*counts[:,0].sum()/counts.sum(),
                    pilots_per_vehicle_slot=z['pilots'][2:].mean(),handovers=z['handover_count'].sum())
                for k,v in computed.items(): np.testing.assert_allclose(v,metrics[k],atol=1e-9,rtol=1e-10)
            if name=='zero_validation' and spec['route']=='legacy': assert row['reference_raw_parity']
            failures+=metrics['optimizer_failures']; calls+=metrics['optimizer_calls']
        grids.append(dict(grid=name,cases=expected,optimizer_calls=calls,optimizer_failures=failures))
    output=dict(passed=True,training=records,evaluation=grids,
        note='Proxy training and directional exact evaluation audited separately; stability is not optimality.')
    atomic_json(root/'audit.json',output)
    return output


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    colors=['#0072B2','#D55E00','#009E73']
    for row,route in enumerate(('legacy','E_all')):
        for color,seed in zip(colors,(11,22,33)):
            folder=root/'training'/route/f'seed{seed}'
            paths=sorted((folder/'updates').glob('*.json'))
            updates=[old.read(p) for p in paths]
            x=np.array([u['round'] for u in updates])
            y=np.array([np.mean([r['mean_event_reward'] for r in u['loads']]) for u in updates])
            axs[row,0].plot(x,y,color=color,alpha=.15,lw=.6,
                            label=f'Seed {seed}' if len(y)<10 else None)
            if len(y)>=10:
                axs[row,0].plot(x[9:],np.convolve(y,np.ones(10)/10,mode='valid'),color=color,label=f'Seed {seed}')
            history=old.read(folder/'validation_history.json')
            axs[row,1].plot([h['round'] for h in history],[h['score'] for h in history],
                            color=color,marker='.',label=f'Seed {seed}')
        axs[row,0].set_title(f'{route}: training reward (10-round moving mean)')
        axs[row,1].set_title(f'{route}: surrogate validation cost')
        axs[row,0].set_ylabel('Mean event reward')
        axs[row,1].set_ylabel('Normalized composite cost (lower is better)')
        for ax in axs[row]:
            ax.grid(alpha=.2); ax.set_xlabel('Additional training round'); ax.legend(frameon=False)
    fig.savefig(root/'training_diagnostics.pdf')
    fig.savefig(root/'training_diagnostics.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=OUTPUT)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--wait',action='store_true',help='Wait up to four hours for the pipeline output')
    args=parser.parse_args()
    if args.wait:
        deadline=time.monotonic()+4*3600
        while not (args.root/'results_summary.json').exists():
            if time.monotonic()>deadline: raise TimeoutError('Pipeline has not completed within four hours')
            time.sleep(30)
    if not args.plot_only: print(audit(args.root))
    plot(args.root)
