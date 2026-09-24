#!/usr/bin/env python3
"""Actor-only depth ablation, including automatic validation and paired tests.

The existing one-layer training is preserved. Both depths select their best
proxy-validation checkpoint per training seed, then select a single policy
across seeds using the same independent exact-validation score. Test data
are not used for model selection. Reruns resume completed stages/cases.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.revision_pipeline import atomic_json, digest

SINGLE = ROOT/'experiment/results/o_mappo_h32_retrained_20260924_v5'
DATA = ROOT/'experiment/results/o_mappo_h32_retrained_20260924_v4'
REFERENCE = ROOT/'experiment/results/o_mappo_h32_retrained_20260924/exact_reference'
TEST_TIMELINE = ROOT/'experiment/results/revision_directional_20260922/test_predictions.pkl'
SEEDS = (11,22,33)
RATES = list(range(1,36,2))
TEST_RATES = (1,5,15,21,25,29)


def read(path):
    return json.loads(path.read_text())


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable,'-u',*map(str,command)],cwd=ROOT,
                       env=dict(os.environ,PYTHONHASHSEED='0'),
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def export_best(source, destination, depth):
    exports=[]
    for seed in SEEDS:
        folder=source/f'train_seed{seed}'
        best=read(folder/'best.json')
        target=destination/f'{depth}_seed{seed}'/'selected.pt'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=digest(folder/'best.pt'):
                raise ValueError('Frozen checkpoint changed')
        else:
            shutil.copyfile(folder/'best.pt',target)
        atomic_json(target.with_suffix('.json'),dict(training_seed=seed,round=best['round'],
            proxy_validation_score=best['score'],source=str(folder/'best.pt'),sha256=digest(target)))
        exports.append(target)
    return exports


def evaluate(root, policies, timeline, rates, seeds, seconds, devices):
    run(['experiment/evaluate_o_mappo_h32_retrained.py','--root',root,'--timeline',timeline,
         '--policies',*policies,'--rates',','.join(map(str,rates)),
         '--seeds',','.join(map(str,seeds)),'--seconds',seconds,'--devices',devices],root/'launcher.log')


def select(root, references):
    ref={(r['rate_mbps'],r['seed'],r['frames']):r for r in
         (read(p) for p in (references/'runs').glob('*.json'))}
    groups={}
    for path in (root/'runs').glob('*.json'):
        row=read(path)
        key=(row['rate_mbps'],row['seed'],row['frames'])
        baseline=ref[key]
        if row['traffic_sha256']!=baseline['traffic_sha256']:
            raise ValueError('Validation traffic not paired')
        m,b=row['metrics'],baseline['metrics']
        ratio=(m['violation_percent']+.02*m['power_w'])/max(b['violation_percent']+.02*b['power_w'],1e-3)
        groups.setdefault(row['checkpoint'],[]).append((key,ratio))
    rankings=[]
    for path,rows in groups.items():
        if len(rows)!=len(RATES) or {x[0][0] for x in rows}!=set(RATES):
            raise ValueError('Incomplete full-load validation')
        ratios=[x[1] for x in rows]
        rankings.append(dict(checkpoint=path,score=.5*(sum(ratios)/len(ratios)+max(ratios)),
                             mean_ratio=sum(ratios)/len(ratios),worst_ratio=max(ratios)))
    rankings.sort(key=lambda x:(x['score'],x['checkpoint']))
    if len(rankings)!=len(SEEDS):
        raise ValueError('Incomplete training-seed comparison')
    atomic_json(root/'selection.json',dict(rule='mean and worst normalized cost, equal weight',
                                         rankings=rankings,selected=rankings[0]))
    return Path(rankings[0]['checkpoint'])


def summarize_test(root):
    cases={}
    for p in (root/'runs').glob('*.json'):
        r=read(p)
        depth=Path(r['checkpoint']).parent.name.split('_seed')[0]
        cases.setdefault((r['rate_mbps'],r['seed']),{})[depth]=r
    rows=[]
    for rate in TEST_RATES:
        by_depth={name:[] for name in ('single','actor2')}
        for seed in (1,2,3):
            pair=cases[rate,seed]
            if pair['single']['traffic_sha256']!=pair['actor2']['traffic_sha256']:
                raise ValueError('Test traffic not paired')
            for depth in by_depth:
                by_depth[depth].append(pair[depth]['metrics'])
        aggregates={}
        for depth,metrics in by_depth.items():
            aggregates[depth]={key:sum(m[key] for m in metrics)/len(metrics) for key in metrics[0]}
        rows.append(dict(rate_mbps=rate,means=aggregates,per_seed=by_depth))
    atomic_json(root/'summary.json',dict(seconds=30,seeds=[1,2,3],warmup_frames=2,rows=rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'experiment/results/o_mappo_actor_depth_20260924')
    parser.add_argument('--workers',type=int,default=36)
    parser.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=dict(single_training=str(SINGLE),seeds=SEEDS,rounds=160,
        actor_hidden_sizes=[64,64],critic_hidden_sizes=[64],
        critic_initialization='same-seed original round0000 critic',
        validation_rates=RATES,validation_seconds=10,validation_seeds=[101],
        test_rates=TEST_RATES,test_seconds=30,test_seeds=[1,2,3],
        selection='best proxy checkpoint per training seed, then minimum independent exact score',
        baseline_checkpoints={str(s):digest(SINGLE/f'train_seed{s}/best.pt') for s in SEEDS},
        code={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),
             ROOT/'experiment/train_o_mappo_hierarchical32.py',ROOT/'utils/o_mappo.py',
             ROOT/'utils/o_mappo_sim.py',ROOT/'utils/hierarchical_beam.py',
             ROOT/'experiment/evaluate_o_mappo_h32_retrained.py']})
    manifest=json.loads(json.dumps(manifest))
    protocol=args.root/'protocol.json'
    if protocol.exists() and read(protocol)!=manifest:
        raise ValueError('Pipeline protocol changed; use a new root')
    atomic_json(protocol,manifest)

    def status(stage,**kwargs):
        value=dict(stage=stage,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
        atomic_json(args.root/'status.json',value)
        print(value,flush=True)

    training=args.root/'actor2_training'
    training.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        command=['experiment/train_o_mappo_hierarchical32.py','train','--root',training,
                 '--data-root',DATA,'--actor-hidden-sizes','64,64','--critic-reference-root',SINGLE,
                 '--training-seeds','11,22,33','--rounds','160','--rollout-seconds','10','--workers',args.workers]
        status('training_and_single_validation')
        future=pool.submit(run,command,training/'training.log')
        single=export_best(SINGLE,args.root/'checkpoints','single')
        evaluate(args.root/'validation_single',single,DATA/'exact_validation.pkl',RATES,[101],10,args.devices)
        selected_single=select(args.root/'validation_single',REFERENCE)
        while not future.done():
            progress=read(training/'progress.json') if (training/'progress.json').exists() else {}
            status('training',round=progress.get('round'))
            time.sleep(30)
        future.result()
    status('actor2_validation')
    actor2=export_best(training,args.root/'checkpoints','actor2')
    evaluate(args.root/'validation_actor2',actor2,DATA/'exact_validation.pkl',RATES,[101],10,args.devices)
    selected_actor2=select(args.root/'validation_actor2',REFERENCE)
    atomic_json(args.root/'selected_policies.json',dict(single=str(selected_single),actor2=str(selected_actor2)))
    status('paired_test')
    evaluate(args.root/'paired_test',[selected_single,selected_actor2],TEST_TIMELINE,TEST_RATES,[1,2,3],30,args.devices)
    summarize_test(args.root/'paired_test')
    status('complete',training_convergence=read(training/'training_complete.json')['seeds'])


if __name__=='__main__':
    main()
