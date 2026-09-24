#!/usr/bin/env python3
"""Paired warm-start PPO for legacy and E_all, with exact validation/test.

No production module or checkpoint is edited. E_all uses the stage-1 bounded
mean-gain context for actor, target optimization and the *fluid approximation*
of RA during training. Exact evaluation always uses unmodified directional,
explicit-RB physical service through the tested stage-1 adapter.
"""
import argparse
import collections
import concurrent.futures
from contextlib import contextmanager, ExitStack
import dataclasses
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from unittest.mock import patch

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['KMP_AFFINITY'] = 'disabled'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiment import train_o_mappo_hierarchical32 as old
from experiment import o_mappo_stage1_consistency as stage1
from experiment.o_mappo_optimizer_information import fixed_allocation
from experiment.o_mappo_target_check import estimate_bounded
from experiment.revision_pipeline import atomic_json, digest, native
from utils import o_mappo as om
from utils.directional_service import beam_average_gain_db

POLICY = stage1.POLICY
DATA_ROOT = ROOT/'experiment/results/o_mappo_h32_retrained_20260924_v4'
OUTPUT = ROOT/'experiment/results/o_mappo_eall_training_20260924'
TEST_TIMELINE = ROOT/'experiment/results/revision_directional_20260922/test_predictions.pkl'
RATES = list(range(1,36,2))
ROUTES = ('legacy','E_all')
SEEDS = (11,22,33)
CONFIGURATIONS = [(r,s) for r in ROUTES for s in SEEDS]


class EAllFluidAdapter:
    """Shared decision estimates, without relabeling allocated RBs as demand.

The old fluid allocator's final greedy scheduling pass is retained. Its input
interference is evaluated at the common E_all occupancy, not a second private
occupancy iteration. Returned actual fluid allocations still determine power,
past RB feedback and congestion reward. Actor/optimizer receive the separately
stored decision estimate. This remains an approximate training environment.
    """
    def __init__(self):
        self.old_step = om.fluid_o_mappo_step
        self.old_allocate = om._fluid_allocation
        self.old_state = om.make_local_state
        self.old_optimize = om.optimize_triggered_targets
        self.frame_count = 0

    def step(self, args, records, learners, queues, rates, previous_load,
             macro_loc, config, tx, rx):
        self.args, self.records, self.learners = args, records, learners
        self.rates = rates
        self.ids = sorted(records, key=str)
        self.state_index = 0
        self.frame_count += 1
        result = self.old_step(args, records, learners, queues, rates, previous_load,
                               macro_loc, config, tx, rx)
        self.step_result = result
        return result

    def allocate(self, args, vehicles, connection, serving, interference, pilots,
                 backlog, initial_load, duration, iterations=5, service_fraction=None):
        self.serving, self.interference, self.connection = serving, interference, connection
        gains = {v:interference[v].copy() for v in vehicles}
        for v in vehicles:
            gains[v][connection[v]] = serving[v]
        self.rb = estimate_bounded(args, connection, gains, interference, self.rates)
        self.caps = np.array([args.num_RB_macro]+[args.num_RB_micro]*4)
        self.load = self.rb/self.caps
        assert np.all((self.load>=0)&(self.load<=1))
        # iterations=0 executes the original final capacity/allocation pass
        # using the provided common estimate, without private load updates.
        return self.old_allocate(args, vehicles, connection, serving, interference,
            pilots, backlog, self.load, duration, iterations=0,
            service_fraction=service_fraction)

    def state(self, *args, **kwargs):
        vehicle = self.ids[self.state_index]
        self.state_index += 1
        assert args[4] == self.connection[vehicle]
        changed = list(args)
        interference = om._interference_db(self.args, args[4], self.interference[vehicle], self.load)
        changed[5] = om.effective_sinr_db(self.args,args[4],self.serving[vehicle],interference)
        changed[8] = self.load
        changed[10] = interference
        return self.old_state(*changed, **kwargs)

    def optimize(self, args, records, learners, triggered, backlog, allocated,
                 load, config, tx, rx, macro_loc, **kwargs):
        assert self.state_index == len(self.ids)
        fixed = fixed_allocation(args, learners, backlog, self.serving,
            self.interference, self.load, self.rb, config)
        return self.old_optimize(args,records,learners,triggered,backlog,fixed,
                                 self.load,config,tx,rx,macro_loc,**kwargs)

    @contextmanager
    def applied(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(om,'no_bf_gain_db',beam_average_gain_db))
            stack.enter_context(patch.object(om,'fluid_o_mappo_step',self.step))
            stack.enter_context(patch.object(om,'_fluid_allocation',self.allocate))
            stack.enter_context(patch.object(om,'make_local_state',self.state))
            stack.enter_context(patch.object(om,'optimize_triggered_targets',self.optimize))
            yield self


@contextmanager
def training_environment(route):
    if route == 'legacy':
        yield None
    elif route == 'E_all':
        with EAllFluidAdapter().applied() as adapter:
            yield adapter
    else:
        raise ValueError('Unknown route')


def initial_policy(seed):
    policy = om.OMAPPPolicy.load(str(POLICY),seed=seed,load_optimizers=False)
    assert policy.config.actor_hidden_sizes == (64,64)
    assert policy.config.batch_size == 256 and policy.config.ppo_epochs == 4
    assert policy.config.state_variant == 'adapted' and policy.config.trigger_gate == 'periodic'
    policy.config.entropy_coefficient = .002
    return policy


def schedule(round_no):
    fraction = min(round_no/120,1)
    return float(1e-5+(1e-4-1e-5)*.5*(1+np.cos(np.pi*fraction)))


def rollout(job):
    path,route,rate,start,length,seed,learn = job
    policy = om.OMAPPPolicy.load(str(path),seed=seed)
    old.bind_worker()
    timeline = old.array_slice(start,length)
    with training_environment(route):
        result,memory = om.run_fluid_o_mappo_episode(old.paper_args(rate*1e6),timeline,
            policy,om.o_mappo_reward_presets()['qos_energy020_load1'],rate,
            seed=seed,learn=learn,collect_only=True)
    if learn:
        for t in memory.transitions:
            t.vehicle = (rate,start,seed,t.vehicle)
        return result,memory.transitions
    states = np.stack([t.local_state for t in memory.transitions])
    return result,states[np.linspace(0,len(states)-1,min(256,len(states)),dtype=int)]


def sources():
    return [Path(__file__),ROOT/'experiment/train_o_mappo_hierarchical32.py',
        ROOT/'experiment/o_mappo_stage1_consistency.py',ROOT/'experiment/o_mappo_target_check.py',
        ROOT/'experiment/o_mappo_optimizer_information.py',ROOT/'experiment/o_mappo_shared_frontend.py',
        ROOT/'experiment/pql_ba_experiment.py',ROOT/'experiment/revision_pipeline.py',
        *sorted((ROOT/'utils').glob('*.py'))]


def manifest(rounds):
    arrays = old.read(DATA_ROOT/'arrays_complete.json')
    return dict(rounds=rounds, routes=list(ROUTES), training_seeds=list(SEEDS), rates=RATES,
        initial_checkpoint=str(POLICY),initial_sha256=digest(POLICY),
        initialization='identical actor AND critic weights; fresh Adam per route/seed',
        batch_size=256,ppo_epochs=4,rollout_seconds=10,validate_every=10,
        learning_rate='cosine 1e-4 to 1e-5 by round120, then 1e-5',entropy_coefficient=.002,
        train_interval=[200,650],proxy_validation_interval=[700,710],
        exact_validation_interval=[710,720],exact_validation_seed=101,
        test_interval=[800,830],test_rates=[5,13,25],test_seeds=[1,2,3],
        selection='best positive-round proxy checkpoint per seed; common exact validation score across 18 loads',
        zero_round='always retained as independent no-training controls; never overwritten',
        reward=dataclasses.asdict(om.o_mappo_reward_presets()['qos_energy020_load1']),
        training_service='frame surrogate; E_all retains fluid greedy allocation with shared bounded estimates',
        evaluation_service='directional explicit RB and slot-level OTR-RA via stage1 tested adapter',
        data_root=str(DATA_ROOT),arrays=arrays,
        exact_validation_sha256=digest(stage1.TIMELINE),test_timeline_sha256=digest(TEST_TIMELINE),
        convergence='report old fixed-window criteria; equal round budget, no claim from round count alone',
        code={str(p.relative_to(ROOT)):digest(p) for p in sources()})


def verify(root):
    frozen = old.read(root/'protocol.json')
    if frozen != manifest(frozen['rounds']):
        raise ValueError('Inputs or code changed; use a fresh experiment directory')
    return frozen


def train(root,workers):
    protocol = verify(root)
    lock=(root/'training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    arrays=protocol['arrays']
    if digest(DATA_ROOT/'channels.npy')!=arrays['channel_sha256'] or digest(DATA_ROOT/'channel_index.npz')!=arrays['index_sha256']:
        raise ValueError('Shared training data changed')
    torch.set_num_threads(1)
    old.single_thread_solvers()
    policies, histories, probes, last_probs, best = {},{},{},{},{}
    # Resume at a completed validation boundary; partially completed updates
    # beyond that point are replayed from the saved optimizer and shuffle RNG.
    progress_path=root/'training_progress.json'
    resume_round=old.read(progress_path)['round'] if progress_path.exists() else -1
    for route,seed in CONFIGURATIONS:
        key=(route,seed)
        folder=root/'training'/route/f'seed{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        if resume_round>=0:
            policies[key]=om.OMAPPPolicy.load(str(folder/f'round{resume_round:04d}.pt'),seed=seed,load_optimizers=True)
            histories[key]=[x for x in old.read(folder/'validation_history.json') if x['round']<=resume_round]
            probes[key]=np.load(folder/'probes.npy')
            with torch.no_grad():
                last_probs[key]=policies[key].actor(torch.from_numpy(probes[key])).softmax(-1).numpy()
            positive=[x for x in histories[key] if x['round']>0]
            best[key]=min([x['score'] for x in positive],default=float('inf'))
        else:
            policies[key]=initial_policy(seed)
            histories[key]=[]
            best[key]=float('inf')
            old.save_policy(policies[key],folder/'round0000.pt')
    started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
            initializer=old.initialize_worker,initargs=(str(DATA_ROOT),)) as pool:
        if (root/'proxy_reference.json').exists():
            reference=np.array(old.read(root/'proxy_reference.json')['costs'])
        else:
            jobs=[(POLICY,'legacy',r,700,10,9900+r,False) for r in RATES]
            result=list(pool.map(rollout,jobs))
            reference=np.maximum([x[0]['queue_violation_percent']+.02*x[0]['average_system_power_w'] for x in result],1e-3)
            atomic_json(root/'proxy_reference.json',dict(costs=reference.tolist(),rows=[x[0] for x in result]))
        for round_no in range(resume_round+1,protocol['rounds']+1):
            if round_no:
                jobs=[]
                lr=schedule(round_no)
                for route,seed in CONFIGURATIONS:
                    folder=root/'training'/route/f'seed{seed}'
                    policy=policies[route,seed]
                    for optimizer in (policy.actor_optimizer,policy.critic_optimizer):
                        for group in optimizer.param_groups:
                            group['lr']=lr
                    old.save_policy(policy,folder/'rollout.pt')
                    for rate in RATES:
                        rng=np.random.default_rng(np.random.SeedSequence([seed,round_no,rate]))
                        start=float(rng.integers(2000,6400))/10
                        jobs.append((folder/'rollout.pt',route,rate,start,10,
                                     seed*100000+round_no*100+rate,True))
                outputs=list(pool.map(rollout,jobs))
                for index,(route,seed) in enumerate(CONFIGURATIONS):
                    group=outputs[index*18:(index+1)*18]
                    memory=om.OMAPPOMemory()
                    for _,transitions in group:
                        memory.transitions.extend(transitions)
                    update=policies[route,seed].update(memory)
                    row=dict(round=round_no,route=route,seed=seed,learning_rate=lr,
                        elapsed_s=time.monotonic()-started,update=update,loads=[x[0] for x in group])
                    atomic_json(root/'training'/route/f'seed{seed}'/'updates'/f'{round_no:04d}.json',row)
                    print('TRAIN',route,seed,round_no,'samples',len(memory),'reward',
                          np.mean([x[0]['mean_event_reward'] for x in group]),flush=True)
                del outputs
            if round_no%10==0:
                jobs=[]
                for route,seed in CONFIGURATIONS:
                    path=root/'training'/route/f'seed{seed}'/f'round{round_no:04d}.pt'
                    old.save_policy(policies[route,seed],path)
                    jobs.extend((path,route,r,700,10,9900+r,False) for r in RATES)
                outputs=list(pool.map(rollout,jobs))
                for index,key in enumerate(CONFIGURATIONS):
                    route,seed=key
                    folder=root/'training'/route/f'seed{seed}'
                    group=outputs[index*18:(index+1)*18]
                    rows=[x[0] for x in group]
                    value,costs=old.score(rows,reference)
                    if key not in probes:
                        probes[key]=np.concatenate([x[1] for x in group])
                        np.save(folder/'probes.npy',probes[key])
                    with torch.no_grad():
                        probs=policies[key].actor(torch.from_numpy(probes[key])).softmax(-1).numpy()
                    previous=last_probs.get(key,probs)
                    row=dict(round=round_no,rows=rows,score=value,costs=costs.tolist(),
                        probe_flip_fraction=float(np.mean(probs.argmax(-1)!=previous.argmax(-1))),
                        probe_probability_change=float(np.abs(probs-previous).mean()))
                    histories[key].append(row)
                    last_probs[key]=probs
                    if round_no>0 and value<best[key]:
                        best[key]=value
                        old.save_policy(policies[key],folder/'best_positive.pt')
                        atomic_json(folder/'best_positive.json',row)
                    atomic_json(folder/'validation_history.json',histories[key])
                    atomic_json(folder/'convergence.json',old.convergence(histories[key]))
                    print('VALIDATE',route,seed,round_no,'score',value,'stable',old.convergence(histories[key])['passed'],flush=True)
                atomic_json(progress_path,dict(round=round_no,elapsed_s=time.monotonic()-started,
                    states={f'{r}_{s}':dict(best_positive_score=(best[r,s] if np.isfinite(best[r,s]) else None),
                        convergence=old.convergence(histories[r,s])) for r,s in CONFIGURATIONS}))
    atomic_json(root/'training_complete.json',dict(rounds=protocol['rounds'],
        seeds={f'{r}_{s}':old.convergence(histories[r,s]) for r,s in CONFIGURATIONS}))


def case_label(label,rate,seed):
    return f'{label}_rate{rate}_seed{seed}'


def exact_case(folder,label,device):
    protocol=old.read(folder/'protocol.json')
    spec=protocol['jobs'][label]
    for p,h in protocol['code'].items():
        if digest(ROOT/p)!=h: raise ValueError('Evaluation code changed')
    policy=Path(spec['policy'])
    if digest(policy)!=spec['policy_sha256']: raise ValueError('Checkpoint changed')
    if digest(Path(protocol['timeline']))!=protocol['timeline_sha256']:
        raise ValueError('Evaluation timeline changed')
    with Path(protocol['timeline']).open('rb') as stream:
        timeline=pickle.load(stream)
    end=min(timeline)+protocol['seconds']
    timeline=collections.OrderedDict((t,r) for t,r in timeline.items() if t<=end+1e-8)
    with patch.object(stage1,'POLICY',policy):
        row,raw,diagnostics,_=stage1.simulate(timeline,
            'baseline' if spec['route']=='legacy' else 'E_all',spec['rate'],spec['seed'],device)
    row.update(label=spec['label'],route=spec['route'],policy=str(policy),
               protocol_sha256=digest(folder/'protocol.json'))
    (folder/'runs').mkdir(exist_ok=True)
    path=folder/'runs'/label
    np.savez_compressed(path.with_suffix('.npz'),**raw)
    atomic_json(path.with_name(label+'_diagnostics.json'),diagnostics)
    row.update(raw_sha256=digest(path.with_suffix('.npz')),
        diagnostics_sha256=digest(path.with_name(label+'_diagnostics.json')))
    # Existing formal baseline must reproduce the unchanged raw arrays.
    if spec['label']=='legacy_zero' and protocol['seconds']==10 and spec['seed']==101:
        ref=stage1.OLD_VALIDATION/f'actor2_seed33_selected_rate{spec["rate"]}_seed101.json'
        assert row['traffic_sha256']==old.read(ref)['traffic_sha256']
        with np.load(ref.with_suffix('.npz')) as z:
            for key in raw: np.testing.assert_array_equal(raw[key],z[key])
        row['reference_raw_parity']=True
    atomic_json(path.with_suffix('.json'),row)
    print('EXACT COMPLETE',label,row['metrics'],flush=True)


def verified_case(folder,label):
    path=folder/'runs'/f'{label}.json'
    if not path.exists(): return False
    r=old.read(path)
    assert r['protocol_sha256']==digest(folder/'protocol.json')
    assert r['raw_sha256']==digest(path.with_suffix('.npz'))
    assert r['diagnostics_sha256']==digest(path.with_name(label+'_diagnostics.json'))
    return True


def exact_grid(folder,policies,timeline,seconds,rates,seeds,devices):
    folder.mkdir(parents=True,exist_ok=True)
    jobs={}
    for label,spec in policies.items():
        for rate in rates:
            for seed in seeds:
                jobs[case_label(label,rate,seed)]=dict(label=label,route=spec['route'],
                    policy=str(spec['path']),policy_sha256=digest(spec['path']),rate=rate,seed=seed)
    protocol=dict(timeline=str(timeline),timeline_sha256=digest(timeline),seconds=seconds,
        jobs=jobs,code={str(p.relative_to(ROOT)):digest(p) for p in sources()})
    if (folder/'protocol.json').exists() and old.read(folder/'protocol.json')!=protocol:
        raise ValueError('Frozen evaluation changed')
    atomic_json(folder/'protocol.json',protocol)
    (folder/'logs').mkdir(exist_ok=True)
    def worker(device,labels):
        for label in labels:
            if verified_case(folder,label): continue
            with (folder/'logs'/f'{label}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),'exact',
                    '--root',str(folder),'--label',label,'--device',device],check=True,cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0'))
            print('DONE',folder.name,label,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures=[pool.submit(worker,d,list(jobs)[i::len(devices)]) for i,d in enumerate(devices)]
        for f in futures: f.result()
    rows=[old.read(folder/'runs'/f'{label}.json') for label in jobs]
    atomic_json(folder/'summary.json',dict(cases=len(rows),runs=rows))


def select(root):
    zero=old.read(root/'zero_validation/summary.json')['runs']
    trained=old.read(root/'trained_validation/summary.json')['runs']
    reference={r['rate_mbps']:r for r in zero if r['label']=='legacy_zero'}
    rankings=[]
    for label in sorted({r['label'] for r in zero+trained}):
        rows=sorted([r for r in zero+trained if r['label']==label],key=lambda r:r['rate_mbps'])
        assert [r['rate_mbps'] for r in rows]==RATES
        ratios=[]
        for r in rows:
            b=reference[r['rate_mbps']]
            assert r['traffic_sha256']==b['traffic_sha256']
            cost=lambda x:x['metrics']['violation_percent']+.02*x['metrics']['power_w']
            ratios.append(cost(r)/max(cost(b),1e-3))
        rankings.append(dict(label=label,route=rows[0]['route'],policy=rows[0]['policy'],
            score=float(.5*(np.mean(ratios)+np.max(ratios))),mean_ratio=float(np.mean(ratios)),
            worst_ratio=float(np.max(ratios))))
    selected={}
    for route in ROUTES:
        candidates=sorted([r for r in rankings if r['route']==route and not r['label'].endswith('_zero')],key=lambda r:(r['score'],r['label']))
        selected[route]=candidates[0]
    atomic_json(root/'selection.json',dict(rankings=rankings,selected_trained=selected,
        note='Both no-training controls retained; a trained candidate need not beat its zero-round control'))
    return selected


def summarize_test(root):
    s=old.read(root/'paired_test/summary.json')
    aggregates=[]
    for rate in (5,13,25):
        for label in ('legacy_zero','E_all_zero','legacy_trained','E_all_trained'):
            rows=sorted([r for r in s['runs'] if r['rate_mbps']==rate and r['label']==label],key=lambda r:r['seed'])
            assert len(rows)==3
            aggregates.append(dict(rate_mbps=rate,label=label,
                mean={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},
                per_seed=[dict(seed=r['seed'],metrics=r['metrics']) for r in rows]))
        for seed in (1,2,3):
            rs=[r for r in s['runs'] if r['rate_mbps']==rate and r['seed']==seed]
            assert len(rs)==4 and len({r['traffic_sha256'] for r in rs})==1
    atomic_json(root/'results_summary.json',dict(selection=old.read(root/'selection.json'),
        convergence=old.read(root/'training_complete.json'),aggregates=aggregates,
        scope='Three fixed test loads, 3 traffic seeds; no formal baseline replacement'))


def pipeline(args):
    root=args.root
    if args.rounds < 10 or args.rounds % 10:
        raise ValueError('Full pipeline requires positive multiples of 10 rounds')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=manifest(args.rounds)
    if (root/'protocol.json').exists() and old.read(root/'protocol.json')!=protocol:
        raise ValueError('Protocol changed; use a new root')
    atomic_json(root/'protocol.json',protocol)
    devices=args.devices.split(',')
    zero={r+'_zero':dict(route=r,path=POLICY) for r in ROUTES}
    def training_job():
        with (root/'training.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'train',
                '--root',str(root),'--workers',str(args.workers)],check=True,cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(training_job)
        exact_grid(root/'zero_validation',zero,stage1.TIMELINE,10,RATES,[101],devices)
        while not future.done():
            if (root/'training_progress.json').exists():
                progress=old.read(root/'training_progress.json')
                print('TRAINING PROGRESS',progress['round'],flush=True)
            time.sleep(30)
        future.result()
    policies={f'{r}_seed{s}':dict(route=r,path=root/'training'/r/f'seed{s}'/'best_positive.pt') for r,s in CONFIGURATIONS}
    exact_grid(root/'trained_validation',policies,stage1.TIMELINE,10,RATES,[101],devices)
    selected=select(root)
    tests=dict(zero)
    tests.update({r+'_trained':dict(route=r,path=Path(selected[r]['policy'])) for r in ROUTES})
    exact_grid(root/'paired_test',tests,TEST_TIMELINE,30,[5,13,25],[1,2,3],devices)
    summarize_test(root)
    print('PIPELINE COMPLETE',root,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','train','exact','summarize'])
    p.add_argument('--root',type=Path,default=OUTPUT)
    p.add_argument('--rounds',type=int,default=160)
    p.add_argument('--workers',type=int,default=36)
    p.add_argument('--label')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    a=p.parse_args()
    if a.command=='run': pipeline(a)
    elif a.command=='train': train(a.root,a.workers)
    elif a.command=='exact': exact_case(a.root,a.label,a.device)
    else: summarize_test(a.root)


if __name__=='__main__':
    main()
