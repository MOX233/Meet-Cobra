#!/usr/bin/env python3
"""Isolated R1 software check; never a replacement for paper experiments.

Uses a small subset of the retained test input for *software testing only*.
The tiny trained predictors and their scores are not scientific results.
Run from the repository's sionna environment. Output must be a NEW directory
directly under experiment/results with a reproduction_smoke_ prefix.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLBACKEND'] = 'Agg'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protected_snapshot():
    paths = set((ROOT/'latexCodes/revision1').rglob('*'))
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    paths.update(ROOT/p for p in tracked if Path(p).suffix in ('.py', '.tex', '.bib', '.cls'))
    assets = json.loads((ROOT/'configs/paper_r1_assets.json').read_text())
    for value in assets['models'].values():
        path = ROOT/value
        paths.update(path.rglob('*') if path.is_dir() else [path])
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--paper-plots', action='store_true', help='Also render retained formal results into scratch output')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.parent != ROOT/'experiment/results' or not out.name.startswith('reproduction_smoke_'):
        parser.error('Use a new experiment/results/reproduction_smoke_NAME directory')
    out.mkdir(exist_ok=False)
    (out/'logs').mkdir()
    os.environ['MPLCONFIGDIR'] = str(out/'mpl-cache')
    before = protected_snapshot()
    report = dict(purpose='Software checks only; test subset is NOT a new training/evaluation dataset',
                  command=sys.argv, device=args.device, checks=[], passed=False)
    started = time.monotonic()

    def run(name, script, *argv):
        command = [sys.executable, '-B', '-u', str(ROOT/script), *map(str, argv)]
        print('START', name, flush=True)
        begin = time.monotonic()
        with (out/'logs'/f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report['checks'].append(dict(name=name, command=command, returncode=result.returncode,
                                     seconds=time.monotonic()-begin))
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out}/logs/{name}.log')
        print('PASS', name, flush=True)

    try:
        import numpy as np
        import torch
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; run on the GPU-enabled host')
        torch.set_num_threads(1)
        if args.device.startswith('cuda'):
            report['gpu'] = torch.cuda.get_device_name(torch.device(args.device))
        from experiment.revision_training import DEFAULT_TEST
        with DEFAULT_TEST.open('rb') as stream:
            source = pickle.load(stream)
        frames = sorted(source)[:21]
        ids = sorted(set.intersection(*(set(source[f]) for f in frames)), key=str)[:32]
        if len(ids) < 12:
            raise ValueError('Insufficient continuous vehicle trajectories for smoke split')
        timeline = {f: {v: source[f][v] for v in ids} for f in frames}
        del source
        prepared = out/'prepared_subset.pkl'
        with prepared.open('wb') as stream:
            pickle.dump(timeline, stream, protocol=4)
        report['input'] = dict(source=str(DEFAULT_TEST), subset_sha256=sha(prepared),
                               frames=len(frames), vehicles=len(ids), seconds=frames[-1]-frames[0])
        data, split = out/'training_data.npz', out/'vehicle_split.npz'
        run('preprocess', 'experiment/revision_training.py', 'data', '--source', prepared, '--output', data)
        run('split', 'experiment/vehicle_split.py', '--data', data, '--output', split, '--seed', 20)
        for task in ('beam', 'desired_gain'):
            stage1, stage2 = out/'stage1'/task, out/'stage2'/task
            run(task+'_finite', 'experiment/train_finite_window_vehicle_split.py',
                '--data', data, '--split-file', split, '--output', stage1, '--task', task,
                '--device', args.device, '--epochs', 1, '--batch-size', 512, '--max-samples', 64)
            run(task+'_stateful', 'experiment/train_stateful_tbptt.py',
                '--data', data, '--split-file', split, '--output', stage2, '--task', task,
                '--device', args.device, '--epochs', 1, '--patience', 2, '--batch-size', 128,
                '--chunk-length', 10, '--learning-rate', '1e-4', '--max-trajectories', 8,
                '--initialization', 'checkpoint', '--initial-checkpoint', stage1/'best.pth')
        training = ['train', '--data', data, '--split-file', split, '--output', out/'interfering_training',
                    '--device', args.device, '--stage1-epochs', 1, '--stage2-epochs', 1,
                    '--batch-size', 128, '--max-samples', 64, '--max-trajectories', 8, '--resume']
        run('interfering_stop', 'experiment/revision_training.py', *training, '--stop-after-epochs', 1)
        run('interfering_resume', 'experiment/revision_training.py', *training)
        run('assemble', 'experiment/revision_training.py', 'assemble', '--base-models', out/'stage2',
            '--training', out/'interfering_training', '--output', out/'models')
        cache_args = ['cache', '--models', out/'models', '--source', prepared, '--output', out/'predictions.pkl',
                      '--device', args.device, '--start', frames[0], '--end', frames[-1]]
        run('cache', 'experiment/revision_training.py', *cache_args)
        cache_sha = sha(out/'predictions.pkl')
        run('cache_idempotence', 'experiment/revision_training.py', *cache_args)
        assert sha(out/'predictions.pkl') == cache_sha
        methods = 'meet_cobra,oracle_mc,reactive_obra,wo_gap_ho,wo_pet_bf,wo_otr_ra'
        run('grid_prepare', 'experiment/revision_pipeline.py', 'prepare', '--root', out/'grid',
            '--cache', out/'predictions.pkl', '--methods', methods, '--rates', 13, '--seeds', 1,
            '--backend', 'cuda' if args.device.startswith('cuda') else 'cpu', '--reactive-input', 'current', '--allow-smoke')
        run('grid_run', 'experiment/revision_pipeline.py', 'run', '--root', out/'grid', '--devices', args.device)
        run('grid_resume', 'experiment/revision_pipeline.py', 'run', '--root', out/'grid', '--devices', args.device)
        run('grid_summary', 'experiment/revision_pipeline.py', 'summarize', '--root', out/'grid')
        # Current literature baselines have separate simulators; the two old
        # implementations in revision_pipeline are deliberately not substituted.
        from experiment import revision_pipeline as grid
        from experiment import mts_hierarchical_tracking as mts
        from experiment import o_mappo_predicted_cross5 as predicted
        from utils.o_mappo import OMAPPPolicy
        mts.shared.single_thread_solvers()
        from utils.compiled_matching import km_algorithm_compiled
        mts.alg.km_algorithm = km_algorithm_compiled
        asset = json.loads((ROOT/'configs/paper_r1_assets.json').read_text())
        # Tiny random-initialized predictors can send every MTS vehicle to the
        # macro BS, making a beam-search check vacuous. Exercise the two current
        # baselines with the selected formal predictors on the SAME small trace.
        run('cache_selected_models', 'experiment/revision_training.py', 'cache',
            '--models', ROOT/asset['models']['nn_bundle'], '--source', prepared,
            '--output', out/'selected_predictions.pkl', '--device', args.device,
            '--start', frames[0], '--end', frames[-1])
        with (out/'selected_predictions.pkl').open('rb') as stream:
            timeline = pickle.load(stream)
        policy = OMAPPPolicy.load(str(ROOT/asset['models']['o_mappo']))
        print('START current_O_MAPPO_and_one_PPO_update', flush=True)
        result, traffic, memory, diagnostic = predicted.simulate(timeline, policy, 13, 1, args.device, learn=True)
        metrics, raw = grid.extract(predicted.paper_args(13e6), result, traffic, 2)
        assert len(memory.transitions) > 0
        previous_count = policy.update_count
        actor_before = {k: v.clone() for k, v in policy.actor.state_dict().items()}
        update = policy.update(memory)
        assert policy.update_count > previous_count
        assert any(not torch.equal(v, actor_before[k]) for k, v in policy.actor.state_dict().items())
        assert all(np.isfinite(v) for v in update.values() if isinstance(v, (float, int)))
        report['o_mappo'] = dict(metrics=metrics, transitions=len(memory.transitions), update=update,
                                  traffic_sha256=traffic['sha256'], weights_saved=False)
        print('PASS current_O_MAPPO_and_one_PPO_update', flush=True)
        print('START current_MTS', flush=True)
        params = mts.shared.paper_args(13e6)
        params.device = torch.device('cpu')
        mts_traffic = mts.make_paired_traffic(params, timeline, 1)
        assert mts_traffic['sha256'] == traffic['sha256']
        np.random.seed(1)
        adapter, service = mts.BeamAdapter('hier32_cross5'), []
        with ExitStack() as stack:
            adapter.activate(stack)
            result = mts.sim.run_sim_mts_report(params, mts.shared.MICRO_BS_LOCATIONS, timeline,
                mts.candidate_configs()['pressure_early'], mts_traffic, seed=1, physics_device=args.device,
                ho_interruption_ms=10, k=5, directional_service=True,
                predicted_ra_interference=True, service_diagnostics=service)
        metrics, raw = grid.extract(params, result, mts_traffic, 2)
        assert len(service) == len(frames)-1
        for row in adapter.frames:
            assert row['probe_count'] == 32*row['acquisition_count']+5*row['five_probe_slots']
        acquisitions = sum(row['acquisition_count'] for row in adapter.frames)
        tracking = sum(row['five_probe_slots'] for row in adapter.frames)
        assert acquisitions > 0 and tracking > 0, 'No actual MTS beam search exercised'
        report['mts'] = dict(metrics=metrics, frames=len(service), traffic_sha256=mts_traffic['sha256'],
                             acquisition_count=acquisitions, five_probe_slots=tracking)
        print('PASS current_MTS', flush=True)
        if args.paper_plots:
            run('paper_system_plots', 'experiment/plot_revision_system_results.py',
                '--figures', out/'paper_figures', '--report', out/'paper_plot_report')
            unified = ROOT/'experiment/results/stateful_tbptt_unified_split_20260913'
            current = ROOT/'experiment/results/revision_directional_20260922'
            run('paper_training_plots', 'experiment/plot_stateful_training_curves.py',
                '--stage1-results', unified/'stage1_finite_window', '--stage2-results', unified/'stage2_stateful_tbptt',
                '--interfering-stage1-results', current/'training/stage1',
                '--interfering-stage2-results', current/'training/stage2',
                '--summary', out/'fig4_summary.json', '--figures', out/'paper_figures')
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        after = protected_snapshot()
        report['protected_files_unchanged'] = before == after
        report['protected_file_count'] = len(before)
        report['elapsed_seconds'] = time.monotonic()-started
        report['passed'] = report['passed'] and before == after
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if before != after:
            raise RuntimeError('Protected source, selected models or submission files changed during check')
    print('PASS', out/'report.json', flush=True)


if __name__ == '__main__':
    main()
