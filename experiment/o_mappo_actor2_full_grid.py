#!/usr/bin/env python3
"""Complete the approved actor2 baseline grid without changing other schemes.

Reuse the 18 compatible paired-test cases, run the remaining 36, and audit
the complete 18-load/3-seed grid against the paper's paired traffic. The fixed
validation-selected policy is used at every load; this entry does not train.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.revision_pipeline import atomic_json, digest

DEPTH = ROOT / 'experiment/results/o_mappo_actor_depth_20260924'
GRID = ROOT / 'experiment/results/revision_directional_20260922/grid'
DEFAULT = ROOT / 'experiment/results/o_mappo_actor2_full_grid_20260924'
RATES = list(range(1, 36, 2))
SEEDS = [1, 2, 3]


def read(path):
    return json.loads(Path(path).read_text())


def audit_results(root, grid=GRID):
    """Recompute every figure metric from raw arrays and verify paired traffic."""
    protocol = read(root / 'protocol.json')
    policy = next(iter(protocol['policies'].values()))
    if digest(Path(policy['path'])) != policy['sha256']:
        raise ValueError('Policy changed')
    if digest(Path(protocol['timeline'])) != protocol['timeline_sha256']:
        raise ValueError('Timeline changed')
    for name, sha in protocol['code'].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'Frozen evaluation code changed: {name}')
    rows, provenance = [], []
    for rate in RATES:
        for seed in SEEDS:
            path = root / 'runs' / f'actor2_seed33_selected_rate{rate}_seed{seed}.json'
            row = read(path)
            reference = read(grid / 'runs' / f'meet_cobra_rate{rate}_seed{seed}.json')
            assert row['frames'] == reference['frames'] == 300
            assert row['checkpoint_sha256'] == policy['sha256']
            assert (row['rate_mbps'], row['seed']) == (rate, seed)
            assert row['traffic_sha256'] == reference['traffic_sha256']
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as raw:
                q, frame = raw['queue_bits'], raw['queue_frame']
                assert np.isfinite(q).all() and (q >= 0).all()
                delay = q[frame >= 2] / (rate * 1e6) * 1000
                violation = np.array([np.mean(q[frame == f] > rate * 1e6 * .02)
                                      for f in range(300)])
                np.testing.assert_allclose(violation, raw['violation_probability'], atol=1e-12)
                counts = raw['association_counts'][2:]
                actual = dict(power_w=raw['energy_j'][2:].mean() / .1,
                    violation_percent=100 * violation[2:].mean(),
                    p90_proxy_ms=np.percentile(delay, 90), p99_proxy_ms=np.percentile(delay, 99),
                    macro_association_percent=100 * counts[:, 0].sum() / counts.sum())
                for name, value in actual.items():
                    np.testing.assert_allclose(value, row['metrics'][name], rtol=1e-10, atol=1e-10)
            rows.append(row)
            provenance.append(dict(file=str(path), sha256=digest(path),
                                   raw_sha256=digest(path.with_suffix('.npz'))))
    groups = []
    for rate in RATES:
        cases = [r for r in rows if r['rate_mbps'] == rate]
        metrics = {}
        for key in cases[0]['metrics']:
            values = [r['metrics'][key] for r in cases]
            metrics[key] = dict(mean=float(np.mean(values)), minimum=min(values), maximum=max(values),
                                per_seed=values)
        groups.append(dict(rate_mbps=rate, metrics=metrics))
    result = dict(cases=len(rows), rates=RATES, seeds=SEEDS, seconds=30, warmup_frames=2,
        policy_sha256=policy['sha256'], protocol_sha256=digest(root / 'protocol.json'),
        paired_with=str(grid), raw_metrics_recomputed=True, groups=groups, provenance=provenance)
    atomic_json(root / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--devices', default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6')
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.audit_only:
        print('AUDITED', audit_results(args.root)['cases'], flush=True)
        return
    source = DEPTH / 'paired_test'
    previous = read(source / 'protocol.json')
    policy = Path(read(DEPTH / 'selected_policies.json')['actor2'])
    paper = read(GRID / 'protocol.json')
    assert paper['rates'] == RATES and paper['seeds'] == SEEDS
    assert paper['warmup_frames'] == 2 and paper['ho_ms'] == 10
    assert previous['timeline_sha256'] == paper['cache_sha256']
    assert previous['seconds'] == 30 and previous['seeds'] == SEEDS
    for name, sha in previous['code'].items():
        assert digest(ROOT / name) == sha, name
    # Everything apart from the deliberately changed O-MAPPO implementation
    # and its launcher must match the original system grid.
    allowed = {'utils/o_mappo.py', 'utils/o_mappo_sim.py', 'experiment/revision_pipeline.py'}
    for name, sha in paper['code_sha256'].items():
        if name not in allowed:
            assert digest(ROOT / name) == sha, name
    assert digest(policy) == previous['policies']['actor2_seed33_selected']['sha256']
    frozen = dict(policy=str(policy), policy_sha256=digest(policy),
        source_protocol=str(source / 'protocol.json'), source_sha256=digest(source / 'protocol.json'),
        paper_protocol=str(GRID / 'protocol.json'), paper_sha256=digest(GRID / 'protocol.json'),
        rates=RATES, seeds=SEEDS, seconds=30, warmup_frames=2,
        code={str(x.relative_to(ROOT)): digest(x) for x in
              [Path(__file__), ROOT / 'experiment/evaluate_o_mappo_h32_retrained.py',
               ROOT / 'experiment/revision_pipeline.py']})
    manifest = args.root / 'extension_protocol.json'
    if manifest.exists() and read(manifest) != frozen:
        raise ValueError('Extension protocol changed; use a new root')
    atomic_json(manifest, frozen)
    destination = args.root / 'runs'
    destination.mkdir(exist_ok=True)
    reused = []
    for src in sorted((source / 'runs').glob('actor2_seed33_selected_*.json')):
        row = read(src)
        assert row['checkpoint_sha256'] == digest(policy) and row['frames'] == 300
        ref = read(GRID / 'runs' / f"meet_cobra_rate{row['rate_mbps']}_seed{row['seed']}.json")
        assert row['traffic_sha256'] == ref['traffic_sha256']
        for file in (src.with_suffix('.npz'), src):
            target = destination / file.name
            if target.exists():
                assert digest(target) == digest(file)
            else:
                temp = target.with_suffix(target.suffix + '.copying')
                shutil.copyfile(file, temp)
                os.replace(temp, target)
        reused.append(dict(source=str(src), sha256=digest(src), raw_sha256=digest(src.with_suffix('.npz'))))
    assert len(reused) == 18
    atomic_json(args.root / 'reuse.json', dict(cases=reused))
    print('REUSED 18 cases; completing 36 additional cases', flush=True)
    command = [sys.executable, '-u', str(ROOT / 'experiment/evaluate_o_mappo_h32_retrained.py'),
        '--root', str(args.root), '--timeline', previous['timeline'], '--policies', str(policy),
        '--rates', ','.join(map(str, RATES)), '--seeds', '1,2,3', '--seconds', '30',
        '--devices', args.devices]
    subprocess.run(command, check=True, cwd=ROOT)
    summary = audit_results(args.root)
    print('COMPLETE AND AUDITED', summary['cases'], flush=True)


if __name__ == '__main__':
    main()
