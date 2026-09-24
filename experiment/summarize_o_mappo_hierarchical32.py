#!/usr/bin/env python3
"""Paired comparison against the preserved directional-service experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiment.revision_pipeline import read, digest, validate, completed
from experiment.revision_training import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'experiment/results/o_mappo_hierarchical32_20260924')
    parser.add_argument('--reference', type=Path, default=ROOT / 'experiment/results/revision_directional_20260922/grid')
    args = parser.parse_args()
    p, sha = validate(args.root / 'grid')
    old_p = read(args.reference / 'protocol.json')
    for field in ('cache_sha256', 'policy_sha256', 'warmup_frames', 'ho_ms', 'backend'):
        assert p[field] == old_p[field], field
    warmup = p['warmup_frames']
    rows = []
    for rate in p['rates']:
        for seed in p['seeds']:
            name = f'o_mappo_rate{rate}_seed{seed}'
            assert completed(args.root / 'grid', name, sha), name
            new = read(args.root / 'grid/runs' / f'{name}.json')
            old = read(args.reference / 'runs' / f'{name}.json')
            meet = read(args.reference / 'runs' / f'meet_cobra_rate{rate}_seed{seed}.json')
            assert new['traffic_sha256'] == old['traffic_sha256'] == meet['traffic_sha256']
            assert new['frames'] == old['frames'] == meet['frames'] == 300
            for record, key in ((old, name), (meet, f'meet_cobra_rate{rate}_seed{seed}')):
                assert digest(args.reference / 'raw' / f'{key}.npz') == record['raw_sha256']
            record = dict(rate_mbps=rate, seed=seed, traffic_sha256=new['traffic_sha256'])
            for method, row in (('exhaustive256', old), ('hierarchical32', new), ('meet_cobra', meet)):
                record[method] = dict(row['metrics'])
                raw_dir = args.root / 'grid/raw' if method == 'hierarchical32' else args.reference / 'raw'
                raw_name = f'meet_cobra_rate{rate}_seed{seed}' if method == 'meet_cobra' else name
                with np.load(raw_dir / f'{raw_name}.npz') as raw:
                    record[method]['handover_count'] = int(raw['handover_count'][warmup:].sum())
            diag = read(args.root / 'grid/diagnostics' / f'{name}.json')['beam_search']
            diag = [d for d in diag if d['frame'] >= warmup]
            loss = np.array([d['exhaustive_gain_db'] - d['gain_db'] for d in diag])
            assert len(loss) == new['metrics']['acquisition_count'] and (loss >= -1e-9).all()
            record['acquisition_quality'] = dict(count=len(loss),
                exact_pair_percent=100 * float(np.mean([(d['tx'], d['rx']) ==
                    (d['exhaustive_tx'], d['exhaustive_rx']) for d in diag])),
                within_1db_percent=100 * float(np.mean(loss <= 1)),
                mean_loss_db=float(loss.mean()), median_loss_db=float(np.median(loss)),
                p90_loss_db=float(np.percentile(loss, 90)))
            record['changes'] = dict(
                power_percent=100 * (new['metrics']['power_w'] / old['metrics']['power_w'] - 1),
                violation_percentage_points=new['metrics']['violation_percent'] - old['metrics']['violation_percent'],
                violation_relative_percent=100 * (new['metrics']['violation_percent'] /
                    old['metrics']['violation_percent'] - 1),
                pilot_percent=100 * (new['metrics']['pilots_per_vehicle_slot'] /
                    old['metrics']['pilots_per_vehicle_slot'] - 1))
            rows.append(record)

    # A complete replay verifies that the default branch retained the old outputs.
    _, reg_sha = validate(args.root / 'legacy_check')
    reg_name = 'o_mappo_rate15_seed1'
    assert completed(args.root / 'legacy_check', reg_name, reg_sha)
    with np.load(args.root / 'legacy_check/raw' / f'{reg_name}.npz') as replay, \
            np.load(args.reference / 'raw' / f'{reg_name}.npz') as original:
        comparison = {}
        for key in original.files:
            if original[key].dtype.kind in 'US':
                np.testing.assert_array_equal(replay[key], original[key])
            else:
                np.testing.assert_allclose(replay[key], original[key], rtol=1e-12, atol=1e-7)
            comparison[key] = bool(np.array_equal(replay[key], original[key]))
    atomic_json(args.root / 'comparison.json', dict(rows=rows,
        legacy_replay_bitwise_equal=comparison,
        reference_protocol_sha256=digest(args.reference / 'protocol.json'),
        hierarchical_protocol_sha256=sha))
    fields = ('power_w', 'violation_percent', 'mean_proxy_ms', 'p90_proxy_ms', 'p99_proxy_ms',
              'pilots_per_vehicle_slot', 'handover_count', 'macro_association_percent')
    with (args.root / 'comparison.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rate_mbps', 'seed', 'method', *fields])
        for row in rows:
            for method in ('exhaustive256', 'hierarchical32', 'meet_cobra'):
                writer.writerow([row['rate_mbps'], row['seed'], method, *[row[method][k] for k in fields]])
    print(json.dumps(dict(rows=rows, legacy_replay_bitwise_equal=comparison), indent=2))

    # Diagnostic figures only: do not overwrite manuscript figures.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    for method, label, marker, color in (
        ('exhaustive256', 'O-MAPPO: exhaustive 256', 's', '#d55e00'),
        ('hierarchical32', 'O-MAPPO: hierarchical 32', 'o', '#0072b2'),
        ('meet_cobra', 'MEET-COBRA', '^', '#009e73')):
        for ax, field in zip(axes, ('power_w', 'violation_percent')):
            ax.plot([r['rate_mbps'] for r in rows], [r[method][field] for r in rows],
                    label=label, marker=marker, color=color, linewidth=1.5, markersize=5)
    for ax in axes:
        ax.set_xlabel('Arrival rate (Mbps)')
        ax.set_xticks(p['rates'])
        ax.grid(True, alpha=.22, which='both')
    axes[0].set_ylabel('Average transmit power (W)')
    axes[1].set_ylabel('Latency violation probability (%)')
    axes[1].set_yscale('log')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.savefig(args.root / 'comparison.pdf')
    fig.savefig(args.root / 'comparison.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
