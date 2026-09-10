#!/usr/bin/env python3
"""Validate saved paired HO runs and plot experimental results (not paper figures)."""

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiment/results_ho_interruption'))
    options = parser.parse_args()
    root = options.output
    protocol = json.loads((root / 'protocol.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((root / 'runs').glob('*.json'))]
    expected = len(protocol['rates_mbps']) * len(protocol['seeds']) * len(protocol['durations_ms']) * 2
    assert len(records) == expected, (len(records), expected)
    repository = Path(__file__).resolve().parents[1]
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((repository / name).read_bytes()).hexdigest() == digest, name
    args = protocol['args']
    dt = args['slot_len'] * args['slots_per_frame']
    warmup = protocol['warmup_frames']
    grouped = collections.defaultdict(list)
    zero_pairs = {}
    traffic_hashes = collections.defaultdict(set)
    bs_rows = []
    for record in records:
        rate, seed, duration, method = (record[k] for k in ('rate_mbps', 'seed', 'ho_ms', 'method'))
        name = f'rate{rate:g}_seed{seed}_ho{duration:g}ms_{method}'
        with np.load(root / 'raw' / (name + '.npz')) as archive:
            # Cache arrays once, rather than decompress the full queue matrix
            # again on every per-frame indexing operation.
            data = {field: archive[field] for field in archive.files}
            for field in data:
                if np.issubdtype(data[field].dtype, np.number):
                    assert np.isfinite(data[field]).all(), (name, field)
            assert np.all(data['queue_bits'] >= 0)
            rb_limit = np.array([args['num_RB_macro']] + [args['num_RB_micro']] * 4)
            assert np.all(data['rb_per_bs'] <= rb_limit + 1e-9)
            assert np.all(data['rb_per_bs'] >= 0)
            powers = np.array([args['p_macro']] + [args['p_micro']] * 4)
            np.testing.assert_allclose(data['energy_j'], data['rb_per_bs'] @ powers * dt,
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(data['blocked_vehicle_slots'],
                                           data['handover_count'] * duration / (1000 * args['slot_len']))
            violation = []
            for frame in range(protocol['frame_count']):
                q = data['queue_bits'][data['queue_frame_index'] == frame]
                threshold = rate * 1e6 * args['lat_slot_ub'] * args['slot_len']
                violation.append(np.mean(q > threshold))
            np.testing.assert_allclose(violation, data['violation_probability'], atol=0, rtol=0)
            np.testing.assert_allclose(np.mean(violation[warmup:]) * 100,
                                       record['metrics']['violation_percent'], atol=0, rtol=0)
            diagnostics = json.loads((root / 'associations' / (name + '.json')).read_text())
            frame_index = data['queue_frame_index']
            serving_bs = np.array([diagnostics[int(i)]['association'][str(veh)]
                                   for i, veh in zip(frame_index, data['queue_vehicle'])])
            by_bs = np.zeros((protocol['frame_count'], 5))
            np.add.at(by_bs, (frame_index, serving_bs),
                      (data['queue_bits'] > threshold).sum(axis=1))
            contributions = 100 * (by_bs[warmup:] /
                                   data['active_vehicle_slots'][warmup:, None]).mean(axis=0)
            bs_power = (data['rb_per_bs'][warmup:] * powers).mean(axis=0)
            np.testing.assert_allclose(contributions.sum(), record['metrics']['violation_percent'])
            for bs in range(5):
                bs_rows.append(dict(rate_mbps=rate, seed=seed, ho_ms=duration, method=method,
                                    bs=bs, violation_contribution_pp=float(contributions[bs]),
                                    power_w=float(bs_power[bs])))
            if duration == 0:
                key = (rate, seed)
                if key in zero_pairs:
                    for field in data:
                        np.testing.assert_array_equal(data[field], zero_pairs[key][field])
                else:
                    zero_pairs[key] = {field: data[field].copy() for field in data}
        traffic_hashes[(rate, seed)].add(record['traffic_sha256'])
        grouped[(rate, duration, method)].append(record['metrics'])
    assert all(len(hashes) == 1 for hashes in traffic_hashes.values())
    validation = dict(runs=expected, source_hashes='matched', finite_outputs='passed',
                      zero_interruption_pair_count=len(zero_pairs),
                      zero_interruption_pairs=('bitwise_identical_all_npz_arrays' if zero_pairs
                                               else 'not_applicable_no_zero_duration'),
                      independent_of_policy_traffic_hashes='matched',
                      queue_violation_recomputation='exact',
                      energy_vs_allocated_rbs='matched', frame_average_rb_caps='passed',
                      blocked_slots_vs_actual_handovers='matched',
                      blocked_service_and_pilots='asserted during each simulation')
    (root / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    environment = dict(python=sys.version, platform=platform.platform(),
                       packages={p: importlib.metadata.version(p)
                                 for p in ('numpy', 'scipy', 'torch', 'matplotlib', 'PuLP')})
    (root / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    with (root / 'per_bs_diagnostics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bs_rows[0]))
        writer.writeheader()
        writer.writerows(bs_rows)
    rates, durations = sorted(protocol['rates_mbps']), sorted(protocol['durations_ms'])
    metrics = [('power_w', 'System transmit power (W)'),
               ('violation_percent', 'Violation probability (%)'),
               ('handovers_per_vehicle_s', 'Handovers per vehicle-second')]
    fig, axes = plt.subplots(3, len(rates), figsize=(4 * len(rates), 8), squeeze=False)
    for column, rate in enumerate(rates):
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, column]
            for method, marker, color, display in (
                    ('original', 'o', '#3b6fb6', 'Original GAP-HO'),
                    ('capacity_corrected', 's', '#c24b40', 'Capacity-corrected GAP-HO')):
                samples = [[p[metric] for p in grouped[(rate, d, method)]] for d in durations]
                ax.errorbar(durations, [np.mean(s) for s in samples],
                            yerr=[np.std(s, ddof=1) if len(s) > 1 else 0 for s in samples],
                            marker=marker, color=color, label=display, capsize=3)
            ax.grid(alpha=.25)
            ax.set_xticks(durations)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            if metric == 'violation_percent' and all(
                    p[metric] == 0 for d in durations
                    for m in ('original', 'capacity_corrected')
                    for p in grouped[(rate, d, m)]):
                ax.set_ylim(0, .01)
            if row == 0:
                ax.set_title(f'{rate:g} Mbps per vehicle')
            if row == 2:
                ax.set_xlabel('HO interruption (ms)')
            if column == 0:
                ax.set_ylabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=1 if len(rates) == 1 else 2, frameon=False)
    title = ('Paired next-frame Oracle evaluation\nError bars show seed standard deviation'
             if len(rates) == 1 else
             'Paired next-frame Oracle evaluation; error bars show seed standard deviation')
    fig.suptitle(title, y=.925 if len(rates) == 1 else .945, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .87 if len(rates) == 1 else .92))
    fig.savefig(root / 'comparison.pdf', bbox_inches='tight')
    fig.savefig(root / 'comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
