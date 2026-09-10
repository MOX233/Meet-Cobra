#!/usr/bin/env python3
"""Validate saved paired HO runs and plot experimental results (not paper figures)."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

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
    for record in records:
        rate, seed, duration, method = (record[k] for k in ('rate_mbps', 'seed', 'ho_ms', 'method'))
        name = f'rate{rate:g}_seed{seed}_ho{duration:g}ms_{method}'
        with np.load(root / 'raw' / (name + '.npz')) as data:
            for field in data.files:
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
            if duration == 0:
                key = (rate, seed)
                if key in zero_pairs:
                    for field in data.files:
                        np.testing.assert_array_equal(data[field], zero_pairs[key][field])
                else:
                    zero_pairs[key] = {field: data[field].copy() for field in data.files}
        traffic_hashes[(rate, seed)].add(record['traffic_sha256'])
        grouped[(rate, duration, method)].append(record['metrics'])
    assert all(len(hashes) == 1 for hashes in traffic_hashes.values())
    validation = dict(runs=expected, source_hashes='matched', finite_outputs='passed',
                      zero_interruption_pairs='bitwise_identical_all_npz_arrays',
                      independent_of_policy_traffic_hashes='matched',
                      queue_violation_recomputation='exact',
                      energy_vs_allocated_rbs='matched', physical_rb_caps='passed',
                      blocked_slots_vs_actual_handovers='matched',
                      blocked_service_and_pilots='asserted during each simulation')
    (root / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
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
            if row == 0:
                ax.set_title(f'{rate:g} Mbps per vehicle')
            if row == 2:
                ax.set_xlabel('HO interruption (ms)')
            if column == 0:
                ax.set_ylabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    fig.suptitle('Paired next-frame Oracle evaluation; error bars show seed standard deviation', y=.945, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(root / 'comparison.pdf', bbox_inches='tight')
    fig.savefig(root / 'comparison.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
