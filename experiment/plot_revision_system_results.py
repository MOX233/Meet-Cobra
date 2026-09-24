#!/usr/bin/env python3
"""Audit the completed directional-service grid and draw publication figures.

This reads, but never modifies, frozen simulations and their original aggregate.
New PDFs/PNGs use the suffix _revision1; submitted and older _WBL plots survive.
Run with --audit to verify all hashes and independently reconstruct raw metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / 'experiment/results/revision_directional_20260922/grid'
FIGURES = ROOT / 'latexCodes/figures'
OMAPPO_RESULTS = ROOT / 'experiment/results/o_mappo_actor2_full_grid_20260924'
REPORT = OMAPPO_RESULTS / 'paper_figures'
METHODS = ('meet_cobra', 'oracle_mc', 'reactive_obra', 'o_mappo',
           'mts_report', 'wo_gap_ho', 'wo_pet_bf', 'wo_otr_ra')
STYLE = {
    'meet_cobra': ('MEET-COBRA', '#c32f27', 'o', '-', 1.8),
    'oracle_mc': ('Oracle-MC', '#222222', 's', '--', 1.2),
    'reactive_obra': ('Reactive-OBRA', '#bc7a00', '^', '-', 1.1),
    'o_mappo': ('O-MAPPO-adapted', '#7046a3', 'D', '-.', 1.2),
    'mts_report': ('MTS-GS-HBF-adapted', '#008477', 'P', '-', 1.3),
    'wo_gap_ho': ('w/o GAP-HO', '#3873b3', 'v', '--', 1.0),
    'wo_pet_bf': ('w/o PET-BF', '#ae718d', '<', ':', 1.1),
    'wo_otr_ra': ('w/o OTR-RA', '#68776a', '>', '-.', 1.0),
}
METRICS = ('power_w', 'violation_percent', 'p90_proxy_ms', 'p99_proxy_ms',
           'macro_association_percent')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def audit(grid, protocol, rows):
    """Check frozen provenance, paired traffic, and every saved raw metric."""
    sha = digest(grid / 'protocol.json')
    for name, expected in protocol['code_sha256'].items():
        assert digest(ROOT / name) == expected, f'Changed frozen source: {name}'
    for name in ('cache', 'policy'):
        assert digest(Path(protocol[name])) == protocol[name + '_sha256']
    for model in protocol['cache_manifest']['models'].values():
        assert digest(Path(model['checkpoint'])) == model['sha256']
    traffic = {}
    for i, row in enumerate(rows, 1):
        name = f"{row['method']}_rate{row['rate_mbps']}_seed{row['seed']}"
        assert row['protocol_sha256'] == sha and row['frames'] == 300, name
        pair = (row['rate_mbps'], row['seed'])
        assert traffic.setdefault(pair, row['traffic_sha256']) == row['traffic_sha256']
        for directory, field, suffix in (
                ('raw', 'raw_sha256', '.npz'), ('diagnostics', 'diagnostics_sha256', '.json')):
            assert digest(grid / directory / (name + suffix)) == row[field], name
        warmup = protocol['warmup_frames']
        with np.load(grid / 'raw' / (name + '.npz'), allow_pickle=False) as raw:
            q = raw['queue_bits']
            frames = raw['queue_frame']
            rate = row['rate_mbps'] * 1e6  # Frozen protocol: common per-vehicle mean.
            assert np.isfinite(q).all() and (q >= 0).all(), name
            delay = q[frames >= warmup].ravel() * (1000 / rate)
            p90, p99 = np.percentile(delay, [90, 99])
            frame_u = np.array([np.mean(q[frames == f] > rate * .020)
                                for f in range(row['frames'])])
            np.testing.assert_allclose(frame_u, raw['violation_probability'], atol=1e-12)
            counts = raw['association_counts'][warmup:]
            actual = dict(power_w=raw['energy_j'][warmup:].mean()/protocol['frame_s'],
                          violation_percent=100*frame_u[warmup:].mean(),
                          p90_proxy_ms=p90, p99_proxy_ms=p99,
                          macro_association_percent=100*counts[:, 0].sum()/counts.sum())
            for metric, value in actual.items():
                np.testing.assert_allclose(value, row['metrics'][metric], rtol=1e-10,
                                           atol=1e-10, err_msg=f'{name}: {metric}')
            if row['method'] == 'oracle_mc':
                for field, metric in (('p2_relaxed_power_w', 'oracle_cr_lb_power_reference_w'),
                                      ('p2_relaxation_feasible', 'oracle_cr_lb_feasible_fraction')):
                    np.testing.assert_allclose(raw[field][warmup:].mean(), row['metrics'][metric])
        diagnostic = read(grid / 'diagnostics' / (name + '.json'))
        service = diagnostic['service'][warmup:]
        for numerator, denominator, metric in (
                ('approximate_interference_sum', 'directional_interference_sum', 'interference_ratio'),
                ('approximate_service_bits', 'directional_service_bits', 'service_capacity_ratio')):
            value = sum(x[numerator] for x in service) / sum(x[denominator] for x in service)
            np.testing.assert_allclose(value, row['comparison'][metric], rtol=1e-10)
        if i % 24 == 0 or i == len(rows):
            print(f'AUDIT {i}/{len(rows)} raw metrics, hashes, paired traffic and diagnostics OK', flush=True)
    return dict(cases=len(rows), protocol_sha256=sha, paired_traffic_groups=len(traffic),
                raw_metrics_recomputed=True, code_and_input_hashes_verified=True)


def load(grid):
    protocol = read(grid / 'protocol.json')
    assert set(protocol['methods']) == set(METHODS)
    assert protocol['rates'] == list(range(1, 36, 2))
    summary = read(grid / 'aggregate/summary.json')
    assert not summary['missing'] and not summary['smoke']
    assert summary['protocol_sha256'] == digest(grid / 'protocol.json')
    rows = [read(grid / 'runs' / f'{m}_rate{r}_seed{s}.json')
            for m in METHODS for r in protocol['rates'] for s in protocol['seeds']]
    curves = {}
    for method in METHODS:
        curves[method] = {}
        for metric in METRICS:
            a = np.array([[next(x for x in rows if x['method'] == method
                               and x['rate_mbps'] == r and x['seed'] == s)['metrics'][metric]
                           for s in protocol['seeds']] for r in protocol['rates']])
            curves[method][metric] = a
            for rate, value in zip(protocol['rates'], a.mean(axis=1)):
                entry = next(g for g in summary['groups']
                             if g['method'] == method and g['rate_mbps'] == rate)
                np.testing.assert_allclose(value, entry['metrics'][metric]['mean'], rtol=1e-12)
    for rate in protocol['rates']:
        group = next(g for g in summary['groups'] if g['method'] == 'oracle_mc' and g['rate_mbps'] == rate)
        for metric in ('oracle_cr_lb_power_reference_w', 'oracle_cr_lb_feasible_fraction'):
            value = np.mean([r['metrics'][metric] for r in rows
                             if r['method'] == 'oracle_mc' and r['rate_mbps'] == rate])
            np.testing.assert_allclose(value, group['metrics'][metric]['mean'], rtol=1e-12)
    return protocol, summary, rows, curves


def replace_o_mappo(root, protocol, rows, curves):
    """Overlay the approved actor2 data, preserving all seven other schemes."""
    result = read(root / 'summary.json')
    assert result['cases'] == 54 and result['raw_metrics_recomputed']
    assert result['rates'] == protocol['rates'] and result['seeds'] == protocol['seeds']
    assert result['seconds'] == 30 and result['warmup_frames'] == protocol['warmup_frames']
    assert result['protocol_sha256'] == digest(root / 'protocol.json')
    manifest = read(root / 'protocol.json')
    assert manifest['timeline_sha256'] == protocol['cache_sha256']
    replacements = {}
    for record in result['provenance']:
        path = Path(record['file'])
        assert digest(path) == record['sha256']
        assert digest(path.with_suffix('.npz')) == record['raw_sha256']
        row = read(path)
        assert row['checkpoint_sha256'] == result['policy_sha256']
        key = (row['rate_mbps'], row['seed'])
        assert key not in replacements
        replacements[key] = row
    updated = []
    for row in rows:
        if row['method'] != 'o_mappo':
            updated.append(row)
            continue
        new = replacements[row['rate_mbps'], row['seed']]
        assert row['traffic_sha256'] == new['traffic_sha256'] and new['frames'] == row['frames']
        updated.append(dict(new, method='o_mappo'))
    for metric in METRICS:
        curves['o_mappo'][metric] = np.array([
            [replacements[rate, seed]['metrics'][metric] for seed in protocol['seeds']]
            for rate in protocol['rates']])
    return updated, dict(root=str(root), summary_sha256=digest(root / 'summary.json'),
                        policy_sha256=result['policy_sha256'], cases=54,
                        raw_metrics_recomputed=True, paired_traffic_verified=True)


def base_axes(figsize=(3.65, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=.145, right=.98, bottom=.14, top=.72)
    ax.set_xlim(.5, 35.5)
    ax.set_xticks([1, 5, 9, 13, 17, 21, 25, 29, 33, 35])
    ax.set_xlabel(r'Mean arrival rate $\lambda$ (Mbps)')
    ax.grid(True, which='major', color='#dddddd', linewidth=.45, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3, width=.6)
    return fig, ax


def draw(ax, curves, metric, rates):
    lines = {}
    # Draw the proposed scheme last, without hiding the seed variability.
    for method in (*METHODS[1:], METHODS[0]):
        label, color, marker, ls, width = STYLE[method]
        a = curves[method][metric]
        ax.fill_between(rates, a.min(axis=1), a.max(axis=1), color=color,
                        alpha=.13, linewidth=0, zorder=1)
        lines[method], = ax.plot(rates, a.mean(axis=1), label=label, color=color,
                                marker=marker, linestyle=ls, linewidth=width,
                                markersize=3.2, markerfacecolor='white',
                                markeredgewidth=.8, markevery=1, zorder=3 if method == 'meet_cobra' else 2)
    return [lines[m] for m in METHODS]


def save(fig, stem, output):
    for suffix in ('pdf', 'png'):
        fig.savefig(output / (stem + '_revision1.' + suffix), dpi=220)
    plt.close(fig)


def plots(curves, summary, rates, output, *, power_only=False, include_oracle_cr_lb=False):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'axes.linewidth': .6})
    specifications = (
        ('power_w', 'power_comparison_curves', r'Average transmit power $\bar P$ (W)'),
        ('violation_percent', 'violation_prob_comparison_curves', r'Latency constraint violation probability $U$ (%)'),
        ('p90_proxy_ms', 'latency_90th_comparison_curves', r'90th-percentile latency proxy $L_{90}$ (ms)'),
        ('p99_proxy_ms', 'latency_99th_comparison_curves', r'99th-percentile latency proxy $L_{99}$ (ms)'),
        ('macro_association_percent', 'BS0_assoc_ratio_comparison_curves', r'Macro-BS association ratio $\rho_0$ (%)'),
    )
    for metric, stem, ylabel in specifications:
        if power_only and metric != 'power_w':
            continue
        fig, ax = base_axes()
        handles = draw(ax, curves, metric, rates)
        ax.set_ylabel(ylabel)
        if metric == 'power_w':
            # Keep the historical diagnostic reproducible, but omit it from
            # the revised paper: it is not a lower bound on realized power.
            if include_oracle_cr_lb:
                ref = [g for g in summary['groups'] if g['method'] == 'oracle_mc']
                ref.sort(key=lambda g: g['rate_mbps'])
                power = [g['metrics']['oracle_cr_lb_power_reference_w']['mean'] for g in ref]
                line, = ax.plot(rates, power, color='#888888', linestyle=':', linewidth=1.0,
                                label='Oracle-CR-LB (P2 reference)', zorder=2)
                handles.append(line)
                # Crosses identify load points containing infeasible P2 instances.
                bad = np.array([g['metrics']['oracle_cr_lb_feasible_fraction']['mean'] < 1 for g in ref])
                ax.plot(np.asarray(rates)[bad], np.asarray(power)[bad], 'x', color='#777777', markersize=4)
            ax.set_ylim(0, 200)
            ax.set_yticks([0, 40, 80, 120, 160, 200])
        elif metric == 'violation_percent':
            # Unlike flooring values before a log plot, symlog preserves true zeros.
            ax.set_yscale('symlog', linthresh=.01, linscale=.7)
            ax.set_ylim(-.001, 50)
            ax.set_yticks([0, .01, .1, 1, 10, 40], ['0', '0.01', '0.1', '1', '10', '40'])
        elif metric in ('p90_proxy_ms', 'p99_proxy_ms'):
            ax.set_yscale('log')
            ax.set_ylim(.9, 4e4 if metric == 'p99_proxy_ms' else 1e4)
            ax.set_yticks([1, 10, 100, 1000, 10000], ['1', '10', r'$10^2$', r'$10^3$', r'$10^4$'])
            ax.axhline(20, color='#555555', linewidth=.9, linestyle=(0, (4, 3)), zorder=1)
            ax.text(.37, 20, '20 ms', transform=ax.get_yaxis_transform(), ha='center', va='bottom',
                    fontsize=7, color='#444444', bbox=dict(facecolor='white', edgecolor='none', pad=.5))
            ax.minorticks_off()
        else:
            ax.set_ylim(-.6, 40)
            ax.set_yticks([0, 10, 20, 30, 40])
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .99),
                   ncol=2, frameon=False, handlelength=2.4, columnspacing=1.1, labelspacing=.45)
        save(fig, stem, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, default=GRID)
    parser.add_argument('--figures', type=Path, default=FIGURES)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--o-mappo-results', type=Path, default=OMAPPO_RESULTS,
                        help='Approved actor2 full-grid results to replace the historical O-MAPPO curve.')
    parser.add_argument('--legacy-o-mappo', action='store_true',
                        help='Reproduce the historical exhaustive-search O-MAPPO curve instead.')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--power-only', action='store_true',
                        help='Regenerate only the power figure; leave the other figures unchanged.')
    parser.add_argument('--include-oracle-cr-lb', action='store_true',
                        help='Include the archived P2 relaxation reference (omitted by default).')
    args = parser.parse_args()
    protocol, summary, rows, curves = load(args.grid)
    args.report.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    if args.audit:
        audit_result = audit(args.grid, protocol, rows)
        (args.report/'audit.json').write_text(json.dumps(audit_result, indent=2)+'\n')
    else:
        audit_path = args.report/'audit.json'
        audit_result = read(audit_path) if audit_path.exists() else None
        if audit_result and audit_result['protocol_sha256'] != digest(args.grid/'protocol.json'):
            audit_result = None
    replacement = None
    if not args.legacy_o_mappo:
        rows, replacement = replace_o_mappo(args.o_mappo_results, protocol, rows, curves)
    comparisons = []
    for rate in protocol['rates']:
        a = [x for x in rows if x['method'] == 'meet_cobra' and x['rate_mbps'] == rate]
        comparisons.append(dict(rate_mbps=rate, seeds=len(a),
            interference_ratio=float(np.mean([x['comparison']['interference_ratio'] for x in a])),
            service_underestimation_percent=float(100*(1-np.mean(
                [x['comparison']['service_capacity_ratio'] for x in a])))))
    result = dict(protocol_sha256=digest(args.grid/'protocol.json'), audit=audit_result,
                  o_mappo_replacement=replacement,
                  power_figure_includes_oracle_cr_lb=args.include_oracle_cr_lb,
                  generated_metrics=['power_w'] if args.power_only else list(METRICS),
                  cases=len(rows), seeds=protocol['seeds'], rates_mbps=protocol['rates'],
                  statistics='Mean across seeds; envelopes show min/max across seeds in one fixed scenario.',
                  latency='Within-seed pooled vehicle-slot percentiles of q_v/lambda_v; then mean across seeds. No extra slot.',
                  warmup_frames=protocol['warmup_frames'], duration_s=30,
                  fixed_decision_comparison= comparisons,
                  l99_first_load_above_20ms={m: next((r for r, v in zip(protocol['rates'],
                      curves[m]['p99_proxy_ms'].mean(axis=1)) if v > 20), None) for m in METHODS})
    # Generated data products, not edits to existing simulation artifacts.
    (args.report/'figure_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.report/'figure_data.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'rate_mbps', 'metric', 'mean', 'seed_min', 'seed_max'])
        for m in METHODS:
            for metric in METRICS:
                for rate, a in zip(protocol['rates'], curves[m][metric]):
                    writer.writerow([m, rate, metric, a.mean(), a.min(), a.max()])
    plots(curves, summary, protocol['rates'], args.figures,
          power_only=args.power_only, include_oracle_cr_lb=args.include_oracle_cr_lb)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
