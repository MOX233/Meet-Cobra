#!/usr/bin/env python3
"""One-time recoverable round-3 relocation of explicitly listed loose assets.

Default is read-only planning. Never touches scientific sources, datasets,
weights, or latexCodes/revision1. Hashes and original names are recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIGURES = [stem+suffix for stem in (
    'BS0_assoc_ratio_comparison_curves', 'latency_90th_99th_comparison_curves',
    'power_comparison_curves', 'violation_prob_comparison_curves')
    for suffix in ('_WBL.pdf', '_WBL_MS.pdf')]
FIGURES += ['violation_by_speed_revision1.pdf', 'violation_by_speed_revision1.png']
MOVES = [(f'latexCodes/figures/{name}', f'archive/figures/revision1_history/{name}') for name in FIGURES]
MOVES += [
    ('response_letter/MEETCOBRA_Response_to_reviewers-commentedSheng.pdf',
     'response_letter/review_notes/MEETCOBRA_Response_to_reviewers-commentedSheng.pdf'),
    ('response_letter/Required Files.png', 'response_letter/submission_support/Required Files.png'),
    ('response_letter/Required Files (2).png', 'response_letter/submission_support/Required Files (2).png'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--receipt', type=Path, default=ROOT/'docs/maintenance/round3_20261008/asset_moves.json')
    args = p.parse_args()
    if args.receipt.exists():
        receipt = json.loads(args.receipt.read_text())
        for row in receipt['moves']:
            assert not (ROOT/row['source']).exists(), row['source']
            assert digest(ROOT/row['destination']) == row['sha256'], row['destination']
        print('Verified previous relocation:', len(receipt['moves']), 'files')
        return
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    tex = '\n'.join(p.read_text() for folder in ('latexCodes', 'response_letter')
                    for p in (ROOT/folder).glob('*.tex'))
    rows = []
    for source, destination in MOVES:
        src, dst = ROOT/source, ROOT/destination
        if source in tracked or src.is_symlink() or dst.exists():
            raise ValueError(f'Unsafe relocation: {source}')
        if source.startswith('latexCodes/figures/') and src.name in tex:
            raise ValueError(f'Figure is still referenced by a top-level TeX source: {source}')
        rows.append(dict(source=source, destination=destination, bytes=src.stat().st_size, sha256=digest(src)))
    print(json.dumps(rows, indent=2))
    if args.apply:
        # Same-filesystem rename; each file is recoverable at its recorded destination.
        for row in rows:
            src, dst = ROOT/row['source'], ROOT/row['destination']
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            assert digest(dst) == row['sha256']
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(dict(moves=rows, deleted_files=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
