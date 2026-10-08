# Experiment cleanup, round 2 — 2026-10-08

## Outcome

Deleted 2,119 explicitly reviewed, untracked files. The allocated storage of
fully removed file objects was 59,939,816,960 bytes (55.823 GiB), counting each
hard-linked inode once. Workspace usage decreased from approximately 107 GiB
to 51 GiB; `experiment/` decreased from approximately 98 GiB to 43 GiB.

| Category | Files |
|---|---:|
| Superseded per-slot raw/sample arrays | 2,113 |
| Obsolete prediction or training caches | 3 |
| Byte-identical validation-cache copies | 3 |

The superseded raw arrays comprise 630 files from the 20260917 system grid,
631 from the 20260918 capped-demand grid, 810 from the earlier multiseed study,
and 42 from the pilot interference study. All associated per-case metrics,
aggregate summaries, protocols, diagnostics, reports and logs remain in place.

The obsolete caches are the old shared-frontend train/test predictions and
the 20260921 interfering-gain training data with the superseded zero-gain
convention. The three validation-cache copies were SHA-256 identical to
`experiment/results/o_mappo_h32_retrained_20260924_v4/exact_validation.pkl`,
which is retained.

## What was not removed

- Current raw channels, prepared simulation inputs and current training data.
- Every file in the formal paper result trees and protected upstream studies.
- All neural-network and RL checkpoints, including historical models.
- Scientific Python code, manuscripts, response letter and submitted artifacts.
- Unique original-paper data outside the explicitly selected superseded studies.

Model pruning was intentionally deferred: checkpoint storage is small compared
with raw arrays, and deleting a weight file could break an initialization chain.
No training, simulation or figure regeneration was performed for this cleanup.

## Verification

- All 2,119 planned targets are absent and the deletion receipt has 2,119 entries.
- All 19,094 remaining experiment files match their pre-cleanup identities;
  11,320 source/report/checkpoint content hashes also match.
- The asset check passed for 50 paths and 41 source/checkpoint hashes; all frozen
  submission checksums and manuscript/response source checksums remain unchanged.
- The official plotting data loader successfully read all 432 final system cases
  (8 methods, 18 traffic loads, 3 seeds), including the final baseline replacements.
- All 54 O-MAPPO and 54 MTS raw-result hashes passed the existing plotting checks.
- The means and seed ranges of all 720 curve rows match the saved paper figure
  CSV at `rtol=1e-12`, `atol=1e-12`. No paper figures were overwritten.
- All 10 maintenance safety tests passed, including a temporary-file test of
  hard-link deletion and exclusion of protected/tracked/model/source files.

## Ledger and recovery limits

`index.json` hashes six explicit target manifests and the dependency-reference
record. `deleted.jsonl` records successful removals. The detailed retained-file
snapshot is local and ignored by Git at
`archive/maintenance/round2_20261008/retained_snapshot.json`; its checksum is in
the index. None of these metadata files is a backup of the deleted arrays.

For old raw arrays, provenance hashes were copied from retained run records
where available; they are **not** claimed to be newly recomputed binary hashes.
Deletion eligibility and unchanged file identities were checked separately.
Duplicate-cache equality and retained source/report/PT/PTH hashes were freshly
verified.

The deleted untracked old arrays cannot be restored by Git. Historical per-slot
analysis requires regenerating those runs; the retained summaries still support
their previously recorded aggregate comparisons. The duplicate validation caches
can be restored exactly by copying the retained v4 cache. Current paper results
require neither regeneration nor restoration.

Only maintenance code, small policies/docs and deletion metadata are versioned.
No datasets, checkpoints or large retained-file snapshot are added to Git.
The existing `.vscode/settings.json` modification is left untouched and unstaged.
