# First workspace cleanup: 2026-10-08

## Deleted files

The exact deletion plan and per-file completion receipt are:

- `cleanup_20261008.json`
- `cleanup_20261008.deleted.jsonl`

207 individually validated files were deleted. Their pre-deletion allocated
sizes total 622.605 GiB:

| Category | Files | Allocated GiB |
|---|---:|---:|
| Obsolete materialized sliding-window datasets | 14 | 502.770 |
| Obsolete simulation-input configurations | 37 | 63.758 |
| Obsolete ray-tracing channel configurations | 40 | 55.327 |
| Obsolete mobility-density trajectory tables | 17 | 0.748 |
| Python bytecode and LaTeX temporary files | 99 | approximately 0.002 |

These were untracked files and cannot be recovered through Git. The plan is an
audit record, not a backup. Current datasets, all experiment results, all model
checkpoints, paper sources and submitted artifacts were excluded from deletion.

## Recoverable moves (contents unchanged)

| Original location | New location |
|---|---|
| Root `exp3_diffInputLen.ipynb`, `gain_block_pred_coordination.ipynb`, `paper_plot_results.ipynb`, `plot_results.ipynb`, `see_sionna_scene.ipynb`, `mox.ipynb` | `notebooks/legacy/`, same filenames |
| `utils/see_sionna_scene.ipynb` | `notebooks/legacy/utils/see_sionna_scene.ipynb` |
| `.ipynb_checkpoints/` | `archive/recovery/root/` |
| `experiment/.ipynb_checkpoints/` | `archive/recovery/experiment/` |
| `_garbage/` | `archive/drafts/pre_restart_20260905/` |
| `paper_figures/` | `archive/figures/original/` |
| Root `power vs lambda.csv`, `violation probability vs lambda.csv` | `archive/tables/`, same filenames |

Already tracked notebooks and old figure PDFs remain tracked as renames. Locally
archived untracked drafts and recovery copies remain outside Git. Moving files
does not create an external backup. Active scientific Python paths, formal result
directories, manuscript sources and `latexCodes/revision1/` are unchanged.

## Checks

`assets_before_20261008.json` and `assets_after_20261008.json` record the verified
50 dependency paths, 41 source/checkpoint hashes and submitted-artifact checksum
checks. The cleanup verifier confirms all 207 targets are absent and the file
identities of all 11 protected datasets are unchanged. Three historical grid
source hashes refer to an earlier Git revision, still available and verified;
this difference predates cleanup and is documented in `../EXPERIMENTS.md`.

Four cleanup safety tests and 26 existing GAP-HO, pipeline-guard and stateful
training unit tests passed. The original whitespace of the three newly versioned
SUMO configuration files was preserved along with their contents. All moved
tracked notebooks and figures are detected by Git as 100% identical renames;
no new data or checkpoint blob is staged.

The observed workspace usage after deletion is approximately 107 GiB. About
98 GiB is under `experiment/`; it includes both current and historical numerical
evidence and training dependencies. This first cleanup intentionally does not
prune that tree. No training or system experiment was rerun for housekeeping.

The pre-existing change to `.vscode/settings.json` was not edited or staged.
