# Round 3 — directory organization and reproduction checks

Completed 2026-10-08. Starting checkpoint: `175a091`; submitted scientific-source
snapshot: `twc-r1-submitted-2026-10-08` (`09f550b`).

## Changes

- Added the executable `docs/R1_RUNBOOK.md`, `configs/r1_workflow.json`, paper-file
  index and local directory READMEs. The active manuscript is unambiguously
  `main_revision1_journal.tex`; `main_revision1.tex` is historical.
- Kept all scientific Python/TeX source paths, current included figures, selected
  models, formal result directories and the submission package unchanged.
- Moved 10 unused WBL/WBL_MS/speed-group figures into the local archive, one
  advisor-annotated PDF into `response_letter/review_notes/`, and two submission
  screenshots into `response_letter/submission_support/`.
  All 13 files remain recoverable; see `asset_moves.json` for paths and SHA-256.
- Added safe data-generation/preprocessing and selected-baseline evaluation
  wrappers. They invoke established algorithms, reject inappropriate output
  roots and preserve source/input hashes; no scientific algorithm was changed.
- External PDFs/screenshots and all generated smoke artifacts are excluded from
  Git. Literature metadata, links and checksums are versioned separately.

## Actual checks

The strengthened end-to-end run is
`experiment/results/reproduction_smoke_20261008_v2/report.json`.
SHA-256: `4f7f350d02356ac44fccb7060c449e2c202d9f1315f487c6e3fae471c80366a4`.

It passed in **149.38 s** on GPU 0, NVIDIA GeForce RTX 3090, using 32 continuous
vehicles over 21 input timestamps / 20 simulated frames (2 s):

1. Build compact beam-average labels and a vehicle-disjoint split from a small
   retained test-input subset, solely for testing software.
2. Train beam and desired-gain predictors for one finite-window epoch and one
   stateful epoch each. Test interfering-gain training across an intentional
   epoch-boundary stop/resume, completing both one-epoch stages.
3. Assemble the three-model bundle; generate a stateful cache; repeat the same
   cache command and verify byte-identical reuse.
4. Run six main schemes at 13 Mbps, seed 1, with physical directional/RB service;
   run again to verify completed-case reuse; compute audited summaries.
5. Generate an additional cache with the selected formal NN models on the same
   small trajectory. Exercise final prediction-input O-MAPPO and one real PPO
   update with 82 transitions (weights changed in memory only; not saved).
6. Exercise current MTS H32-cross5 with nonzero beam-search activity: 588
   acquisitions and 57,832 cross-5 tracking slots. Verify probe accounting,
   physical RB constraints, power/queue metric consistency and paired traffic.
7. Regenerate formal system plots and NN training curves in scratch output.
   The system `figure_data.csv` is byte-identical to the retained formal CSV.
8. Check that 259 protected source/model/submission files are unchanged.

The first smoke run also passed but tiny untrained predictions made MTS stay on
the macro BS. The second check deliberately uses formal predictors for the two
baseline checks so beam-search coverage is not vacuous. Neither run supplies new
paper metrics; test-subset training is only a software fixture.

Additional actual checks:

- `scripts/r1_baselines.py prepare/run/run/plot` passed on the first short main
  grid; both selected baselines ran, the second run skipped complete cases, and
  eight-scheme figures/CSV were generated under `r1_baselines_smoke_20261008/`.
- `scripts/r1_data.py system-input` completed on the same small raw-H subset.
  Its preprocessing is array-identical to the old implementation in the parity
  unit test, including zero channels and a newly entering vehicle.
- **38 unit tests passed**: 4 round-1 safety, 6 round-2 safety, 2 new wrapper
  safety/parity, and 26 existing GAP/guard/stateful-training tests.
- Round-2 retained-asset verification passed: 19,094 retained experiment files
  unchanged, 11,320 content hashes verified; deleted 2,119 targets remain absent.
- Asset verification passed: 50 required paths, 41 source/model hashes,
  3 older source blobs available in Git, frozen submission checksums unchanged.
- Source syntax, `git diff --check`, all 13 relocation hashes and formal figure
  numerical CSV equivalence passed.

## Explicit limits

No full 100+100-epoch training, 160-update RL training, full-load system grid,
SUMO mobility generation or full RT regeneration was run in this round.
The new runbook identifies prerequisites and restart boundaries without claiming
that these expensive stages were revalidated end to end.

The old main-grid protocol expects three earlier source versions. Current-code
strict audit of that old protocol still refuses the mismatch; no guard or saved
hash was changed. New runs get fresh protocols; exact historical reruns require
the corresponding Git version and separately backed-up assets.

No dataset, checkpoint, result array, current figure or submission artifact was
deleted in round 3. Local smoke outputs occupy about 0.16 GiB and remain clearly
named, ignored, disposable test outputs—not another formal dataset/model.
The pre-existing `.vscode/settings.json` user modification is not part of this
checkpoint and was not edited or staged.
