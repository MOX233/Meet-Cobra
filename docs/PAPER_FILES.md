# Paper files: one active source, one frozen submission

## Which manuscript is current?

| Role | Path | Policy |
|---|---|---|
| Active journal-layout revision | `latexCodes/main_revision1_journal.tex` | Only current manuscript source |
| Active response | `response_letter/response_letter.tex` | Only current response source |
| Original reviewed submission | `latexCodes/main.tex` | Historical reference; do not synchronize edits |
| Pre-journal-layout revision | `latexCodes/main_revision1.tex` | Historical reference, not another current version |
| Submitted R1 | `latexCodes/revision1/` | Immutable, including PDFs, ZIP and source package |
| Earlier drafts/recovery copies | `archive/drafts/`, `archive/recovery/` | Local archival material |

The historical top-level TeX files stay in place because their relative figure
and bibliography paths and experiment provenance refer to these locations.
Do not maintain both revision TeX files in parallel. A future revision should use
a new revision number and a new submission directory, not overwrite `revision1/`.

## Figures

`latexCodes/figures/` keeps figures required by the active journal source and
the original manuscript. PNG companions are previews; the included PDF is the
scientific figure. A filename ending in `_revision1` alone does not establish
that a figure is included: the TeX `includegraphics` statements are authoritative.

| Current figure(s) | Generator / origin |
|---|---|
| `system_model_v3.pdf`, `NNmodel.pdf`, `flowchart.pdf` | Authored vector diagrams; retain original assets separately |
| `Sionna Simulation.png` | Scene screenshot; retain image and scene inputs |
| `NN_training_curves(a).pdf`, `(b).pdf` | `experiment/plot_stateful_training_curves.py`; use corrected interfering-model overrides |
| `vehicle_speed_cdf_revision1.pdf` | `experiment/plot_mobility_paper_figures.py` |
| Five `*_comparison_curves_revision1.pdf` files | `experiment/plot_revision_system_results.py` for frozen submitted results; `scripts/r1_baselines.py plot` for a new combined run |
| `gain_error_sensitivity_revision1.pdf`, `gain_error_power_revision1.pdf` | `experiment/plot_gain_error_paper.py` |
| `response_letter/figures/r2c1_interference_validation.pdf` | `response_letter/plot_interference_validation.py` |

Ten unreferenced WBL/WBL_MS and discarded speed-group figure files were moved to
`archive/figures/revision1_history/`. Older Python plotting scripts can still
generate those old names; they were **not** edited. Do not use these historical
plots as the submitted R1 results. Original figure backups remain under
`archive/figures/original/`.

## External material and submission support

- `response_letter/literature/`: three full-text papers, plus a versioned index
  with source links, DOI and SHA-256. The PDFs are local, excluded from Git.
- `response_letter/review_notes/`: advisor-annotated response PDF; not a response
  source and not a file to submit.
- `response_letter/submission_support/`: screenshots of the submission form;
  they document that submission, not general journal requirements.
- The actual conference-paper PDF, prior-publication statement, submitted source
  archive and reviewer PDF remain in **frozen** `latexCodes/revision1/`.
- `response_letter/Reviewer_comments.txt`, its Chinese translation and
  `paper_revision_policy.md` retain their existing paths.

Round-3 moves are recoverable by reversing the source/destination pairs in
`docs/maintenance/round3_20261008/asset_moves.json`; all moved bytes were hashed.
No source, data, model or submitted file was deleted or moved in this round.

## Separate backups

Git is not the backup for full-text papers, advisor annotations, screenshots,
raw/generated datasets, models, experiment outputs or future submission PDFs.
Keep the frozen submission and these local materials in an external backup.
The [runbook](R1_RUNBOOK.md#backup-and-recovery) identifies the numerical assets
needed to recover the exact submitted results rather than regenerate a new run.
