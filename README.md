# MEET-COBRA

IEEE TWC first revision submitted on 2026-10-08. Submission snapshot:
`twc-r1-submitted-2026-10-08` (`09f550b`).

## Start here

- [Current assets and result locations](configs/paper_r1_assets.json)
- [Reproduction guide](docs/REPRODUCING_R1.md)
- [Executable R1 runbook: order, commands, resume and backup](docs/R1_RUNBOOK.md)
- [Current and historical manuscript assets](docs/PAPER_FILES.md)
- [Experiment report index](docs/EXPERIMENTS.md)
- [Workspace maintenance](docs/WORKSPACE_MANAGEMENT.md)
- [Round-2 cleanup and result retention](docs/maintenance/round2_20261008/REPORT.md)
- [Submitted artifacts](latexCodes/revision1/SUBMISSION_SNAPSHOT.md)

The active manuscript is `latexCodes/main_revision1_journal.tex`; the response
source is `response_letter/response_letter.tex`. The `latexCodes/revision1/`
submission is frozen. Do not overwrite its PDFs or source archive.

## Layout

| Location | Role |
|---|---|
| `utils/` | Models, algorithms and physical/system simulation |
| `experiment/*.py` | Training, experiment, analysis and plotting entrypoints |
| `test_*.py`, `tests/` | Existing scientific tests and new maintenance tests |
| `scripts/` | Safe reproduction wrappers, smoke checks and maintenance |
| `configs/` | Small, versioned asset manifests |
| `docs/` | Current guides and indexes; historical reports retain their original paths |
| `sionna_result/`, `sumo_data/` | Retained raw data for the current configuration |
| `data4sim/` | Retained simulation inputs for the current configuration |
| `experiment/results/` | Models, compact datasets, caches and result records; not bulk-added to Git |
| `NN_result/` | Historical models; retained for provenance and legacy utilities |
| `notebooks/legacy/` | Historical interactive work, not the current reproduction entrypoint |
| `archive/` | Small historical drafts/figures and notebook recovery copies |
| `latexCodes/`, `response_letter/` | Manuscript and reviewer-response sources |

Existing Python source and formal result paths are intentionally unchanged:
frozen protocols check source hashes and some scripts import helpers from older
baseline modules. In particular, PQL/DQL-named files are not necessarily unused.

## Quick integrity and software checks

Run from the repository root in the existing `sionna` environment:

```bash
python -B scripts/check_paper_assets.py
python -B -m unittest discover -s tests -p 'test_workspace_maintenance.py' -v
python -B -m unittest test_gap_rb_usage test_gap_refinement test_revision_pipeline_guards test_stateful_tbptt -v
```

The asset check does not retrain models or run simulations. Large datasets and
model files remain local, outside Git. A Git checkout alone does not restore the
exact submitted numerical results; retain current raw data, checkpoints and
result records or regenerate them as described in the guide.

For an actual small end-to-end check on an available GPU (new output name each
time; around three minutes on the checked RTX 3090):

```bash
python -B -u scripts/smoke_r1.py --output experiment/results/reproduction_smoke_MY_CHECK --device cuda:0 --paper-plots
```

This intentionally uses a tiny test-data subset to test software, not to report
prediction accuracy or paper performance. All outputs stay outside Git.
For new full grids, run the **six** main schemes with `revision_pipeline.py`,
then the two final baselines with `scripts/r1_baselines.py`; see the runbook.
Do not use the two historical baseline choices in `revision_pipeline.py` as the
final paper baselines. Scientific source files and frozen paths remain unchanged.
