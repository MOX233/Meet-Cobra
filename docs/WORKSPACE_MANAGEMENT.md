# Workspace and file-management policy

## Rules

1. Keep source code, tests, small configuration files, environment notes and
   reproducibility instructions in Git. Do not bulk-add data, result directories,
   checkpoints, caches or future compiled submissions. Existing Git history,
   including the submitted revision, is not rewritten.
2. The submitted revision is immutable. Current manuscript/source files stay in
   their established locations. New revisions get separate submission folders.
3. Formal experiment directories, their `protocol.json`, model-selection records,
   histories, seeds and raw evidence are retained. A failed or superseded approach
   is not grounds to erase all experimental evidence.
4. Dataset obsolescence is determined by configuration and formal dependencies,
   not modification time. Keep the current 32-transmit/8-receive-antenna, 28-GHz,
   vehicle-arrival-rate-1.00 raw data and required compact training data.
5. Build LaTeX into ignored `build/` directories. Never clean all `.bbl` files:
   `latexCodes/revision1/Manuscript_LaTeX/main.bbl` belongs to the submitted source.
6. New runs use a unique output directory and record the Git commit, inputs,
   hashes, random seeds and configuration. Do not overwrite a frozen protocol.
7. Before any further deletion, review the explicit plan, dependency checks and
   active processes. No `git clean -fdx` or repository-wide recursive deletion.

## 2026-10-08 cleanup

The deletion scope is limited to obsolete top-level dataset files and disposable
untracked build files. All experimental results and all checkpoints remain intact.
The historical sliding-window files in `prepared_dataset/` are not inputs to the
current finite-window/stateful training pipeline, which constructs windows from
compact chronological data at runtime.

The protected list is in `configs/paper_r1_assets.json`. The maintenance script
additionally checks formal/upstream JSON metadata for dataset references, excludes
Git-tracked files, rejects symlinks, requires unchanged file identities and refuses
active project jobs or open targets. It never recursively deletes a directory.

The reviewed plan and completion receipt are under `docs/maintenance/`.
The plan records names, configuration-bearing filenames, sizes and timestamps;
it is not a backup of deleted data. Old untracked data cannot be restored through
Git. Re-running a stochastic simulator is not a guarantee of byte-identical data.

For a future cleanup, create a **new** plan filename and review it before applying:

```bash
python -B scripts/workspace_maintenance.py plan --plan docs/maintenance/NEW_PLAN.json
# Review the file list and protected datasets, then use its printed SHA-256:
python -B scripts/workspace_maintenance.py apply --plan docs/maintenance/NEW_PLAN.json --confirm-sha256 REVIEWED_HASH
python -B scripts/workspace_maintenance.py verify --plan docs/maintenance/NEW_PLAN.json
python -B scripts/check_paper_assets.py
```

## Why existing Python files are not relocated

Formal protocols hash `utils/*.py` and experiment entrypoints, and baselines share
functions through historical module names. Moving everything into a new package
would require a separate tested code refactor. This cleanup organizes documentation,
notebooks and historical draft assets without changing scientific source hashes.

Historical notebooks should be run with the repository root as the working
directory. Recovery notebooks are preserved rather than discarded because their
contents differ from the saved notebooks.
