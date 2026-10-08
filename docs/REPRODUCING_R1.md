# Reproducing the submitted R1 work

This guide indexes the existing scientific code without changing its algorithms,
parameters or frozen result directories. Cleanup checks are not new training runs.
Use the existing `sionna` environment; observed package versions are recorded in
[`environment-observed-20261008.txt`](environment-observed-20261008.txt).

## 1. Current raw data and regeneration code

The retained configuration is 28 GHz, 32 transmit antennas, 8 receive antennas,
and vehicle arrival rate `Lambda=1.00`. This mobility parameter is **not** the
per-vehicle traffic load swept in the system experiments.

- `sumo_data/trajectory_Lbd1.00.csv`: current mobility trajectory.
- `sionna_result/trajectoryInfo_lbd1.00_200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10.pkl`:
  training/validation channels.
- The corresponding `800_830`, `800_900` and `800_950` raw files are retained,
  including inputs required by current tests and their provenance.
- `data4sim/` retains the current `800_830` and `800_950`, `Np8`, `mode0`,
  `lookahead10` prepared inputs. The prediction cache requires these prepared
  records, including `CSI_preprocessed`; a raw channel pickle is not a drop-in
  replacement.

Dataset generation is implemented in `utils/sumo_utils.py`,
`utils/data_utils.py`, `generate_data_3Dbeam.py`,
`generate_data_3Dbeam_subprocess.py` and `utils/sim_utils.py`.
Scene XML files and `meshes/` are versioned. The current SUMO network, route,
vehicle-type and `.sumocfg` files are small configuration inputs, not datasets.

The legacy top-level generation script hard-codes its time interval and GPU.
Its configurable underlying function is `utils.data_utils.run_sionna_sim`.
Do not run it blindly over the retained raw files: configure a separate run and
explicit intervals/GPU first. Retaining the existing trajectory and channels is
necessary for exact reuse of the submitted realization; regenerating them with
stochastic simulation is not guaranteed to produce identical files.

## 2. Compact training data and vehicle split

The current trainers construct windows from compact chronological trajectories;
they do not need the deleted multi-hundred-GiB `prepared_dataset/` files.
Two compact files are intentionally retained:

1. `experiment/results/stateful_tbptt_20260912/training_trajectories.npz`, used
   for the final beam and desired-gain models;
2. `experiment/results/revision_directional_20260922/training_data.npz`, whose
   interfering-gain labels use the revised beam-average definition.

Their identical file sizes do not mean identical labels. Both stages use
`experiment/results/stateful_tbptt_unified_split_20260913/vehicle_split_seed20.npz`.

Regeneration entrypoints, with a **new output directory** chosen by the operator:

```bash
python experiment/prepare_stateful_trajectories.py --help
python experiment/revision_training.py data --help
python experiment/vehicle_split.py --help
```

The first supports `--interference-label legacy-max`; `revision_training.py data`
builds the revised interference labels. Use `--seed 20 --train-fraction 0.7`
when rebuilding the vehicle split from the same source vehicles.

## 3. Training and prediction cache

For beam and desired-gain training, the existing sequence is:

1. `experiment/train_finite_window_vehicle_split.py`: finite-window training.
2. `experiment/train_stateful_tbptt.py`: initialize from the selected first-stage
   checkpoint; carry recurrent state along each trajectory, with TBPTT length 10.

Both stages use 100 epochs in the formal run. Recover the complete configuration,
including the stage-specific batch size, optimizer and checkpoint selection,
from each retained `metadata.json`; do not silently replace it with CLI defaults.

For the revised interfering-gain model, use `experiment/revision_training.py train`.
Its `assemble` subcommand combines the final beam and desired-gain models with
the new interfering-gain model. Its `cache` subcommand performs stateful inference
on the prepared system-test input. Use `--help` on each subcommand for arguments.
Do not rebuild or overwrite a formal cache merely for workspace housekeeping.

The selected three-model bundle is
`experiment/results/revision_directional_20260922/models`.
The corresponding `selected_models` copy is used for timing; the asset checker
verifies that the three checkpoint pairs are byte-identical.

Final O-MAPPO training and evaluation use
`experiment/train_o_mappo_predicted_cross5.py`. Its initialization depends on
earlier trained models; see the upstream dependencies in the asset manifest.
Do not treat this fine-tuning entrypoint as training from scratch.

## 4. System experiments, plotting and source versions

Use [`EXPERIMENTS.md`](EXPERIMENTS.md) to select the correct entrypoint/results.
The main system entrypoint `experiment/revision_pipeline.py` provides
`prepare`, `case`, `run`, `status` and `summarize` commands. A new run should have
a new root, explicit methods, all intended traffic loads, and explicit seeds.
Final O-MAPPO and MTS run through their own entrypoints rather than the older
baseline implementations in the main frozen grid.

The old grid and final baselines were produced at different source revisions.
For exact frozen reruns, retain their source hashes and use the recorded Git
revision; do not disable guards or overwrite recorded hashes. The final paper
plots combine those retained results. No full numerical rerun was performed
during cleanup.

## 5. Integrity checks and what Git does not store

```bash
python -B scripts/check_paper_assets.py
python -B -m unittest discover -s tests -p 'test_workspace_maintenance.py' -v
python -B -m unittest test_gap_rb_usage test_gap_refinement test_revision_pipeline_guards test_stateful_tbptt -v
```

Git stores source/configuration, not raw channels, trajectory tables, checkpoints
or caches. For exact-result recovery, back up the protected local assets separately.
A fresh Git checkout supports rebuilding the workflow but does not contain the
large data or selected model weights. Historical experimental entrypoints that
used deleted obsolete datasets require those datasets to be regenerated.

After cleanup round 2, selected superseded system runs retain their configuration
and per-case metrics but no longer retain the large per-slot arrays. The current
paper grids and all model weights remain intact. See the retention table in
`EXPERIMENTS.md` and the exact deletion ledger under `docs/maintenance/` before
trying to audit or resume an older run.
