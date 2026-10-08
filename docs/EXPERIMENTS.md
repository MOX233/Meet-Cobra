# Paper R1 experiment index

The submitted paper does **not** use every directory under `experiment/results/`.
The current dependency manifest is [`../configs/paper_r1_assets.json`](../configs/paper_r1_assets.json).
Paths below are relative to the repository root. They are deliberately preserved
because result protocols, plotting code and training metadata reference them.

| Purpose | Current result directory | Main entrypoint |
|---|---|---|
| Beam and desired-gain two-stage training | `experiment/results/stateful_tbptt_unified_split_20260913` | `train_finite_window_vehicle_split.py`, `train_stateful_tbptt.py` |
| Revised interfering-gain training, assembled models, cache, main system grid | `experiment/results/revision_directional_20260922` | `revision_training.py`, `revision_pipeline.py` |
| Final O-MAPPO: predicted association inputs, H32 and cross-5 tracking | `experiment/results/o_mappo_predicted_cross5_20260925` | `train_o_mappo_predicted_cross5.py` |
| Final MTS: H32 and cross-5 tracking | `experiment/results/mts_h32_full_grid_20260925` | `mts_h32_full_grid.py` |
| GAP-HO iteration sensitivity | `experiment/results/gap_refinement_directional_20260922` | `gap_refinement_directional.py` |
| Mobility distributions | `experiment/results/mobility_audit_20260926` | `audit_mobility_data.py` |
| Mobility-conditioned evaluation | `experiment/results/mobility_conditioned_20260927` | `mobility_conditioned_evaluation.py` |
| Gain-prediction error sensitivity | `experiment/results/gain_error_sensitivity_20260929` | `gain_error_sensitivity.py` |

All entrypoints in this table are under `experiment/`. Plotting uses:

- `experiment/plot_stateful_training_curves.py`;
- `experiment/plot_revision_system_results.py`;
- `experiment/plot_mobility_paper_figures.py`;
- `experiment/plot_gain_error_paper.py`;
- `response_letter/plot_interference_validation.py`.

The final system plots combine the main grid with the **separate final O-MAPPO
and MTS directories**. Do not substitute the older baseline results embedded in
the main grid. Figure numbers changed during revision; identify a figure by its
LaTeX label and plotting entrypoint, not an old figure number alone.

## Historical but still needed

O-MAPPO pretraining proceeds through earlier H32 training, actor-depth evaluation
and `o_mappo_eall_training_20260924` before the final predicted-input fine-tuning.
The MTS full grid also references its pilot run
`mts_hierarchical_tracking_20260925_v2`. These are upstream dependencies, not
disposable duplicates. The manifest lists them explicitly.

Other `experiment/results*` directories are historical studies. This cleanup
does **not** delete their models or numerical evidence. A later cleanup should
distinguish reproducible caches from model initialization dependencies and unique
experimental records before removing anything.

## Frozen source versions

The main grid records Git revision `0e9474fbf781cfa78c29d9a81c3db44d4de093b3`.
Three of its source files were later extended for the final baselines:
`experiment/revision_pipeline.py`, `utils/o_mappo.py`, `utils/o_mappo_sim.py`.
The submitted source snapshot is `twc-r1-submitted-2026-10-08` (`09f550b`).

The maintenance asset check verifies both the current submitted source and the
availability of the older exact source blobs in Git. This does not make the
older grid's live-source hash check pass on the newer sources. For exact reruns,
use the recorded source revision in a separate worktree; for a new run, create
a new protocol. Never edit the old protocol merely to suppress a hash mismatch.
