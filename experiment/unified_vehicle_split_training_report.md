# Unified Vehicle-Split Two-Stage NN Training

Date: 2026-09-13

## Protocol

One deterministic split is created from the 706 vehicles that have valid
next-frame records in the 200--800 s channel trace. The split contains 494
training vehicles and 212 validation vehicles, with no vehicle overlap. It is
fixed before either training dataset is constructed.

- Stage I constructs finite-window samples of one to ten CSI frames from the
  assigned trajectories and trains each model from random initialization for
  100 epochs.
- The selected Stage-I checkpoint initializes Stage II. Stage II organizes the
  same vehicle partitions as continuous trajectories, retains the LSTM state
  across frames, and fine-tunes for another 100 epochs using TBPTT with a
  truncation length of ten frames.
- Both stages use AdamW, seed 20, and the same split file. The Stage-I and
  Stage-II learning rates are respectively $10^{-3}$ and $10^{-4}$.

Validation uses finite windows in Stage I and stateful trajectory processing
in Stage II. The vehicle partitions are identical, but the available history
and sequence processing differ. The comparison also includes additional
training epochs and a different learning-rate schedule; it does not isolate
the contribution of matching training to stateful inference.

Both stages select the beam checkpoint by maximum validation Top-1 accuracy
and each gain checkpoint by minimum validation MAE. After Stage II, the
training script reloads `best.pth`, rather than `last.pth`, for the reported
Top-$M_{\rm P}$ evaluation.

The compact chronological dataset contains 646,747 next-frame samples in 722
trajectory segments. Its SHA-256 is
`71cf39f1edb208b29896bc485ee8e597f3e316ab054e5ce8475c08e939335c1e`.
The split file SHA-256 is
`6af64204659e247f692634efc9db5ecb9d9d1beaefa94986a0265acb7770dc7c`.

## Validation results

| Predictor | Stage-I best validation result | Best Stage-II validation result | Stage-II best epoch |
|---|---:|---:|---:|
| Beam pair | 86.683% Top-1 | 89.164% Top-1 | 88 |
| Desired-link gain | 4.790 dB MAE | 3.111 dB MAE | 59 |
| Interfering-link gain | 3.676 dB MAE | 2.348 dB MAE | 91 |

The maximum Top-3 validation accuracy observed during Stage II is 98.655% at
epoch 18. At the selected beam checkpoint, Top-$M_{\rm P}$ accuracy first
exceeds 99% at $M_{\rm P}=4$ and first exceeds 99.9% at $M_{\rm P}=17$.

The selected beam checkpoint is from Stage-II epoch 88 (cumulative epoch 188),
with 89.164306% Top-1 and 98.651073% Top-3 accuracy. The Top-3 star in Fig. 4
instead marks epoch 18 (cumulative epoch 118), with 98.654519% Top-3 accuracy.
Both Top-3 values round to 98.65% at the manuscript's display precision;
the stars denote metric-wise extrema, not a common checkpoint.

## Independent chronological check

The selected Stage-II checkpoints were additionally evaluated on the 800--950
s trace, which was not used for either training stage. Stateful prediction on
all 730,924 links gives 88.808% Top-1 accuracy, 98.316% Top-3 accuracy,
3.231 dB desired-link MAE, and 2.538 dB interfering-link MAE. This check is
kept as an implementation audit and is not used for the Fig. 4 annotations.

## Stateful inference timing

The same three selected Stage-II checkpoints were benchmarked with persistent
recurrent states on 2026-09-13. On an AMD EPYC 7742 CPU with one thread, FP32,
and batch size one, 14,166 vehicle-frame measurements give a combined median
latency of 5.328958 ms and a 95th percentile of 5.743067 ms. The protocol,
checks, and artifact paths are documented in
[the stateful overhead report](nn_overhead_stateful_report.md).

## Pending system-level integration

As checked on 2026-09-13, the default predictor-loading path in
`utils/sim_utils.py` still points to the original 2025 finite-window models.
Fig. 5--Fig. 8 have not yet been regenerated using the selected Stage-II
checkpoints. Their future rerun must use the three `best.pth` files under
the Stage-II artifact directory below (beam epoch 88, desired-link gain epoch
59, and interfering-link gain epoch 91), with stateful inference. These are
the same checkpoints used for the Top-$M_{\rm P}$ evaluation and the independent
chronological prediction check; no model is to be reselected by Top-3,
candidate-list length, traffic load, or system-level performance.

## Artifacts

- Fixed split: `experiment/results/stateful_tbptt_unified_split_20260913/vehicle_split_seed20.npz`
- Stage I: `experiment/results/stateful_tbptt_unified_split_20260913/stage1_finite_window/`
- Stage II: `experiment/results/stateful_tbptt_unified_split_20260913/stage2_stateful_tbptt/`
- Independent check: `experiment/results/stateful_tbptt_unified_split_20260913/heldout_test_800_950/`
- Fig. 4 summary: `experiment/results/stateful_tbptt_unified_split_20260913/stage2_stateful_tbptt/fig4_summary.json`
- Updated figures: `latexCodes/figures/NN_training_curves(a).pdf` and
  `latexCodes/figures/NN_training_curves(b).pdf`

The directory
`stage1_finite_window_incomplete_fullwindow/` is an interrupted diagnostic run
that excluded trajectories shorter than ten frames. It is retained only for
traceability and is not part of the reported model or figure.
