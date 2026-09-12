# Stateful TBPTT training from scratch: 100-epoch schedule and normalization study

Date: 2026-09-12

## Protocol

- Initialization: random (no finite-window checkpoint).
- Recurrent training: state carried across frames; computation graph detached every 10 frames (TBPTT length 10); state reset only at trajectory boundaries.
- Data: 200--800 s trajectories; vehicle-level split with seed 20 (507 training and 215 validation trajectories).
- Optimizer: AdamW, weight decay `1e-4`, gradient clipping at 1.0.
- Schedule: five-epoch linear warm-up followed by cosine decay to `1e-6`; all runs complete 100 epochs.
- Peak learning rates compared: `3e-4` and `1e-3`.
- Selected normalization: each of the 128 input features is standardized using mean and standard deviation computed only from the training trajectories. The original internal BatchNorm running statistics are kept fixed to avoid trajectory-batch-dependent running-statistic drift; their affine parameters remain trainable.
- Independent test: chronological 800--950 s trace, 182,731 vehicle frames and 730,924 links. The reported test values use the stateful predictor.

The six generated input-normalization files have the identical SHA-256 digest
`a4a44b533c020411be91a78c81ada49e6ad31f5608cd905c31b34c55d96d4fba`.

## Normalization screening

Replacing every internal BatchNorm layer with LayerNorm was tested first. It was stopped after epoch 17 because it removed information needed to regress absolute channel-gain levels. At epoch 17, the `3e-4` run obtained 74.21% beam Top-1 accuracy, 35.70 dB desired-gain MAE, and 29.41 dB interfering-gain MAE. These pilots are retained under `scratch_ln_cosine_*` but are excluded from final model selection.

## Best validation checkpoints over 100 epochs

| Peak LR | Beam Top-1 (%) | Epoch | Desired MAE (dB) | Epoch | Interfering MAE (dB) | Epoch |
|---:|---:|---:|---:|---:|---:|---:|
| `3e-4` | 89.3955 | 85 | 2.4754 | 99 | 1.9167 | 98 |
| `1e-3` | 89.3732 | 79 | 2.4963 | 100 | 1.9698 | 100 |

The smaller peak learning rate is slightly better on the within-period vehicle-held-out validation split for all three tasks.

## Independent chronological test

| Training configuration | Top-1 (%) | Top-3 (%) | Top-5 (%) | Desired MAE (dB) | Interfering MAE (dB) |
|---|---:|---:|---:|---:|---:|
| Two-stage model currently selected for the paper | 88.4729 | 98.6263 | 99.5882 | 3.4056 | 2.6299 |
| Previous scratch, fixed/plateau LR `1e-4` | 83.0551 | 95.9037 | 97.6476 | 5.4402 | 4.5359 |
| Previous scratch, fixed/plateau LR `3e-4` | 84.4605 | 97.1234 | 98.5885 | 5.6011 | 5.3225 |
| Scratch, standardized input, cosine peak `3e-4` | 86.8150 | 96.6715 | 97.9384 | 4.3797 | **2.6135** |
| Scratch, standardized input, cosine peak `1e-3` | **88.0806** | **97.9241** | **99.0649** | **3.5251** | 2.7562 |

Although `3e-4` is slightly better on validation, `1e-3` transfers better to the later time interval for beam and desired-gain prediction. Relative to the previous `3e-4` scratch run, the new `1e-3` configuration improves Top-1 by 3.62 percentage points and reduces desired/interfering MAE by 2.08/2.57 dB.

Relative to the selected two-stage model, the new `1e-3` scratch model differs by -0.392 percentage points in Top-1 and +0.119/+0.126 dB in desired/interfering MAE. Vehicle-cluster bootstrap 95% intervals for these differences are [-1.044, 0.310] percentage points, [-0.283, 0.583] dB, and [-0.195, 0.504] dB, respectively. Thus these three differences are not resolved from zero by this trace. The two-stage model remains clearly better in Top-3 and Top-5 accuracy, with scratch-minus-two-stage differences of -0.702 and -0.523 percentage points and corresponding 95% intervals [-1.120, -0.288] and [-0.740, -0.352].

## Conclusion

The scratch-specific schedule and training-set feature standardization remove most of the performance gap caused by using a fine-tuning-oriented setup from random initialization. Nevertheless, the existing two-stage model remains the safer paper model because it has the best overall independent-test result, especially for Top-3 and Top-5 beam accuracy. The `1e-3` scratch result shows that the finite-window checkpoint is helpful but not indispensable; a properly configured one-stage stateful training process can reach nearly the same Top-1 and gain-prediction accuracy in 100 epochs.

## Artifacts

- `experiment/results/stateful_tbptt_20260912/scratch_std_cosine_lr3e-4/`
- `experiment/results/stateful_tbptt_20260912/scratch_std_cosine_lr1e-3/`
- `experiment/results/stateful_tbptt_20260912/test_scratch_std_cosine_lr3e-4_cpu/`
- `experiment/results/stateful_tbptt_20260912/test_scratch_std_cosine_lr1e-3_cpu/`
- Training script: `experiment/train_stateful_tbptt.py`
- Evaluation script: `experiment/compare_stateful_prediction.py`

