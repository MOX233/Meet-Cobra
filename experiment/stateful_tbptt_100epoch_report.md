# Stateful TBPTT training: 100-epoch run

## Protocol

The three paper models were statefully fine-tuned for 100 epochs with the same
vehicle-disjoint split, TBPTT length of 10, initialization, optimizer settings,
and random seeds used in the selected 30-epoch experiment. Training ran in
parallel on three GPUs. The retained checkpoint for each task is the one with
the best validation metric, rather than the final-epoch checkpoint.

## Validation results

| Predictor | Best 30-epoch result | Best 100-epoch result | Best epoch |
|---|---:|---:|---:|
| Beam-pair index (top-1 accuracy) | 91.333% | 91.366% | 97 |
| Desired-link gain (MAE) | 3.237 dB | 3.072 dB | 94 |
| Interfering-link gain (MAE) | 2.521 dB | 2.430 dB | 99 |

The 100-epoch extension gives only a small beam-accuracy improvement, but both
gain predictors continue to improve after epoch 30.

## Held-out chronological test

The selected checkpoints were evaluated with persistent per-vehicle recurrent
states on the same independent 800--950 s trace used for the 30-epoch audit.
The test contains 182,731 vehicle frames and 730,924 BS--vehicle links.

| Metric | 30-epoch TBPTT | 100-epoch TBPTT | Change |
|---|---:|---:|---:|
| Beam top-1 accuracy | 88.412% | 88.473% | +0.061 percentage points |
| Beam top-3 accuracy | 98.633% | 98.626% | -0.007 percentage points |
| Beam top-5 accuracy | 99.586% | 99.588% | +0.002 percentage points |
| Desired-link gain MAE | 3.570 dB | 3.406 dB | -0.164 dB |
| Interfering-link gain MAE | 2.765 dB | 2.630 dB | -0.136 dB |

The 100-epoch model is therefore retained. Its top-1 accuracy is slightly
higher, its ranked beam-candidate coverage is effectively unchanged, and its
two gain-prediction errors are lower than those of the 30-epoch model.

## Artifacts

- Training histories and checkpoints:
  `experiment/results/stateful_tbptt_20260912/finetune_100ep_lr1e-4/`
- Held-out test:
  `experiment/results/stateful_tbptt_20260912/test_100ep_lr1e-4_cpu/`
- Fig. 4 summary:
  `experiment/results/stateful_tbptt_20260912/finetune_100ep_lr1e-4/fig4_summary.json`
- Reproducible plotting script:
  `experiment/plot_stateful_training_curves.py`
- Updated figure files:
  `latexCodes/figures/NN_training_curves(a).pdf` and
  `latexCodes/figures/NN_training_curves(b).pdf`
