# Stateful versus sliding-window LSTM inference

## Purpose

This experiment compares two inference implementations of the three frozen
MEET-COBRA neural networks:

- **Sliding window:** each call processes the latest `min(age, 10)` CSI frames
  with zero initial hidden and cell states. This is the current simulator
  implementation.
- **Stateful streaming:** each vehicle and each neural network retain their own
  hidden and cell states across frames, and only the newest CSI frame is
  processed. States are reset when a vehicle first appears or re-enters.

Both methods use exactly the same checkpoints, chronological CSI observations,
and next-frame labels. No model is retrained or tuned.

## Evaluation protocol

- Training period of the frozen checkpoints: 200--800 s.
- Test trace: 800--950 s (`lbd1.00`, 1501 frames at 0.1-s spacing).
- Prediction task: CSI observations through frame `x` predict the labels at
  frame `x+1` for vehicles present in both frames.
- Sample size: 182,731 vehicle--frame predictions, 730,924 BS links, and 272
  vehicles.
- Gain targets reproduce the respective training scripts. In particular, the
  interference-gain target is the maximum channel-matrix element magnitude,
  not the cached `g_avg` field.
- Confidence intervals are paired vehicle-cluster bootstrap intervals with
  2,000 replicates. Whole vehicle trajectories are resampled to avoid treating
  temporally correlated frames as independent observations.

Approximately 28.2% of BS links have an exactly zero channel. For these links,
the cached optimal beam index defaults to index zero even though no beam is
physically distinguishable. Results are therefore reported both over all links
and over nonzero-channel links.

## Main results

### All test links

| Metric | Sliding window | Stateful | Stateful minus window |
|---|---:|---:|---:|
| Beam top-1 accuracy | 88.054% | 85.957% | -2.097 pp |
| Beam top-3 accuracy | 98.373% | 97.991% | -0.382 pp |
| Beam top-5 accuracy | 99.377% | 99.166% | -0.211 pp |
| Beam top-18 accuracy | 99.909% | 99.909% | +0.0003 pp |
| Desired-gain MAE | 4.322 dB | 5.161 dB | +0.839 dB |
| Interference-gain MAE | 3.396 dB | 3.763 dB | +0.367 dB |

The paired 95% intervals for the stateful-minus-window differences are
[-2.589, -1.745] percentage points for top-1 accuracy, [0.365, 1.216] dB for
desired-gain MAE, and [0.025, 0.606] dB for interference-gain MAE.

### Nonzero-channel links

| Metric | Sliding window | Stateful | Stateful minus window |
|---|---:|---:|---:|
| Beam top-1 accuracy | 85.593% | 82.849% | -2.744 pp |
| Beam top-3 accuracy | 98.016% | 97.452% | -0.564 pp |
| Beam top-5 accuracy | 99.289% | 98.956% | -0.333 pp |
| Beam top-18 accuracy | 99.874% | 99.874% | +0.0008 pp |
| Desired-gain MAE | 3.655 dB | 4.161 dB | +0.507 dB |
| Interference-gain MAE | 2.978 dB | 3.306 dB | +0.328 dB |

The paired 95% intervals are [-3.369, -2.321] percentage points for top-1
accuracy, [0.185, 0.751] dB for desired-gain MAE, and [0.081, 0.502] dB for
interference-gain MAE. Excluding zero-channel links therefore strengthens,
rather than removes, the observed accuracy loss.

## Dependence on retained history

| Continuous vehicle age | Links | Top-1 change | Desired-gain MAE change | Interference-gain MAE change |
|---|---:|---:|---:|---:|
| 1--10 frames | 10,864 | 0.000 pp | +0.000 dB | -0.000 dB |
| 11--20 frames | 10,776 | +0.065 pp | -0.162 dB | -0.047 dB |
| 21--50 frames | 31,600 | -1.544 pp | +0.562 dB | +0.161 dB |
| 51--100 frames | 51,048 | -2.936 pp | +0.880 dB | +0.432 dB |
| 101--300 frames | 188,556 | -2.084 pp | +1.133 dB | +0.496 dB |
| More than 300 frames | 438,080 | -2.149 pp | +0.774 dB | +0.339 dB |

Within the first ten frames, both implementations use the same complete
prefix and give identical beam decisions. The tiny gain-output differences
are floating-point batching effects. The stateful method is also numerically
equivalent to replaying the same full prefix (maximum relative L2 difference
across the explicit checks: `1.12e-5`). Thus, the later performance difference
is caused by retaining observations older than ten frames, not by an incorrect
state-update implementation.

## Conclusion

The two implementations are mathematically equivalent only when they process
the same sequence from the same initial state. The current sliding-window
implementation discards observations older than ten frames and resets the
state, whereas unrestricted streaming retains them. The checkpoints were
trained with effective sequence lengths from one to ten frames. Carrying their
states far beyond that range changes the inference distribution and, on this
test trace, materially reduces top-1 beam accuracy and increases both gain
MAEs. Therefore, unrestricted cross-frame state retention should **not** replace
the current ten-frame sliding-window implementation for these checkpoints.

Stateful one-frame inference remains a plausible low-complexity architecture,
but it would need training that matches persistent-state deployment (for
example, stateful training with truncated backpropagation and explicit reset
rules) followed by a new held-out evaluation. The present experiment does not
test such a retrained model.

## Reproducibility artifacts

- Script: `experiment/compare_stateful_prediction.py`
- Unit tests: `test_stateful_prediction.py`
- Full metrics: `experiment/results/stateful_prediction_20260912/full_test/summary.json`
- Tabular metrics: `experiment/results/stateful_prediction_20260912/full_test/metrics.csv`
- Paired predictions: `experiment/results/stateful_prediction_20260912/full_test/predictions.npz`
- Inputs, hashes, checkpoints, checks, and protocol:
  `experiment/results/stateful_prediction_20260912/full_test/metadata.json`

