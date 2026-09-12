# Stateful TBPTT training and frozen-test comparison

## Outcome

Vehicle-stream stateful fine-tuning with a TBPTT length of 10 removes the
accuracy degradation previously observed when the paper checkpoints were used
with unrestricted recurrent state. On the held-out 800--950 s trace, the new
stateful models roughly match the paper model's top-1 beam accuracy, modestly
improve top-3/top-5 accuracy, and substantially reduce both gain-prediction
MAEs.

## Reproducible training protocol

- Source channel trace: 200--800 s, 707 vehicles and 647,470 vehicle frames.
- Supervised samples: 646,747 chronological `x -> x+1` transitions in 722
  continuous trajectory segments. Sixteen discontinuities were split rather
  than carrying state across a gap.
- Split: seed 20, disjoint vehicles; 494 vehicles (507 segments, 450,832
  transitions) for training and 212 vehicles (215 segments, 195,915
  transitions) for validation.
- Architecture: unchanged paper LSTM and prediction heads.
- Initialization: the three paper checkpoints.
- Stateful rule: hidden/cell values persist through the complete vehicle
  trajectory and are zeroed only at a trajectory start.
- TBPTT: update and detach hidden/cell states every 10 frames. Detaching cuts
  gradients but does not reset state values.
- Input noise: a fresh pilot-noise realization per training epoch; fixed seed
  for validation.
- Optimization: AdamW, weight decay `1e-4`, gradient clipping at 1.0, up to 30
  epochs. Pretrained BatchNorm running statistics are frozen; affine parameters
  remain trainable.
- Model selection: two learning rates (`1e-4` and `3e-5`) evaluated on the
  disjoint-vehicle validation set. The test set was evaluated only after the
  validation choice.

The selected learning rate was `1e-4` for all three tasks:

| Model | Metric before stateful fine-tuning | Selected validation metric | Best epoch |
|---|---:|---:|---:|
| Beam-pair index | 88.298% top-1 | 91.333% top-1 | 29 |
| Desired gain | 4.890 dB MAE | 3.237 dB MAE | 30 |
| Interference gain | 3.490 dB MAE | 2.521 dB MAE | 29 |

## Held-out test protocol

- Frozen checkpoints; no test-set tuning.
- Chronological 800--950 s trace, 1,500 next-frame transitions.
- 182,731 vehicle--frame samples, 730,924 BS links, and 272 vehicles.
- The paper model uses its original zero-state, latest-10-frame sliding window.
- The new model processes one current CSI frame and preserves per-vehicle state.
- Inputs, labels, vehicles, and time indices match exactly sample by sample.
- Paired 95% intervals resample whole vehicle trajectories (10,000 bootstrap
  replicates).

## Test results

### Four-way diagnostic

| Checkpoint and inference | Beam top-1 | Beam top-3 | Beam top-5 | Desired-gain MAE | Interference-gain MAE |
|---|---:|---:|---:|---:|---:|
| Paper checkpoint, 10-frame window | 88.054% | 98.373% | 99.377% | 4.322 dB | 3.396 dB |
| Paper checkpoint, persistent state | 85.957% | 97.991% | 99.166% | 5.161 dB | 3.763 dB |
| TBPTT checkpoint, 10-frame window | 88.124% | 98.505% | 99.491% | 4.420 dB | 3.640 dB |
| **TBPTT checkpoint, persistent state** | **88.412%** | **98.633%** | **99.586%** | **3.570 dB** | **2.765 dB** |

Stateful fine-tuning improves the persistent-state deployment itself by 2.455
percentage points in top-1 accuracy and reduces the two MAEs by 1.591 dB and
0.998 dB relative to applying persistent state to the unadapted paper
checkpoints.

### Primary comparison: paper window versus TBPTT persistent state

| Metric | Paper window | TBPTT stateful | Difference |
|---|---:|---:|---:|
| Beam top-1 accuracy | 88.054% | 88.412% | +0.358 pp |
| Beam top-3 accuracy | 98.373% | 98.633% | +0.260 pp |
| Beam top-5 accuracy | 99.377% | 99.586% | +0.208 pp |
| Desired-gain MAE | 4.322 dB | 3.570 dB | -0.752 dB |
| Interference-gain MAE | 3.396 dB | 2.765 dB | -0.630 dB |

The paired 95% interval for the top-1 change is [-0.301, 0.766] percentage
points, so the small top-1 increase is not statistically resolved at the
vehicle-cluster level. The top-3 and top-5 intervals are [0.162, 0.382] and
[0.068, 0.440] percentage points. The desired- and interference-gain MAE
intervals are [-1.073, -0.497] dB and [-0.926, -0.415] dB, respectively.

On nonzero-channel links, top-1 changes from 85.593% to 85.844% (paired interval
[-0.633, 0.772] percentage points), while desired- and interference-gain MAE
decrease from 3.655 to 3.242 dB and from 2.978 to 2.584 dB. The gain improvements
therefore do not arise from zero-channel labels.

For continuously retained states older than ten frames, the primary comparison
still gives +0.284 percentage points in top-1 accuracy and MAE reductions of
0.732 dB and 0.621 dB. Over nonzero links older than 300 frames, the beam top-1
change is essentially neutral (-0.059 percentage points), while both MAEs
remain lower by about 0.42 and 0.41 dB.

## Interpretation and scope

The experiment supports persistent one-frame inference for the statefully
fine-tuned checkpoints. It does not show a statistically clear top-1 advantage
over the existing window pipeline, but it recovers the old model's state-drift
loss, improves ranked beam coverage, and gives clear gain-prediction benefits.

The new checkpoints received 30 additional fine-tuning epochs and use a
deployment-matched chronological training organization. Hence this is a
comparison of the two practical training-and-inference pipelines, not an
equal-training-budget ablation that attributes every improvement solely to
TBPTT. A strict causal ablation would train window and stateful models from the
same initialization for the same number of updates and noise realizations.

## Artifacts

- Baseline Git snapshot: `2710ecf`
- Data builder: `experiment/prepare_stateful_trajectories.py`
- Stateful trainer: `experiment/train_stateful_tbptt.py`
- Direct comparison: `experiment/compare_tbptt_results.py`
- Unit tests: `test_stateful_tbptt.py`
- Training data and metadata:
  `experiment/results/stateful_tbptt_20260912/training_trajectories.{npz,json}`
- Selected runs: `experiment/results/stateful_tbptt_20260912/finetune_lr1e-4/`
- New-model test: `experiment/results/stateful_tbptt_20260912/test_lr1e-4_cpu/`
- Direct paired result:
  `experiment/results/stateful_tbptt_20260912/comparison_paper_window_vs_tbptt_stateful/`

