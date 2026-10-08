# Fig. 4 update with the beam-averaged interference predictor

Date: 2026-09-22.

## Scope and sources

The completed run is `results/revision_directional_20260922/training`.
Both stages completed 100 epochs; this is not the earlier run with the
changed zero-channel floor. Only the interfering-link gain predictor was
retrained. Beam and desired-link gain histories and selected checkpoints
remain those in `results/stateful_tbptt_unified_split_20260913`.

The new interfering-gain target is the uniform DFT beam-average gain.
The dB convention is `20*log10(sqrt(mean(abs(H)**2))+1e-9)`, retaining
-180 dB for a zero channel. Its dataset SHA-256 is
`4fb18c3292a047ec2a70fb1bd42b218f122d5281dc5358434142656a5b4c9c14`.
The vehicle split SHA-256, shared with the retained models, is
`6af64204659e247f692634efc9db5ecb9d9d1beaefa94986a0265acb7770dc7c`.

The actual new training configuration uses batch size 128 in both stages,
seed 20, initial learning rates 0.001 and 0.0001, AdamW weight decay 0.0001,
and stage-II TBPTT length 10. The retained models used batch size 512 in
stage I and 128 in stage II. No training settings were changed in this update.

## Validation results used in Fig. 4

| Predictor/metric | Best stage-I validation value | Best stage-II validation value | Stage-II epoch |
|---|---:|---:|---:|
| Beam Top-1 | 86.682936% | 89.164306% | 88 |
| Beam Top-3 | 97.928566% | 98.654519% | 18 |
| Desired-link gain MAE | 4.789721 dB | 3.111440 dB | 59 |
| Beam-averaged interfering-link gain MAE | 3.850237 dB | 2.508201 dB | 100 |

The selected interference checkpoint is stage-II epoch 100, i.e., plotted
epoch 200. Its SHA-256 is
`af270976bfcd91d9661faa0bb752cc87eab85322b0da1dad3958f7d62249e5cb`.
The stage-I checkpoint used to initialize it is also epoch 100, with SHA-256
`52279c41120db1a5e27da38c508ea1893f5bbdbdac180e968770a54779cfe9ac`.

The beam checkpoint remains selected by Top-1 at stage-II epoch 88;
the Top-3 star is an independent metric maximum, not a second deployed model.
Its Top-M_P thresholds remain M_P=4 for greater than 99% and M_P=17 for
greater than 99.9%. The old interfering MAE of 2.348142 dB concerned a
different label and must not be interpreted as a directly comparable score.

`results/revision_directional_20260922/fig4_summary.json` records the exact
metrics, history paths, and history hashes. The plot retains the existing
two-stage layout, colors, fonts, legends and reference lines. A small right
margin prevents the new epoch-200 star from being clipped. No curve smoothing
or substitution of test-set scores was performed.

## Stateful inference timing

The final three-model bundle is saved in
`results/revision_directional_20260922/selected_models`, with source paths
and checkpoint hashes in `bundle.json`. The assembly checks completed
training, label convention, shared vehicle split, and checkpoint hashes.

The existing CPU benchmark was repeated with the same data, trajectory
selection, seed, CPU affinity and protocol as the prior timing report:
AMD EPYC 7742, one thread, PyTorch 2.6.0, FP32, batch size one, 16 continuous
trajectories, 4,722 frames per round, three rounds, and 100 warmup steps per
round. The 14,166 timed inferences give:

- Combined median: 5.384275 ms (5.38 ms in manuscript).
- Combined 95th percentile: 5.7215435 ms (5.72 ms in manuscript).
- Three-model matrix FLOPs: 4,198,416, unchanged.
- Total trainable parameters: 2,127,912, unchanged.

Timing includes conversion of preprocessed CSI, all three stateful forward
passes, state retention, ranked top-5 selection and gain denormalization.
Acquisition, preprocessing, report transmission and system decisions are
excluded. It is a server CPU measurement, not a vehicle-device energy test.
Normal timing variability must not be interpreted as a weight-induced speedup.

Incremental outputs were checked against full-prefix inference at lengths
1, 10, 11, 21, 100 and 301. Ranked beam indices match; maximum gain deviation
is 0.000030518 dB. Trajectory resets match a fresh state. Parameters and
buffers remained unchanged. Raw timings and checks are under
`results/revision_directional_20260922/nn_overhead_cpu_1thread`.

## Reproduction

Run from the repository root in the `sionna` environment. Figure generation
updates the two Fig. 4 PDF assets but does not overwrite old training histories
or their summary. For the timing command, select a new output directory if the
documented directory already exists; the benchmark refuses to overwrite it.

```bash
python experiment/plot_stateful_training_curves.py --interfering-stage1-results experiment/results/revision_directional_20260922/training/stage1 --interfering-stage2-results experiment/results/revision_directional_20260922/training/stage2 --summary experiment/results/revision_directional_20260922/fig4_summary.json --figures latexCodes/figures
python experiment/revision_training.py assemble --training experiment/results/revision_directional_20260922/training --output experiment/results/revision_directional_20260922/selected_models
python experiment/benchmark_stateful_nn_overhead.py --checkpoint-root experiment/results/revision_directional_20260922/selected_models --output experiment/results/revision_directional_20260922/nn_overhead_cpu_1thread --trajectories 16 --min-frames 200 --rounds 3 --warmup 100 --threads 1 --affinity 0,1,2,3 --seed 20260913
python -m unittest test_plot_stateful_training_curves test_stateful_nn_overhead -v
```

Nine regression checks passed. Checkpoint hashes, plotted metrics, retained
beam and desired-gain results, and all original reviewer quotations were
checked. Both LaTeX documents compile successfully (15 manuscript pages and
24 response pages); the plotted figure and its manuscript placement were
visually inspected. The existing notation-table overfull-box warning is
unrelated to this update and was not changed. Revised manuscript and response numbers are
marked red; original reviewer quotations and submitted-text deletion records
are preserved. R2C1 now reports the new prediction result; R1C7, R2C6 and
R3 Major C2 use the remeasured inference times. Static model counts and report
payloads are unchanged.

This update does not generate simulation caches, run system experiments or
update Fig. 5–Fig. 8. The full directional-interference evaluation, L1038
quantitative range and final baseline conclusions remain pending.
