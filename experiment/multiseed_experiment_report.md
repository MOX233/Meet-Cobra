# Multi-seed evaluation report

## Scope

This experiment replaces each single-seed 30-s curve point with the mean of five
independent evaluation seeds for all methods that will appear in the revised
paper figure set:

- MEET-COBRA;
- Oracle-MC and Oracle-CR-LB;
- Reactive-OBRA;
- w/o GAP-HO, w/o PET-BF, and w/o OTR-RA;
- MTS-GS-HBF-adapted;
- O-MAPPO-adapted.

The traffic-rate grid is 1, 3, ..., 35 Mbps (18 points).  In total, the data set
contains 9 methods x 18 rates x 5 seeds = 810 complete simulation points.

## Protocol

- Evaluation seeds: 1, 2, 3, 4, and 5.
- Test interval: 800--830 s of the paper mobility trace.
- Learned predictors and the O-MAPPO policy are frozen across evaluation seeds.
- Each method/rate run resets the requested seed.  The seeds change stochastic
  traffic arrivals and Rician fading; the mobility trace is held fixed.
- The LSTM gain/beam/interference predictions were precomputed on GPU 0.  The
  cache was checked against the original per-vehicle GPU inference path, with
  exact agreement for all eight simulator output arrays in an end-to-end check.
- A historical MEET-COBRA seed-1, 1-Mbps point was reproduced exactly before the
  multi-seed run.  All eight raw arrays agreed element by element, including the
  original average power of 4.0095050167 W and violation probability of
  0.1096468325%.
- No curve interpolation or post-hoc smoothing filter is applied.

For each seed-level manuscript metric, the reported center is the arithmetic
mean.  Uncertainty is a two-sided 95% Student-t confidence interval,

\[
\bar{x} \mathbin{\pm} t_{0.975,4}\,s/\sqrt{5}, \qquad
t_{0.975,4}=2.776445.
\]

This interval quantifies evaluation randomness conditional on the fixed
mobility trace and frozen learned models.  It is not a training-seed confidence
interval.

## Validation

The validation script checked all 810 point JSON/NPZ pairs, their complete rate
grids, expected frame counts, finite raw arrays, and the stored manuscript
metrics.  It then recomputed power, queue violation probability, mean normalized
backlog proxy, its 90th and 99th percentiles, and the macro-BS association ratio
from the raw arrays.  All methods passed 90/90 points, and the maximum absolute
error between a recomputed metric and its stored value was 0.

Across the non-lower-bound methods, the validation covers approximately 2.94
billion normalized-backlog samples.  The complete multi-seed result directory
occupies approximately 18 GiB.

## Representative results

Values below are five-seed means with 95% confidence half-widths.

| Rate | Method | Power (W) | Violation (%) | P99 backlog proxy (ms) |
|---:|:---|---:|---:|---:|
| 15 Mbps | MEET-COBRA | 33.8592 +/- 0.0018 | 0.16894 +/- 0.00023 | 2.10386 +/- 0.00015 |
| 15 Mbps | MTS-GS-HBF-adapted | 40.0997 +/- 0.2185 | 0.04916 +/- 0.00129 | 5.28735 +/- 0.03223 |
| 15 Mbps | O-MAPPO-adapted | 71.4615 +/- 0.6985 | 1.9011 +/- 0.0552 | 170.158 +/- 6.824 |
| 25 Mbps | MEET-COBRA | 97.1097 +/- 0.0732 | 0.72729 +/- 0.00946 | 11.3201 +/- 0.3111 |
| 25 Mbps | MTS-GS-HBF-adapted | 111.3743 +/- 0.1876 | 1.6612 +/- 0.0631 | 80.8964 +/- 2.4037 |
| 25 Mbps | O-MAPPO-adapted | 147.4030 +/- 0.9134 | 14.4262 +/- 0.3841 | 2814.622 +/- 202.208 |

Using the joint descriptive criterion of mean violation <= 1% and mean P99
backlog proxy <= 20 ms, the largest tested admissible rates are:

| Method | Largest tested admissible rate |
|:---|---:|
| Oracle-MC | 29 Mbps |
| MEET-COBRA | 27 Mbps |
| MTS-GS-HBF-adapted | 21 Mbps |
| w/o GAP-HO | 19 Mbps |
| w/o PET-BF | 17 Mbps |
| w/o OTR-RA | 15 Mbps |
| Reactive-OBRA | 13 Mbps |
| O-MAPPO-adapted | 11 Mbps |

This criterion is a compact diagnostic and is not an additional optimization
constraint used during simulation.

## Did multi-seed averaging make the curves smoother?

It reduced random seed-to-seed fluctuation, but the visual effect is modest.
Relative to seed 1, the mean normalized second-difference roughness averaged
over eligible methods changed by approximately 0.0% for power, -0.2% for
violation probability, -0.9% for P90, -3.6% for P99, and -4.1% for macro-BS
association ratio.  Confidence intervals for MEET-COBRA and the oracle methods
are especially narrow.

This is scientifically plausible: each 30-s point already averages hundreds of
frames and millions of vehicle-slot samples, while the same mobility trace and
frozen policies are used across seeds.  The remaining bends and non-monotonic
segments therefore mostly reflect load-transition thresholds and the fixed
trace, rather than Monte-Carlo noise.  Increasing the number of evaluation
seeds alone would narrow confidence intervals but is unlikely to make these
structural transitions much smoother.  If stronger generalization evidence is
needed, the next statistically distinct experiment should vary mobility/channel
traces; training-seed robustness for RL should be reported separately.

## Artifacts

- `results_multiseed/validation_report.json`: complete raw-data validation.
- `results_multiseed/aggregate/seed_level_curve_data.csv`: 810 seed-level rows.
- `results_multiseed/aggregate/multiseed_curve_data.csv`: 162 aggregate rows,
  including mean, sample standard deviation, SEM, and 95% CI half-width.
- `results_multiseed/aggregate/protocol.json`: machine-readable aggregation
  protocol.
- `../latexCodes/figures/*_WBL_MS.pdf`: four multi-seed paper figures with 95%
  confidence bands.

