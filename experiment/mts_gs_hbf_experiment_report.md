# MTS-GS-HBF-adapted full-grid experiment report

Date: 2026-09-02

## Objective

This experiment evaluates whether a non-reinforcement-learning joint handover
and beamforming method can provide a credible additional baseline for
MEET-COBRA.  The implementation is an adaptation, rather than a literal copy,
of the multi-timescale GS/HBF architecture to the manuscript's one-macro/four-
micro BS system model and its queue, energy, interference, and OTR-RA models.

## Adapted algorithm

The controller has three time scales and contains no learned model, training
data, replay buffer, actor, critic, or value function.

1. Slow-time handover/user association uses capacity-aware many-to-one
   Gale--Shapley deferred acceptance.  A vehicle ranks BSs using predicted
   per-RB rate, estimated BS load, energy per delivered Mbit, and handover
   cost.  A BS ranks vehicles using queue urgency, rate, RB demand, and a
   stay bonus.  Continuous per-vehicle RB demands are admitted against each
   BS's RB budget.
2. Medium-time analog beamforming uses the same 32-by-8 DFT codebooks as the
   paper.  A full beam sweep is performed at association epochs; intervening
   frames use 3-by-3 local beam-pair tracking.
3. Fast-time RB allocation uses the manuscript's common OTR-RA routine so that
   the comparison isolates association/beamforming differences.

Every command inferred from frame `t` is applied in frame `t+1`; the simulator
therefore does not use same-frame future channel information.

## Pressure-adaptive extension

Four static designs first established the energy/reliability boundary.  Two
additional designs then continuously changed their preference weights using
the observable queue ratio, without conditioning on the experiment's nominal
traffic-rate label.  The selected `pressure_early` design uses:

- association and full sweep every 5 frames, local tracking every frame;
- queue pressure increasing linearly from queue ratio 0.25 to 1.0;
- energy weight changing from 4.0 to 0.25 as pressure rises;
- load weight changing from 1.0 to 2.5;
- BS-side queue weight changing from 2.0 to 4.0;
- handover penalty/hysteresis changing from 0.50/0.25 to 0.10/0.05; and
- queue-drain weight changing from 0.5 to 1.0.

This makes lightly queued vehicles prefer low-energy, stable links, while
urgent vehicles shift toward high-rate, load-balanced associations.

## Screening

The screening interval was 800--803 s, with seed 1 and traffic rates 1, 19,
and 27 Mbps.  The table entries are `(power W, violation %, mean queue proxy
ms)`.

| Candidate | 1 Mbps | 19 Mbps | 27 Mbps | Selection score |
|---|---:|---:|---:|---:|
| qos_fast | (3.00, 0.038, 9.53) | (101.22, 2.650, 5.48) | (139.38, 13.760, 46.74) | 18.0748 |
| balanced | (3.01, 0.031, 9.51) | (80.10, 2.888, 6.96) | (119.44, 17.941, 117.80) | 22.2688 |
| energy_sticky | (3.01, 0.038, 9.50) | (51.25, 2.494, 7.94) | (126.97, 19.616, 98.62) | 23.3948 |
| slow_energy | (2.85, 0.016, 9.44) | (50.32, 3.651, 10.60) | (97.85, 27.265, 174.14) | 32.0517 |
| pressure_adaptive | (2.99, 0.035, 9.51) | (59.03, 1.395, 4.11) | (135.95, 11.649, 40.41) | 14.4001 |
| pressure_early | (2.99, 0.035, 9.51) | (59.03, 1.395, 4.11) | (136.93, 11.089, 36.95) | **13.8444** |

The score is the sum of the 19/27-Mbps violation percentages plus 0.02 times
mean power and 0.002 times mean queue proxy.  It is only a deterministic
screening rule; there is no gradient-based or RL training.  However, this was
an exploratory screen on the first three seconds of the test trajectory, so
it must not be described as protocol-clean validation in a manuscript.

## Exact 30-second full-grid evaluation

Protocol: the common trajectory from 800--830 s, seed 1, Rician fading, 100
slots/frame, two-frame queue-metric warm-up, and the common OTR-RA allocator.
Each point has 300 frames and 4,089,000 per-vehicle/per-slot queue samples.

| Rate (Mbps) | Power (W) | Violation (%) | Mean queue (ms) | P90 (ms) | P99 (ms) | Macro association (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.584 | 0.037 | 9.332 | 16.473 | 18.876 | 0.770 |
| 3 | 6.510 | 0.002 | 3.770 | 6.170 | 7.443 | 0.673 |
| 5 | 11.142 | 0.005 | 2.576 | 3.967 | 4.790 | 0.538 |
| 7 | 15.814 | 0.011 | 2.098 | 3.040 | 3.738 | 0.550 |
| 9 | 20.380 | 0.015 | 1.845 | 2.542 | 3.298 | 0.562 |
| 11 | 25.567 | 0.021 | 1.693 | 2.235 | 3.733 | 0.827 |
| 13 | 32.608 | 0.032 | 1.603 | 2.028 | 4.383 | 1.646 |
| 15 | 39.869 | 0.051 | 1.558 | 1.877 | 5.328 | 2.238 |
| 17 | 50.486 | 0.134 | 1.597 | 1.767 | 6.557 | 3.671 |
| 19 | 62.537 | 0.311 | 1.803 | 1.683 | 10.021 | 5.221 |
| 21 | 79.742 | 0.497 | 2.103 | 1.617 | 15.528 | 7.919 |
| 23 | 93.711 | 1.092 | 2.905 | 1.570 | 45.581 | 9.327 |
| 25 | 111.513 | 1.608 | 4.043 | 1.530 | 79.133 | 11.636 |
| 27 | 133.944 | 2.309 | 5.379 | 1.500 | 116.225 | 14.304 |
| 29 | 157.879 | 4.129 | 11.964 | 1.498 | 338.133 | 16.999 |
| 31 | 181.990 | 10.781 | 94.120 | 39.434 | 2683.610 | 19.782 |
| 33 | 182.966 | 32.908 | 559.279 | 1672.553 | 8051.374 | 20.213 |
| 35 | 183.031 | 42.657 | 1091.498 | 3787.168 | 11993.224 | 20.543 |

### Comparison with the existing 30-second curves

| Rate | Method | Power (W) | Violation (%) | P99 queue proxy (ms) |
|---:|---|---:|---:|---:|
| 1 | MEET-COBRA | 4.010 | 0.110 | 18.874 |
| 1 | DQL-HBT-adapted | 14.147 | 0.032 | 18.092 |
| 1 | O-MAPPO-adapted | 5.437 | 0.060 | 18.242 |
| 1 | **MTS-GS-HBF-adapted** | **2.584** | 0.037 | 18.876 |
| 19 | MEET-COBRA | 44.378 | 0.340 | 1.916 |
| 19 | DQL-HBT-adapted | 134.626 | 7.812 | 1032.400 |
| 19 | O-MAPPO-adapted | 93.561 | 4.369 | 409.904 |
| 19 | **MTS-GS-HBF-adapted** | 62.537 | **0.311** | 10.021 |
| 27 | **MEET-COBRA** | **125.815** | **0.681** | **10.048** |
| 27 | DQL-HBT-adapted | 182.128 | 32.481 | 9597.058 |
| 27 | O-MAPPO-adapted | 166.103 | 20.355 | 3629.080 |
| 27 | MTS-GS-HBF-adapted | 133.944 | 2.309 | 116.225 |

At 19 Mbps, MTS-GS-HBF-adapted uses 53.5% less power than DQL-HBT-adapted
and 33.2% less than O-MAPPO-adapted, while reducing their violation rates by
96.0% and 92.9%, respectively.  At 27 Mbps, the corresponding power reductions
are 26.5% and 19.4%, and the violation-rate reductions are 92.9% and 88.7%.
At 1 Mbps its violation rate is 0.005 percentage points above DQL-HBT-adapted,
but it uses 81.7% less power.

Across the complete 18-point grid, MTS-GS-HBF-adapted has lower power and
violation probability than O-MAPPO-adapted at all 18 points; it also has lower
mean queue, P90, and P99 at 15, 9, and 13 points, respectively.  Against
DQL-HBT-adapted it has lower power at all 18 points and lower violation
probability at 17 points.  These results support retaining O-MAPPO as the
representative RL baseline while replacing the redundant DQL curve with the
methodologically complementary MTS baseline.

MEET-COBRA retains the important high-load advantage.  At 27 Mbps it uses
6.1% less power than MTS-GS-HBF-adapted and reduces the violation rate from
2.309% to 0.681%; its P99 queue proxy is also 10.048 ms instead of 116.225 ms.
At 19 Mbps the two methods have similar violation rates, but MEET-COBRA uses
29.0% less power and has a much shorter P99 tail.

## Diagnostics and validation

| Rate (Mbps) | HO / vehicle / s | Beam switches / vehicle / s | GS fallbacks / association epoch | Mean overflow RB on triggered frames |
|---:|---:|---:|---:|---:|
| 1 | 0.0239 | 1.0985 | 0.0 | 0.0 |
| 19 | 0.0229 | 0.9587 | 0.0 | 0.0 |
| 27 | 0.0670 | 0.8437 | 0.1 | 0.2265 |

- All saved raw arrays are finite and nonnegative.
- Independent recomputation from the NPZ files exactly reproduces every
  reported manuscript metric (maximum absolute error 0).
- All 18 required traffic rates are present exactly once.  Every point has
  300 frames and 4,089,000 per-vehicle/per-slot queue samples.
- Power is monotonically nondecreasing over the complete load grid.
- There are no matching fallbacks at 1 or 19 Mbps.  At 27 Mbps there are only
  six fallback events over 60 association epochs.
- Actual OTR-RA allocations never exceed 133 macro or 66 micro RBs.
- Six unit tests cover configuration validation, deferred acceptance,
  overload fallback, causal command execution, link-candidate finiteness, and
  minimal end-to-end exact simulation.

As a basic check against the short screening segment driving the conclusion,
the frame-averaged metrics were also recomputed using only 805--830 s.  At
1/19/27 Mbps, respectively, the resulting `(power W, violation %, mean queue
ms)` values are `(2.558, 0.0367, 9.300)`, `(63.264, 0.2135, 1.575)`, and
`(134.223, 1.4529, 2.264)`.  The qualitative conclusions therefore persist
outside the 800--803 s screening portion, although this check is not a
substitute for a fully disjoint validation protocol.

The current Python implementation is research code rather than an optimized
real-time implementation.  Mean measured controller time is about 83--89 ms
per frame on this machine; association epochs average about 233--248 ms and
local-tracking epochs about 42--53 ms.  These timings include Python loops and
full per-vehicle channel/beam computations and should not be presented as an
implementation latency claim without optimization.

## Assessment

`pressure_early` is a promising and appropriately strong non-RL joint HO--BF
baseline.  It has a clearly different mechanism from DQL-HBT and O-MAPPO,
substantially outperforms both RL baselines at the representative medium/high
loads, and still leaves MEET-COBRA a clear energy/tail-reliability advantage
in the high-load regime.  The complete 18-point curve supports its formal
inclusion as the non-RL joint HO--BF baseline, replacing the methodologically
redundant and generally weaker DQL-HBT-adapted curve.

The full curve uses one random seed, matching the existing paper/baseline
curve protocol.  The exploratory candidate selection used 800--803 s; this
must not be called disjoint validation.  For the cleanest formal methodology,
confirm the frozen configuration on a separate pre-800-s validation segment,
without changing it after observing the full test curve.  The paper and
response letter should explicitly call the method an adaptation and explain
that the source method's fast digital-HBF/user-scheduling stage is mapped to
the common OTR-RA stage required by the MEET-COBRA system model.

## Reproducibility artifacts

- Algorithm: `utils/mts_gs_hbf.py`
- Exact simulator: `utils/mts_gs_hbf_sim.py`
- Experiment driver: `experiment/mts_gs_hbf_experiment.py`
- Tests: `test_mts_gs_hbf.py`
- Screening outputs: `experiment/results_mts_gs_hbf/screen/` and
  `experiment/results_mts_gs_hbf_adaptive/screen/`
- Selected 30-second results and raw arrays:
  `experiment/results_mts_gs_hbf/pressure_early/exact_800_830_seed1/`
- Updated full-grid figures:
  `latexCodes/figures/{power,violation_prob,latency_90th_99th,BS0_assoc_ratio}_comparison_curves_WBL.pdf`
