# R2C1 interference validation: generated result tables

Completed cases: 21/21. Protocol SHA256: `7ba6aa652a6055c62e2d5830fdaca965654f6939659157e22cb30ea0217cf343`.

All legacy cases exactly reproduce every saved array of the capped-GAP full-grid MEET-COBRA run.
The tables average per-seed statistics. U is the existing queue-length-based latency violation metric.
Only micro users with allocated RBs enter the matched-schedule physical error statistics.

## Fixed legacy schedules: manuscript surrogate vs explicit directional interference

| Mbps | Seeds | Mean-I ratio | Median absolute aggregate-SINR error (dB) | P95 absolute error (dB) | Potential-service ratio | RB-overlap mean error (%) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | [1] | 1.8035 | 2.668 | 9.509 | 0.9227 | -0.0679 |
| 13 | [1, 2, 3] | 0.8682 | 7.843 | 13.624 | 0.8076 | 0.0057 |
| 23 | [1] | 0.9141 | 7.326 | 11.951 | 0.8057 | 0.0000 |
| 29 | [1, 2, 3] | 2.4332 | 7.492 | 11.342 | 0.8026 | 0.0004 |
| 35 | [1] | 4.6117 | 7.572 | 11.116 | 0.8104 | 0.0000 |

## Separating the approximations on the same schedules

| Mbps | Comparison | Mean-I ratio | Median abs. aggregate-SINR error (dB) | P95 abs. error (dB) | Potential-service ratio |
|---:|---|---:|---:|---:|---:|
| 1 | gain_only | 1.7773 | 2.044 | 9.432 | 0.9511 |
| 1 | overlap_only_directional | 1.0148 | 0.423 | 7.794 | 0.9701 |
| 1 | overlap_only_average | 0.9993 | 2.106 | 7.443 | 0.9509 |
| 1 | legacy_to_frame_mean | 1.5816 | 1.741 | 4.280 | 0.9521 |
| 1 | frame_to_slot_mean | 0.9990 | 0.050 | 0.235 | 1.0000 |
| 13 | gain_only | 0.8684 | 5.765 | 9.068 | 0.9157 |
| 13 | overlap_only_directional | 0.9997 | 3.639 | 16.688 | 0.8820 |
| 13 | overlap_only_average | 1.0000 | 2.586 | 13.060 | 0.9145 |
| 13 | legacy_to_frame_mean | 1.8291 | 3.303 | 5.296 | 0.8932 |
| 13 | frame_to_slot_mean | 1.0001 | 0.113 | 0.346 | 0.9999 |
| 23 | gain_only | 0.9159 | 6.263 | 9.166 | 0.8823 |
| 23 | overlap_only_directional | 0.9981 | 2.240 | 10.493 | 0.9132 |
| 23 | overlap_only_average | 1.0001 | 0.109 | 1.304 | 0.9925 |
| 23 | legacy_to_frame_mean | 2.2398 | 3.306 | 5.317 | 0.8825 |
| 23 | frame_to_slot_mean | 1.0000 | 0.115 | 0.353 | 0.9999 |
| 29 | gain_only | 2.4346 | 6.864 | 9.164 | 0.8424 |
| 29 | overlap_only_directional | 0.9995 | 1.572 | 7.052 | 0.9527 |
| 29 | overlap_only_average | 1.0001 | 0.112 | 1.223 | 0.9948 |
| 29 | legacy_to_frame_mean | 2.1994 | 3.154 | 5.138 | 0.8961 |
| 29 | frame_to_slot_mean | 1.0000 | 0.112 | 0.342 | 0.9999 |
| 35 | gain_only | 4.6136 | 7.272 | 9.829 | 0.8310 |
| 35 | overlap_only_directional | 0.9996 | 0.929 | 4.125 | 0.9752 |
| 35 | overlap_only_average | 1.0000 | 0.000 | 0.000 | 1.0000 |
| 35 | legacy_to_frame_mean | 2.1401 | 2.953 | 4.949 | 0.9078 |
| 35 | frame_to_slot_mean | 1.0000 | 0.109 | 0.331 | 0.9999 |

## Closed-loop system metrics (optimizer inputs unchanged)

| Mbps | Service evaluator | Seeds | Power (W) | U (%) | P99 proxy (ms) |
|---:|---|---|---:|---:|---:|
| 1 | legacy | [1] | 4.9258 | 0.1951 | 18.8790 |
| 13 | legacy | [1, 2, 3] | 29.6650 | 0.2420 | 2.3212 |
| 13 | average_expected | [1, 2, 3] | 27.1843 | 0.2274 | 2.3057 |
| 13 | directional_explicit | [1, 2, 3] | 24.9769 | 0.1836 | 2.3168 |
| 23 | legacy | [1] | 72.2333 | 0.7312 | 11.1271 |
| 29 | legacy | [1, 2, 3] | 157.5207 | 0.8986 | 16.0527 |
| 29 | average_expected | [1, 2, 3] | 148.2759 | 0.1499 | 1.5431 |
| 29 | directional_explicit | [1, 2, 3] | 144.7598 | 0.0999 | 1.4556 |
| 35 | legacy | [1] | 185.7687 | 38.8945 | 11621.0066 |

## Interpretation boundaries

- Mean-I ratios are RB-weighted ratios of total interference; SINR errors first average interference over the user's allocated RBs. They are not per-RB SINR quantiles.
- Explicit service sums each RB's Shannon service with the original pilot efficiency, preserving nonlinear effects of interference variability.
- Gain-only holds expected overlaps fixed; overlap-only holds directional or average gains fixed.
- Closed-loop variants change physical micro-tier queue service, not information supplied to the predictors or optimizers. They diagnose robustness, not a fully harmonized new system implementation.
- An unchanged fixed schedule necessarily has unchanged transmit power; only separately run closed-loop variants support power/U comparisons.
- Original simulator uses a frame-level maximum-element interference gain. The manuscript surrogate uses a slot-level Frobenius mean. The frame-average bridge separates the aggregation and temporal aspects.
- Existing Sionna RT channels, slot fading, beam report timing, independent data streams and frequency-flat links are retained. No new ray tracing or packet-level/waveform simulation is claimed.
- All seeds share the same map, vehicle trajectory collection and model weights. These results do not establish cross-deployment robustness.
