# Queue/load/interference/energy-aware PQL-BA-adapted: experiment report

Date: 2026-08-29

## Executive conclusion

Adding queue, serving-BS load, interference, and energy information materially improves PQL-BA, but the resulting methods are still not competitive enough to serve as the principal new baseline in the MEET-COBRA revision.

The strongest tested method is a hierarchical PQL-BA-adapted variant. Its Q learner selects one of five BSs, while an event-triggered local search selects the best 32×8 TX/RX beam pair within the chosen micro BS. On the disjoint 800–830 s test trace it reduces queue violation from 39.54%/85.41% for the previous adapted PQL-BA to 7.30%/53.86% at 1/19 Mbps. This is a large reliability improvement, but it raises power to 58.63/185.13 W. Reactive-OBRA achieves 12.28/78.11 W and 0.157%/8.09% under the same two loads, while MEET-COBRA achieves 4.01/44.38 W and 0.110%/0.340%.

Thus, PQL-BA-adapted is useful as an algorithmic diagnostic or supplementary baseline, but it should not replace a published queue/load-aware joint HO–BF method as the requested strong literature baseline. The extensive adaptations also mean it is no longer the algorithm reported in the source PQL-BA paper.

## Adapted state, reward, and training environment

The source method is Huynh et al., “Optimal Beam Association for High Mobility mmWave Vehicular Networks: Lightweight Parallel Reinforcement Learning Approach,” IEEE Transactions on Communications, 2021 ([arXiv manuscript](https://arxiv.org/abs/2005.00694)). The shared asynchronous tabular-Q structure and distance-triggered decisions are retained.

### Contextual state

In addition to quantized serving-link RSSI, current action, heading, and coarse 2D location, the adapted state contains:

- normalized queue ratio, (Q_v/Q_v^{\mathrm{ub}});
- serving-BS RB-load ratio;
- serving-link interference-to-noise ratio bin;
- optionally, offered-traffic class. The final model omits the traffic class so experience at different offered rates can update shared Q states.

The queue, load, and interference bins are deliberately coarse to limit state explosion. The final location grid is 100 m and the decision distance is 10 m.

### Multi-objective reward

The per-frame reward accumulated between two decision events is

\[
r_v = c + w_s\min\!\left(\frac{S_v}{\lambda_v T_f},2\right)
-w_q\min\!\left(\frac{Q_v}{Q_v^{\rm ub}},10\right)
-w_{\rm vio}{\bf 1}\{Q_v>Q_v^{\rm ub}\}
-w_E P_v-w_L\rho_{b_v}.
\]

Here (S_v) is useful service, (P_v) is the RB-weighted transmit power attributed to the vehicle, and (ho_{b_v}) is the serving-BS load. The selected reward uses (w_s=1), (w_q=0.5), (w_{\rm vio}=4), (w_E=0.2), and (w_L=0). The action-independent offset (c=10) keeps ordinary learned values above the zero initialization of unseen actions and does not change action ordering for equal-duration transitions. Frozen selection is additionally restricted to visited actions in a known state.

### Fluid training surrogate

Putting the full 100-slot OTR-RA simulator inside every Q update would require several hours per candidate. Training therefore uses a frame-level fluid surrogate that retains:

- the selected beam gain and no-BF interfering gains from the same traces;
- fixed-point micro-BS activity and inter-cell interference;
- macro/micro bandwidth, RB count, power, noise figure, and beam-pilot overhead;
- OTR-like capacity-priority RB allocation;
- per-vehicle traffic, queue evolution, and the same 20 ms normalized queue threshold.

All final validation and test claims use the original slot-level OTR-RA simulator with Poisson arrivals and Rician fading; the fluid model is used only for training and candidate screening.

## Candidate screening

Screening used 200–320 s for training, 320.1–340 s for proxy validation, four epochs with a 1/7/13/19 Mbps load schedule, and seed 1. Test data were never used for candidate selection.

### Spatial-state resolution

All three candidates include queue, load, interference, and heading states and use the QoS reward.

| Location state | 1 Mbps: power / violation | 19 Mbps: power / violation | Known-state coverage at 1/19 Mbps |
|---|---:|---:|---:|
| None | 53.65 W / 48.85% | 123.31 W / 80.39% | 71.3% / 86.5% |
| 100 m grid | 53.93 W / **38.99%** | 179.06 W / **66.70%** | 43.1% / 42.8% |
| 200 m grid | 53.13 W / 49.75% | 170.69 W / 69.20% | 55.4% / 59.5% |

The 100 m grid has lower coverage but the best reliability, confirming that spatial information is valuable and that coverage alone does not determine performance.

### Reward variants with the original 129-action design

The original adapted action space contains one macro action and 4×32 micro-BS/TX-beam actions. The following proxy results use the 100 m state.

| Reward | 1 Mbps: power / violation | 19 Mbps: power / violation |
|---|---:|---:|
| QoS, no energy term | 53.93 W / 38.99% | 179.06 W / 66.70% |
| QoS + energy 0.05 | 53.49 W / 46.51% | 185.01 W / 67.03% |
| QoS + energy 0.20 | 53.50 W / 43.97% | 178.97 W / **63.41%** |
| QoS + energy 0.50 | 53.74 W / 42.66% | 184.41 W / 66.89% |
| Strong queue/violation | 53.72 W / 44.67% | 185.27 W / 65.32% |
| Strong queue/violation + energy 0.20 | 54.20 W / 40.97% | 174.88 W / 66.86% |
| Delay-heavy + energy 0.10 | 53.67 W / 45.47% | 184.74 W / 67.91% |

No weight is uniformly best. Energy weight 0.20 improves the high-load violation rate but does not lower both power and violation. Simply increasing queue/violation penalties is not monotonic.

After eight full epochs on 200–740 s, the 129-action QoS and QoS+energy-0.20 models contain about 68,400 states and 145,900 visited state-action pairs. On independent 740.1–800 s proxy validation, their high-load violation rates are 87.22% and 88.23%, respectively. A 5 s slot-level validation gives:

| 129-action method | 1 Mbps: power / violation | 19 Mbps: power / violation |
|---|---:|---:|
| Context + QoS | 52.96 W / 28.23% | 97.46 W / 77.12% |
| Context + QoS + energy 0.20 | 53.18 W / 29.07% | 120.77 W / 72.21% |

The known-state ratios are 77–94%, so poor performance is not primarily an unseen-state fallback problem. The 129-action table remains too sparse at the action level and holds exact TX beams too long under mobility.

## Hierarchical PQL-BA-adapted

### Action and beam-search design

The hierarchical variant reduces the Q action space from 129 to five serving-BS actions. For a selected micro BS, the vehicle searches its 32×8 TX/RX DFT pairs at an event and tracks the selected pair until the next event. This is still joint HO–BF coordination, but BF is a deterministic local subproblem rather than part of the Q action.

An exhaustive search requires 256 measurements. Putting all 256 pilots in one slot yields 100% overhead and a zero-capacity RB in the existing scheduler. The implementation therefore spreads them as 55, 55, 55, 55, and 36 pilots over five slots, followed by one tracking pilot per remaining slot. Every slot stays strictly below 100% overhead and the total search cost remains 256 pilots.

Under the same short screening protocol, the corrected hierarchical QoS+energy-0.20 model obtains 51.25 W/5.59% at 1 Mbps and 181.27 W/32.38% at 19 Mbps. The corresponding 129-action candidate obtains 53.50 W/43.97% and 178.97 W/63.41%. Most of the improvement therefore comes from action factorization and current local BF, not merely adding more variables to the Q state.

### Explicit serving-load penalties

Two additional rewards directly penalize the serving-BS load:

| Load weight | 1 Mbps: power / violation | 19 Mbps: power / violation |
|---:|---:|---:|
| 0 | **51.25 W / 5.59%** | **181.27 W / 32.38%** |
| 2 | 55.45 W / 7.80% | 184.65 W / 39.94% |
| 5 | 54.87 W / 5.43% | 184.33 W / 41.93% |

Direct load penalties are not beneficial: they increase high-load switching and worsen the high-load queue metric. The final reward therefore keeps load in the state but sets the explicit load weight to zero.

## Full hierarchical training and validation

The final candidate was trained for eight epochs on 200–740 s, cycling through 1/7/13/19 Mbps twice. It finishes with:

- epsilon 0.0627;
- 9,718 Q states and 21,375 visited state-action pairs;
- known-state proxy-validation coverage of 96.8%/99.0% at 1/19 Mbps;
- proxy-validation power of 55.97/185.67 W;
- proxy-validation violation of 6.23%/45.58%.

In the second load cycle, exploration-period violation falls from 23.08% to 8.28% at 1 Mbps, from 24.03% to 15.14% at 7 Mbps, and from 27.06% to 23.79% at 13 Mbps. At 19 Mbps it only falls from 40.56% to 39.21%, indicating that the learned association remains inadequate near network saturation.

A 10 s slot-level validation on 740.1–750 s produces 56.34/183.47 W and 7.12%/41.90% violation. This is reasonably consistent with the fluid validation and supports using the surrogate for candidate screening.

## Final 30 s test

The policy is frozen and tested on the disjoint 800–830 s trace with the same seed, Poisson traffic, Rician fading, interference, OTR-RA, and queue threshold as the manuscript experiments.

| Traffic | Method | Power (W) | Queue violation (%) |
|---:|---|---:|---:|
| 1 Mbps | MEET-COBRA | 4.010 | 0.110 |
|  | Reactive-OBRA | 12.283 | 0.157 |
|  | Previous PQL-BA adaptation | 53.499 | 39.539 |
|  | **Hierarchical PQL-BA-adapted** | **58.627** | **7.300** |
| 19 Mbps | MEET-COBRA | 44.378 | 0.340 |
|  | Reactive-OBRA | 78.111 | 8.090 |
|  | Previous PQL-BA adaptation | 82.400 | 85.415 |
|  | **Hierarchical PQL-BA-adapted** | **185.126** | **53.862** |

Relative to the previous PQL-BA adaptation, the hierarchical method reduces violation by 32.24 percentage points (81.5% relative) at 1 Mbps and 31.55 points (36.9% relative) at 19 Mbps. The cost is 9.6% more power at 1 Mbps and 124.7% more at 19 Mbps.

Additional final-test statistics are:

| Traffic | Queue proxy | Known-state ratio | HO/(vehicle·s) | Beam switches/(vehicle·s) | Pilots/(vehicle·slot) | Macro association |
|---:|---:|---:|---:|---:|---:|---:|
| 1 Mbps | 189.7 ms | 79.2% | 0.312 | 0.369 | 0.449 | 63.49% |
| 19 Mbps | 1927.8 ms | 98.5% | 0.634 | 0.682 | 0.744 | 39.15% |

At low load, the policy uses the macro BS as a high-power reliability fallback. At high load, it associates fewer vehicles with the macro but drives the whole system close to the approximately 185 W fully occupied-RB power ceiling. Even at that power, more than half of queue observations violate the threshold. Therefore, the method is not energy efficient at comparable QoS.

## Scientific interpretation and recommendation

1. **Context helps:** queue/load/interference state and a QoS-aware reward improve PQL-BA, especially at low load.
2. **Energy weighting is a tradeoff, not a cure:** moderate energy weighting can improve one load point but does not produce a Pareto improvement across power and reliability.
3. **Action factorization is the largest gain:** reducing Q learning to five BS actions and solving BF locally is far more effective than learning among 129 exact TX-beam actions.
4. **Local rewards cannot fully coordinate congestion:** every vehicle optimizes its own queue/service/power using a shared table, but there is no instantaneous joint action or global congestion credit assignment. At high load, the policies overuse already saturated resources and switch frequently.
5. **The improved method is no longer source PQL-BA:** queue/load/interference states, energy/QoS reward, heterogeneous data macro, fluid loaded training, and hierarchical BF are substantial new design choices.

Recommendation:

- Do not use hierarchical PQL-BA-adapted as the sole or principal new strong baseline. It remains substantially worse than Reactive-OBRA and MEET-COBRA and could distract from the revision with extensive implementation caveats.
- It can be reported as a supplementary sensitivity study showing that a carefully strengthened PQL family still struggles with network-wide energy–latency coordination.
- If a PQL-derived method must appear in the main manuscript, label the original source-faithful adaptation and the hierarchical extension separately. Do not imply that the hierarchical extension is the published PQL-BA algorithm.
- Further improvement would require a global or difference reward, coordinated multi-agent actions, and function approximation/generalization across actions. At that point the method is closer to a new MARL baseline than a PQL-BA adaptation; a published queue/load-aware joint HO–BF method remains preferable for satisfying the reviewer.

## Implementation and valid artifacts

Code:

- `utils/pql_ba.py`: contextual state support, visited-action frozen selection, hierarchical BS actions, and exhaustive beam-pair search.
- `utils/pql_ba_adapted.py`: fluid training environment and multi-objective reward presets.
- `utils/pql_ba_sim.py`: exact contextual decisions and multi-slot sweep-pilot evaluation.
- `experiment/pql_ba_adapted_experiment.py`: screening, exact validation, training, and testing driver.
- `test_pql_ba.py`: eleven regression tests, including backward-compatible policy loading.

Primary result directories:

- `experiment/results_pql_ba_adapted/full_tuning_top2_context_z10/`: full 129-action contextual policies.
- `experiment/results_pql_ba_adapted/exact_full_context129_5s/`: exact diagnostic validation for the 129-action policies.
- `experiment/results_pql_ba_adapted/full_hierarchical_energy020_spread/`: final hierarchical policy and training/validation history.
- `experiment/results_pql_ba_adapted/exact_full_hierarchical_10s/`: exact 10 s validation.
- `experiment/results_pql_ba_adapted/evaluation_hierarchical_energy020_30s/`: final 30 s test results.
- `experiment/results_pql_ba_adapted/screen_hierarchical_load_rewards/` and `screen_hierarchical_energy020_spread_control/`: explicit-load-penalty controls.

The early `screen_hierarchical_location100/` policies were trained before the 256-pilot multi-slot correction and are diagnostic only. The failed `exact_screen_hierarchical_5s/` directory and the interrupted `tuning_context_shared_z10/` directory must not be used for claims. The final policy and all final reported results use the corrected spread-pilot protocol.

All eleven unit tests pass, and all PQL-BA/PQL-BA-adapted Python modules pass `py_compile`.
