# Adapted PQL-BA baseline: experiment report

Date: 2026-08-29

## Executive conclusion

The adapted PQL-BA implementation is functional: its training reward improves, the frozen policy generalizes to a disjoint test trace under its own single-link throughput objective, and its handover/beam actions can be evaluated in the same loaded-network simulator as MEET-COBRA. However, it is not a suitable **primary or sole strong baseline** for the revised manuscript. Even the validation-selected version produces queue-violation probabilities of 39.54% at 1 Mbps and 85.41% at 19 Mbps, compared with 0.110%/0.340% for MEET-COBRA and 0.157%/8.090% for Reactive-OBRA.

PQL-BA remains defensible as a **secondary literature/diagnostic baseline** if the reviewer explicitly asks for this named method. If included, the manuscript must state that the original algorithm maximizes useful single-user data and does not observe queues, multi-user load, inter-cell interference, or energy. It should not be presented as a load-aware energy-efficiency competitor.

## Source method and adaptations

The source is Huynh et al., “Optimal Beam Association for High Mobility mmWave Vehicular Networks: Lightweight Parallel Reinforcement Learning Approach,” IEEE Transactions on Communications, 2021 ([arXiv manuscript](https://arxiv.org/abs/2005.00694)). It is an event-driven semi-Markov beam-association method: multiple vehicles asynchronously update one shared tabular Q function, the state contains quantized serving-link RSSI and the current beam, the action selects a serving mmWave beam, and the reward is useful data delivered over an event interval.

The following changes are required to make the method executable under the MEET-COBRA system model.

| Item | Source PQL-BA | Adaptation used here | Reason |
|---|---|---|---|
| Macro BS | LTE macro is control-only | Action 0 makes the macro BS a data-serving option | The manuscript has a heterogeneous data plane with one macro and four micro BSs |
| Beam action | Select one mmBS beam | 129 actions: one macro action plus 4 micro BSs × 32 TX beams | Jointly represents HO and transmit beam selection |
| RX beam | Not a separate action in the reported table | For a selected TX beam, the receiver locally sweeps all 8 RX beams | Produces a concrete 32×8 beam pair without expanding the Q action space to 1025 actions |
| Mobility state | One-dimensional road zones | Optional four-bin heading and coarse two-dimensional location | Removes state aliasing in the bidirectional 2D road topology and when the macro is serving |
| Decision timing | Event at a zone crossing | Event after a configured cumulative traveled distance; command from frame x is applied in frame x+1 | Preserves event-driven operation and matches the manuscript's causal reassociation timing |
| Reward | Useful delivered data | Single-user full-band SNR capacity, including beam-pilot overhead | Preserves the source objective; queues/load/energy are deliberately not added |
| Loaded evaluation | No common MEET-COBRA load model | Freeze Q, then apply common OTR-RA, Rician fading, interference, traffic, and queue evolution | Makes reported power and queue results directly comparable to the manuscript simulator |
| HO interruption | Source model includes an interruption term when changing mmBS | Set to zero | The common manuscript system model does not assign an HO interruption time to any scheme |

At every event, a micro action refreshes the local RX combiner even if the TX action is unchanged. The first slot uses eight sweep pilots and the other slots use one tracking pilot. This behavior is covered by a regression test because omitting the same-action RX refresh unfairly degrades PQL-BA.

## Experimental protocol

- Training trace: 200.0–800.0 s from `sionna_result/trajectoryInfo_lbd1.00_200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10.pkl`.
- Hyperparameter training interval: 200.0–740.0 s.
- Disjoint validation interval: 740.1–800.0 s.
- Final training interval: 200.0–800.0 s.
- Disjoint test interval: 800.0–830.0 s from `data4sim/lbd1.00_800_950_tx(1,32)_rx(1,8)_freq2.8e+10_Np8_mode0_lookahead10.pkl`.
- Candidate selection metric: frozen-policy validation average full-band rate, which is PQL-BA's source objective. Test queue results were never used to select a candidate.
- Common network evaluation: seed 1, 100 slots/frame, 1 ms/slot, OTR-RA, Rician fading, inter-micro-BS interference, and 20 ms normalized-backlog threshold.
- The 1 Mbps and 19 Mbps cases were selected as a low-load and a representative moderate-load audit. Existing paper curves at the same rates provide the comparison values.

## State and decision-zone selection

All state candidates below use a 10 m decision zone and five training epochs.

| State candidate | Validation rate (Mbps) | Link availability |
|---|---:|---:|
| Source RSSI + current beam | 354.885 | 72.314% |
| + heading + 25 m location bins | 348.024 | 82.027% |
| + heading + 50 m location bins | 415.587 | 85.453% |
| + heading + 100 m location bins | **448.067** | **88.653%** |

The 100 m location state was then tested with three decision-zone sizes under the same training/validation split and five-epoch budget.

| Zone size | Validation rate (Mbps) | Link availability | System HO/s | System beam switches/s |
|---:|---:|---:|---:|---:|
| 5 m | 444.252 | 87.916% | 159.583 | 229.833 |
| 10 m | 448.067 | 88.653% | 81.252 | 115.776 |
| 20 m | **464.584** | **89.050%** | **40.651** | **57.947** |

The validation rule therefore selects heading + 100 m location bins + a 20 m decision zone. The 5 m result also shows that merely refreshing beams more often does not solve the mismatch and nearly doubles switching relative to 10 m.

## Final training and frozen-policy test

The selected 20 m policy was retrained for ten epochs on 200.0–800.0 s. It made 300,080 Q updates and ended with 26,257 stored states and epsilon 0.052. Mean event reward increased from 467.24 Mbit in epoch 1 to 860.26 Mbit in epoch 10; the final four epochs lie between 829.08 and 862.92 Mbit. This is empirical reward stabilization, not a claim of formal tabular-Q convergence: the table is sparse, the environment is stochastic, and a fixed learning rate is used.

On the disjoint 800.0–830.0 s test trace, the frozen policy obtains:

- average single-user full-band rate: 402.965 Mbps;
- link availability: 89.092%;
- 2,070 decisions over 4,102.5 vehicle-seconds;
- 47.3 system handovers/s and 65.3 system beam switches/s.

For sensitivity, the 10 m final model obtains 462.520 Mbps and 88.363% link availability on the same test trace. The 20 m model was retained because it was selected on validation data; choosing 10 m after viewing test performance would introduce test-set selection bias.

## Loaded-network results

The table compares the validation-selected 20 m PQL-BA policy with values already stored in the manuscript figure CSV files.

| Traffic per vehicle | Method | Average system power (W) | Queue violation (%) |
|---:|---|---:|---:|
| 1 Mbps | MEET-COBRA | 4.010 | 0.110 |
|  | Reactive-OBRA | 12.283 | 0.157 |
|  | Adapted PQL-BA | **53.499** | **39.539** |
| 19 Mbps | MEET-COBRA | 44.378 | 0.340 |
|  | Reactive-OBRA | 78.111 | 8.090 |
|  | Adapted PQL-BA | **82.400** | **85.415** |

At 1 Mbps, PQL-BA uses 13.34× the power of MEET-COBRA and 4.36× that of Reactive-OBRA, while its violation probability is higher by 39.43 and 39.38 percentage points, respectively. At 19 Mbps, it uses 85.7% more power than MEET-COBRA and 5.5% more than Reactive-OBRA, while its violation probability is higher by 85.07 and 77.32 percentage points.

Additional PQL-BA statistics are:

| Traffic | Queueing proxy | HO/(vehicle·s) | Beam switches/(vehicle·s) | Pilots/(vehicle·slot) | Macro association |
|---:|---:|---:|---:|---:|---:|
| 1 Mbps | 1363.6 ms | 0.316 | 0.449 | 0.975 | 2.841% |
| 19 Mbps | 5777.1 ms | 0.316 | 0.449 | 0.975 | 2.841% |

The association policy does not observe traffic, so its HO, beam-switch, pilot, and macro-association statistics are identical across offered rates. At 19 Mbps, the comparatively moderate 82.4 W must not be interpreted as good energy efficiency: the 85.4% violation rate shows that much of the offered traffic is not being served. Energy is meaningful only under a comparable QoS level.

The 10 m sensitivity model gives 53.629 W/31.012% at 1 Mbps and 100.210 W/84.450% at 19 Mbps. Thus, changing the zone size trades switching, power, and outage behavior, but neither tested final model approaches the queue reliability of MEET-COBRA or Reactive-OBRA.

## Why the source objective fails in this scenario

1. PQL-BA observes only the serving-link state; it cannot distinguish a lightly loaded BS from a congested one or react to a vehicle's normalized backlog.
2. Its reward values single-user useful data, not energy per delivered QoS, deadline reliability, or network-wide resource consumption.
3. Inter-cell interference and multi-user RB competition appear only after the policy is frozen, so the learned beam/BS ranking is not the ranking relevant to loaded operation.
4. A high average single-user rate can coexist with intermittent fixed-beam outages. Those outages dominate a 20 ms queue-violation metric even at low average traffic.
5. The source paper's one-dimensional, homogeneous-mmBS/control-macro assumptions require nontrivial state and action adaptations in the heterogeneous 2D system. Further adding queue/load/energy to the state and reward would produce a new PQL-derived algorithm rather than a faithful PQL-BA baseline.

## Recommendation for the revision

- Do **not** use PQL-BA as the only new “strong learning/optimization” baseline. Its severe objective mismatch could be criticized as a straw-man comparison, and its performance is already worse than Reactive-OBRA.
- Keep it as a reproducible secondary baseline or supplementary negative result if the editor/reviewer specifically values comparison with PQL-BA. In that case, report the adaptations and source-objective limitation explicitly and include the single-link sanity check to demonstrate that the implementation learned what PQL-BA was designed to optimize.
- Prefer a queue/load-aware joint HO–BF learning or optimization method as the principal added baseline. If no credible such method can be adapted, PQL-BA can still satisfy the literal “joint HO–BF and learning-based” category, but it should be accompanied by a clear scope statement rather than advertised as equally matched to MEET-COBRA's objective.

## Reproducibility and valid artifacts

Implementation:

- `utils/pql_ba.py`: adapted shared tabular-Q policy, training, and queue-free rollout.
- `utils/pql_ba_sim.py`: frozen-policy loaded-network evaluation.
- `experiment/pql_ba_experiment.py`: tuning, final training, and evaluation driver.
- `test_pql_ba.py`: action mapping, RX sweep, same-action RX refresh, state, and Q-update tests.

Primary valid artifacts:

- `experiment/results_pql_ba/tuning_location_rxrefresh/`: 10 m state-resolution study.
- `experiment/results_pql_ba/tuning_zone_sensitivity_rxrefresh/`: 5/20 m zone sensitivity; combine with the preceding 10 m study.
- `experiment/results_pql_ba/final_location100_z20_rxrefresh/`: validation-selected final policy and training history.
- `experiment/results_pql_ba/evaluation_z20_final_30s/`: final 30 s test results.
- `experiment/results_pql_ba/final_location100_rxrefresh/` and `evaluation_corrected_30s_powerall/`: 10 m sensitivity model and corrected 30 s results.

Directories created before the same-action RX-beam-refresh correction, or directories explicitly named as short diagnostics, must not be used for manuscript claims.

The combined PQL-BA/PQL-BA-adapted implementation now passes all eleven unit tests in `test_pql_ba.py`; the source-faithful tests remain included, and older saved policies are backward compatible.
