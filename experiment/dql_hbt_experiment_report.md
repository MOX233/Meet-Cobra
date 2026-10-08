# DQL-HBT-adapted for MEET-COBRA: algorithm and experiment report

Date: 2026-08-30

## Executive conclusion

**DQL-HBT-adapted is suitable for inclusion as the principal literature-derived joint handover--beamforming baseline in the MEET-COBRA revision.** It is much more credible than the previously tested PQL-BA adaptations: its action semantics directly retain the published DQL-HBT structure, neural function approximation generalizes across locations and network states, and it achieves a meaningful reliability--power tradeoff in the common simulator.

The recommended 8-epoch policy obtains, over three independent traffic/fading seeds on the disjoint 800--830 s test trace:

| Traffic | Power (W), mean $\pm$ 95% CI | Queue violation (%), mean $\pm$ 95% CI | Queue proxy (ms), mean $\pm$ 95% CI |
|---:|---:|---:|---:|
| 1 Mbps | 14.096 $\pm$ 0.672 | 0.0212 $\pm$ 0.0456 | 6.351 $\pm$ 0.276 |
| 19 Mbps | 133.336 $\pm$ 2.818 | 7.7288 $\pm$ 0.2738 | 78.050 $\pm$ 1.938 |

The confidence intervals use a two-sided Student-$t$ critical value with two degrees of freedom, rather than a large-sample normal approximation.

At 1 Mbps, it is close to Reactive-OBRA in power while attaining essentially zero queue violation. At 19 Mbps, it lowers violation from 53.86% for hierarchical PQL-BA-adapted to 7.73%, while also reducing power from 185.13 W to 133.34 W. It therefore strictly dominates the tested PQL adaptation at both operating points. Relative to Reactive-OBRA, it trades substantially greater power (133.34 W versus 78.11 W) for a slightly smaller violation probability (7.73% versus 8.09%). MEET-COBRA still provides a clear Pareto improvement at 19 Mbps (44.38 W and 0.34%).

The DQL baseline does not solve network-wide congestion. At 27/35 Mbps, the recommended policy reaches 182.13/185.17 W and 32.48%/60.88% violation. A separate 14-epoch training curriculum covering traffic up to 35 Mbps is not a uniform improvement: it gives 182.07/184.69 W and 35.14%/59.35% at 27/35 Mbps, and it greatly raises the 19 Mbps power to 167.38 W for only a small reliability gain. This negative control confirms that adding high-load experience alone does not replace global congestion-aware coordination.

The baseline must be called **DQL-HBT-adapted**, not simply DQL-HBT, because queue, load, interference, energy, the heterogeneous macro tier, and the common OTR-RA scheduler are substantial adaptations. Before inserting its curve into the manuscript, the frozen 8-epoch policy should be evaluated on the manuscript's complete 1--35 Mbps grid. No further reward or policy selection should use that test grid.

## Source algorithm and retained structure

The source is Khosravi et al., “Reinforcement Learning-based Joint Handover and Beam Tracking in Millimeter-wave Networks” ([arXiv:2301.05305](https://arxiv.org/abs/2301.05305)). Its essential structure is retained:

- a shared DQL policy makes a high-level decision at each event;
- action 0 tracks the beam of the serving BS;
- each other action hands over to a specified target BS and performs full beam training;
- tracking searches only a neighborhood around the previous beam direction;
- the source-like state contains location, serving BS, serving-link SNR/SINR, and a tracking indicator;
- the source-like reward favors link spectral efficiency and penalizes a below-threshold link.

The source problem is a single-UE, homogeneous mmWave, equal-resource, noise-limited setting. MEET-COBRA instead has many vehicles, one sub-6 GHz macro BS and four mmWave micro BSs, unequal RB bandwidths and powers, queues, loaded inter-cell interference, and beam-training overhead. A literal, unmodified implementation would therefore not be a meaningful like-for-like comparator.

## System adaptation

### Action and beam-training semantics

The DQN has six outputs:

1. track the currently serving link;
2. hand over to the macro BS;
3. hand over to micro BS 1;
4. hand over to micro BS 2;
5. hand over to micro BS 3;
6. hand over to micro BS 4.

The action that “hands over” to the current serving BS is masked. A micro-BS handover performs an exhaustive $32\times8=256$ TX/RX-beam sweep. The sweep is spread over five slots as 55, 55, 55, 55, and 36 pilots so that no slot has 100% pilot overhead. A tracking action searches the wrapped $3\times3$ neighborhood around the previous TX/RX pair and costs nine pilot measurements. Tracking the omnidirectional macro link has no beam sweep.

Decisions are triggered every 10 m of traveled distance. The observation in frame $x$ produces a command that is applied in frame $x+1$, so evaluation is causal. Threshold-triggered variants were also tested and were inferior.

### Adapted state

The selected continuous state has 24 features:

- normalized 2D position;
- one-hot serving-BS identity;
- normalized serving SINR and previous tracking indicator;
- sine/cosine heading and normalized speed;
- normalized queue ratio and offered traffic rate;
- five current BS-load ratios;
- interference-to-noise ratio;
- cyclic sine/cosine encodings of the current TX and RX beam indices.

The load vector is obtained from the common RB-demand estimator and can be viewed as BS-broadcast context. The action remains a per-vehicle decision; the DQN does not observe or choose the simultaneous actions of other vehicles. This distinction is central to interpreting its high-load behavior.

### Adapted reward

The selected per-frame reward, averaged between two decision events, is

\[
r_v = 10
+ \min\!\left(\frac{S_v}{\lambda_v T_f},2\right)
-0.5\min\!\left(\frac{Q_v}{Q_v^{\rm ub}},10\right)
-4{\bf 1}\{Q_v>Q_v^{\rm ub}\}
-0.2P_v
-0.25{\bf 1}\{\mathrm{HO}\}
-0.1\frac{N_{\rm sweep}}{256}.
\]

Here $S_v$ is useful served traffic in one frame, $P_v$ is the RB-weighted transmit power attributed to vehicle $v$, and $N_{\rm sweep}$ is the number of beam measurements caused by the applied action. The constant offset stabilizes learning across variable event durations and does not affect the ordering of equal-duration actions.

The explicit serving-load penalty was tested separately. Load is retained in the state, but its reward weight is zero because a direct local load penalty worsened candidate screening. The load still affects reward indirectly through interference, service, queue, and power.

### DQL implementation

- dueling MLP with two 128-unit ReLU hidden layers;
- Double-DQN target;
- replay capacity 200,000 and warm-up 2,000 transitions;
- batch size 256, Adam learning rate $3\times10^{-4}$;
- discount factor 0.95;
- target-network update every 250 gradient steps;
- epsilon decays exponentially from 1.0 to 0.05 over 150,000 decisions;
- Smooth-L1 loss and gradient-norm clipping at 10.

One network is shared by all vehicles. This gives far better state/action generalization than the sparse tabular PQL policy while retaining decentralized execution.

## Training and evaluation protocol

Training uses the frame-level fluid surrogate previously audited for PQL-BA-adapted. It retains selected beam gains, no-BF interference gains, macro/micro bandwidth and power, fixed-point cell activity, pilot overhead, OTR-like capacity-priority allocation, per-vehicle queues, and the same 20 ms normalized queue threshold. It is used only for policy learning and candidate screening.

All claimed validation and test metrics use the original slot-level simulator with:

- 100 slots per 100 ms frame and 1 ms slots;
- Poisson traffic arrivals;
- Rician fading;
- loaded inter-cell interference;
- the same OTR-RA routine as the manuscript;
- the same SUMO/Sionna RT channel and mobility traces;
- explicit multi-slot full-sweep and local-tracking overhead.

The data partition is:

| Purpose | Interval | Trace |
|---|---:|---|
| Short training | 200--320 s | training trajectory |
| Short proxy validation | 320.1--340 s | training trajectory |
| Full candidate training | 200--740 s | training trajectory |
| Exact candidate validation | 740.1--750 s | training trajectory |
| Final retraining | 200--800 s | training trajectory |
| Frozen test | 800--830 s | separate test file |

The shared 800 s boundary frame only initializes the test state; reported test dynamics begin at the following frame, and two queue warm-up frames are discarded for non-power metrics. Candidate and reward selection never use test results.

## Candidate screening

Seven variants were screened:

- source state/reward with SINR-threshold decisions;
- source state/reward with periodic decisions;
- adapted state with QoS reward;
- adapted state with QoS and energy weight 0.20;
- the preceding reward plus explicit load weight 1;
- the preceding energy reward plus HO and sweep penalties;
- adapted state with a SINR/queue-triggered event rule.

Short training on 200--320 s and proxy validation on 320.1--340 s produced:

| Candidate | 1 Mbps: power / violation | 19 Mbps: power / violation |
|---|---:|---:|
| Source, threshold trigger | 21.44 W / 0.846% | 154.52 W / 38.06% |
| Source, periodic | 11.18 W / 0% | 149.13 W / 28.36% |
| Adapted QoS | 14.20 W / 0.0045% | 150.85 W / 25.80% |
| Adapted QoS + energy 0.20 | **9.85 W / 0.0089%** | 152.08 W / **18.39%** |
| Adapted + energy + load 1 | 13.69 W / 0.0090% | 151.54 W / 22.88% |
| Adapted + energy + HO/sweep | 9.86 W / 0.0089% | 152.05 W / 18.43% |
| Adapted, threshold/queue trigger | 33.18 W / 5.62% | 48.79 W / 48.55% |

The threshold/queue-triggered candidate saves power by making few tracking decisions but allows stale beams and queues to grow. The explicit load term is also detrimental. Energy weight 0.20 is the best short-screen compromise.

## Full candidate validation

The source-periodic, energy-0.20, and energy-0.20+HO/sweep candidates were trained for eight epochs on 200--740 s, cycling through 1/7/13/19 Mbps twice. Proxy validation selected the HO/sweep-regularized reward. The independent 10 s slot-level validation gives:

| Candidate | 1 Mbps: power / violation | 19 Mbps: power / violation |
|---|---:|---:|
| Source-like DQL-HBT | 20.43 W / 0% | 133.42 W / 66.17% |
| Adapted, QoS + energy | 18.22 W / 0.119% | 158.50 W / 18.72% |
| **Adapted, QoS + energy + HO/sweep** | **16.17 W / 0%** | 166.34 W / **8.36%** |

The source-like policy converges to tracking in 98.8% of decisions. That behavior is reasonable for its link-centric reward but disastrous under congestion: it almost never changes association, and high-load violation reaches 66.17%. Adding queue, load, interference, and energy context reduces this by 47.45 percentage points. Adding HO/sweep regularization improves reliability further despite penalizing switching, because it learns more stable, useful associations rather than oscillating among cells.

## Final 30 s multi-seed test

The selected reward was retrained for eight epochs on 200--800 s and frozen. Three independent seeds change Poisson arrivals and Rician fading but use the same mobility/channel trace.

### Per-seed results

| Traffic | Seed | Power (W) | Queue violation (%) | Queue proxy (ms) |
|---:|---:|---:|---:|---:|
| 1 Mbps | 1 | 14.147 | 0.0318 | 6.427 |
|  | 2 | 13.803 | 0 | 6.223 |
|  | 3 | 14.336 | 0.0318 | 6.403 |
| 19 Mbps | 1 | 134.626 | 7.812 | 78.049 |
|  | 2 | 132.889 | 7.770 | 78.831 |
|  | 3 | 132.494 | 7.604 | 77.270 |

The high-load violation spans only 0.21 percentage points across seeds. The principal result is therefore stable to the evaluated stochastic arrivals and fading.

### Comparison with existing methods

The following table uses the three-seed DQL mean and the already reported common-simulator results for the other methods.

| Traffic | Method | Power (W) | Queue violation (%) |
|---:|---|---:|---:|
| 1 Mbps | MEET-COBRA | 4.010 | 0.110 |
|  | Reactive-OBRA | 12.283 | 0.157 |
|  | Hierarchical PQL-BA-adapted | 58.627 | 7.300 |
|  | **DQL-HBT-adapted** | **14.096** | **0.021** |
| 19 Mbps | MEET-COBRA | 44.378 | 0.340 |
|  | Reactive-OBRA | 78.111 | 8.090 |
|  | Hierarchical PQL-BA-adapted | 185.126 | 53.862 |
|  | **DQL-HBT-adapted** | **133.336** | **7.729** |

Relative to hierarchical PQL-BA-adapted, DQL-HBT-adapted reduces power by 76.0% at 1 Mbps and 28.0% at 19 Mbps. It reduces violation by 99.7% and 85.7%, respectively. Neural state generalization, the explicit tracking-versus-handover action, and the smaller six-action space are decisive improvements over the sparse tabular PQL design.

At 19 Mbps, MEET-COBRA reduces DQL-HBT-adapted power by 66.7% and its violation probability by 95.6%. This is a useful comparison for the paper: DQL provides coordinated learned HO--BF, while MEET-COBRA additionally performs globally coupled predictive association and queue-aware resource coordination.

### Execution and control overhead

| Traffic | Decisions/(veh·s) | Tracking action | HO/(veh·s) | Beam switches/(veh·s) | Pilots/(micro-link slot) | Inference (ms/decision) | Macro association |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Mbps | 0.964 | 62.38% | 0.361 | 0.418 | 0.504 | 0.264 | 54.96% |
| 19 Mbps | 0.964 | 77.33% | 0.218 | 0.460 | 0.856 | 0.275 | 19.23% |

The network inference itself is comfortably below the 100 ms frame duration. Beam measurements and signaling, rather than DQN execution, are the larger overhead. These quantities make DQL-HBT-adapted useful for the reviewer's requested like-for-like overhead comparison.

## High-load robustness and negative control

The 8-epoch policy is trained on 1/7/13/19 Mbps. Its frozen extrapolation is:

| Policy | Traffic | Power (W) | Queue violation (%) | Queue proxy (ms) | HO/(veh·s) | Macro association |
|---|---:|---:|---:|---:|---:|---:|
| Recommended 8-epoch | 27 Mbps | 182.128 | 32.481 | 460.36 | 0.383 | 22.54% |
|  | 35 Mbps | 185.170 | 60.882 | 2278.44 | 0.615 | 25.55% |

For comparison, a fresh 14-epoch universal policy was trained on the sequence 1/7/13/19/25/31/35 Mbps twice:

| Policy | Traffic | Power (W) | Queue violation (%) | Queue proxy (ms) | HO/(veh·s) | Macro association |
|---|---:|---:|---:|---:|---:|---:|
| Full-load 14-epoch | 1 Mbps | 12.879 | 0 | 6.51 | 0.611 | 52.02% |
|  | 19 Mbps | 167.381 | 6.894 | 150.50 | 0.244 | 33.15% |
|  | 27 Mbps | 182.074 | 35.139 | 987.90 | 0.292 | 35.98% |
|  | 35 Mbps | 184.691 | 59.345 | 2364.60 | 0.397 | 33.30% |

The full-load curriculum is not Pareto superior. It slightly improves 1 and 35 Mbps, but at 19 Mbps it spends 32.76 W more to gain only 0.92 percentage points of reliability, and at 27 Mbps it increases violation by 2.66 points. The 8-epoch policy is therefore retained. The negative result also shows that the high-load deficiency is architectural, not merely a shortage of high-load samples.

Each vehicle receives common load context but optimizes an individual action/reward. There is no joint action, centralized critic, difference reward, or global capacity constraint inside the DQN. Near saturation, independent decisions can move too many users toward the same resource and cannot coordinate simultaneous re-association. MEET-COBRA's GAP-HO explicitly solves a global capacitated association problem, which explains the large separation at 27 Mbps.

## Recommendation for the revision

### Include it as a formal baseline

DQL-HBT-adapted should replace PQL-BA as the main new learning-based HO--BF baseline because:

1. it directly answers both reviewers asking for a coordinated HO--BF method from existing literature;
2. it preserves the source algorithm's defining track-versus-handover DQL structure;
3. it is evaluated under exactly the same mobility, ray-traced desired links, queues, interference, beam overhead, and OTR-RA as MEET-COBRA;
4. it is substantially stronger than PQL-BA-adapted and competitive with Reactive-OBRA in reliability at 19 Mbps;
5. it exposes a scientifically interpretable advantage of MEET-COBRA: global, predictive load coordination rather than merely better local beam tracking;
6. its measured inference, HO, and pilot statistics support the requested overhead discussion.

### Required presentation safeguards

- Name it **DQL-HBT-adapted** everywhere.
- Cite the source paper and state which action/state/beam-tracking elements are retained.
- Explicitly disclose the heterogeneous macro tier, queue/load/interference state, energy/QoS reward, and common OTR-RA adaptations.
- State that OTR-RA is held fixed so that the comparison isolates learned HO--BF coordination; the baseline does not jointly learn RA.
- Include the source-like ablation, at least in the response letter or supplementary material, to show why adaptation is necessary.
- Do not claim that the adapted numerical results are reproduced results from the source paper.
- Report HO and pilot overhead alongside power and queue violation.
- Use the preselected 8-epoch policy. Do not select per-load policies using final test results.
- Before producing the manuscript figure, evaluate this frozen policy at every traffic point used by the existing main curves. The current experiments establish suitability, but the 18-point plotting run remains to be completed.

## Implementation and artifacts

Code:

- `utils/dql_hbt.py`: state/action construction, local beam tracking, reward presets, dueling Double DQN, replay learning, and fluid training.
- `utils/dql_hbt_sim.py`: causal slot-level evaluation with Rician fading, loaded interference, OTR-RA, queues, and spread sweep overhead.
- `experiment/dql_hbt_experiment.py`: candidate tuning, exact validation, final training, multi-seed aggregation, and frozen testing.
- `experiment/plot_dql_hbt_results.py`: reproducible endpoint comparison plot.
- `test_dql_hbt.py`: action, state, local tracking, replay/gradient, masking, and checkpoint regression tests.

Primary result directories:

- `experiment/results_dql_hbt/screen_120s/`: seven-candidate short screening.
- `experiment/results_dql_hbt/full_top3_8ep/`: full training of the top three designs.
- `experiment/results_dql_hbt/full_top3_exact_10s/`: independent slot-level candidate validation.
- `experiment/results_dql_hbt/final_adapted_8ep/`: recommended final policy.
- `experiment/results_dql_hbt/final_test_30s_3seeds/`: primary three-seed 1/19 Mbps results.
- `experiment/results_dql_hbt/final_original_high_load_control_seed1/`: recommended-policy 27/35 Mbps control.
- `experiment/results_dql_hbt/final_adapted_full_load_14ep/`: full-load curriculum policy.
- `experiment/results_dql_hbt/final_full_load_test_30s_seed1/`: full-load-policy 1/19/27/35 Mbps negative control.
- `experiment/results_dql_hbt/dql_hbt_endpoint_comparison.pdf` and `.png`: endpoint comparison plot.

All reported JSON files include the policy path, interval, seed, Rician-fading flag, and relevant hyperparameters. Result directories are intentionally ignored by Git because they contain generated checkpoints and outputs; source code, tests, plotting code, and this report remain visible to version control.
