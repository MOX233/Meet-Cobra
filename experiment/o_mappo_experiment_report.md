# O-MAPPO-adapted for MEET-COBRA: reproduction, adaptation, and experiment report

Date: 2026-08-30

## Executive conclusion

**O-MAPPO-adapted is suitable for inclusion as a second formal literature-derived joint HO--BF learning/optimization baseline, alongside DQL-HBT-adapted.** It should not be presented as a verbatim reproduction of the source code: the source has no public implementation, and the dynamic vehicular system requires substantial but explicit adaptations. The defining O-MAPPO structure is nevertheless retained: decentralized binary handover-trigger actors, centralized training, and a constrained optimizer that selects the target BS/beam for all simultaneously triggered users.

The recommended queue/load/interference/energy-aware policy was trained only on the 200--800 s training trace, frozen, and tested on the disjoint 800--830 s trace. Over three independent Poisson-arrival/Rician-fading seeds, it obtains:

| Traffic | Power (W), mean $\pm$ 95% CI | Queue violation (%), mean $\pm$ 95% CI | Queue proxy (ms), mean $\pm$ 95% CI |
|---:|---:|---:|---:|
| 1 Mbps | 5.420 $\pm$ 0.135 | 0.0601 $\pm$ 0.0008 | 8.110 $\pm$ 0.038 |
| 19 Mbps | 92.576 $\pm$ 2.513 | 4.3191 $\pm$ 0.1160 | 22.304 $\pm$ 1.247 |

The confidence intervals use a two-sided Student-$t$ critical value with two degrees of freedom. The three 19 Mbps violations are 4.369%, 4.277%, and 4.311%; the result is therefore stable across the evaluated stochastic arrivals and fast fading.

Relative to DQL-HBT-adapted at 19 Mbps, O-MAPPO-adapted reduces power by 30.6%, violation by 44.1%, and queue proxy delay by 71.4%. This is a useful architectural comparison rather than a redundant second DQL curve: DQL directly selects a per-vehicle track/target action, whereas O-MAPPO learns only cooperative binary triggers and delegates simultaneous target choices to a network-wide capacitated optimizer. MEET-COBRA still has a clear advantage at 19 Mbps: using the existing common-simulator result of 44.378 W/0.340%, it reduces O-MAPPO-adapted power and violation by 52.1% and 92.1%, respectively.

## Source and reproduction boundary

The source is Ruiyu Wang, Yao Sun, Chao Zhang, Bowen Yang, Muhammad Imran, and Lei Zhang, “A novel handover scheme for millimeter wave network: an approach of integrating reinforcement learning and optimization,” *Digital Communications and Networks*, vol. 10, no. 5, pp. 1493--1502, 2024 ([DOI](https://doi.org/10.1016/j.dcan.2023.08.002), [author-accepted manuscript](https://eprints.gla.ac.uk/306180/1/306180.pdf)).

The source O-MAPPO method has the following defining structure:

- each UE actor chooses a binary action: keep the serving link or trigger handover;
- actors execute with local observations, while training uses a centralized global state/critic;
- the local observation includes previous handover delay, previous system throughput, previous bandwidth allocation, and public BS load/user-count information;
- the cooperative reward uses system-throughput and delay thresholds, together with a handover cost;
- when handover is triggered, a lower-level optimizer chooses among three target links and allocates bandwidth under BS capacity and per-UE QoS constraints;
- PPO uses a 0.2 clipping ratio, discount 0.9, GAE, 64-unit ReLU actor/critic networks, Adam, and a reported learning rate of $5\times10^{-4}$.

The source scenario has a fixed ten-UE population, one microwave MBS plus six mmWave SCBSs, perfect beam tracking, 2 m/s random walk, a noise-limited SCBS model, and source-specific absolute throughput/delay thresholds. The official repository states that no associated data/code are available. Consequently, the work here is an independent source-guided reimplementation, not a run of author-released code.

## Retained algorithmic structure

### Multi-agent trigger policy

At each 10 m distance-zone crossing, every eligible vehicle supplies a local state to a binary actor:

1. action 0 keeps the serving BS and locally tracks its beam;
2. action 1 triggers handover and exposes that vehicle to the joint target optimizer.

All vehicles share actor parameters because vehicle identities enter and leave the trace. This is the standard parameter-sharing adaptation for a changing homogeneous-agent population. During execution, an actor sees only its local/broadcast features. During training, a centralized critic receives a permutation-invariant pooled global state consisting of the mean, minimum, and maximum of all active local states plus normalized active-user count. This replaces the fixed-length concatenation used for ten source UEs while preserving CTDE.

The observation in frame $t$ produces a command applied at frame $t+1$, so both training and exact evaluation are causal.

### Target-BS/beam optimizer

For each triggered vehicle, the current serving BS is excluded and at most three alternative links are retained. A macro candidate uses its 3GPP UMa gain. A micro candidate uses the best $32\times8$ DFT TX/RX beam pair and includes loaded inter-cell interference and sweep overhead in its per-RB capacity.

The optimizer jointly assigns all vehicles triggered in the same frame. It minimizes a normalized combination of:

- RB demand/current transmission-time proxy;
- current target-cell load price;
- required-RB transmit energy.

Per-BS RB capacities are explicit constraints. Continuous overflow slack, heavily penalized in the objective, keeps the problem solvable when offered traffic exceeds total physical capacity. Training uses a deterministic capacity-aware greedy solver for speed. Frozen-policy validation/test uses SciPy's mixed-integer solver with binary link variables, three candidates per triggered UE, a 2 s safety limit, and a deterministic greedy fallback. No fallback was used in any reported exact run.

The source paper also optimizes continuous bandwidth. In the common MEET-COBRA comparison, O-MAPPO's optimizer stops at association/beam/capacity planning and the final per-slot allocation is always the manuscript's common OTR-RA scheduler. This deliberate adaptation prevents O-MAPPO from being evaluated with a different RA rule. It must be disclosed in the paper.

### Beam training

- A triggered handover to a micro BS performs a $32\times8=256$-pair exhaustive TX/RX sweep.
- The sweep is spread over several slots so that no slot has 100% pilot overhead.
- A non-trigger action on a micro link searches the wrapped $3\times3$ neighborhood of the previous pair, costing nine measurements.
- The macro link is omnidirectional and incurs no beam sweep.

The source assumes perfect tracking and does not charge this overhead. Explicitly modeling it is necessary for a fair comparison with MEET-COBRA and DQL-HBT-adapted.

## State and reward variants

### Source-like state and reward

The 13-feature source-like local state contains:

- previous HO-delay indicator;
- previous system served/offered throughput ratio;
- previous own RB-allocation fraction;
- five public BS user-count/load features;
- one-hot serving-BS identity.

The source's absolute 1.8/2.3 Gbps thresholds cannot be transferred to a changing 100+ vehicle population. They are normalized by current offered traffic while retaining the source three-level reward: $10\delta$, $\delta$, or $-\delta$ according to high/intermediate/poor system throughput-delay conditions, minus handover cost.

Both source-gated and periodic source-like candidates converge to “never trigger” after freezing. On the 19 Mbps proxy validation they remain on the macro BS and reach approximately 65.1% queue violation; on the independent exact validation the result is 66.0%. The source global threshold reward is too weak for per-agent credit assignment in the 100+ vehicle setting.

### Adapted state

The selected 31-feature local state retains all source-like features and adds:

- normalized 2D position, sine/cosine heading, and speed;
- serving SINR;
- normalized queue and traffic rate;
- five RB-load ratios;
- interference-to-noise ratio;
- cyclic encodings of the current TX/RX beams.

BS load/counts are broadcast quantities. The actor remains decentralized; it does not observe other vehicles' private queues or simultaneous actions.

### Adapted cooperative reward

For vehicle $v$, a local QoS/energy reward is formed from served/offered traffic, normalized queue, queue-threshold violation, attributed RB power, congestion, handover, and sweep costs. The training reward is a 50/50 mixture of this local reward and its system-wide mean. This retains cooperation while restoring per-agent credit:

\[
r_v = \tfrac12 r_v^{\rm local} + \tfrac12 \bar r^{\rm local},
\]

\[
r_v^{\rm local} =
2\min\!\left(\frac{S_v}{\lambda_vT_f},2\right)
-\min\!\left(\frac{Q_v}{Q_v^{\rm ub}},10\right)
-5\mathbf{1}\{Q_v>Q_v^{\rm ub}\}
-0.2P_v
-C_{\rm load}
-0.05\mathbf{1}\{\mathrm{HO}\}
-0.02\frac{N_{\rm sweep}}{256}.
\]

The selected congestion term is the mean squared BS utilization plus any estimated overflow. Squared utilization is used because the physical OTR scheduler caps actual load at one; a pure $\max(\rho-1,0)$ penalty would otherwise be identically zero before the target optimizer.

## PPO implementation

- parameter-shared binary actor and shared pooled centralized critic;
- one 64-unit ReLU hidden layer in each network;
- Adam learning rate $5\times10^{-4}$ for actor and critic;
- discount factor 0.9 and GAE $\lambda=0.5$;
- PPO clipping 0.2, entropy coefficient 0.01;
- four PPO passes per collected episode, minibatch 256;
- gradient-norm clipping at 10.

Each on-policy transition spans two consecutive distance-trigger events of the same vehicle. Rewards are averaged over the intervening frames, and a departing vehicle terminates its trajectory. All activity in one episode is collected with a fixed actor before the PPO update.

## Training and evaluation protocol

Training uses the same audited frame-level fluid surrogate as the DQL/PQL work. It retains beam gains, macro/micro bandwidth and power, loaded inter-cell interference, pilot overhead, fixed-point BS activity, finite RB capacities, vehicle queues, and 20 ms queue threshold. It is used only for learning and inexpensive candidate screening.

All claimed validation/test metrics use exact slot-level simulation with:

- 100 slots per 100 ms frame and 1 ms slots;
- Poisson traffic arrivals;
- Rician fading;
- loaded micro-cell interference;
- the same ray-traced desired channels and mobility traces;
- the common OTR-RA routine;
- explicit multi-slot full sweeps and local tracking.

| Purpose | Interval | Trace |
|---|---:|---|
| Candidate training | 200--240 s | training trajectory |
| Proxy validation | 240.1--250 s | training trajectory |
| Exact candidate validation | 740.1--745 s | training trajectory |
| Final retraining | 200--800 s | training trajectory |
| Frozen test | 800--830 s | separate test file |

The shared 800 s boundary only initializes test state; reported dynamics begin in the next frame. Two frames are discarded as queue warm-up for non-power metrics. Neither candidate selection nor reward selection uses the final test file.

## Candidate experiments

### Fluid proxy screening

Seven candidate combinations were trained for four epochs on 200--240 s, cycling through 1/7/13/19 Mbps. The principal endpoint results on 240.1--250 s are:

| Candidate | 1 Mbps: power / violation | 19 Mbps: power / violation |
|---|---:|---:|
| Source-like, overlap/SINR gate | 18.92 W / 0% | 133.00 W / 65.09% |
| Source-like, periodic | 18.92 W / 0% | 133.00 W / 65.09% |
| Adapted QoS, source target objective | 17.97 W / 0% | 169.15 W / 9.16% |
| Adapted QoS, load-priced target | 18.92 W / 0% | 170.84 W / 20.48% |
| Adapted QoS + energy 0.05 | 13.61 W / 0% | 170.89 W / 5.95% |
| Adapted QoS + energy 0.20 | 9.77 W / 0.009% | 142.03 W / 2.75% |
| **Adapted QoS + energy 0.20 + congestion** | **3.97 W / 0.027%** | **112.35 W / 2.66%** |

Simply adding a load price to the target objective worsens queue performance because it encourages simultaneous movement without giving the actor a useful local credit signal. A congestion term in the cooperative reward is much more effective: the actor learns when to trigger, while the optimizer handles simultaneous targets.

### Independent exact candidate validation

The frozen candidates were then tested on 740.1--745 s with Rician fading, OTR-RA, and the MILP target solver:

| Candidate | 1 Mbps: power / violation | 19 Mbps: power / violation |
|---|---:|---:|
| Source-like | 20.55 W / 0% | 133.00 W / 66.02% |
| Adapted QoS, source target objective | 18.11 W / 0% | 165.01 W / 5.04% |
| Adapted QoS, load-priced target | 20.22 W / 0% | 167.81 W / 17.09% |
| Adapted QoS + energy 0.05 | 15.90 W / 0.049% | 173.85 W / 9.54% |
| Adapted QoS + energy 0.20 | 13.12 W / 0.108% | 158.41 W / 2.34% |
| **Adapted QoS + energy 0.20 + congestion** | **5.07 W / 0.016%** | **116.75 W / 1.89%** |

No candidate had a MILP failure. The queue/energy/congestion variant was selected before the final test.

## Final training

The selected candidate was retrained from scratch for eight full 200--800 s epochs, using the fixed schedule 1/7/13/19 Mbps twice. Every epoch generated 61,176 completed on-policy transitions. Total training time was 1479.7 s on the current machine.

| Epoch | Traffic | Fluid power (W) | Fluid violation (%) | Exploration trigger ratio |
|---:|---:|---:|---:|---:|
| 1 | 1 | 15.26 | 0.66 | 0.504 |
| 2 | 7 | 66.38 | 1.74 | 0.465 |
| 3 | 13 | 109.20 | 2.39 | 0.444 |
| 4 | 19 | 144.39 | 3.48 | 0.413 |
| 5 | 1 | 6.87 | 0.21 | 0.298 |
| 6 | 7 | 43.48 | 1.09 | 0.360 |
| 7 | 13 | 75.82 | 1.90 | 0.351 |
| 8 | 19 | 112.40 | 3.28 | 0.329 |

The second curriculum cycle lowers both power and violation at 1/7/13 Mbps and lowers high-load power substantially without increasing violation. Actor/critic losses remain finite throughout.

## Final 30 s multi-seed test

### Per-seed results

| Traffic | Seed | Power (W) | Queue violation (%) | Queue proxy (ms) |
|---:|---:|---:|---:|---:|
| 1 Mbps | 1 | 5.384 | 0.0599 | 8.118 |
|  | 2 | 5.395 | 0.0605 | 8.120 |
|  | 3 | 5.483 | 0.0599 | 8.093 |
| 19 Mbps | 1 | 93.553 | 4.369 | 22.874 |
|  | 2 | 91.533 | 4.277 | 22.107 |
|  | 3 | 92.642 | 4.311 | 21.930 |

### Comparison with DQL-HBT-adapted

| Traffic | Method | Power (W) | Violation (%) | Queue proxy (ms) | HO/(veh·s) | Pilot count/link-slot | Macro association |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 Mbps | O-MAPPO-adapted | 5.420 | 0.060 | 8.110 | 0.094 | 1.011 | 1.56% |
|  | DQL-HBT-adapted | 14.096 | 0.021 | 6.351 | 0.361 | 0.504 | 54.96% |
| 19 Mbps | O-MAPPO-adapted | 92.576 | 4.319 | 22.304 | 0.204 | 0.949 | 8.80% |
|  | DQL-HBT-adapted | 133.336 | 7.729 | 78.050 | 0.218 | 0.856 | 19.23% |

At 1 Mbps, DQL has a numerically smaller violation/delay, but both policies are below 0.1% violation. O-MAPPO uses 61.5% less power and 73.9% fewer handovers by keeping most vehicles on micro cells and using local tracking. This raises per-link pilot count because far more active links are directional micro links.

At 19 Mbps, O-MAPPO improves power, violation, and queue delay simultaneously. It uses 6.5% fewer HOs, but 6.3% more beam switches and 10.9% more pilot measurements. The improvement is therefore not free of control overhead, and pilot/beam statistics should accompany the main performance curves.

### Execution time

| Traffic | Decisions/(veh·s) | Trigger ratio | Actor time (ms/decision frame) | MILP time (ms/trigger frame) | Full sweep/(veh·s) | Local sweep/(veh·s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 Mbps | 0.964 | 0.098 | 1.422 | 3.944 | 0.080 | 0.865 |
| 19 Mbps | 0.964 | 0.212 | 1.127 | 4.247 | 0.123 | 0.756 |

The actor and MILP latency are comfortably below the 100 ms frame duration. The exact simulator's wall time is dominated by the 100 Rician/OTR/queue slot updates per frame, not by online O-MAPPO decisions.

## High-load extrapolation

The frozen policy, trained only up to 19 Mbps, was evaluated at 27/35 Mbps with seed 1. These points were not used for selection or retraining.

| Method | Traffic | Power (W) | Violation (%) | Queue proxy (ms) | HO/(veh·s) | Optimizer overflow (RB) |
|---|---:|---:|---:|---:|---:|---:|
| O-MAPPO-adapted | 27 Mbps | 166.577 | 20.355 | 138.60 | 0.333 | 21.34 |
| DQL-HBT-adapted | 27 Mbps | 182.128 | 32.481 | 460.36 | 0.383 | -- |
| O-MAPPO-adapted | 35 Mbps | 183.701 | 60.828 | 1325.93 | 0.584 | 479.29 |
| DQL-HBT-adapted | 35 Mbps | 185.170 | 60.882 | 2278.44 | 0.615 | -- |

O-MAPPO's capacity-aware target selection retains a meaningful advantage at 27 Mbps. At 35 Mbps, both learning baselines saturate and have essentially the same violation/power. The large optimizer slack confirms that target rearrangement cannot overcome insufficient total radio capacity. This is a useful boundary result: O-MAPPO improves coordination but does not eliminate the network-wide capacity limit.

## Recommendation for the revision

### Include O-MAPPO-adapted as a formal baseline

The baseline should be included because:

1. it is derived from a published joint handover/beam/resource optimization method and directly addresses the reviewer request;
2. it is architecturally distinct from DQL-HBT-adapted;
3. it preserves the source learning--optimization decomposition and CTDE semantics;
4. the adaptation is evaluated under exactly the common mobility, channels, queues, interference, pilot overhead, and OTR-RA;
5. it is stronger than DQL-HBT-adapted at 19 and 27 Mbps, making it a credible rather than token comparison;
6. MEET-COBRA retains a large reliability/energy advantage at 19 Mbps, so the new baseline strengthens rather than erases the paper's claim;
7. measured actor/MILP, HO, beam-switch, and pilot statistics support the requested complexity/overhead discussion.

### Required naming and disclosure

- Call it **O-MAPPO-adapted**, not simply O-MAPPO.
- Cite the source paper and explicitly state that no author code was available.
- State that parameter sharing and pooled centralized state handle the dynamic vehicle population.
- State that the source absolute thresholds were normalized and the adapted reward adds queue, load, interference, and energy information.
- State that the source SQP/implicit-enumeration bandwidth solver is mapped to a three-candidate MILP association/beam solver, with common OTR-RA retained for fair final allocation.
- Do not claim that the numerical values reproduce the source paper.
- Include the source-like collapse as an ablation in the response letter or supplement; it justifies the adaptation.
- Report pilot/beam/HO overhead and both actor and optimizer time.
- Use the fixed selected policy for the paper. Do not select separate policies from final test rates.

### Remaining work before manuscript insertion

Suitability is established at 1/19 Mbps and the high-load boundary at 27/35 Mbps. Before generating the final manuscript curve, the frozen policy should be evaluated at every traffic point used in the existing figure. That is a mechanical plotting run and must not be used to change the reward or candidate. Manuscript and response-letter edits were intentionally not made in this task.

## Verification

- Five O-MAPPO unit tests pass: state/global dimensions, source gate, command/beam semantics, greedy/MILP target constraints, PPO update/checkpoint round trip.
- Seven existing DQL-HBT tests also pass after the new implementation was added.
- Both Python modules and the experiment driver compile.
- All exact candidate/final/high-load runs report zero MILP failures.
- The final three-seed JSON uses Student-$t$ rather than normal 95% intervals.

## Implementation and artifacts

Code:

- `utils/o_mappo.py`: actor/critic, PPO, source/adapted states and rewards, target optimization, fluid training.
- `utils/o_mappo_sim.py`: exact causal slot-level evaluation.
- `experiment/o_mappo_experiment.py`: screening, exact validation, final training, test, and aggregation protocol.
- `experiment/plot_o_mappo_results.py`: O-MAPPO/DQL endpoint and high-load comparison figure.
- `test_o_mappo.py`: focused implementation tests.

Principal outputs:

- `experiment/results/o_mappo/tuning/tuning_results.json`
- `experiment/results/o_mappo/tuning_load1/tuning_results.json`
- `experiment/results/o_mappo/exact_validation/exact_validation.json`
- `experiment/results/o_mappo/exact_validation_load1/exact_validation.json`
- `experiment/results/o_mappo/final_load1/final_policy.pt`
- `experiment/results/o_mappo/final_load1/final_training.json`
- `experiment/results/o_mappo/final_test/loaded_network_results.json`
- `experiment/results/o_mappo/high_load_test/loaded_network_results.json`
- `experiment/results/o_mappo/o_mappo_dql_comparison.pdf`
- `experiment/results/o_mappo/o_mappo_dql_comparison.png`
