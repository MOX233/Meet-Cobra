# O-MAPPO improvement experiments: tail-risk reward, full candidates, feasibility feedback, and recurrence

Date: 2026-09-02

## Executive conclusion

Two exploratory O-MAPPO variants were implemented, trained, and evaluated in the MEET-COBRA system model:

1. **Exp1 (CVaR + full candidates + stronger MLP):** a 95%-CVaR penalty on queue-constraint excess, all four alternative BSs, and two-layer 128-unit actor/critic networks.
2. **Exp2 (feasibility + optimizer feedback + GRU):** Exp1 plus per-BS feasibility features, causal feedback from the preceding target optimizer, and length-4 recurrent actor/critic policies with 128-unit GRUs.

Neither variant should replace the current formal **O-MAPPO-adapted** baseline. In the common 30 s exact simulation, the current baseline Pareto-dominates Exp1 and Exp2 at 19 and 27 Mbps. At 1 Mbps, Exp1 and Exp2 reduce the single-seed violation estimate by only 0.00005 and 0.00127 percentage points while increasing power by 81% and 104%; this is not a useful operating-point improvement. Exp2 improves on Exp1's reliability at 19 and 27 Mbps, but only as a local power--reliability tradeoff and not enough to beat the current baseline.

The experiments reveal a consistent failure mode. A tail-risk penalty encourages the policy to use the high-capacity macro BS much more frequently. This raises power without preventing persistent queues under sustained medium/high load. Candidate-feasibility features and recurrence partially mitigate the early macro collapse after a complete 12-stage curriculum, but do not remove the mismatch between fluid training and the discrete, interference-coupled exact simulator.

The current formal baseline and its stored results were not modified.

## Implemented variants

### Exp1: queue-tail/CVaR reward, all candidate BSs, and stronger networks

The adapted O-MAPPO reward was extended with the system-tail term

\[
  -w_{\rm CVaR}\operatorname{CVaR}_{0.95}
  \left(\left[\min(Q_v/Q_v^{\rm ub},10)-1\right]^+\right),
\]

where the empirical CVaR is the mean of the largest 5% queue-excess ratios among active vehicles in the current training step. The common tail penalty keeps persistently starved vehicles visible in a 100+ agent reward.

Four CVaR weights, 0.5, 1, 2, and 3, were screened. The proxy selection scores were 13.708, 9.016, 10.466, and 10.434, respectively, so the full experiment used `cvar_weight=1` and `cvar_alpha=0.95`.

Other Exp1 changes were:

- all four alternative BSs are exposed to the joint target optimizer (the current BS remains excluded);
- actor and centralized critic use 128--128 MLPs instead of the formal baseline's smaller networks;
- PPO uses learning rate `3e-4`, discount 0.98, GAE 0.9, four PPO epochs, and minibatches of 256;
- the remaining reward terms and load/energy-aware greedy target optimizer are unchanged from O-MAPPO-adapted.

### Exp2: candidate feasibility, optimizer feedback, and recurrent policy

Exp2 retains Exp1's reward and four alternative targets. Its local state grows from 31 to 55 features by adding, for each of the five BSs:

- candidate SINR;
- normalized RB demand;
- residual RB capacity;
- demand--capacity feasibility margin.

Four causal optimizer-feedback features are also included: total normalized overflow, selected-target load, selected-target overflow, and preceding optimizer success. Feedback comes only from the previously completed optimization; no future channel or traffic information is used.

Both actor and centralized critic use a 128-unit GRU over the most recent four decision-event observations followed by a 128-unit head. Left zero padding is used at sequence start. PPO uses three epochs and minibatches of 512 to control recurrent-training cost.

## Training protocol

Both final policies use:

- training trace: `sionna_result/trajectoryInfo_lbd1.00_200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10.pkl`;
- training interval: 200--800 s;
- seed: 1;
- load curriculum (Mbps): 1, 7, 13, 19, 25, 31, 7, 19, 1, 31, 13, 25;
- 12 stages, one complete fluid episode per stage;
- greedy constrained target optimization during training.

Exp2 was initially inspected after stage 8 and exhibited severe macro bias in the exact simulator. It was then resumed through all 12 stages to exclude slow recurrent convergence as the cause. The four resumed-stage fluid results were:

| Stage | Load (Mbps) | Power (W) | Violation (%) | CVaR excess | Trigger ratio |
|---:|---:|---:|---:|---:|---:|
| 9 | 1 | 9.58 | 0.310 | 0.300 | 0.439 |
| 10 | 31 | 184.27 | 34.907 | 8.999 | 0.462 |
| 11 | 13 | 114.79 | 2.283 | 2.338 | 0.411 |
| 12 | 25 | 182.83 | 15.169 | 8.452 | 0.399 |

At the same final curriculum positions, Exp1 obtained 184.22 W/31.159% at 31 Mbps, 103.72 W/1.963% at 13 Mbps, and 180.89 W/10.917% at 25 Mbps. Thus Exp2 did not outperform Exp1 in the fluid training environment at these medium/high-load points.

## Exact-simulator protocol

Frozen policies were evaluated on the disjoint 800--830 s trace with seed 1, Rician fading, the common OTR-RA scheduler, and the MILP target optimizer. Every 30 s run contains 300 frames; the first two frames are excluded from queue means and violation probability in accordance with the existing baseline scripts.

Exp1 was evaluated at 1, 7, 9, 19, 27, and 35 Mbps. Exp2 was evaluated at 1, 19, and 27 Mbps after its complete 12-stage curriculum. These are exploratory single-seed comparisons, not paper-ready confidence-interval results.

## Exp1 30 s exact results

| Load (Mbps) | Power (W) | Violation (%) | Mean queue proxy (ms) | p90 (ms) | p99 (ms) | Macro association (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.858 | 0.060 | 7.260 | 14.306 | 18.137 | 21.97 |
| 7 | 76.727 | 0.141 | 1.968 | 2.584 | 3.391 | 37.41 |
| 9 | 106.327 | 0.215 | 2.010 | 2.192 | 2.848 | 41.72 |
| 19 | 169.353 | 8.602 | 151.448 | 7.069 | 5572.746 | 35.99 |
| 27 | 182.435 | 26.472 | 282.626 | 702.370 | 5638.938 | 33.73 |
| 35 | 183.154 | 54.700 | 1430.673 | 4482.645 | 15640.078 | 38.46 |

At 7 and 9 Mbps, Exp1 lowers violation relative to the formal baseline but increases power by 100.5% and 133.5%, respectively. At 19 and 27 Mbps it is strictly worse in both power and violation. At 35 Mbps it lowers violation by 6.13 percentage points at essentially unchanged saturated power, but its mean queue and p99 remain worse. The low p90 but extremely high p99 at 19 Mbps identifies a small set of persistently starved vehicles that the single-step CVaR reward did not control over time.

## Common 30 s comparison

| Load | Method | Power (W) | Violation (%) | Mean queue (ms) | p99 (ms) | Macro association (%) |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | Current O-MAPPO-adapted | 5.437 | 0.060 | 8.118 | 18.242 | 1.89 |
| 1 | Exp1 | 9.858 | 0.060 | 7.260 | 18.137 | 21.97 |
| 1 | Exp2 | 11.095 | 0.059 | 6.975 | 18.143 | 29.51 |
| 19 | Current O-MAPPO-adapted | 93.561 | 4.369 | 22.874 | 409.904 | 9.26 |
| 19 | Exp1 | 169.353 | 8.602 | 151.448 | 5572.746 | 35.99 |
| 19 | Exp2 | 171.831 | 6.391 | 67.394 | 1345.016 | 34.05 |
| 27 | Current O-MAPPO-adapted | 166.103 | 20.355 | 138.603 | 3629.080 | 15.19 |
| 27 | Exp1 | 182.435 | 26.472 | 282.626 | 5638.938 | 33.73 |
| 27 | Exp2 | 184.096 | 24.241 | 327.342 | 7658.132 | 32.72 |

Relative to the current formal baseline, Exp2 changes power/violation by:

- 1 Mbps: +104.1% power and -0.001 percentage points violation;
- 19 Mbps: +83.7% power and +2.021 percentage points violation;
- 27 Mbps: +10.8% power and +3.886 percentage points violation.

Consequently, Exp2 is strictly dominated at the common medium/high-load points. At 1 Mbps its minute numerical violation reduction is obtained at more than twice the baseline power, so it remains an uncompetitive tradeoff even though the point is not strictly Pareto-dominated.

## Why the 5 s screen was insufficient

The final Exp2 checkpoint looked promising in a 5 s screen:

| Load (Mbps) | 5 s power (W) | 5 s violation (%) | 30 s power (W) | 30 s violation (%) |
|---:|---:|---:|---:|---:|
| 1 | 8.925 | 0.000 | 11.095 | 0.059 |
| 19 | 137.973 | 0.772 | 171.831 | 6.391 |
| 27 | 180.216 | 14.710 | 184.096 | 24.241 |

The longer trajectory exposes persistent queues that have not yet developed during initialization. Exp1 exhibited the same effect: at 19 Mbps, its violation increased from 1.671% over 5 s to 8.602% over 30 s. Short exact runs remain useful for rejecting obvious failures, but they cannot establish a queue-tail improvement.

## Interpretation and recommendation

1. **CVaR does change the operating point, but in the wrong direction.** The common system-tail penalty encourages more aggressive triggering and macro association. It can reduce violations at selected low/high loads, but usually by spending substantially more energy.
2. **Single-step empirical CVaR does not control temporal starvation.** The p90/p99 separation shows that a small subset of vehicles can remain backlogged across many frames even when the current-step team tail is penalized.
3. **Feasibility features are informative but easy to exploit as a shortcut.** The policy learns that the macro BS often has the largest immediate feasibility margin. Without an explicit long-horizon macro-energy constraint, it overuses that option.
4. **Recurrence does not repair model mismatch by itself.** Completing all 12 stages corrected the extreme stage-8 macro collapse, but the final policy remains dominated in the exact environment and is substantially more expensive to train.
5. **Recommendation:** retain the existing O-MAPPO-adapted formal baseline. Keep Exp1 and Exp2 as diagnostic research branches only; do not add their curves to the paper in their current form.

A future improvement attempt should target the root mismatch rather than simply enlarge the network: for example, constrained/Lagrangian risk and macro-energy budgets, temporally accumulated per-vehicle starvation state, and training-domain calibration against short exact-simulator rollouts.

## Reproducibility and validation

Main runner:

- `experiment/o_mappo_improvement_experiment.py`

Implementation:

- `utils/o_mappo.py`
- `utils/o_mappo_sim.py`

Stored artifacts:

- Exp1 policy/history/protocol: `experiment/results/o_mappo_improvements/exp1_cvar_full_strong/`
- Exp1 30 s exact results: `experiment/results/o_mappo_improvements/exp1_cvar_full_strong/exact_800_830_seed1/`
- Exp2 policy/history/protocol: `experiment/results/o_mappo_improvements/exp2_feasibility_recurrent/`
- Exp2 30 s exact results: `experiment/results/o_mappo_improvements/exp2_feasibility_recurrent/exact_800_830_seed1/`
- screening artifacts: `experiment/results/o_mappo_improvements/screen_exp1/`

Validation completed successfully:

- six O-MAPPO unit tests passed, including feasibility-state dimensions, recurrent checkpoint round-trip, PPO update, optimizer constraints, and command execution;
- all Exp1 and Exp2 long-window raw numeric arrays are finite;
- every long-window run contains 300 frames;
- all reported CSV metrics were recomputed exactly from the stored NPZ traces (maximum absolute discrepancy 0);
- no MILP optimizer failures occurred in any of the nine long-window runs.
