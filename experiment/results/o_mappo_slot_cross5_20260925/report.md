# O-MAPPO HO32 + per-slot cross-five pilot

## Protocol

- Six arrival rates: 1, 5, 15, 21, 25, 29 Mbps; traffic/fading seed 1; 30 s; first two frames excluded.
- Frozen approved two-layer E_all actor (training seed 11, round 20). No retraining, checkpoint search or load-dependent policy selection.
- HO to a micro BS: 32 hierarchical probes in the first service slot after 10 ms interruption. Thereafter five measured pairs per slot: current, Tx-minus, Tx-plus, Rx-minus, Rx-plus. No extra current-beam pilot, no diagonal pairs, no additional event-driven nine-pair search.
- Tracking runs even when no 10 m actor decision occurs, and carries the final beam into the next frame. Unlike the selected MTS variant, it does not restart hierarchical search every frame.
- tracking_pilots changes from 1 to 5 in the existing cost formulas; actor dimensions/weights, target optimizer objective/candidate count, E_all interference/load rule and OTR-RA logic are unchanged. Realized states, triggers and associations may change through closed-loop feedback.
- Actual slot-selected beams determine both desired and cross-link gains in the explicit-RB service evaluator. Final beam states are committed after each frame, not before actor decisions.
- The current formal baseline retains ideal current-frame planning CSI, including its nominal post-HO planning beam and candidate-BS gain estimates. Physical service uses the slot-level search above. Extra CSI acquisition for planning remains uncharged, as in the formal baseline; this is not the equal-information/prediction-report actor variant.
- The current O-MAPPO and MEET references use the same seed, cache, 300 frames, warmup, directional service and HO model. The current O-MAPPO at 15 Mbps was rerun and every saved raw array matched its formal reference exactly.
- Twenty-five unit tests plus a 10-frame GPU smoke passed. Every simulated serving-link slot checked paid probe counts and delivered gains. All six full cases passed independent raw-metric recalculation.

## Power and violation probability (single paired seed)

| Mbps | Current P (W) | New P (W) | MEET P (W) | Current U (%) | New U (%) | MEET U (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.04673 | 1.85156 | 5.43911 | 0.01740 | 0.00151 | 0.25147 |
| 5 | 17.54519 | 8.19712 | 10.94677 | 0.04579 | 0.00000 | 0.21686 |
| 15 | 52.60380 | 26.99889 | 26.84886 | 0.18706 | 0.00000 | 0.22621 |
| 21 | 108.11838 | 39.81495 | 38.12318 | 1.28800 | 0.06016 | 0.25259 |
| 25 | 150.57801 | 65.88940 | 66.28013 | 3.83117 | 0.29386 | 0.18845 |
| 29 | 177.15328 | 135.84863 | 127.74085 | 10.04344 | 0.91947 | 0.19366 |

## Tail and association diagnostics

| Mbps | Current L99 (ms) | New L99 (ms) | MEET L99 (ms) | Current macro (%) | New macro (%) | MEET macro (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.66000 | 18.85200 | 18.93200 | 0.43083 | 0.06893 | 0.07140 |
| 5 | 4.35521 | 4.41500 | 5.01480 | 1.12755 | 0.06893 | 0.07632 |
| 15 | 2.17123 | 2.09598 | 2.18868 | 3.09707 | 0.06893 | 0.09109 |
| 21 | 38.34793 | 1.71587 | 1.82037 | 12.75019 | 0.15756 | 0.14033 |
| 25 | 411.22071 | 1.63621 | 1.66102 | 19.13883 | 3.78641 | 4.35510 |
| 29 | 3104.47049 | 15.01477 | 1.54152 | 22.63473 | 13.69064 | 13.61186 |

## Interpretation boundaries

- Loads with both lower mean power and lower U than the current O-MAPPO: [1, 5, 15, 21, 25, 29].
- Five probes every active slot increase the recurring pilot fraction from 2/112 to 10/112 (about 1.79% to 8.93%), excluding acquisition/event slots. The experiment measures the balance between better tracking and this extra cost.
- This pilot evaluates deployment of the same frozen actor under a changed BF mechanism, not the best achievable policy after dedicated retraining.
- One paired seed and six load points are preliminary evidence, not a multi-seed claim or proof of statistical significance.
- U=0 means no violations were observed in the evaluated samples; it is not a guarantee of zero violation probability.
- No paper text, response letter, formal figures, trained checkpoints or production defaults were modified.

## Reproduction

```bash
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/o_mappo_slot_tracking.py run
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/o_mappo_slot_tracking.py audit
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/analyze_o_mappo_slot_tracking.py
```

Code checkpoint: 7f88aa2. Original-code tag: pre-o-mappo-slot-cross5-20260925. Raw arrays and detailed diagnostics remain under runs/.

## Discussion

1. The new beam procedure reduces mean power and U relative to the current O-MAPPO at all six tested loads, with power reductions of 23.3%–69.4%. At 21 Mbps, power falls from 108.12 W to 39.81 W, U from 1.288% to 0.0602%, and L99 from 38.35 ms to 1.716 ms.
2. Against MEET-COBRA, the new variant has lower power and U at 1 and 5 Mbps. At 15 and 21 Mbps, it spends slightly more power (27.00 versus 26.85 W; 39.81 versus 38.12 W) while having lower U. At 25 Mbps it uses slightly less power but has higher U. At 29 Mbps MEET has lower power and U, with L99 of 1.542 ms versus 15.015 ms. Thus the pilot does not support uniform superiority of either scheme.
3. The reduced macro-tier reliance is a material observed contributor to power savings. At 21 Mbps, macro association falls from 12.75% to 0.158%, and the macro component of average transmit power falls from 59.73 W to 0.77 W. At 15 Mbps the corresponding macro power falls from 10.51 W to 0.27 W. These are closed-loop changes despite identical actor weights and optimizer rules.
4. Handover counts (including the initial frames and first macro-to-micro transitions) fall from 535 to 203 at 15 Mbps and from 423 to 210 at 21 Mbps. This is consistent with continuous local beam updates reducing the need for HO, but the experiment does not separately identify the causal contribution of tracking frequency, neighborhood shape, or physical acquisition timing.
5. The gain is not explained by charging fewer pilots: ordinary active micro slots now pay five rather than one probe, and HO acquisition still pays 32. The 5-point local set is smaller than the old event-triggered 9-point set, but updates occur much more frequently.
6. These results justify a full-load, multi-seed evaluation of the new version before any formal replacement. Dedicated actor retraining could be studied afterwards but is not necessary to establish the observed improvement of this frozen-policy pilot. The current paper and formal result figures remain unchanged.
