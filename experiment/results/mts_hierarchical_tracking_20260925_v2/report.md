# MTS hierarchical acquisition and five-point tracking pilot

## Protocol

- Arrival rates: 1, 5, 15, 21, 25, 29 Mbps; traffic seed 1.
- Same 800--830 s trace, prediction cache and GPU fading as the formal grid. The first two service frames are omitted from aggregate metrics.
- Control: five predicted candidates per frame, then fixed beam with one tracking observation per slot.
- H32-hold: sixteen wide and sixteen fine probes in the first available slot of each frame; one current-beam observation in later slots.
- H32-cross5: the same initialization, followed by current, Tx-left, Tx-right, Rx-left and Rx-right probes in each later slot. Neighbourhoods wrap around the DFT grid.
- HO interruption: 10 ms. No search during interruption; acquisition starts at the first available slot. No extra five-probe search is added to the acquisition slot.
- GS preferences, association period (five frames), prediction reports and OTR-RA are retained. Candidate-link RB demand estimates use the new nominal frame-average BF overhead. For the two new arms, bounded occupancy estimation also includes that overhead. The formal control is reproduced unchanged.
- Expected candidate gains remain predicted optimal gains, not free measurements of candidate BSs. Only paid serving-link probes affect actual beam choice.
- Actual per-slot selected transmit and receive beams and explicit RB assignment determine the interference and service entering every queue update.
- No actor training or O-MAPPO changes are included in this pilot. Formal manuscript figures are untouched.

## Results

| Mbps | Variant | Power (W) | U (%) | L99 (ms) | Macro association (%) | Probes per active micro-link slot |
|---:|---|---:|---:|---:|---:|---:|
| 1 | report_hold | 5.7173 | 0.22645 | 18.9170 | 2.962 | 1.0400 |
| 1 | hier32_hold | 6.0139 | 0.23926 | 18.8960 | 3.496 | 1.3104 |
| 1 | hier32_cross5 | 6.0670 | 0.23384 | 18.8380 | 3.705 | 5.2703 |
| 1 | meet_cobra_reference | 5.4391 | 0.25147 | 18.9320 | 0.071 | — |
| 5 | report_hold | 11.0498 | 0.21943 | 5.0098 | 0.537 | 1.0400 |
| 5 | hier32_hold | 11.3075 | 0.21536 | 5.0022 | 0.537 | 1.3101 |
| 5 | hier32_cross5 | 11.7761 | 0.21345 | 4.4275 | 0.623 | 5.2701 |
| 5 | meet_cobra_reference | 10.9468 | 0.21686 | 5.0148 | 0.076 | — |
| 15 | report_hold | 29.0977 | 0.21342 | 2.1780 | 0.746 | 1.0400 |
| 15 | hier32_hold | 29.7646 | 0.21360 | 2.1754 | 0.746 | 1.3101 |
| 15 | hier32_cross5 | 32.4034 | 0.21921 | 2.1047 | 0.968 | 5.2701 |
| 15 | meet_cobra_reference | 26.8489 | 0.22621 | 2.1887 | 0.091 | — |
| 21 | report_hold | 44.8741 | 0.23904 | 1.7655 | 1.677 | 1.0400 |
| 21 | hier32_hold | 49.4730 | 0.24692 | 1.8376 | 2.528 | 1.3101 |
| 21 | hier32_cross5 | 55.1781 | 0.25446 | 1.7960 | 3.200 | 5.2701 |
| 21 | meet_cobra_reference | 38.1232 | 0.25259 | 1.8204 | 0.140 | — |
| 25 | report_hold | 67.6674 | 0.41016 | 1.6746 | 4.449 | 1.0400 |
| 25 | hier32_hold | 72.5997 | 0.43400 | 1.9939 | 5.261 | 1.3101 |
| 25 | hier32_cross5 | 91.3232 | 0.47150 | 1.9908 | 8.523 | 5.2701 |
| 25 | meet_cobra_reference | 66.2801 | 0.18845 | 1.6610 | 4.355 | — |
| 29 | report_hold | 108.7578 | 0.58323 | 1.5854 | 9.963 | 1.0400 |
| 29 | hier32_hold | 114.0461 | 0.62121 | 2.0260 | 10.675 | 1.3101 |
| 29 | hier32_cross5 | 135.5270 | 0.78587 | 3.0280 | 14.097 | 5.2701 |
| 29 | meet_cobra_reference | 127.7408 | 0.19366 | 1.5415 | 13.612 | — |

## Audit and interpretation limits

All 18 cases passed raw power, queue, violation-probability, latency-proxy and RB-capacity checks. The six control cases reproduce the saved formal results within numerical tolerance. All traffic hashes match the formal paired cases.

These are single-seed, closed-loop comparisons: different BF outcomes may alter queues and later associations. Mean serving gain across a variant is descriptive, not a fixed-association gain comparison. U and L99 retain the manuscript's queue-based latency-proxy definitions. No multi-seed or scenario-level robustness conclusion follows from this pilot.

Per-link probing overhead excludes common reporting/prediction observations. For a full, uninterrupted frame the counts are 104, 131 and 527, respectively. PET-BF uses at most 500 probes per such frame and may stop early.

## Interpretation

The pilot does not support replacing the formal report-based MTS baseline. H32-cross5 consumes more power at all six tested loads and has a higher violation probability at five of them. Its low-load tail-delay improvement does not persist at 21, 25 and 29 Mbps.

At 25 Mbps, H32-cross5 raises power from 67.67 to 91.32 W (+34.96%), U from 0.4102% to 0.4715%, and L99 from 1.6746 to 1.9908 ms. At 29 Mbps, power rises from 108.76 to 135.53 W (+24.61%), U from 0.5832% to 0.7859%, and L99 from 1.5854 to 3.0280 ms. H32-hold is less costly than H32-cross5, but also fails to improve both power and latency over the formal baseline at these loads.

The probe cost is substantial: approximately 9.41% of active micro-link slot time for H32-cross5, versus 1.86% for report-hold and 2.34% for H32-hold. The macro association share also increases, from 4.45% to 8.52% at 25 Mbps and from 9.96% to 14.10% at 29 Mbps. These are observable closed-loop changes consistent with increased micro-tier resource demand; this pilot does not separately identify the causal contribution of overhead, beam selection and changed association.

In particular, a local maximum among five current measurements is not necessarily the full-codebook optimum. Hierarchical acquisition can select a different sector from the prediction-based candidate set. The initial global limitation is not removed merely by probing more frequently. A fixed-association measurement study would be required to attribute gain changes independently of association.

## O-MAPPO design discussion (not run)

HO-triggered 32-probe acquisition followed by five-point tracking in every subsequent available slot is a reasonable separate adaptation. HO trigger decisions may retain the 10 m movement interval; beam tracking need not wait for that decision. Unlike the tested MTS variant, O-MAPPO would not reacquire every frame if the serving BS is unchanged. Initial micro-link acquisition is also required if no beam is yet available.

The serving-link observations, optimizer's predicted probing cost and RL training environment must agree with this new timing. On a frame with no acquisition the five-point policy uses 5 probes per active slot, substantially more than the old routine one-pilot observation. Existing actor weights can be used for a frozen-policy diagnostic but do not establish the performance of a properly retrained policy. No O-MAPPO results or official manuscript assets were changed here.

## Reproduction

```bash
/home/ubuntu/anaconda3/envs/sionna/bin/python -m unittest test_mts_hierarchical_tracking test_mts_report test_mts_report_bounded test_revision_directional test_o_mappo_hierarchical -q
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/mts_hierarchical_tracking.py run
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/analyze_mts_hierarchical_tracking.py
```

The run command resumes at completed-case boundaries and validates frozen source/input hashes. All 31 related unit tests passed. The audit generator regenerates the protocol/results sections; the interpretation above records the analysis of this completed run.
