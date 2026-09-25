# MTS H32-cross5 full-load, multi-seed results

## Configuration and preservation

- All 18 arrival rates from 1 to 35 Mbps in steps of 2; paired seeds 1, 2, 3; 30 seconds per case; first two frames excluded.
- 54 validated cases: six unchanged pilot cases reused with source hashes, 48 additional cases run.
- Five-frame GS association and predicted channel gains retained. BF does not use the predicted beam-pair list or PET-BF.
- Every frame: 32 hierarchical probes in the first available slot, then five axial-neighbour probes in each later available slot. No probes during the 10 ms HO interruption.
- Matching demand and bounded occupancy estimates include the new average BF overhead. Common OTR-RA and explicit directional RB service are retained.
- Raw power, U, latency-proxy quantiles, RB capacities, beam-probe accounting and paired traffic hashes were independently checked for all 54 cases.
- Original MTS data and pilot results remain intact. Previous paper figures and CSV are saved in pre_update_figures. The seven other curves, including the approved E_all O-MAPPO, are unchanged (630/630 figure-data rows).
- The experiments do not retrain or change O-MAPPO. Its separately discussed per-slot five-point tracking remains a future experiment.

## Three-seed means

| Mbps | MEET P (W) | MTS P (W) | MEET U (%) | MTS U (%) | MEET L99 (ms) | MTS L99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.43880 | 6.01411 | 0.25057 | 0.23141 | 18.93233 | 18.84033 |
| 3 | 7.95265 | 8.32519 | 0.21649 | 0.21275 | 7.97133 | 7.09594 |
| 5 | 10.94790 | 11.77763 | 0.21680 | 0.21333 | 5.01480 | 4.42688 |
| 7 | 13.94511 | 15.35601 | 0.21552 | 0.22345 | 3.59879 | 3.43439 |
| 9 | 16.91373 | 19.26070 | 0.21450 | 0.22952 | 3.06145 | 2.99222 |
| 11 | 19.82179 | 23.06603 | 0.21854 | 0.22165 | 2.60888 | 2.54048 |
| 13 | 23.32832 | 27.22245 | 0.22257 | 0.21485 | 2.41920 | 2.29648 |
| 15 | 26.84789 | 32.39781 | 0.22613 | 0.22298 | 2.18897 | 2.10494 |
| 17 | 30.38280 | 37.66653 | 0.23108 | 0.22229 | 2.05547 | 2.00465 |
| 19 | 33.79626 | 46.71541 | 0.24028 | 0.23108 | 1.99681 | 1.83030 |
| 21 | 38.12365 | 55.22189 | 0.25348 | 0.25947 | 1.82065 | 1.79671 |
| 23 | 46.17300 | 68.37649 | 0.24117 | 0.38802 | 1.76799 | 1.84222 |
| 25 | 66.28335 | 91.43900 | 0.18955 | 0.47078 | 1.66154 | 1.99041 |
| 27 | 95.02903 | 109.99352 | 0.16113 | 0.59750 | 1.58751 | 2.00488 |
| 29 | 127.74587 | 135.90794 | 0.19387 | 0.76872 | 1.54204 | 3.00994 |
| 31 | 157.63031 | 162.41868 | 0.27438 | 1.12413 | 1.54379 | 43.35916 |
| 33 | 172.81369 | 177.02582 | 0.52378 | 2.33204 | 5.00045 | 267.53715 |
| 35 | 175.79630 | 180.60077 | 1.14974 | 5.92507 | 29.49671 | 1136.33566 |

## Boundaries on interpretation

- MTS has lower mean power than MEET at these evaluated loads: [].
- MTS has lower mean U than MEET at these evaluated loads: [1, 3, 5, 13, 15, 17, 19].
- First evaluated load with mean L99 above 20 ms: {'meet_cobra': 35, 'oracle_mc': None, 'reactive_obra': 19, 'o_mappo': 21, 'mts_report': 31, 'wo_gap_ho': 25, 'wo_pet_bf': 33, 'wo_otr_ra': 33}.
- Means and min/max envelopes use the same three paired seeds; envelopes are not confidence intervals. Vehicle trajectories and deployment are fixed.
- L90/L99 are computed within each seed from q/lambda samples and then averaged across seeds; U is the frame-averaged proxy violation probability.
- Relative results describe this explicitly adapted baseline, not a complete reproduction of the original hybrid analog/digital beamformer.

## Interpretation

- The new MTS beamformer removes the predicted beam-candidate list and PET-BF from beam selection, but retains predicted gains for association and the common OTR-RA. It should not be described as independent of all MEET-COBRA modules.
- For a full micro-link frame without interruption, 32 acquisition probes plus 99 times 5 tracking probes give 527 probes, compared with 5 acquisition probes plus 99 single-pair measurements (104 probes) in the previous predicted-candidate implementation. With eta=2/112, their average modeled probing fractions are approximately 9.41% and 1.86%. The smaller remaining data-service fraction is a plausible contributor to the higher power and stronger high-load queue tails; the comparison does not isolate it from changes in selected beams and subsequent associations.
- At 33 Mbps the new MTS has mean power 177.03 W, U=2.332%, and L99=267.54 ms, versus 160.99 W, 1.080%, and 33.54 ms for the previous MTS. This change is reported internally as a consequence of the selected, more independent beam-search design, not as a performance improvement over the previous MTS.
- Low-load U improvements over MEET are small and do not establish statistical significance with only three seeds. Higher-load comparisons and figures retain every evaluated load and seed.

## Reproduction

```bash
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/mts_h32_full_grid.py run
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/mts_h32_full_grid.py audit
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/plot_revision_system_results.py
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/report_mts_h32_full_grid.py
```

The run command verifies and skips completed cases. To recreate the previous MTS curves, use --legacy-mts and separate --figures and --report output directories.
