# Prediction-only O-MAPPO: completed experiment

All values below are three-seed means. Each run lasts 30 s; the first two frames are omitted.
The baseline uses one common validation-selected checkpoint at every test load. No paper files were changed.

Selected: seed11; checkpoint SHA-256 `16cef09a17207776f9a371d136a01d8bbe1643277dae3528d0c29a7255ab7f85`.

## Power / violation probability

| Mbps | Prediction, fine-tuned | Prediction, no fine-tuning | True-CSI cross5 | MEET-COBRA | Oracle-MC |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.889 W / 0.2539% | 5.639 W / 0.2349% | 1.859 W / 0.0016% | 5.439 W / 0.2506% | 1.725 W / 0.0269% |
| 3 | 8.760 W / 0.2452% | 8.163 W / 0.2076% | 4.841 W / 0.0000% | 7.953 W / 0.2165% | 4.350 W / 0.0000% |
| 5 | 12.385 W / 0.2289% | 12.184 W / 0.2115% | 8.200 W / 0.0000% | 10.948 W / 0.2168% | 7.387 W / 0.0000% |
| 7 | 16.082 W / 0.2382% | 15.822 W / 0.2208% | 11.590 W / 0.0000% | 13.945 W / 0.2155% | 10.432 W / 0.0000% |
| 9 | 19.855 W / 0.2514% | 19.801 W / 0.2353% | 15.336 W / 0.0000% | 16.914 W / 0.2145% | 13.514 W / 0.0000% |
| 11 | 23.751 W / 0.2701% | 23.564 W / 0.2519% | 18.912 W / 0.0000% | 19.822 W / 0.2185% | 16.408 W / 0.0000% |
| 13 | 27.904 W / 0.2844% | 27.762 W / 0.2626% | 22.663 W / 0.0000% | 23.328 W / 0.2226% | 20.111 W / 0.0000% |
| 15 | 32.222 W / 0.3043% | 32.536 W / 0.2881% | 26.983 W / 8.22e-06% | 26.848 W / 0.2261% | 23.750 W / 0.0000% |
| 17 | 36.376 W / 0.3362% | 36.683 W / 0.3214% | 31.054 W / 0.0063% | 30.383 W / 0.2311% | 27.419 W / 0.0019% |
| 19 | 40.785 W / 0.3875% | 41.104 W / 0.3763% | 34.945 W / 0.0230% | 33.796 W / 0.2403% | 30.713 W / 0.0133% |
| 21 | 45.728 W / 0.5165% | 46.572 W / 0.4825% | 39.824 W / 0.0706% | 38.124 W / 0.2535% | 34.694 W / 0.0324% |
| 23 | 51.835 W / 0.8214% | 57.790 W / 0.7060% | 47.711 W / 0.2393% | 46.173 W / 0.2412% | 39.464 W / 0.0575% |
| 25 | 60.714 W / 1.6552% | 70.347 W / 1.0435% | 65.344 W / 0.3213% | 66.283 W / 0.1895% | 46.747 W / 0.0708% |
| 27 | 77.606 W / 2.6949% | 103.491 W / 1.2892% | 93.616 W / 0.6282% | 95.029 W / 0.1611% | 65.996 W / 0.0906% |
| 29 | 107.397 W / 4.1799% | 136.539 W / 1.5763% | 127.780 W / 1.0077% | 127.746 W / 0.1939% | 94.872 W / 0.0885% |
| 31 | 140.106 W / 4.4262% | 175.661 W / 2.1532% | 173.298 W / 1.2412% | 157.630 W / 0.2744% | 126.628 W / 0.1161% |
| 33 | 162.862 W / 4.1863% | 181.572 W / 5.1326% | 181.870 W / 3.3861% | 172.814 W / 0.5238% | 157.366 W / 0.2903% |
| 35 | 179.334 W / 7.7503% | 180.736 W / 7.0305% | 179.841 W / 5.8927% | 175.796 W / 1.1497% | 174.735 W / 0.8171% |

## Training stability

Separate selection-interval score: best fine-tuned 1.19652; no fine-tuning 1.72503. The all-candidate validation choice is seed11. This choice does not use test metrics.
- Seed 11: selected positive update 40; validation score 1.0000 -> 1.1374; last 1.4572; stability criterion passed: False.
- Seed 22: selected positive update 20; validation score 1.0000 -> 1.0046; last 2.4649; stability criterion passed: False.
- Seed 33: selected positive update 150; validation score 1.0000 -> 1.0126; last 1.0272; stability criterion passed: False.

## Scope and interpretation

- Prediction-only inputs do not imply prediction-only physical simulation: serving-link measurements and actual directional service are retained, with their probe costs.
- True-CSI cross5 uses the frozen approved actor. The trained prediction-input version also changes the actor weights and uses the exact rollout environment. Their difference is not a one-variable causal estimate of CSI error.
- Mean ± one sample standard deviation bands summarize three traffic/fading seeds, not uncertainty over deployments or independent channel environments.
- Zero observed violations do not establish zero violation probability.
- The maximum-gain forecast is a proxy for the subsequent hierarchical search; no true-channel calibration is inserted into candidate selection.
- Raw queues, power, RB capacity bounds, hashes and paired arrivals were independently checked for all compared cases.

Detailed protocol: `experiment/o_mappo_predicted_cross5_report.md`.
