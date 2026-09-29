# R1C4 / R2C3 gain-error sensitivity study

The experiment uses the formal `revision_directional_20260922` MEET-COBRA cache and simulator, without retraining or changing the paper. Git rollback point: `cf1f708`, tag `pre-gain-error-study-20260929`. Initial study implementation: `04aaa6d`.

## Design

- Rates: 9, 19, 29, 35 Mbps. Paired seeds: 1, 2, 3.
- Noise: independent Gaussian draws in dB with standard deviation 0 through 10 dB, separately on desired-link and beam-averaged interfering-link reports for the four micro BSs.
- The same normalized draws are reused across standard deviations and loads. The two gain types have independent random streams, distinct from traffic, fading and RB mapping.
- A report is changed once and reused consistently at every causal consumer. Physical channels, candidate beam-pair indices and macro-BS gains are unchanged. Actual selected beams, probing counts, associations and queues evolve normally.
- 252 unique full simulations: 12 common zero-noise controls plus 240 perturbed runs. Each contains 300 frames, with two initial frames excluded from system metrics.
- All 12 zero-noise runs must reproduce the original full raw arrays before any full perturbation cases are launched. Full runs cannot be replaced by smoke tests.
- `prediction_statistics.json` covers all labeled cached reports, including warmup. `prediction_statistics_eval_window.json` additionally matches the system metric window and should be used for comparisons with previously reported test MAEs.
- All outputs are under `experiment/results/gain_error_sensitivity_20260929/`; no existing model, paper, figure or simulation result is overwritten.

## Commands

Run from `/home/ubuntu/niulab/Meet_Cobra`, using the `sionna` environment. GPU execution must have access to the server's NVIDIA driver.

```bash
python -m unittest test_gain_error_sensitivity test_revision_directional test_gap_rb_usage -v
python -u experiment/gain_error_sensitivity.py prepare
python -u experiment/gain_error_sensitivity.py case --kind desired --sigma 0 --rate 19 --seed 1 --device cuda:0 --smoke-frames 8
python -u experiment/gain_error_sensitivity.py case --kind interfering --sigma 10 --rate 35 --seed 1 --device cuda:1 --smoke-frames 8
python -u experiment/gain_error_sensitivity.py run --workers-per-device 4 --detach
```

The default device list is `cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6`. Set `--devices` and `--workers-per-device` according to available resources. The launcher has a directory lock, individual case locks, a 3600 s per-case limit, and stops scheduling new cases if one fails. Repeating the same `run` command resumes incomplete cases and verifies saved results before skipping them; do not start another launcher while one is active.

```bash
python experiment/gain_error_sensitivity.py status
tail -f experiment/results/gain_error_sensitivity_20260929/queue.log
python experiment/analyze_gain_error_sensitivity.py --allow-partial
```

After completion, run the independent queue/RB audit and produce diagnostic figures (not manuscript figures):

```bash
python experiment/analyze_gain_error_sensitivity.py --audit
```

`summary.json` and `curves.csv` report per-seed values and three-seed means/min/max. `raw/` contains queues, RB occupancy, per-vehicle serving BSs, power, probes and HO counts. `diagnostics/` contains the per-frame HO/GAP and explicit-directional-service checks. `independent_audit.json` independently checks the requested metrics and pairing. The figure files and experiment report stay inside the study directory.
