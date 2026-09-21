# Configurable GAP-HO refinement study

Date: 2026-09-16

## Scope and version protection

The pre-study code checkpoint is `a5e50ff57d92cc96813d97266c89693379a9a109`.
It includes the previously untracked experiment and baseline source files,
without committing unrelated manuscript edits, figure edits, raw data, or
discarded files. The legacy two-pass GAP-HO implementation remains the default.

This study implements option B: a configurable maximum number of alternating
RB-demand/association updates with an optional absolute load-change tolerance.
No manuscript, response letter, or paper figure is modified by this study.

## Implementation

- Entry: `HO_EE_GAP_APX_SINR_conservative_adaptive` in `utils/alg_utils.py`.
- Opt-in helper: `utils/gap_refinement.py`.
- Simulator option: `gap_refinement_config`, accepting a configuration object
  or a dictionary with `max_iterations`, `tolerance_rb`, and
  `relaxation_factor`.
- `tolerance_rb=None` disables early stopping. A tolerance of zero instead
  stops only at exact equality; the two settings are not interchangeable.
- Each round recomputes the frame-average RB demands, solves GAP with the
  current demands, and updates the implied frame-average BS loads. The
  residual is the maximum absolute load change across all BSs, in RBs.
- Initial loads are the physical BS capacities. Micro-BS loads determine the
  initial full-load interference estimate. Macro-BS load does not enter
  interference, but its initialization makes the all-BS stopping test explicit.
- Final feasibility repair runs once, after termination of the refinement.
  Recorded convergence refers to the pre-repair update and does not establish
  a fixed point of the final repaired assignment.
- The HO factor multiplies capacity coefficients only. Power costs and the
  interference feedback use uninflated frame-average RB demand.
- The existing adaptive capacity reserve, measured pilot-count estimate,
  fallback without an interference predictor, and OTR-RA are retained.
  These remaining paper/implementation differences are not silently changed.
- No damping, objective-based iterate selection, or cycle-breaking changes
  are introduced. Repeated iterates are recorded diagnostically, not treated
  as convergence.

## Verification

Fifteen unit tests in `test_gap_refinement.py` and `test_ho_interruption.py`
pass. They cover bounds and validation, inclusive absolute tolerance, disabled
versus zero tolerance, nonconvergent oscillation, legacy dispatch, real-solver
two-pass regression, empty inputs, capacity-only HO corrections, and repair.

The end-to-end regression executes the checkpoint's HO source without changing
the working tree. With zero and 10-ms interruption, all eight simulator
outputs are bitwise identical for the checkpoint, unchanged default, and
new two-iteration/no-early-stop path, including full queues and associations.
The initial three-run smoke test also passed independent recomputation of
violation probabilities, energy from allocated RBs, physical capacity limits,
blocked slots from actual handovers, and all stopping-rule traces.

The nine 30-s legacy runs at 21, 27, and 35 Mbps and seeds 1--3 also match
all saved arrays of the earlier HO-capacity-correction experiment bit for bit.
The dataset fingerprint matches its previously recorded SHA-256,
`034b9575072dcd00dedf3afbd0605f35cfea07ce2e8d78692b9566e7b907ffc0`.

## Experimental protocol

The study uses existing RT traces beginning at 800 s. Oracle next-frame gains
and interference are used for continuing vehicles, with the existing departure
fallback, so no NN inference or model reselection is involved. The existing
Oracle simulator uses RT frame channels without additional random slot-level
Rician draws. Entry queues and independent Poisson arrivals are paired across
all modes by vehicle and slot, with matching SHA-256 fingerprints.

All compared modes use 10-ms HO interruption and capacity correction,
`M_P=5`, the same traffic trace, the same reserve rule, and the same OTR-RA.
Two initial frames are excluded from reported performance metrics. The
decision-time measurements cover the CPU reference HO implementation and
diagnostic collection, under concurrent workers; they are comparative runtime
measurements, not deployment latency guarantees.

Initial screening uses 10 s, seed 1, and 5, 21, 27, and 35 Mbps per vehicle.
It includes the legacy path; fixed budgets of 1, 2, 3, 5, and 10 rounds;
three-round budgets with tolerances 0.1, 0.5, and 1 RB; and a five-round
budget with tolerance 0.1 RB. This is a local parameter study, not a full
load-grid or learned-predictor evaluation.

```bash
python -m unittest -v test_gap_refinement test_ho_interruption
python experiment/gap_refinement_experiment.py --seconds 3 --rates 21 \
  --modes legacy iter2_off iter3_off --workers 3 --regression \
  --output experiment/results_gap_refinement_smoke_20260916
python experiment/gap_refinement_experiment.py --seconds 10 \
  --rates 5 21 27 35 --seeds 1 --workers 12 \
  --output experiment/results_gap_refinement_screen_20260916
```

Each output directory records an immutable protocol, source hashes, dataset
hash, paired-traffic hashes, per-seed metrics, raw slot queues, per-frame
associations, and per-decision refinement traces. Resume only under an
identical protocol. `experiment/analyze_gap_refinement.py --output PATH`
independently validates saved results and writes `validated_summary.json`.

## Results

### Ten-second screening (seed 1)

All 40 runs completed and passed independent validation. The actual evaluated
interval is 9.7 s after excluding the initial two simulation frames.
The three-second smoke test is an implementation check, not a parameter-
selection result. Values below are system power in W and violation probability
in percent; these are not the subsequently confirmed three-seed results.

| Load (Mbps) | Legacy two passes, P / U | Three fixed rounds, P / U | Five fixed rounds, P / U | Ten fixed rounds, P / U |
|---:|---:|---:|---:|---:|
| 5 | 8.37678 / 0 | 8.37701 / 0 | 8.37645 / 0 | 8.37645 / 0 |
| 21 | 44.93757 / 0.007400 | 44.92841 / 0.012369 | 44.92305 / 0.017953 | 44.92305 / 0.017953 |
| 27 | 109.62435 / 0.010304 | 109.85584 / 0.010457 | 109.81786 / 0.010457 | 109.73233 / 0.010688 |
| 35 | 185.72629 / 27.005315 | 185.70039 / 26.805931 | 185.69524 / 26.929085 | 185.69513 / 26.942891 |

With a three-round budget, tolerance 0.1 RB preserved the fixed-three-round
system metrics at all four screened loads. At 27 Mbps it reduced the average
number of rounds from 3 to 2.773. Tolerances 0.5 and 1 RB further reduced this
number to 2.206 and 2.186, with a small change in power to 109.89095 W; the
violation probability remained 0.010457%. At the other three loads these
tolerances did not reduce the executed number of rounds below three.

With a five-round budget and tolerance 0.1 RB, average rounds were 4.021,
4.175, 3.113, and 4.433 at 5, 21, 27, and 35 Mbps. The first three loads
retained the fixed-five-round system metrics; at 35 Mbps the violation
probability changed from 26.929085% to 26.962246%.

The saved `tolerance_stop_percent` includes decisions that first meet the
tolerance on their final allowed round. It must not be read as the percentage
of decisions that saved computation. `early_stop_before_cap_percent_mean`
in the independent analysis counts only stops strictly before the budget.

The tenth-round residual still exceeded 0.1 RB in 14/97 evaluated decisions
at 27 Mbps and 5/97 at 35 Mbps. Thus a larger finite budget does not establish
convergence for every discrete assignment update. Smaller load residuals also
do not imply monotonic improvements in system power or queue violations.

### Thirty-second, three-seed follow-up

This follow-up compares legacy two-pass execution with budgets of three and
five rounds, both using tolerance 0.1 RB, at 21, 27, and 35 Mbps. All use the
same 10-ms interruption and HO-aware capacity correction. Seeds are 1, 2,
and 3, with a 29.7-s evaluated interval. All 27 runs completed.

| Load (Mbps) | Legacy, P (W) / U (%) | Three rounds, epsilon=0.1 RB | Five rounds, epsilon=0.1 RB |
|---:|---:|---:|---:|
| 21 | 46.51392 / 0.006318 | 46.54126 / 0.008223 | 46.54511 / 0.009915 |
| 27 | 113.21266 / 0.004774 | 113.34323 / 0.007068 | 113.24949 / 0.007045 |
| 35 | 185.73112 / 34.217141 | 185.74550 / 34.394583 | 185.74625 / 34.505973 |

These are arithmetic means across the three evaluation seeds. The maximum
power increase of the three-round configuration is about 0.115% (27 Mbps).
Its increases in U are 0.001905, 0.002295, and 0.177442 percentage points at
21, 27, and 35 Mbps. The paired confidence intervals are recorded in
`validated_summary.json`; they condition on the fixed RT trace, Oracle
prediction, and current traffic model, not on independent deployments or
NN training runs.

Not every delay statistic worsens. At 35 Mbps, mean P99 of the queue-length-
based latency proxy decreases from 9387.72 ms to 9271.63 ms for three rounds
and 9292.24 ms for five rounds, while U increases. Hence this study does not
establish dominance over every metric; it does show no improvement in the
power/U pair that motivates the parameter choice.

At this overloaded 35-Mbps point, all these variants can finish repair with
an infeasible *planning* assignment. This is inherited behavior and is
recorded explicitly. Actual slot allocations still respect physical RB
capacities, as checked from the saved arrays. Bounded fixed-point refinement
does not make an overloaded planning problem feasible.

| Load (Mbps) | Average rounds, three-round cap | Average rounds, five-round cap | Legacy HO median (ms) | Three-round HO median (ms) | Five-round HO median (ms) |
|---:|---:|---:|---:|---:|---:|
| 21 | 2.997 | 4.054 | 809.3 | 1223.7 | 1625.2 |
| 27 | 2.902 | 3.310 | 1073.5 | 1563.9 | 1609.7 |
| 35 | 3.000 | 4.495 | 1112.7 | 1683.4 | 2426.6 |

Runtime entries are the means of the three seed-level medians for the
single-threaded Python reference HO implementation, measured with concurrent
simulation workers. They are not NN inference times. Even the legacy code
does not demonstrate a 100-ms decision deadline on this implementation;
solver optimization and a dedicated end-to-end runtime study would be
separate work. No solver implementation was changed to improve this timing.

```bash
python experiment/gap_refinement_experiment.py --seconds 30 \
  --rates 21 27 35 --seeds 1 2 3 \
  --modes legacy iter3_eps0p1 iter5_eps0p1 --workers 12 \
  --output experiment/results_gap_refinement_confirm_20260916
```

### Zero-interruption diagnostic (10 s, seed 1)

A separate nine-run diagnostic removes the HO interruption, keeping all other
settings paired. At zero interruption, capacity factors are exactly one.
This checks whether the direction of the refinement effect is independent of
the HO setting; it is not a second three-seed confirmation study.

| Load (Mbps) | Legacy, P (W) / U (%) | Three rounds, epsilon=0.1 RB | Five rounds, epsilon=0.1 RB |
|---:|---:|---:|---:|
| 21 | 44.90087 / 0.009369 | 44.89668 / 0.009145 | 44.88864 / 0.008990 |
| 27 | 108.40186 / 0.004522 | 108.28637 / 0.004138 | 108.24812 / 0.004214 |
| 35 | 185.74746 / 26.587145 | 185.79843 / 26.577479 | 185.79843 / 26.543938 |

There are small improvements at 21 and 27 Mbps without interruption. Thus
the experiments do not support a universal claim that additional rounds
always help or always hurt. Parameter choice must be made under the intended
HO setting, rather than inferred from a shrinking load residual alone.

```bash
python experiment/gap_refinement_experiment.py --seconds 10 \
  --rates 21 27 35 --seeds 1 --ho-ms 0 \
  --modes legacy iter3_eps0p1 iter5_eps0p1 --workers 9 \
  --output experiment/results_gap_refinement_zero_ho_20260916
```

## Conclusions and recommendation

1. Option B is implemented and tested, but an implementation generalization
   should not be presented as a demonstrated performance improvement.
2. The load-change tolerance controls consistency of the estimated load
   update, not the queue-violation objective. Integer GAP rounding, final
   repair, subsequent handovers, and slot scheduling prevent an inference
   that a smaller fixed-point residual must improve actual system metrics.
3. The tested 10-ms-interruption, 30-s three-seed results favor retaining the
   legacy two-pass behavior for the power/U pair. Do not automatically adopt
   three or five rounds simply to preserve the manuscript's historical
   `N_iter=3` statement. In the subsequent discussion on 2026-09-16, the user
   approved a fixed two-round paper configuration, retaining the general
   loop but removing its tolerance-based stopping test. The optional code
   and all recorded experimental configurations remain unchanged.
4. Keep the configurable implementation for subsequent experiments. Its
   explicit two-round/no-early-stop setting is a verified way to reproduce
   the legacy behavior. Tolerance 0.1 RB is a tested candidate, not a recovered
   legacy value or a proven optimum. Setting 0.5 or 1 RB has only been screened
   on the 10-s, single-seed grid here.
5. The existing reserve and measured-pilot assumptions remain untouched.
   This study therefore resolves the missing configurable iteration mechanism,
   not every difference between the manuscript and the simulator. Formal
   parameter adoption should also use the selected stateful NN checkpoints
   and the final agreed system model; this study intentionally uses Oracle data.

Across the smoke test, screening, three-seed follow-up, and zero-interruption
diagnostic, 79 simulation runs are saved. Independent validators check source
hashes, paired traffic, physical capacities, energy, queue violations, blocked
slots, and the iteration/stopping traces. The report and compact protocols,
per-run metrics, and validated summaries are suitable for Git; large raw NPZ
queue arrays and full decision traces remain in their experiment directories.

## Using the optional implementation

Pass this dictionary to `run_sim_withUMa` to reproduce the three-round trial:

```python
gap_refinement_config = {
    "max_iterations": 3,
    "tolerance_rb": 0.1,
    "relaxation_factor": 1.1,
}
```

`gap_refinement_diagnostics` optionally accepts a list to collect the
per-decision traces. There is no change to the default caller behavior.

## Returning to the old behavior

Omit `gap_refinement_config` (or set it to `None`). No Git reset is required.
For the new implementation with equivalent two-pass behavior, explicitly use
`max_iterations=2` and `tolerance_rb=None`. The default HO interruption and
capacity-correction options are also unchanged from the pre-study checkpoint.
