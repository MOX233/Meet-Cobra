# HO interruption: original versus capacity-corrected GAP-HO

## Purpose and scope

This is a small paired Oracle experiment to decide whether multiplying only
GAP-HO's capacity coefficients by an HO interruption factor is useful. It does
not update the paper, response letter, official baseline curves, or NN models.

The two policies see the same interruption duration, RT trajectory, Oracle
information, per-vehicle arrival arrays, entry queues, BF overhead model, and
OTR-RA implementation. Original GAP-HO's history-dependent resource reservation
is retained in both policies. It can evolve differently because the policies
produce different queues; this is part of the closed-loop response.

## Model and algorithm switches

`run_sim_withUMa` accepts two independent, opt-in keyword arguments:

```python
ho_interruption_ms=0.0       # Physical execution interruption; default is zero.
ho_capacity_correction=False  # Whether GAP-HO anticipates that interruption.
```

- An actual data-BS change at a frame boundary blocks the first `L` slots of
  that frame, where `L = ho_interruption_ms / (1000 * args.slot_len)`.
- Durations must be slot-aligned and less than a frame. No silent rounding.
- Unchanged association and initial access do not incur HO interruption.
- The same duration applies to macro-to-micro, micro-to-macro, and micro-to-micro
  changes. Preparation is outside data resources; execution succeeds after the
  gap. Failure, retransmission, and control-plane energy are not added here.
- During the gap, an interrupted vehicle receives no data RBs and incurs no
  beam-search overhead, but its queue continues to accumulate arrivals.
- Its serving BS can allocate data RBs to other available vehicles. Power and
  realized interference use actual allocated RBs, not a post hoc time penalty.
- At present nonzero interruption is deliberately restricted to Oracle runs.
  Calling this path with NN predictors raises an error. Existing learned and
  baseline simulations remain unchanged; the new mode is not advertised as a
  validated extension of those runners.

For candidate `(m, v)`, the corrected policy uses

```text
factor = 1                                    if m is the current serving BS
factor = 1 / (1 - L / slots_per_frame)         otherwise
capacity coefficient = original average RB demand * factor
power cost           = original average RB demand * power_per_RB
average BS load      = sum(original average RB demand * association)
```

Both GAP iterations and capacity-feasibility repair use the corrected capacity
coefficients. The matching objective and repair destination costs use the
uninflated power costs. Interference-load updates use full-frame average demand.
The existing greedy repair ordering is retained, rather than redesigned.
The corrected constraint is a conservative planning approximation, not an exact
characterization of every possible slot-level schedule.

## Oracle and pairing details

The experimental driver reads the existing 800--830 s RT data interval. A frame
is 100 ms and a slot is 1 ms. There are 299 simulated frames because the first
data frame initializes history. All reported main metrics exclude the first
two simulated frames, leaving 29.7 s of evaluated operation.

The new `oracle_ho_cache` supplies **next-frame** desired-link gains, interfering
gains, positions, and beam-search counts for continuing vehicles. Thus it avoids
both NN inference and a current-frame holdover being mistaken for perfect
next-frame prediction. Vehicles absent from the next frame retain the existing
current-vehicle planning convention with a current-frame fallback. Newly entering
vehicles follow the existing initial-association rule. The last computed command
is unused because no subsequent frame is simulated.

Current-frame service uses the existing Oracle BF path. Oracle best beams are
known, but the existing pilot overhead is still charged in available slots.
As in the existing Oracle implementation, the RT channel is constant within a
frame and no extra slot-level Rician draws are added. This is an isolation test,
not a numerical replacement for the learned MEET-COBRA curves.

Arrivals and initial queues are generated once per `(load, seed)` by independent
random streams, then replayed for every duration and both policies. A SHA-256
fingerprint is stored and checked across each paired group. Seeds vary traffic
and entry queues, not the shared SUMO trajectory or RT geometry.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest -v test_ho_interruption

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiment/ho_interruption_experiment.py --regression-only \
  --seconds 1.3 --output experiment/results_ho_interruption_checks

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiment/ho_interruption_experiment.py --seconds 30 --workers 4

python experiment/analyze_ho_interruption.py
```

The default small grid is 5, 21, and 35 Mbps per vehicle; seeds 1, 2, and 3;
and interruption durations 0, 5, and 10 ms. Both policies are run at every point,
including zero, for 54 runs in total. These durations are diagnostic settings,
not a claim of standardized or measured HO interruption values.

After the initial 10-ms comparisons showed improvements at 21 Mbps but a small
violation increase at 35 Mbps, a targeted follow-up was added at the paper's
illustrative 27-Mbps operating point: three seeds, 10-ms interruption, and both
policies (six additional runs). This follow-up is reported separately, without
discarding the unfavorable high-load observations.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiment/ho_interruption_experiment.py --seconds 30 --workers 3 \
  --rates 27 --seeds 1 2 3 --durations-ms 10 \
  --output experiment/results_ho_interruption_27Mbps
```

The driver resumes completed cases only when the protocol and source hashes
match. Use another `--output` directory for different settings.
Workers use independent `spawn` initialization. The auxiliary
`run_selected_ho_cases.py` launcher was used to finish disjoint late-queue cases
early under the same protocol; the primary sweep subsequently reuses those
files. This changes execution order, not experimental settings or results.

## Version management and restoration

- `76078e5`: checkpoint of the exact pre-experiment simulator and algorithms,
  including earlier batching-related modifications that were already present.
- Tag `meet-cobra-pre-ho-interruption-20260910`: points to that checkpoint.
- `9f46f9b`: opt-in interruption implementation, paired driver, and unit tests.
- `088d304`: independent worker initialization and analysis infrastructure.
- Tag `meet-cobra-ho-interruption-oracle-20260910`: completed experiment snapshot,
  including the report, compact results, validation, and reproduction helpers.
- The user's unrelated staged `.gitignore` change and all manuscript edits were
  left out of these commits.

**No restoration is needed to run the old physical model.** Existing callers use
the unchanged defaults (zero interruption and no capacity correction). If a
literal source restoration is wanted later, the two original files can be
restored selectively, without resetting the worktree or touching the manuscript:

```bash
git restore --source=meet-cobra-pre-ho-interruption-20260910 -- \
  utils/alg_utils.py utils/sim_utils.py
```

This command is documented, not executed. New helper and experiment files can
remain, but the new experiment driver needs the extended simulator version.
Save any later uncommitted edits to these two files before restoring their source.

## Verification

Six unit tests cover slot-duration validation, stay versus switch factors,
initial access, independent GAP power and capacity coefficients, both GAP
iterations, feasibility-repair costs, average-load accounting, and traffic replay.

The regression check executes the historical simulator and GAP code directly
from the checkpoint using `git show`, without checking out or overwriting files.
It compares all eight returned objects, including full queue trajectories and
HO commands, at zero interruption. A forced-switch integration test also checks
zero RBs, zero pilots, and exact arrival accumulation in every blocked slot.

Saved-result validation independently recomputes `U` from the queue arrays,
reconciles energy with allocated RBs, checks finite outputs and recorded
frame-average RB allocations against physical RB caps,
verifies blocked-slot counts against actual HO counts, and requires all paired
zero-interruption arrays to be bitwise identical.

All six unit tests and the historical regression checks passed. Saved-result
validation passed for all 54 main-sweep runs and six follow-up runs; all nine
full-length zero-interruption pairs had bitwise-identical saved arrays.

## Results

All entries below are three-seed means over the same 29.7-s evaluated interval.
`U (%)` is the queue-length-based latency violation metric already used by the
simulator. HO counts exclude warmup and count actual BS changes, not proposed
commands or beam changes. Within each listed case, the HO count is identical
across the three seeds for a given policy.

| Load (Mbps per vehicle) | HO gap (ms) | Original power (W) | Corrected power (W) | Original U (%) | Corrected U (%) | HO count, original → corrected |
|---|---|---|---|---|---|---|
| 5 | 5 | 8.5464 | 8.5464 | 0 | 0 | 698 → 698 |
| 5 | 10 | 8.5469 | 8.5469 | 0 | 0 | 698 → 698 |
| 21 | 5 | 46.4338 | 46.4487 | 0.014356 | 0.006558 | 948 → 886 |
| 21 | 10 | 46.4389 | 46.5139 | 0.028381 | 0.006318 | 948 → 862 |
| 27 (follow-up) | 10 | 111.9765 | 113.2127 | 0.019510 | 0.004774 | 1907 → 981 |
| 35 | 5 | 185.7633 | 185.7619 | 34.030758 | 34.052437 | 2064 → 1394 |
| 35 | 10 | 185.7635 | 185.7311 | 34.124187 | 34.217141 | 2064 → 1026 |

### What the paired results establish

1. **No effect at low load.** At 5 Mbps, all saved raw arrays of the original
   and corrected policies are identical at both nonzero interruption durations,
   for all three seeds. This is consistent with ample capacity and an unchanged
   minimum-power association. The physical gap still occurs, but the capacity
   correction does not change decisions in these cases.
2. **Useful improvements at 21 Mbps.** At 5 ms, U falls by 54.32% relative to the
   original policy, for a 0.032% power increase; HOs fall by 6.54%. At 10 ms,
   U falls by 77.74%, power increases by 0.162%, and HOs fall by 9.07%. The
   respective absolute reductions in U are 0.007798 and 0.022064 percentage
   points. All three seeds show the same improvement direction.
3. **The benefit also appears at the 27-Mbps follow-up point.** At 10 ms,
   U falls by 75.53% (0.014736 percentage points), while power increases by
   1.2362 W, or 1.104%. HOs fall by 48.56%. The pooled p99 queueing-delay proxy
   decreases from approximately 2.240 ms to 1.631 ms. These are improvements
   in latency performance in exchange for modest additional transmit power,
   not a simultaneous reduction of both objectives.
4. **Less handover does not ensure lower U under saturation.** At 35 Mbps,
   HOs decrease by 32.46% and 50.29% for the 5-ms and 10-ms gaps, respectively.
   However, U increases by 0.021680 and 0.092954 percentage points. The changes
   in power are negligible. All three seeds show this same tradeoff.

The configured full-RB transmit-power ceiling is
`133 * 1 W + 4 * 66 * 0.2 W = 185.8 W`. The 35-Mbps cases operate almost exactly
at this ceiling. Reducing interruptions cannot remove their overall capacity
shortage. The correction also makes switching harder in the capacity constraint,
so fewer handovers can coexist with less favorable load redistribution. This is
an interpretation consistent with the results, not a proof that every changed
association has that effect.

For example, at 35 Mbps, 10 ms, seed 1, the macro-BS contribution to network-wide
U decreases from 6.2833 to 5.7597 percentage points, but the combined micro-BS
contribution increases from 27.8359 to 28.4533 percentage points. Thus, the
aggregate deterioration is not contradicted by the reduction in HO count.
The per-BS diagnostic CSV retains this decomposition for every case.

### Recommendation

**The capacity-only correction brings real but load-dependent benefits.** It is
not uniformly better than keeping GAP-HO unchanged, and the current power model
does not support claiming a material power saving from this correction.

For the next revision experiment, we favor keeping the capacity-corrected
GAP-HO as the preferred minimal HO-aware candidate. The 21- and 27-Mbps results
show that the correction can substantially reduce U at small power cost without
redesigning GAP-HO. Retain the original GAP-HO as an internal control and retain
both configuration switches; the highest-load counterexamples should not be
hidden or described as improved latency performance.

This Oracle isolation test does **not** establish the same gains with NN
prediction errors and slot-level Rician fading, nor does it validate O-MAPPO or
MTS-GS-HBF with the added interruption. Before adopting this extension in the
manuscript and regenerating its performance curves, verify the selected setting
in the learned MEET-COBRA implementation and apply a consistent physical gap to
the compared methods. No manuscript or response-letter changes have been made.

## Saved artifacts

The main sweep is stored in `experiment/results_ho_interruption/`; the additional
point is stored in `experiment/results_ho_interruption_27Mbps/`.
The shared RT cache's path, size, and SHA-256 fingerprint are recorded separately
in `experiment/ho_interruption_dataset_manifest.json`. Check that fingerprint
before resuming after data regeneration; source-hash validation alone does not
detect a changed RT file.

- `protocol.json`: model settings, sampling window, and experimental source hashes.
- `runs/*.json`: per-case metrics, traffic fingerprint, and elapsed wall time.
- `raw/*.npz`: full queue arrays with vehicle and frame indices, frame energy,
  U, HO counts, pilot counts, RB allocations, and blocked-vehicle-slot counts.
- `associations/*.json`: actual serving BSs and actual switched vehicles by frame.
- `per_seed.csv`, `aggregate.csv`, and `summary.json`: per-seed and aggregate data.
- `paired_differences.csv`: corrected minus original, including paired-seed
  differences and descriptive 95% Student-t intervals for their mean.
- `per_bs_diagnostics.csv`: each BS's contribution to U and transmit power.
- `validation.json`, `environment.json`: consistency checks and software versions.
- `comparison.pdf`, `comparison.png`: experimental figures, separate from all
  manuscript figures.

The intervals describe traffic-seed variation conditional on one fixed RT
trajectory; three seeds do not establish robustness to other trajectories or
traffic distributions. The raw queues and association logs remain on disk;
compact protocols, summaries, figures, and validation records are versioned.
The initial fork timing runs (`results_ho_interruption_initial_fork`) and separate
timing/progress probes are diagnostics only and are not pooled into these results.
