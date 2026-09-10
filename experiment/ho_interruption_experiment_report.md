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

The driver resumes completed cases only when the protocol and source hashes
match. Use another `--output` directory for different settings.

## Version management and restoration

- `76078e5`: checkpoint of the exact pre-experiment simulator and algorithms,
  including earlier batching-related modifications that were already present.
- Tag `meet-cobra-pre-ho-interruption-20260910`: points to that checkpoint.
- `9f46f9b`: opt-in interruption implementation, paired driver, and unit tests.
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
reconciles energy with allocated RBs, checks finite outputs and physical RB caps,
verifies blocked-slot counts against actual HO counts, and requires all paired
zero-interruption arrays to be bitwise identical. Results and their interpretation
are recorded below after the sweep finishes.
