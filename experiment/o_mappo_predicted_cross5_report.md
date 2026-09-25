# Prediction-only O-MAPPO HO32 / cross5 experiment

Status: three-seed training and all-load evaluations completed on 2026-09-25.
All 324 scheme/load/seed records passed independent raw-data and paired-traffic
checks; all 27 relevant regression tests passed again after the experiments.

## Scope

The user approved prediction-only association decisions and corresponding PPO
training, followed by complete-load, multiple-seed evaluation. No manuscript,
response letter, formal checkpoint, production default or paper figure is changed.

Rollback tag: `pre-o-mappo-predicted-cross5-20260925` (commit `33a10aa`).
The four pre-existing tracked working-tree modifications are left untouched.

## Protocol

- Shared frozen gain-prediction frontend: `revision_directional_20260922/models`.
  The NN weights are not retrained. Fresh chronological prediction reports are
  generated for the existing training trajectories. The test cache is exactly
  the one used by the current paper grid.
- Actor: 31 -> 64 -> 64 -> 2; critic: 94 -> 64 -> 1; original feature scaling,
  binary HO trigger, optimizer objective, three alternative candidates, capacity
  penalty, and `qos_energy020_load1` reward retained.
- Initialize both networks from the approved E_all checkpoint, with fresh Adam
  optimizers for training seeds 11, 22 and 33. Minibatch size 256, four PPO epochs
  per update. No arbitrary minibatch-size changes.
- Every update contains an equal-duration rollout for each of all 18 arrival
  rates 1, 3, ..., 35 Mbps. Five-second chronological segments are drawn from
  200--650 s. Plan: 160 updates; cosine learning rate 1e-4 to 1e-5 by update 120.
- Use the **same exact slot simulator** for training and evaluation: independent
  paired Poisson arrivals, per-slot Rician fading, explicit orthogonal RB maps,
  directional cross-link interference and actual service feeding the queues.
  This replaces the earlier frame-fluid training approximation; it is disclosed
  rather than treated as an information-source-only controlled ablation.
- Validate every ten updates at all 18 loads on 700--710 s. Retain the best
  positive-update checkpoint per training seed, then choose one common model
  using all-load evaluation at 710--720 s. Test once at 800--830 s for all 18
  loads and traffic seeds 1, 2, 3; omit the initial two frames in summaries.
- Selection cost: U in percent + 0.02 P in watts, normalized separately by load
  using the fixed prediction-input control with no additional fine-tuning. The
  control retains the already-trained E_all actor; it is not randomly initialized.
  Score is half
  the mean normalized cost plus half the worst normalized cost.
- Convergence is assessed from five validation checkpoints, per-load cost span,
  aggregate cost span and policy changes. A prescribed update count alone is
  **not** reported as proof of convergence.

## Information boundary

1. Actor and target optimizer accept only whitelisted position/motion fields,
   predicted desired/interfering gains, queues, association and historical
   service feedback. H, optimal-beam labels, raw pilots and predicted beam lists
   are not accepted by the decision boundary.
2. Reports formed at x predict x+1 for association decisions. Current-frame RA
   receives the report formed at x-1. The bounded occupancy refinement is kept,
   but its channel inputs are predictions rather than private true gains.
3. RA retains measured serving-link gain, but uses predicted interfering gains.
4. A HO changes the BS without obtaining a free nominal planning beam. Physical
   acquisition occurs with 32 probes only after the ten interrupted slots.
   Otherwise five cross-shaped neighbors are probed in every active micro slot.
   Beams carry across frames. No PET-BF module is inserted into the baseline.
5. Private batched physical response buffers accelerate the simulator. Candidate
   selection consults only the 16 coarse + 16 fine or five paid measurements;
   unprobed responses never enter an actor state, critic state or optimizer.

Predicted maximum desired gain is a link-quality proxy; it is not assumed to
equal the eventual hierarchical-search outcome. Both candidate-BS and retained-
association estimates use the report-based convention. The macro link retains
the common position/path-loss model.

## Checks completed before the full run

- 27 relevant tests passed, including unchanged existing slot-tracking,
  predicted-state and HO-interruption tests.
- Counterfactual H and oracle-label changes do not change public/predicted
  actor inputs, critic inputs, occupancy estimates or association decisions.
- A 100-slot comparison reproduces original OTR-RA integer allocations and
  DirectionalService queues, including interrupted users and pilot costs.
- Batched BF reproduces the original paid-probe selections, gains and counts;
  perturbing future physical samples cannot change past chosen beams.
- A five-second GPU rollout and PPO update passed (494 executed-action
  transitions in that smoke run). Its numbers are validation-smoke diagnostics,
not final test evidence.

## Completed training and model selection

All three training seeds completed 160 PPO updates. Each update pooled the 18
loads before optimization; this is not 160 supervised-learning epochs. Training
took approximately 53.6 minutes, excluding data preparation and final evaluation.
The three best positive-update checkpoints were selected at updates 40, 20 and
150 for seeds 11, 22 and 33, respectively. Evaluation on the separate 710--720 s
selection interval chose seed 11, update 40. Its all-load score was 1.19652,
compared with 1.72807 and 1.79365 for the other two candidates. The no-additional-
fine-tuning control scored 1.72503 on this same interval. Lower is better.

None of the three training runs passed the predefined stability criterion.
Mean training reward over the last 20 updates was also not consistently higher
than over the first 20. Therefore, these results must not be described as a
converged policy or as a consistent improvement from continued training. The
chosen checkpoint is validation-selected, not the last checkpoint or a model
selected separately for each test load.

## Complete test comparisons

The selected prediction-input model, the prediction-input control without
additional fine-tuning, and the frozen true-CSI HO32/cross5 control each have
18 loads times three traffic/fading seeds, i.e., 54 runs of 30 s. Six verified
true-CSI pilot cases were reused; the remaining 48 were newly run. Existing
MEET-COBRA, Oracle-MC and formal O-MAPPO results were retained as references.
The no-additional-fine-tuning control was declared while training was ongoing
and cannot change the frozen positive-update training protocol. Its independent
validation score, not its test results, was used when comparing model choices.

The complete numerical tables, plots, raw-result audit and checkpoint hashes are
stored under `experiment/results/o_mappo_predicted_cross5_20260925/`:

- `report.md`, `comparison.csv`, and `analysis.json`: full-load results and provenance.
- `comparison.pdf` and `training_curves.pdf`: experimental figures only.
- `selection.json`, `training_complete.json`, and the three validation histories:
  model selection and stability diagnostics.
- `test/runs`, `zero_test/runs`, and `true_control/runs`: per-case JSON summaries,
  diagnostics and compressed raw queues, RB allocations and other system records.
- `training/seed11/best_positive.pt`: the common selected test checkpoint.

### Main findings

All numbers below are averages over three test seeds. U is expressed in percent,
not as a fraction.

- At 1, 5, 15 and 21 Mbps, the fine-tuned prediction-input model uses 5.889,
  12.385, 32.222 and 45.728 W, respectively, versus 5.439, 10.948, 26.848 and
  38.124 W for MEET-COBRA. The earlier low-load power advantage of the
  CSI-privileged cross5 version is not retained.
- MEET-COBRA has lower mean U at all 18 loads. At 25--33 Mbps the fine-tuned
  prediction-input model uses less power than MEET-COBRA, but with substantially
  larger violations. At 29 Mbps, for example, the comparison is 107.397 W /
  4.1799% versus 127.746 W / 0.1939%. This is not an energy advantage at matched
  latency performance.
- Continued training does not uniformly improve the prediction-input baseline.
  At 29 Mbps, retaining the original actor weights gives 136.539 W / 1.5763%,
  whereas the validation-selected fine-tuned model gives 107.397 W / 4.1799%.
  Its macro-BS association share falls from 13.61% to 7.74%, and its L99 proxy
  rises from 91.38 to 364.76 ms. These are observed tradeoffs, not proof that one
  specific training or estimation mechanism caused the performance change.
- The forecast-based decision interface is feasible and removes privileged
  candidate-channel access. It does not by itself establish that this PPO
  configuration generalizes robustly across loads. No final baseline or paper
  result has been replaced on the basis of this experiment.

### Interpretation limits

The true-CSI versus prediction-input comparison changes channel-derived actor
features, target-BS estimates, occupancy estimates and the interference input
used by RA. The fine-tuned version additionally changes actor weights and uses
exact slot-level training instead of the older frame-fluid approximation.
Therefore, the overall difference must not be attributed solely to actor input
noise or to the HO optimizer in isolation.

Three test seeds vary arrivals and small-scale fading over the same 30 s vehicle
trajectory interval; they are not three independent propagation environments.
Plot bands represent one sample standard deviation, not confidence intervals.
The current experiment supports a transparent information-matched comparison,
not a claim of universal or statistically established superiority.

## Subsequent paper promotion (2026-09-25)

After reviewing the experiment, the user approved the fine-tuned prediction-input
version as the formal O-MAPPO-adapted baseline. This is a separate promotion step;
the experiment above did not itself edit the paper. No simulations or NN training
were repeated during promotion.

The formal Fig. 5--8 assets (five PDFs, including separate L90 and L99 panels) now
use the selected seed-11/update-40 model at all 18 loads and all three test seeds.
`paper_figures/figure_replacement_audit.json` verifies all 90 O-MAPPO metric rows
against the approved analysis and confirms that the other seven schemes' 630
rows are unchanged. Figure formatting, mean/min/max conventions and captions are
unchanged. The old figures, working manuscript, response and figure data were
copied to `pre_promotion_archive/` before replacement.

The manuscript's baseline description now specifies prediction inputs and paid
HO32/cross5 search. Its performance discussion, L99 threshold crossing and
conclusion, together with R2C5 and R3 Major C3, reflect the observed power/latency
tradeoff rather than claiming lower power at every high load. No convergence or
algorithmic-performance-limit claim is added. The user's existing MTS paragraph
and Fig. 4 edits are preserved. Both LaTeX documents compile; ten figure-loader
regression tests pass, including legacy result formats and selection mismatch
rejection.

The default plotting entry point is now:

```bash
/home/ubuntu/anaconda3/envs/sionna/bin/python experiment/plot_revision_system_results.py
```

### Original experiment commands

```bash
/home/ubuntu/anaconda3/envs/sionna/bin/python -u experiment/train_o_mappo_predicted_cross5.py launch
tail -f experiment/results/o_mappo_predicted_cross5_20260925/pipeline.log
```

After interruption, the same launch command resumes from a complete validation
boundary and skips verified completed evaluation cases. Concurrent launches are
blocked by a process lock. Never change code or inputs in a frozen result root.
