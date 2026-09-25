# Prediction-only O-MAPPO HO32 / cross5 experiment

Status: implementation and small-scale checks passed; full training/evaluation pending.

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
  using the fixed untrained prediction-input validation control. Score is half
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

## Commands

```bash
/home/ubuntu/anaconda3/envs/sionna/bin/python -u experiment/train_o_mappo_predicted_cross5.py launch
tail -f experiment/results/o_mappo_predicted_cross5_20260925/pipeline.log
```

After interruption, the same launch command resumes from a complete validation
boundary and skips verified completed evaluation cases. Concurrent launches are
blocked by a process lock. Never change code or inputs in a frozen result root.
