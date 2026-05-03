# Skill: profile_lr_sweep

When to use: a new recipe family, a new dataset_snapshot, or a new
adapter rank — anything that meaningfully changes the optimization
landscape and warrants verifying that "training is at least sane"
before committing to a full run.

## Inputs

- A `dataset_snapshot` record id (or path to train.jsonl).
- A baseline recipe (the `recipe_proposal` you're about to test, OR
  the current default in `train/optimizer.yaml`).
- A small sweep grid: typically 3 LRs spanning ~1.5 orders of
  magnitude (e.g. 5e-6, 2e-5, 8e-5).

## Procedure

1. `mem_search(<recipe family>, kind="profile_run", top_k=5)` — has
   this LR range been profiled on similar data? If yes, link your
   new findings as confirmation/refutation rather than redoing.
2. For each LR:
   - `run_short_training(recipe_diff=<set lr to this value>,
     max_steps=200, log_every=10)` — AppliedScientist can do this; full
     training is MachineLearningEngineer's job.
   - Capture: loss trajectory, gradient-norm trajectory (if available),
     final-step loss, NaN events.
3. `plot_loss_curve(training_run_ids=[...])` — get a single PNG.
4. Apply the sanity tests:
   - **Loss decreases**: final-step loss < starting loss × 0.9 over
     200 steps. Failing this is a red flag.
   - **No NaN / Inf**: any NaN within 200 steps → recipe is broken.
   - **Not flat**: if loss change is < 1% over 200 steps, the LR is
     too low (or data is broken).
   - **Not exploding**: if loss > 2× starting loss anywhere → too high.
   - **Train ≠ random val**: sample 50 train and 50 val rows, compute
     loss on each — if they're identical to 3 decimals, you may have
     a data leak or a frozen model.

## Output

Write 1 `profile_run` per LR in the sweep, plus 1 summary record (also
`profile_run`) that compares them.

```yaml
# Per-LR record
kind: profile_run
title: "Profile: lr=<LR> on <snapshot_id> — <verdict>"
body: |
  Recipe diff: lr: <LR>  (others unchanged from baseline)
  Dataset:     <snapshot_id>
  Steps:       200
  Loss start:  <L0>
  Loss end:    <Lf>
  Decrease:    <(L0-Lf)/L0*100>%
  Grad norm trajectory: stable | growing | spiking
  NaN events:  none | <count>
  Sanity checks: pass | fail (<which>)
  Verdict: usable | too-low | too-high | broken
tags: ["profile", "lr_sweep", <recipe_family>]
refs: [<snapshot_id>]

# Summary record
kind: profile_run
title: "LR sweep summary on <snapshot_id>: best <LR>, range <lo>-<hi>"
body: |
  Tested: 5e-6, 2e-5, 8e-5 (steps=200 each)
  Best by loss-decrease: <LR>
  Loss-curve PNG: <png path>
  Recommendation for full training: lr=<LR>, with <warmup_advice>.
  Caveat: 200-step profile predicts full-training behavior weakly;
  profile is for catching broken recipes, not picking the optimum.
tags: ["profile", "lr_sweep", "summary", <recipe_family>]
refs: [<all per-LR profile_run ids>]
```

## Anti-patterns

- Do NOT recommend an LR for full training based on a 200-step
  profile alone. Profile rules out broken; it does not pick the
  best.
- Do NOT skip the train/val identical-loss check. This catches data
  leaks that profile-by-loss-curve would miss.
- Do NOT extend the sweep beyond 3 points "to be sure" — that's
  MachineLearningEngineer's job (they can run cv on the top 2 picks).
- Do NOT write a `recipe_proposal` from your findings. Write
  `profile_run` records and let ResearchScientist read them.
