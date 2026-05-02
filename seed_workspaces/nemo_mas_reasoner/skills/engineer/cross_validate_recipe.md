# Skill: cross_validate_recipe

When to use: a `training_run` looks promising (latest `eval_report`
shows a meaningful gain over the prior best) AND the Orchestrator
spawns you to confirm stability before promoting. This is the last
step before a recipe gets recommended for the leaderboard
submission.

## Inputs

- The promising `training_run_id` (its recipe + dataset_snapshot
  defines what to rerun).
- N **training** seeds (default 3) — the Orchestrator's task message
  specifies. **Note**: eval is deterministic (temp=0.0); same ckpt +
  same eval split → same score. So CV variance comes from the
  training seed (data shuffle order, LoRA init), NOT from eval. Each
  rerun produces a different ckpt, and we eval each ckpt once.
- M splits (default 1, the same kaggle_dev_local; can be expanded
  to alternate held-out splits if available).
- Stability threshold for the verdict (default std/mean < 0.02).

## Cost reality check

CV is expensive. 3 seeds × 1 split = 3× the compute of one
training_run. Before launching, confirm the budget actually
allows this — if not, propose a 2-seed CV in your final response
and let Orchestrator decide.

## Procedure

1. `mem_get(<training_run_id>)` — extract recipe overlay path,
   dataset_snapshot id, recipe_proposal id.
2. `mem_get(<recipe_proposal_id>)` and `mem_get(<dataset_snapshot_id>)`
   to confirm both are pinned (no mutation since the original run).
3. `rerun_recipe_with_seeds(recipe_path=<overlay>,
   data_path=<from snapshot>, seeds=[<seed1>, <seed2>, <seed3>],
   splits=[<split>])` → list of new training_run ids.
4. For each rerun, after it completes: spawn an Analyst (in your
   final response, request this — Engineer doesn't spawn
   Analysts) to run eval and write `eval_report`.

   *Note*: this skill PRODUCES the training_runs and the
   `cv_result`. Engineer doesn't run the eval itself — that's
   Analyst's tool. So the flow is: launch reruns → wait for
   completion → write a `cv_result` placeholder that lists the
   training_run_ids and asks Orchestrator to spawn Analyst evals.

   Once Analyst eval_reports are in, a follow-up Engineer spawn
   updates the cv_result placeholder via `mem_link` to add
   `refs` to the eval_reports, OR writes a new finalized
   `cv_result`.
5. `compute_stability(training_run_ids=<list>,
   eval_report_ids=<list once available>)` → table of per-seed
   scores, mean, stddev.

## Output

Initial placeholder (after launching reruns):

```yaml
kind: cv_result
title: "CV launched for rec_<orig> — <N> seeds, awaiting evals"
body: |
  Origin training_run:  <training_run_id>
  Recipe proposal:      <recipe_proposal_id>
  Dataset snapshot:     <dataset_snapshot_id>
  Seeds:                [<s1>, <s2>, <s3>]
  Splits:               [<split>]
  Rerun training_runs:  [<rerun_1>, <rerun_2>, <rerun_3>]
  Status:               training-complete, awaiting eval
  Stability threshold:  std/mean < 0.02
tags: ["cv", "pending"]
refs: [<training_run_id>, <recipe_proposal_id>, <dataset_snapshot_id>,
       <rerun_1>, <rerun_2>, <rerun_3>]
```

Final cv_result (after eval_reports are in, written by a follow-up
Engineer spawn, NOT a mem_link patch — refs are append-only and
the prior placeholder stays as-is for audit):

```yaml
kind: cv_result
title: "CV: rec_<orig> — mean <M>, std <SD>, <verdict>"
body: |
  Origin training_run:  <training_run_id>
  Recipe proposal:      <recipe_proposal_id>
  Dataset snapshot:     <dataset_snapshot_id>
  Seeds × splits:
    seed=<s1>, split=<sp>: score=<x1>  (eval_report=<er1>)
    seed=<s2>, split=<sp>: score=<x2>  (eval_report=<er2>)
    seed=<s3>, split=<sp>: score=<x3>  (eval_report=<er3>)
  Mean:    <M>
  Stddev:  <SD>
  std/mean: <ratio>
  Threshold: <0.02>
  Verdict: stable | unstable
  Promotion recommendation:
    If stable: PROMOTE — submit recipe rec_<proposal> to leaderboard.
    If unstable: DO NOT PROMOTE — Theorist should propose
                 (a) more data, (b) lower LR, or (c) larger
                 LoRA rank to stabilize. Cite this cv_result.
tags: ["cv", "final", <verdict>]
refs: [<training_run_id>, <recipe_proposal_id>, <dataset_snapshot_id>,
       <rerun_1>, <rerun_2>, <rerun_3>, <er1>, <er2>, <er3>]
```

## Anti-patterns

- Do NOT compute a `cv_result` from a single seed. Hard rule.
- Do NOT modify the recipe between reruns (different LR per seed
  is not CV; it's a sweep). Same recipe, different **training**
  seed.
- Do NOT "rerun eval with different seeds" — eval is deterministic
  at temp=0.0; that's a no-op and Analyst should refuse.
- Do NOT skip the stability threshold in the body. The threshold
  is part of the result; future Theorists need to know what gate
  was applied.
- Do NOT promote based on mean alone — std must clear the
  threshold. A 0.04 mean improvement with 0.10 std isn't a
  signal.
