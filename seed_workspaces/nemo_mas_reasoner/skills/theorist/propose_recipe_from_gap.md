# Skill: propose_recipe_from_gap

When to use: a fresh `data_gap` record exists and the Orchestrator
has spawned you to turn it into an actionable `recipe_proposal`.
This is the most common Theorist task.

## Inputs

- The `data_gap` record id.
- The current cycle's most recent `eval_report`.
- Recent `cv_result` records (to know baseline stability).

## Procedure

1. `mem_get(<gap_id>)` — read it carefully. The gap should already
   have a "Proposed next batch" section. Your job is NOT to override
   that proposal; it's to (a) sanity-check it, (b) decide what
   recipe-side changes complement it, (c) write the proposal in
   formal form.
2. `mem_search(<category>, kind="recipe_proposal", top_k=8)` — has
   a similar proposal been tried? If yes:
   - Look for its `cv_result` (via `mem_search(<proposal_id>,
     kind="cv_result")`).
   - If it failed, your new proposal must explain why this attempt
     is different (different prompt source? different teacher?
     different mix weight?).
   - If it succeeded but the gap recurs, the issue isn't recipe —
     surface this as a `breakthrough` instead.
3. Decide the change scope. Two flavors:
   - **Pure-data change**: the proposal is just "commission the
     batch in the gap and re-mix". Touch `data/recipes/default.yaml`
     (e.g. raise `solver_upsample` for the gap's category) only if
     the gap evidence supports upsampling; otherwise leave the
     recipe alone.
   - **Data + recipe change**: the gap evidence shows the model
     learned but couldn't reproduce within budget — adjust
     `train/optimizer.yaml` (LR / warmup) or
     `data/recipes/default.yaml::filters::max_cot_tokens`.
4. Compose the diff. Use `render_recipe_diff` to format it.
5. Write the `recipe_proposal`.

## Output

```yaml
kind: recipe_proposal
title: "Proposal: <one-line action> — addresses <gap_id>"
body: |
  Motivation:
    Triggering gap:    <gap_id>
    Latest eval:       <eval_report_id>  score=<S>
    Prior attempts:    <list of prior recipe_proposals if any, with cv outcomes>

  Proposed change (single change — see hard rules):
    <unified diff or yaml block>

  Predicted effect:
    Direction:  <up | down | unclear>
    Magnitude:  <e.g. "+0.005 to +0.015 on the kaggle metric">
    Reasoning:  <one paragraph tying the gap to the change>

  Smallest test:
    1. DataEngineer commissions <batch spec from gap_id>.
    2. Engineer runs SFT on the new dataset_snapshot.
    3. Analyst evaluates on <split>.
    4. If score Δ > <threshold>, Engineer runs cv_result with 2
       seeds.

  Risks:
    - <e.g. "upsampling cryptarithm to 12 may starve other
       categories — check eval per-category accuracy doesn't
       regress more than 0.005">

  Rollback criterion:
    If <metric> drops by > <threshold>, revert to recipe rec_X.
tags: [<category>, "data-driven" | "recipe-driven", <change_axis>]
refs: [<gap_id>, <latest_eval_report_id>]
```

## Anti-patterns

- Do NOT propose more than one change in one record. Two changes →
  two proposals. (Hard rule.)
- Do NOT reference a `data_gap` you haven't read in full
  (`mem_get`, not just snippet).
- Do NOT predict effect with no reasoning — if you can't articulate
  why the change should help, it shouldn't be proposed.
- Do NOT skip the rollback criterion. Theorist's job is to plan for
  failure too.
- Do NOT propose recipe-side changes when the evidence points at
  data. Data first; recipe second; if both look right, the issue
  may be the model itself (which this MAS doesn't change — surface
  as a `breakthrough`).
