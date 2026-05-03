# Skill: run_training_stage

When to use: a `recipe_proposal` is accepted and a
`dataset_snapshot` is ready. This skill drives one full
training_run end-to-end: launch via the platform `StageRegistry`,
let the platform enforce divergence kills, and record the result.

## Execution model — platform runners

Training runs through the platform's stage runners under
`agent_evolve/model/runners/stages/*.py`. You do NOT invoke a
workspace-local script. `launch_training` delegates to
`backend.run_trial`, which dispatches through `StageRegistry` to the
correct `@register_stage` implementation (`sft`, `rl`,
`teacher_distill`, `solver_distill`, `data_merge`, `generate`).

## Inputs

- `recipe_proposal_id` (ResearchScientist's proposal you're executing)
- `dataset_snapshot_id` (DataScientist's data you're training on)
- Stage to run (`sft` or `rl`) — derived from
  `train/pipeline.yaml::stages` in the recipe

## Procedure

1. `mem_get(<recipe_proposal_id>)` — read the diff. Apply it by
   copying the relevant YAMLs into a `cycles/<cycle_id>/` overlay
   dir and editing them there. Do NOT mutate the workspace YAMLs
   in place — the overlay keeps each training_run reversible.
2. `mem_get(<dataset_snapshot_id>)` — get the path to the
   `train.jsonl`.
3. `launch_training(recipe_path=cycles/<cycle_id>/,
   data_path=<from snapshot>,
   ckpt_out=cycles/<cycle_id>/ckpt/,
   monitor=true)`.
4. The platform's stage runner enforces the hard rules:
   - Kill if loss > 2× starting loss for 50 consecutive steps.
   - Kill on NaN.
   - For RL: kill if KL-to-reference grows > 5× initial.
5. On completion: inspect the JSON returned by `launch_training`
   (`status`, `ckpt_path`, `metric_name`, `metric_value`,
   `cost`). For deeper logs, `read_training_log(<job_id>)` and
   `read_checkpoint_metric(<ckpt_path>)`.

## Output

If the run returns `status == "success"`:

```yaml
kind: training_run
title: "<stage> run: <ckpt_path> — <metric_name> <metric_value>"
body: |
  Stage:              <sft | rl>
  Recipe proposal:    <recipe_proposal_id>
  Dataset snapshot:   <dataset_snapshot_id>
  Recipe overlay:     cycles/<cycle_id>/
  Dispatched via:     StageRegistry → agent_evolve/model/runners/stages/<stage>.py
  Wallclock:          <h:mm:ss>
  GPU-hours:          <#>
  Final step:         <N>
  Primary metric:     <metric_name> = <metric_value>
  RL extras (if rl):
    Mean reward:      <R>
    KL to ref:        <KL>
    Rollouts:         <count>
  Checkpoint path:    <ckpt_out>
  Status:             success
tags: ["<stage>", <recipe_family>]
refs: [<recipe_proposal_id>, <dataset_snapshot_id>]
```

If `launch_training` returns a non-success status (divergence, OOM,
missing platform stage), write a `failed_attempt` instead — NOT a
`training_run`:

```yaml
kind: failed_attempt
title: "<stage> failed: <status> (<kill_reason>)"
body: |
  Recipe proposal:   <recipe_proposal_id>
  Dataset snapshot:  <dataset_snapshot_id>
  Status:            <train_failed | oom | invalid>
  Kill reason:       <loss-explosion | NaN | KL-blowup | OOM | missing-stage>
  Loss trajectory:   <last 10 logged losses, if available>
  Hypothesis:        <e.g. "LR too high — see profile_run rec_X
                       which suggested 5e-6 not 2e-5">
  Recommendation:    <e.g. "ResearchScientist should propose a revised
                       recipe with lower LR before retrying">
tags: ["divergence", "<stage>"]
refs: [<recipe_proposal_id>, <dataset_snapshot_id>]
```

## Hard rules (re-stated)

- One training_run = one recipe = one dataset_snapshot = one refs
  link triplet. NEVER batch.
- Use a per-cycle overlay dir for the recipe; never mutate
  workspace YAML in place during a run.
- Non-success runs are NOT `training_run` records — they are
  `failed_attempt`. ResearchScientist treats them differently.
- Training is implemented in `agent_evolve/model/runners/stages/`.
  Do NOT create or edit workspace-local runner scripts.

## Anti-patterns

- Do NOT skip the monitor (`monitor=false`). The platform stage
  runner is what catches divergence; bypassing it burns a full
  training_run's GPU-hours for no record.
- Do NOT continue past a NaN. NaN means weights are undefined —
  any further training corrupts more.
- Do NOT race two training runs on overlapping GPUs. Confirm
  via the backend that the GPU set is free before launching.
- Do NOT modify the proposed recipe to "make it work" if it
  diverges. Surface the divergence; ResearchScientist proposes a fix.
- Do NOT attempt to scaffold a workspace-local runner as a
  workaround for a missing platform stage. Write a
  `failed_attempt` with a concrete "needs @register_stage('<X>')"
  body so ResearchScientist / platform owners know.
