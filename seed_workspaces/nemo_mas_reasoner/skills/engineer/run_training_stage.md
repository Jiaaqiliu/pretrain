# Skill: run_training_stage

When to use: a `recipe_proposal` is accepted and a
`dataset_snapshot` is ready. This skill drives one full
training_run end-to-end: launch, monitor, kill on divergence,
record.

## Inputs

- `recipe_proposal_id` (Theorist's proposal you're executing)
- `dataset_snapshot_id` (DataEngineer's data you're training on)
- Stage to run (`sft` or `rl`) — derived from
  `train/pipeline.yaml::stages` in the recipe

## Procedure

1. `mem_get(<recipe_proposal_id>)` — read the diff. Apply it to
   the workspace YAML files (Theorist's proposals are the source
   of truth; you write the YAMLs from the diff, then run).
   Actually — DO NOT mutate the YAMLs in place; instead, copy the
   relevant YAMLs into a `cycles/<cycle_id>/` overlay dir, apply
   the diff there, and pass that overlay to the runner. This
   keeps the workspace itself clean and reversible.
2. `mem_get(<dataset_snapshot_id>)` — get the path to the
   `train.jsonl`.
3. `mem_recent(kind="runner_capability", k=1)` — confirm the
   stage you need is covered. If not, route via
   `scaffold_sft_runner` / `scaffold_rl_runner` first.
4. `launch_training(runner_path=runner/<stage>_runner.py,
   recipe_path=cycles/<cycle_id>/<stage>.yaml,
   data_path=<from snapshot>,
   ckpt_out=cycles/<cycle_id>/ckpt/,
   monitor=true)`.
5. While running, the monitor enforces hard rules from
   engineer.md:
   - Kill if loss > 2× starting loss for 50 consecutive steps.
   - Kill on NaN.
   - For RL: kill if KL-to-reference grows > 5× initial.
6. On completion: `read_training_log(<job_id>)` and
   `read_checkpoint_metric(<ckpt_out>)`.

## Output

If the run completed successfully:

```yaml
kind: training_run
title: "<stage> run: <ckpt_path> — final loss <L>"
body: |
  Stage:              <sft | rl>
  Recipe proposal:    <recipe_proposal_id>
  Dataset snapshot:   <dataset_snapshot_id>
  Runner used:        runner/<stage>_runner.py
  Recipe overlay:     cycles/<cycle_id>/
  Command line:       <verbatim>
  Wallclock:          <h:mm:ss>
  GPU-hours:          <#>
  Final step:         <N>
  Final train loss:   <L>
  RL extras (if rl):
    Mean reward:      <R>
    KL to ref:        <KL>
    Rollouts:         <count>
  Checkpoint path:    <ckpt_out>
  metric.json hash:   <sha>
  Status:             success
tags: ["<stage>", <recipe_family>]
refs: [<recipe_proposal_id>, <dataset_snapshot_id>]
```

If the monitor killed the run, write a `failed_attempt` instead
(NOT a `training_run` — Theorist must know it diverged):

```yaml
kind: failed_attempt
title: "<stage> diverged: <kill_reason> at step <N>"
body: |
  Recipe proposal:   <recipe_proposal_id>
  Dataset snapshot:  <dataset_snapshot_id>
  Killed at step:    <N>
  Kill reason:       <loss-explosion | NaN | KL-blowup>
  Loss trajectory:   <last 10 logged losses>
  Hypothesis:        <e.g. "LR too high — see profile_run rec_X
                       which suggested 5e-6 not 2e-5">
  Recommendation:    <e.g. "Theorist should propose a revised
                       recipe with lower LR before retrying">
tags: ["divergence", "<stage>"]
refs: [<recipe_proposal_id>, <dataset_snapshot_id>]
```

## Hard rules (re-stated)

- One training_run = one recipe = one dataset_snapshot = one refs
  link triplet. NEVER batch.
- Use a per-cycle overlay dir for the recipe; never mutate
  workspace YAML in place during a run.
- Killed runs are NOT `training_run` records. They are
  `failed_attempt`. Theorist treats them differently.

## Anti-patterns

- Do NOT skip the monitor (`monitor=false`) to "let it run". The
  monitor is what catches the failures Theorist needs to know
  about.
- Do NOT continue past a NaN. NaN means weights are undefined —
  any further training corrupts more.
- Do NOT race two training runs on overlapping GPUs. Confirm
  via the backend that the GPU set is free before launching.
- Do NOT modify the proposed recipe to "make it work" if it
  diverges. Surface the divergence; Theorist proposes a fix.
