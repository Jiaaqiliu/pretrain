# Skill: run_training_stage

When to use: a `recipe_proposal` is accepted and a `dataset_snapshot` is
ready. This skill drives one full training_run end-to-end via the
benchmark's backend CLI, monitors for divergence, and records the result.

## Execution model — benchmark backend

Training runs through the benchmark-specific k8s backend at
`agent_evolve/backends/nemo_reasoner/k8s/submit.sh`. You do NOT invoke a
workspace-local script. The backend reads the training recipe you pass,
loads the base model, applies the LoRA patches (huikang's recipe), and
writes artifacts to the directory you specify via `--out`.

## Inputs

- `recipe_proposal_id` (Planner's proposal you're executing)
- `dataset_snapshot_id` (DataWorker's snapshot — a data_recipe file
  under `recipes/data/` + materialized JSONL under `artifacts/data/<hash>/`)
- `run_name` — human label, e.g. `cycle12_lr3e4`. Used for job name +
  artifacts sub-directory.

## Procedure

1. `mem_get(<recipe_proposal_id>)` — read the diff. The Planner will
   have written a new training recipe at
   `recipes/train/<run_name>.yaml`. Confirm it exists.
2. `mem_get(<dataset_snapshot_id>)` — confirm
   `recipes/data/<data_recipe>.yaml` exists.
3. Resolve paths (exports set by the harness):
   ```bash
   BACKEND=/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner
   FORK=$NEMO_MAS_WORKSPACE   # absolute path to the per-cycle fork root
   ```
4. Launch:
   ```bash
   $BACKEND/k8s/submit.sh train \
       --train-recipe $FORK/recipes/train/<run_name>.yaml \
       --data-recipe  $FORK/recipes/data/<data_recipe>.yaml \
       --out          $FORK/artifacts/sft/<run_name> \
       --name         <run_name>
   ```
   Optional overrides: `--lr`, `--steps`, `--save-every`, `--seed`.
5. Monitor:
   - Tail `$FORK/artifacts/sft/<run_name>/train.log` (stdout captured).
   - Or `kubectl logs -f job/ne-train-<run_name_hyphenated>`.
6. Hard kill rules (enforced by the entry script, not you):
   - NaN loss → process exits.
   - Grad norm NaN → process exits.
7. On completion:
   - Adapters in `$FORK/artifacts/sft/<run_name>/step_{N}/` and `final/`.
   - `train.log` has per-step loss, grad, lr.

## Output

If training completed (adapter at `final/` exists):

```yaml
kind: training_run
title: "sft run: artifacts/sft/<run_name>/final — loss=<final_loss>"
body: |
  Recipe proposal:    <recipe_proposal_id>
  Dataset snapshot:   <dataset_snapshot_id>
  Run name:           <run_name>
  Train recipe:       recipes/train/<run_name>.yaml
  Data recipe:        recipes/data/<data_recipe>.yaml
  Artifacts root:     artifacts/sft/<run_name>/
  Wallclock:          <h:mm:ss>
  GPU-hours:          <#>
  Final step:         <N>
  Final loss:         <final_loss>
  Periodic ckpts:     step_{50,100,...}/
  Final ckpt:         artifacts/sft/<run_name>/final/
  Status:             success
tags: ["sft", <recipe_family>]
refs: [<recipe_proposal_id>, <dataset_snapshot_id>]
```

If training died (NaN, OOM, missing adapter), write a `failed_attempt`
instead of a `training_run`:

```yaml
kind: failed_attempt
title: "sft failed: <status> (<kill_reason>)"
body: |
  Recipe proposal:   <recipe_proposal_id>
  Dataset snapshot:  <dataset_snapshot_id>
  Status:            <train_failed | oom | invalid>
  Kill reason:       <NaN-loss | NaN-grad | OOM | other>
  Loss trajectory:   <last 10 logged losses, if available>
  Hypothesis:        <e.g. "LR too high — cycle8 with lr=1e-4 worked">
  Recommendation:    <e.g. "Planner should propose a revised
                       recipe with lower LR before retrying">
tags: ["divergence", "sft"]
refs: [<recipe_proposal_id>, <dataset_snapshot_id>]
```

## Hard rules (re-stated)

- One `training_run` = one recipe = one dataset_snapshot = one `refs`
  link triplet. NEVER batch.
- The training recipe under `recipes/train/<run_name>.yaml` is the
  single source of truth for this run. Do NOT edit it mid-run.
- Non-success runs are NOT `training_run` records — they are
  `failed_attempt`. Planner treats them differently.
- Training code lives in
  `agent_evolve/backends/nemo_reasoner/k8s/entries/train_unsloth.py`.
  Do NOT create or edit workspace-local runner scripts.

## Anti-patterns

- Do NOT continue past a NaN. NaN means weights are undefined —
  any further training corrupts more.
- Do NOT race two training runs on overlapping GPUs. The backend
  scheduler bin-packs across H200 nodes; confirm the intended pod is
  scheduled before logging as running.
- Do NOT modify the proposed recipe to "make it work" if it
  diverges. Surface the divergence; Planner proposes a fix.
- Do NOT attempt to scaffold a workspace-local runner. Write a
  `failed_attempt` with concrete next steps instead.
