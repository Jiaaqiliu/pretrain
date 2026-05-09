---
name: trainer-launch-stage
description: Launch one training stage (sft / rl / teacher_distill / …) through the platform StageRegistry and record it in memory as a `training_run`. Use when the orchestrator hands you a recipe + dataset and asks for one full training execution. Do NOT use for multi-seed reruns (that's trainer-cross-validate).
---

You are the Trainer. This skill runs ONE full training execution end-to-end and writes a `training_run` record. It does not choose the recipe or audit the data — those are Planner / Reviewer jobs.

## Inputs you must have before running

- `recipe_id`    — `rec_…` of the `recipe_proposal` you're executing (from the task brief)
- `dataset_id`   — `rec_…` of the `dataset_snapshot` you're training on
- `recipe_path`  — workspace-relative path to the recipe YAML, e.g. `train/recipes/sft_v3.yaml`
- `data_path`    — workspace-relative path to the JSONL, e.g. `data/final/train.jsonl`
- `ckpt_out`     — workspace-relative output dir, e.g. `checkpoints/adapters/sft_v3/`
- `max_steps`    — optional cap; pass through from the task brief if set

If any of these are missing from the task brief, STOP and write a `failed_attempt` (see bottom) instead of training.

## Environment contract

Assume the harness already set:

- `NEMO_MAS_WORK_DIR`         — run root
- `NEMO_MAS_WORKSPACE_ROOT`   — the forked workspace for THIS cycle
- `NEMO_MAS_MEMORY_PATH`      — `<work_dir>/memory/records.jsonl`
- `NEMO_MAS_COMPUTE_BACKEND`  — `k8s` or `local`

If `NEMO_MAS_COMPUTE_BACKEND` is unset, refuse and write a `failed_attempt` saying "compute backend unset; Orchestrator must set NEMO_MAS_COMPUTE_BACKEND before spawning trainer."

## Steps

### 1 — Verify the referenced records exist

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$RECIPE_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$DATASET_ID"
```

Both must return `"ok": true` with the expected `kind`. If either is missing or wrong kind, write a `failed_attempt` and stop.

### 2 — Launch the stage

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli train launch \
  --recipe "$RECIPE_PATH" \
  --data   "$DATA_PATH" \
  --out    "$CKPT_OUT" \
  ${MAX_STEPS:+--max-steps $MAX_STEPS} \
  --monitor
```

This blocks until the run finishes (or the backend surfaces a failure). Capture stdout into `$RESULT_JSON`; it is a single-line JSON object with `job_id`, `status`, `ckpt_path`, `metric_name`, `metric_value`, `cost`.

If `status != "success"` → go to "Failure handling" below; do NOT write a `training_run`.

### 3 — Read the sidecar metric (defensive)

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli metric read --ckpt "$CKPT_OUT"
```

Use this as ground truth for the primary metric. The `launch` return value can carry a stale value when the stage writes metrics after vLLM eval completes.

### 4 — Build the body file

Write a plain text file to `/tmp/training_run_body.md` with:

```
recipe_path: <path>
data_path: <path>
ckpt_out: <path>
max_steps: <int-or-null>
stage: <stage name from mutation_plan or recipe>
job_id: <from step 2>
status: success
wall_seconds: <from metric read, if present>
gpu_hours: <from cost.gpu_hours if present>
final_ckpt_path: <from step 2>
primary_metric_name: <from step 3 or step 2>
primary_metric_value: <from step 3 or step 2>
train_metric_trajectory: <one-line summary, e.g. "loss 2.1 → 1.4 over 1200 steps">
notes: <any observations, e.g. kernel choice, save_every_steps used>

```json
{"recipe": {"base_model": "<family + adapter shape>", "data_mix": "<one-line breakdown>", "training": "<steps, lr, KL>", "quality_gate": "<cp_* id or 'n/a'>"}}
```
```

The fenced JSON block at the end is REQUIRED — the trace viewer parses it. Do NOT skip it.

### 5 — Append the record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind training_run \
  --title "<stage>: <short description>" \
  --body-file /tmp/training_run_body.md \
  --ref "$RECIPE_ID" \
  --ref "$DATASET_ID"
```

You MUST pass BOTH `--ref "$RECIPE_ID"` AND `--ref "$DATASET_ID"`. The schema rejects a `training_run` missing either.

On `"ok": false`, fix the body/refs (do NOT retry blindly) and try again.

## Failure handling

Any of: compute backend unset · records missing · `launch` returns non-success · OOM · diverged loss →

1. Write `/tmp/failed_attempt_body.md` with: what was attempted, the exact error from the CLI, recipe_path, data_path, job_id (if any).
2. Append as `failed_attempt`:
   ```bash
   python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
     --role trainer --kind failed_attempt \
     --title "launch_failed: <short cause>" \
     --body-file /tmp/failed_attempt_body.md \
     --ref "$RECIPE_ID"
   ```
3. Return the failed_attempt record id to the Orchestrator. Planner needs it.

## Anti-patterns

- ❌ Do NOT edit files under `runner/` or anywhere in the workspace that duplicates platform runner logic.
- ❌ Do NOT modify `data/final/train.jsonl` (DataWorker's territory).
- ❌ Do NOT modify `data/recipes/default.yaml` or `train/*.yaml` yourself — those come from `recipe_proposal`. If incomplete, refuse.
- ❌ Do NOT batch multiple recipe variants into one `training_run`. One run = one recipe = one refs pair.
- ❌ Do NOT call `pack_submission` here. That's `trainer-pack-submission`.
- ❌ Do NOT call Kaggle. That's the Reviewer's job.
