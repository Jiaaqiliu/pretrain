---
name: trainer-collect-results
description: Harvest pending k8s Jobs that were submitted async by `trainer-launch-stage` / `trainer-run-eval`. For each marker under `<work_dir>/.pending_jobs/`, check Job status; on success, write the corresponding `training_run` or `eval_report`; on failure, write a `failed_attempt`. Idempotent — safe to invoke between cycles or from a Stop hook.
---

You are the Trainer. This skill is the harvester half of the async-launch pattern. It writes the records that `trainer-launch-stage` and `trainer-run-eval` deferred.

## When to invoke

- The lead asks "any pending jobs done?" or "collect results."
- Before a planner cycle, so the planner reads the latest `eval_report` / `training_run`.
- From a `Stop` hook (optional automation).

## Environment

- `NEMO_MAS_WORK_DIR`        — run root
- `NEMO_MAS_WORKSPACE_ROOT`  — forked workspace
- `NEMO_MAS_MEMORY_PATH`     — ledger
- `KUBECTL_CTX="arn:aws:eks:ap-southeast-3:801953956576:cluster/p5-llm"`

## Steps

### 1 — Enumerate markers

```bash
PENDING_DIR="$NEMO_MAS_WORK_DIR/.pending_jobs"
DONE_DIR="$NEMO_MAS_WORK_DIR/.pending_jobs/done"
mkdir -p "$DONE_DIR"
ls -1 "$PENDING_DIR"/*.json 2>/dev/null | grep -v '/done/' || echo "no pending jobs"
```

For each marker file, read it and decide which path to take based on `kind`.

### 2 — Check Job status

```bash
JOB="<job_name from marker>"
SUCC=$(kubectl --context "$KUBECTL_CTX" get job "$JOB" \
         -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "")
FAIL=$(kubectl --context "$KUBECTL_CTX" get job "$JOB" \
         -o jsonpath='{.status.failed}' 2>/dev/null || echo "")
```

Three outcomes:

- **`SUCC=1`** → harvest success path (step 3a or 3b).
- **`FAIL≥1`** → harvest failure path (step 4).
- Neither (still Running, or controller hasn't recorded counts yet) → skip this marker; next collect pass will retry.

If the Job is gone entirely (TTL deleted) but the on-disk artifacts are present, treat as success and harvest. If artifacts are missing too, treat as failure (write `failed_attempt`, do not retry).

### 3a — Harvest a finished training_run (kind=="training_run")

Read sidecar metric:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli metric read \
  --ckpt "<context.ckpt_out>/final"
```

Write `/tmp/training_run_body_<run_name>.md`:

```
recipe_path: <context.recipe_path>
parent_recipe: <context.parent_recipe>
diff_summary: <context.diff_summary>
data_path: <context.data_path>
ckpt_out: <context.ckpt_out>
job_id: <job_name>
node_name: <marker.node_pin or "scheduler-picked">
status: success
wall_seconds: <from metric read>
gpu_hours: <if available>
final_ckpt_path: <ckpt_out>/final
primary_metric_name: <from metric read>
primary_metric_value: <from metric read>
train_metric_trajectory: <best-effort one-liner from pod log if available>
notes: <e.g. "harvested asynchronously by trainer-collect-results">

```json
{"recipe": <context.recipe_json>}
```
```

Append:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind training_run \
  --title "sft: <diff_summary> (<run_name>)" \
  --body-file "/tmp/training_run_body_<run_name>.md" \
  --ref "<context.recipe_id>" \
  --ref "<context.dataset_id>"
```

### 3b — Harvest a finished eval_report (kind=="eval_report")

Read the eval Job's outputs from `<context.out_dir>`:

- `metrics.json` → per-bucket breakdown (the JSON metrics block source — DO NOT round, DO NOT fabricate)
- `predictions.jsonl` → per-row predictions for sampling failures
- `eval.log` → vLLM stdout

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data sample \
  --path "<context.out_dir>/predictions.jsonl" -n 30 --seed 0
```

Read the wrong rows; group by failure shape (wrong answer · wrong format · overlong · no `\boxed{}`).

Write `/tmp/eval_report_body_<run_name>.md`:

```
<one-sentence score_note — first line>

- <bullet 1>
- <bullet 2>
- <bullet 3>

Cross-tab (category × bucket):
  <table>

```json
{"metrics": {"local": <from metrics.json>, "breakdown": {...}}}
```
```

Append:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind eval_report \
  --title "eval balanced_dev726 · <run_name>" \
  --body-file "/tmp/eval_report_body_<run_name>.md" \
  --ref "<context.parent_id>"
```

If the parent record's `kind` is `profile_run` and the schema rejects, fall back to refing the `recipe_proposal` directly and note the deviation in the body — don't fabricate a `training_run`.

### 4 — Harvest a failure

```bash
kubectl --context "$KUBECTL_CTX" logs "job/$JOB" --tail=200 > "/tmp/trainer/${JOB}_pod.log" 2>&1 || true
kubectl --context "$KUBECTL_CTX" describe job "$JOB" | tail -60 >> "/tmp/trainer/${JOB}_pod.log"
```

Write `/tmp/failed_attempt_body_<run_name>.md`:

```
attempted: <kind from marker>
job_name: <job>
recipe_id (or parent_id): <id>
recipe_path / ckpt_path: <from context>
node: <marker.node_pin>
failure: <best-guess from log: OOM / NaN / divergence / ImagePullBackOff / other>
pod_log: /tmp/trainer/<job>_pod.log
notes: harvested asynchronously by trainer-collect-results
```

Append:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind failed_attempt \
  --title "<kind>_failed: <short cause>" \
  --body-file "/tmp/failed_attempt_body_<run_name>.md" \
  --ref "<context.recipe_id or context.parent_id>"
```

Then delete the Job to free the queue:

```bash
kubectl --context "$KUBECTL_CTX" delete job "$JOB" --ignore-not-found
```

### 5 — Move the marker to `done/` (idempotency)

After a successful append (success or failure path), move the marker out of the pending dir:

```bash
mv "$PENDING_DIR/<JOB>.json" "$DONE_DIR/<JOB>.json"
```

This makes the skill safe to invoke repeatedly — already-harvested markers are skipped automatically.

## Report back

A small table per invocation:

| job | kind | status | record_id | note |
|---|---|---|---|---|
| ne-train-a813b66f | training_run | success | rec_… | harvested |
| ne-eval-a813b66f-step100 | eval_report | success | rec_… | harvested |
| ne-train-… | training_run | running | — | left for next cycle |
| ne-eval-… | eval_report | failed | rec_… (failed_attempt) | OOM at row 312 |

## Anti-patterns

- ❌ Do NOT delete a marker without writing the corresponding ledger record. The marker is the only handle the harness has on the job.
- ❌ Do NOT block waiting for "still Running" Jobs — leave them for the next collect pass.
- ❌ Do NOT fabricate metrics. If `metrics.json` is missing on a Job marked Succeeded, treat as failure (something off in the runner) and write a `failed_attempt`.
- ❌ Do NOT fabricate a `training_run` to host an eval that lives under a `profile_run`. Use the `profile_run` ref or fall back to the `recipe_proposal`.
