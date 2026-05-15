---
name: trainer-launch-stage
description: Submit one SFT-LoRA training Job to the k8s p5-llm cluster, watch it for ~60s to catch fast-fail (ImagePullBackOff, immediate Failed), then drop a pending-job marker and return — the trainer stays free for other work. The companion `trainer-collect-results` skill writes the final `training_run` once the Job completes. Scope: SFT with LoRA only.
---

You are the Trainer. This skill submits ONE training Job and returns fast — it does NOT wait for completion. Use `trainer-collect-results` later to harvest the result.

## Inputs

- `recipe_id`     — `rec_…` of the `recipe_proposal` you're executing
- `dataset_id`    — `rec_…` of the `dataset_snapshot` you're training on
- `data_path`     — absolute path to the JSONL (read from the snapshot body's `path:`)
- `data_recipe`   — absolute path to `recipes/data/<name>.yaml` (informational)
- `ckpt_out`      — absolute output dir, e.g. `<workspace>/artifacts/sft/<short_id>/`
- `node_name`     — optional k8s node pin (planner names this in the wave plan)
- `run_name`      — short id for the Job + wandb name, e.g. `a813b66f`

If anything is missing, STOP and write a `failed_attempt` instead of launching.

## Environment

Harness sets:
- `NEMO_MAS_WORK_DIR`        — run root
- `NEMO_MAS_WORKSPACE_ROOT`  — forked seed workspace
- `NEMO_MAS_MEMORY_PATH`     — `<work_dir>/memory/records.jsonl`

You also need:
- `WANDB_API_KEY` — required for `--wandb`; if unset, pass `--no-wandb`
- `KUBECTL_CTX="arn:aws:eks:ap-southeast-3:801953956576:cluster/p5-llm"`

Pending markers live at `<NEMO_MAS_WORK_DIR>/.pending_jobs/<job_name>.json`.

## Steps

### 1 — Verify referenced records

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$RECIPE_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$DATASET_ID"
```

Both must return `"ok": true` with the right `kind`. If not, write a `failed_attempt` and stop.

### 2 — Apply the planner's diff to a sibling child YAML

The planner writes diffs; the executor writes files. Do not mutate `recipes/train/default.yaml`. Write `recipes/train/default_<short_id>.yaml`:

```bash
SHORT_ID=$(echo "$RECIPE_ID" | sed 's/^rec_//' | cut -c1-8)
DEFAULT_YAML="$NEMO_MAS_WORKSPACE_ROOT/recipes/train/default.yaml"
NEW_YAML="$NEMO_MAS_WORKSPACE_ROOT/recipes/train/default_${SHORT_ID}.yaml"
cp "$DEFAULT_YAML" "$NEW_YAML"
# apply the proposal's one-knob diff to $NEW_YAML using Edit
```

Sanity-check (must show exactly the one knob):

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli recipe diff --a "$DEFAULT_YAML" --b "$NEW_YAML"
```

### 3 — Reject if a marker already exists for this job

```bash
JOB="ne-train-$RUN_NAME"
MARKER="$NEMO_MAS_WORK_DIR/.pending_jobs/${JOB}.json"
mkdir -p "$NEMO_MAS_WORK_DIR/.pending_jobs"
[ -f "$MARKER" ] && { echo "marker already exists for $JOB — collect first"; exit 2; }
```

### 4 — Submit via submit.sh (non-blocking)

```bash
BACKEND=/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner
WANDB_FLAG=$([ -n "${WANDB_API_KEY:-}" ] && echo --wandb || echo --no-wandb)
NODE_FLAG=$([ -n "${NODE_NAME:-}" ] && echo "--node $NODE_NAME" || echo "")

mkdir -p /tmp/trainer
"$BACKEND/k8s/submit.sh" train \
  --train-recipe "$NEW_YAML" \
  --data-recipe  "$NEMO_MAS_WORKSPACE_ROOT/recipes/data/default_data.yaml" \
  --out          "$CKPT_OUT" \
  --name         "$RUN_NAME" \
  $NODE_FLAG \
  $WANDB_FLAG  2>&1 | tee "/tmp/trainer/${RUN_NAME}_submit.log"
```

`submit.sh train` is non-blocking — it `kubectl apply`s the Job manifest and returns immediately.

**Do NOT use `nemo-mas train launch`** — it goes through the legacy bridge wired to the old `train/pipeline.yaml` shape and returns `train_failed` instantly on `train-1.1` forks.

### 5 — 60-second fast-fail watch

For 60s, poll every 10s and exit early on fast-fail signatures. Anything past 60s is "submitted, training" — drop a marker and return.

```bash
for i in 1 2 3 4 5 6; do
  sleep 10
  PHASE=$(kubectl --context "$KUBECTL_CTX" get pods -l job-name="$JOB" \
            -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")
  REASON=$(kubectl --context "$KUBECTL_CTX" get pods -l job-name="$JOB" \
            -o jsonpath='{.items[0].status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || echo "")
  RESTARTS=$(kubectl --context "$KUBECTL_CTX" get pods -l job-name="$JOB" \
            -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}' 2>/dev/null || echo "0")
  case "$REASON" in
    ImagePullBackOff|ErrImagePull|CrashLoopBackOff|CreateContainerConfigError)
      echo "FAST-FAIL: $REASON"; FAST_FAIL=1; break;;
  esac
  if [ "$PHASE" = "Failed" ] || [ "${RESTARTS:-0}" -gt 0 ]; then
    echo "FAST-FAIL: phase=$PHASE restarts=$RESTARTS"; FAST_FAIL=1; break
  fi
done
```

If `FAST_FAIL=1`, capture the pod log and write a `failed_attempt` immediately:

```bash
kubectl --context "$KUBECTL_CTX" logs "job/$JOB" --tail=200 > "/tmp/trainer/${RUN_NAME}_pod.log" 2>&1 || true
kubectl --context "$KUBECTL_CTX" describe job "$JOB" | tail -60 >> "/tmp/trainer/${RUN_NAME}_pod.log"
kubectl --context "$KUBECTL_CTX" delete job "$JOB" --ignore-not-found
# write /tmp/failed_attempt_body_${RUN_NAME}.md and mem append (see Failure handling below)
exit 0
```

### 6 — Drop a pending-job marker and return

Write `<NEMO_MAS_WORK_DIR>/.pending_jobs/<JOB>.json`:

```json
{
  "kind": "training_run",
  "job_name": "ne-train-<run_name>",
  "submitted_at": "<ISO-8601 UTC>",
  "node_pin": "<node or empty>",
  "refs": ["<recipe_id>", "<dataset_id>"],
  "context": {
    "recipe_id": "<recipe_id>",
    "dataset_id": "<dataset_id>",
    "recipe_path": "recipes/train/default_<short_id>.yaml",
    "parent_recipe": "recipes/train/default.yaml",
    "diff_summary": "<one-knob change, e.g. 'batching.num_steps 460 -> 350'>",
    "data_path": "<abs JSONL path>",
    "ckpt_out": "<abs ckpt_out>",
    "run_name": "<run_name>",
    "recipe_json": {"base_model": "<...>", "data_mix": "<...>", "training": "<...>"}
  }
}
```

Then return control. Do NOT block on Job completion.

## Failure handling (fast-fail only)

Within the 60s watch:

```bash
cat > "/tmp/failed_attempt_body_${RUN_NAME}.md" <<EOF
attempted: SFT launch via submit.sh train
recipe_id: $RECIPE_ID
dataset_id: $DATASET_ID
recipe_path: $NEW_YAML
job_name: $JOB
node: ${NODE_NAME:-<scheduler>}
fast_fail_reason: <reason>
pod_log: /tmp/trainer/${RUN_NAME}_pod.log
EOF

python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind failed_attempt \
  --title "launch_failed: <short cause>" \
  --body-file "/tmp/failed_attempt_body_${RUN_NAME}.md" \
  --ref "$RECIPE_ID"
```

Failures past the 60s window (NaN at step 300, OOM at step 500) are caught by `trainer-collect-results` later — that's the cost of going async; the planner sees the failure one cycle later.

## Running multiple proposals concurrently

Run step 4 (submit.sh) for all proposals in a single message via `Bash(run_in_background: true)`. Then run the 60s watch in parallel for each. Each writes its own marker independently. With this skill the trainer can launch a 3-wide slate in ~70s wall-clock and stay free for the next task.

## Anti-patterns

- ❌ Do NOT call `nemo-mas train launch` — broken bridge.
- ❌ Do NOT mutate `recipes/train/default.yaml`. Always write a sibling.
- ❌ Do NOT edit `agent_evolve/backends/nemo_reasoner/k8s/` — surface backend bugs as `failed_attempt`s.
- ❌ Do NOT block past the 60s watch waiting for completion. Drop the marker and let `trainer-collect-results` do it.
- ❌ Do NOT call `pack_submission` or Kaggle from this skill.
