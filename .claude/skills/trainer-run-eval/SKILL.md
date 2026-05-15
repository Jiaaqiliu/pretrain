---
name: trainer-run-eval
description: Submit one eval Job to the k8s p5-llm cluster, watch ~60s for fast-fail, then drop a pending-job marker and return — the trainer stays free for other work. The companion `trainer-collect-results` skill writes the final `eval_report` once the Job completes.
---

You are the Trainer. This skill submits ONE eval Job and returns fast — it does NOT wait for completion. Use `trainer-collect-results` later to harvest the result.

## Inputs

- `training_run_id` — `rec_…` of the run being evaluated. **Optional** when evaluating a partial checkpoint from an in-flight run; in that case pass `profile_run_id` instead and write the `profile_run` first (see role contract).
- `ckpt_path`       — absolute path to the adapter dir (must contain `adapter_config.json`)
- `out_dir`         — absolute output dir for eval artifacts, e.g. `<workspace>/artifacts/eval/<short_id>_<step>/`
- `run_name`        — short id for the eval Job + result subdir, e.g. `a813b66f-step100`
- `node_name`       — optional pin (planner's wave plan reserves a node for serialized eval; for partial-ckpt eval, pin to a node that isn't running the same proposal's training pod)
- `tp`              — tensor_parallel_size (default 1; Nemotron-3-Nano-MoE is decode-bound, TP=1 is fastest)

`KUBECTL_CTX` must be set: `arn:aws:eks:ap-southeast-3:801953956576:cluster/p5-llm`.

Pending markers live at `<NEMO_MAS_WORK_DIR>/.pending_jobs/<job_name>.json`.

## Steps

### 1 — Verify the parent record (training_run or profile_run)

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$PARENT_ID"
```

Confirm `kind == "training_run"` (`status: success`) OR `kind == "profile_run"`. If neither, refuse — write a `failed_attempt`.

Confirm the checkpoint dir exists:

```bash
ls "$CKPT_PATH/adapter_config.json"
```

Missing → `failed_attempt` (the parent record claims a checkpoint that isn't on disk).

### 2 — Reject if a marker already exists

```bash
JOB="ne-eval-$RUN_NAME"
MARKER="$NEMO_MAS_WORK_DIR/.pending_jobs/${JOB}.json"
mkdir -p "$NEMO_MAS_WORK_DIR/.pending_jobs"
[ -f "$MARKER" ] && { echo "marker already exists for $JOB — collect first"; exit 2; }
```

### 3 — Submit via submit.sh (non-blocking)

```bash
BACKEND=/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner
NODE_FLAG=$([ -n "${NODE_NAME:-}" ] && echo "--node $NODE_NAME" || echo "")

mkdir -p /tmp/trainer
"$BACKEND/k8s/submit.sh" eval \
  --adapter "$CKPT_PATH" \
  --out     "$OUT_DIR" \
  --name    "$RUN_NAME" \
  --tp      "${TP:-1}" \
  $NODE_FLAG  2>&1 | tee "/tmp/trainer/${RUN_NAME}_eval_submit.log"
```

submit.sh is non-blocking. **Do NOT use `nemo-mas eval run`** — same broken bridge.

### 4 — 60-second fast-fail watch

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

If `FAST_FAIL=1`, capture log + write a `failed_attempt` (refing the parent record), delete the Job, and return.

### 5 — Drop a pending-job marker and return

Write `<NEMO_MAS_WORK_DIR>/.pending_jobs/<JOB>.json`:

```json
{
  "kind": "eval_report",
  "job_name": "ne-eval-<run_name>",
  "submitted_at": "<ISO-8601 UTC>",
  "node_pin": "<node or empty>",
  "refs": ["<parent_id>"],
  "context": {
    "parent_id": "<training_run_id or profile_run_id>",
    "parent_kind": "training_run | profile_run",
    "ckpt_path": "<abs ckpt path>",
    "out_dir": "<abs out_dir — eval results land here>",
    "run_name": "<run_name>",
    "split": "balanced_dev726",
    "tp": 1
  }
}
```

Return control. Do NOT block on Job completion.

## Concurrent evals

Submit multiple `submit.sh eval` calls in a single message via `Bash(run_in_background: true)`, then run the 60s watch in parallel for each. Each writes its own marker.

## Anti-patterns

- ❌ Do NOT call `nemo-mas eval run` — broken bridge.
- ❌ Do NOT block past the 60s watch.
- ❌ Do NOT round-up metrics. The numbers come from `metrics.json` produced by the eval Job — collect skill reads them.
- ❌ Do NOT eval a `training_run` whose status is anything other than `success`. (`profile_run` parents are fine for partial-checkpoint evals.)
- ❌ Do NOT patch backend code on eval failures — surface as `failed_attempt`.
