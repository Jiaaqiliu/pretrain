---
name: dw-pipeline-launch
description: Submit one solver-as-hint distillation pipeline run inside a detached tmux session, watch ~60s for fast-fail (UNREACHABLE endpoint, FATAL log line, Stage 2 pass-rate halt), then drop a pending-job marker and return — the data_worker stays free for other work. The companion `dw-pipeline-collect` skill writes the final `distill_batch` once the pipeline finishes. Generation-only — does NOT mix into the training set.
---

You are the Data Worker. This skill submits ONE pipeline run and returns
fast — it does NOT wait for completion. Use `dw-pipeline-collect` later
to harvest the result.

## Inputs

- `pipeline_name`    — must match `name:` in the config's `pipeline.yaml`
                       (drives `${name}` substitution + tmux session name)
- `domain`           — must match `domain:` in the config
- `config_path`      — absolute path to a `pipeline.yaml`. Use the
                       canonical `agent_evolve/model/data/pipelines/<domain>/pipeline.yaml`
                       OR a one-off override (the override is preferred
                       on this driver host because endpoints need to be
                       `localhost:18000` / `localhost:18001`, not
                       cluster-DNS).
- `templates_path`   — absolute path to the matching `prompt_templates.yaml`
- `from_stage`       — optional; default is **auto-resume**:
                       if `<out_dir>/stage1.jsonl` exists, use 2; else 1.
- `to_stage`         — optional; default 5 (the driver caps at 5).
- `limit`            — optional row cap for smoke runs (omit / 0 for full)
- `spec_id`          — optional `rec_…` to ref. If absent, this is a
                       lead-authorized one-off and the marker stores
                       `null` for refs.

If `pipeline_name` / `config_path` / `templates_path` is missing, STOP
and write a `failed_attempt`.

## Environment

The harness sets:
- `NEMO_MAS_WORK_DIR`         — run root (markers live under
  `<work_dir>/.pending_jobs/`)
- `NEMO_MAS_WORKSPACE_ROOT`   — forked workspace
- `NEMO_MAS_MEMORY_PATH`      — `<work_dir>/memory/records.jsonl`

You also need:
- `AWS_REGION=us-west-2` for Stage 4 (Bedrock Opus). Export it before
  launch so the tmux child inherits it.

Pipeline outputs land at
`<NEMO_MAS_WORKSPACE_ROOT>/artifacts/generation/<pipeline_name>/`.
Markers live at `<NEMO_MAS_WORK_DIR>/.pending_jobs/distill-<pipeline_name>.json`.

## Steps

### 1 — Verify port-forwards

```bash
/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner/k8s/serving/portforward.sh status
```

Both `teacher_120b` and `self_30b` (if Stage 3 is enabled in the config)
must report `endpoint=ok`. If either is `unreachable`, run
`portforward.sh start` and re-check; if still failing, write a
`failed_attempt` and stop. Do NOT bypass with hand-rolled
`kubectl port-forward`.

### 2 — Dry-run validation

```bash
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
  -m agent_evolve.model.data.pipelines.legacy.shared.run_pipeline \
  --config "$CONFIG_PATH" --templates "$TEMPLATES_PATH" --dry-run
```

Confirms YAML parses, source CSV exists, every `enabled: true` stage's
endpoint is reachable. Anything `UNREACHABLE` here is a fast-fail —
write `failed_attempt` and stop.

### 3 — Reject if a marker already exists for this pipeline

```bash
SESSION="ne-distill-$PIPELINE_NAME"
MARKER="$NEMO_MAS_WORK_DIR/.pending_jobs/distill-${PIPELINE_NAME}.json"
mkdir -p "$NEMO_MAS_WORK_DIR/.pending_jobs"
[ -f "$MARKER" ] && {
  echo "marker already exists for $PIPELINE_NAME — run dw-pipeline-collect first"
  exit 2
}
# Also clear any leftover tmux session under the same name (orphan from
# a prior crash). Safe — if there's no session, has-session returns 1.
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
```

### 4 — Auto-resume detection

Read the config to find the Stage 1 out_path; if `stage1.jsonl` exists
on disk and `--from-stage` was not specified by the spec, default to
`--from-stage 2` so we don't re-walk the CSV (~12 min for bits).

```bash
OUT_DIR="$NEMO_MAS_WORKSPACE_ROOT/artifacts/generation/$PIPELINE_NAME"
mkdir -p "$OUT_DIR"
LOG_PATH="$OUT_DIR/run.log"
EXIT_PATH="$OUT_DIR/.exit_code"
rm -f "$EXIT_PATH"   # clear stale sentinel from prior runs

FROM_STAGE_DEFAULT=1
[ -s "$OUT_DIR/stage1.jsonl" ] && FROM_STAGE_DEFAULT=2
FROM_STAGE="${FROM_STAGE:-$FROM_STAGE_DEFAULT}"
TO_STAGE="${TO_STAGE:-5}"
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && [ "${LIMIT}" -gt 0 ] 2>/dev/null && LIMIT_FLAG="--limit $LIMIT"
```

### 5 — Launch in a detached tmux session

The session runs the pipeline driver, tees output to disk, and writes
an exit-code sentinel on completion. The sentinel is what
`dw-pipeline-collect` uses to distinguish clean exit from "tmux died."

```bash
PY=/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python
# `set -o pipefail` is critical — without it, `python … | tee log` returns
# tee's exit code (always 0) and the sentinel lies about success. With
# pipefail, $? reflects python's real exit so dw-pipeline-collect can
# distinguish clean finish (exit=0) from threshold-halt (exit≠0).
CMD="set -o pipefail; export AWS_REGION=us-west-2; \
$PY -m agent_evolve.model.data.pipelines.legacy.shared.run_pipeline \
  --config '$CONFIG_PATH' --templates '$TEMPLATES_PATH' \
  --from-stage $FROM_STAGE --to-stage $TO_STAGE $LIMIT_FLAG \
  2>&1 | tee '$LOG_PATH'; \
echo EXIT_CODE=\$? > '$EXIT_PATH'"

tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""
```

Confirm the session is up:

```bash
tmux has-session -t "$SESSION" && echo "session alive" || { echo "FAILED to launch tmux"; exit 1; }
```

You can attach later with `tmux attach -t ne-distill-<pipeline_name>`
to inspect live progress (use `Ctrl-B D` to detach without killing).

### 6 — 60-second fast-fail watch

Poll every 10s for 60s. Exit early on any of:

- tmux session disappeared (process crashed before any meaningful work)
- log contains `FATAL` (driver halt)
- log contains `endpoint check.*UNREACHABLE` (dry-run passed but live
  endpoint died between dry-run and launch)
- log contains `Traceback` followed within 5 lines by `Error`
- log contains a Stage 2 halt line: `pass rate .* < threshold`

```bash
FAST_FAIL=""
for i in 1 2 3 4 5 6; do
  sleep 10
  # session still alive?
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    if [ -f "$EXIT_PATH" ] && grep -q '^EXIT_CODE=0$' "$EXIT_PATH"; then
      # rare: tiny pipeline finished inside 60s. Not a fast-fail.
      echo "session ended cleanly within 60s"
      break
    fi
    FAST_FAIL="session died early"
    break
  fi
  # log scan
  if grep -qE 'FATAL|UNREACHABLE|pass rate .* < threshold' "$LOG_PATH" 2>/dev/null; then
    FAST_FAIL=$(grep -m1 -E 'FATAL|UNREACHABLE|pass rate .* < threshold' "$LOG_PATH")
    break
  fi
done
```

### 7 — Fast-fail handling

If `FAST_FAIL` is non-empty: capture log tail, kill the session, write a
`failed_attempt`, and exit. **Do NOT drop a pending marker** — the
failure is already recorded in the ledger.

```bash
if [ -n "$FAST_FAIL" ]; then
  tail -200 "$LOG_PATH" > "/tmp/dw_distill_${PIPELINE_NAME}_tail.log" 2>/dev/null || true
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  cat > "/tmp/dw_distill_${PIPELINE_NAME}_fail.md" <<EOF
attempted: dw-pipeline-launch $PIPELINE_NAME
config: $CONFIG_PATH
templates: $TEMPLATES_PATH
from_stage: $FROM_STAGE  to_stage: $TO_STAGE  limit: ${LIMIT:-0}
fast_fail_reason: $FAST_FAIL
log_tail: /tmp/dw_distill_${PIPELINE_NAME}_tail.log
EOF
  /fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
    -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
    --role data_worker --kind failed_attempt \
    --title "pipeline-launch_failed: $PIPELINE_NAME ($FAST_FAIL)" \
    --body-file "/tmp/dw_distill_${PIPELINE_NAME}_fail.md" \
    ${SPEC_ID:+--ref "$SPEC_ID"}
  exit 0
fi
```

### 8 — Drop a pending-job marker and return

Write `<NEMO_MAS_WORK_DIR>/.pending_jobs/distill-<pipeline_name>.json`:

```json
{
  "kind": "distill_batch",
  "session_name": "ne-distill-<pipeline_name>",
  "submitted_at": "<ISO-8601 UTC>",
  "refs": ["<spec_id or null>"],
  "context": {
    "pipeline_name": "<pipeline_name>",
    "domain": "<domain>",
    "config_path": "<abs config path>",
    "templates_path": "<abs templates path>",
    "from_stage": <int>,
    "to_stage": <int>,
    "limit": <int or 0>,
    "out_dir":   "<abs artifacts/generation/<name>/>",
    "log_path":  "<abs run.log>",
    "exit_path": "<abs .exit_code>"
  }
}
```

The marker is the **only handle** the rest of the system has on the
in-flight run. Do NOT delete it; `dw-pipeline-collect` moves it to
`done/` after harvesting.

Then return control. Report back to the lead in one sentence:

> launched ne-distill-bits_solver_hinted_v1, marker dropped, attach with `tmux attach -t ne-distill-bits_solver_hinted_v1`

## Anti-patterns

- ❌ Do NOT block past the 60s watch waiting for completion. The whole
  point of the launch/collect split is to keep this skill fast.
- ❌ Do NOT lower `expected_pass_rate` / `pass_threshold` to dodge a
  Stage 2 / Stage 4 halt. Those gates exist to catch broken prompts.
- ❌ Do NOT bypass the persistent port-forward by hand-rolling
  `kubectl port-forward`. The supervisor at
  `agent_evolve/backends/nemo_reasoner/k8s/serving/portforward.sh`
  manages state and self-heals.
- ❌ Do NOT edit `pipeline.yaml`, `prompt_templates.yaml`, or the
  override config. If something is wrong with them, write a
  `failed_attempt` and stop — they belong to the Planner.
- ❌ Do NOT write a `distill_batch` from this skill. That happens in
  `dw-pipeline-collect` after the run actually finishes.
