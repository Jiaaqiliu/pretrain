---
name: dw-pipeline-collect
description: Harvest pending distillation pipeline runs that were submitted async by `dw-pipeline-launch`. For each marker under `<work_dir>/.pending_jobs/distill-*.json`, check tmux session liveness + exit-code sentinel + curated JSONL on disk; on success write a `distill_batch`, on failure write a `failed_attempt`. Idempotent — safe to invoke any time, between cycles, or from a Stop hook.
---

You are the Data Worker. This skill is the harvester half of the
async-launch pattern. It writes the records that `dw-pipeline-launch`
deferred. Mirror of `trainer-collect-results` for the data pipeline.

## When to invoke

- The lead asks "any pipeline runs done?" / "collect distill results."
- Before another launch, so the in-flight one is cleared from pending.
- From a `Stop` hook (optional automation).

## Environment

- `NEMO_MAS_WORK_DIR`         — run root (markers live under
  `<work_dir>/.pending_jobs/`)
- `NEMO_MAS_WORKSPACE_ROOT`   — forked workspace
- `NEMO_MAS_MEMORY_PATH`      — ledger

## Steps

### 1 — Enumerate markers

```bash
PENDING_DIR="$NEMO_MAS_WORK_DIR/.pending_jobs"
DONE_DIR="$PENDING_DIR/done"
mkdir -p "$DONE_DIR"
ls -1 "$PENDING_DIR"/distill-*.json 2>/dev/null | grep -v '/done/' \
  || echo "no pending pipeline runs"
```

For each marker, parse the JSON and decide which path to take based on
session liveness, the `.exit_code` sentinel, and the curated JSONL.

### 2 — Per-marker status check

For each marker file:

```bash
SESSION="<session_name from marker>"
LOG_PATH="<context.log_path from marker>"
EXIT_PATH="<context.exit_path from marker>"
OUT_DIR="<context.out_dir from marker>"
PIPELINE_NAME="<context.pipeline_name from marker>"
DOMAIN="<context.domain from marker>"

# tmux session alive?
SESSION_ALIVE=0
tmux has-session -t "$SESSION" 2>/dev/null && SESSION_ALIVE=1

# exit-code sentinel present?
EXIT_CODE=""
[ -f "$EXIT_PATH" ] && EXIT_CODE=$(grep '^EXIT_CODE=' "$EXIT_PATH" | head -1 | cut -d= -f2)

# curated jsonl exists?
CURATED=$(ls "$OUT_DIR"/curated/*/${DOMAIN}_distilled.jsonl 2>/dev/null | head -1)
```

Three outcomes:

| session_alive | exit_code | curated | Outcome | Action |
|---|---|---|---|---|
| 1 | (n/a) | (any) | **Running** | skip; next collect pass will retry |
| 0 | `0` | exists | **Success** | step 3a |
| 0 | `0` | missing | **Soft fail** (driver reported success but no curated file — Stage 5 must have been skipped or empty) | step 3b |
| 0 | non-zero | (any) | **Hard fail** | step 3b |
| 0 | empty | (any) | **Crashed** (sentinel never written) | step 3b |

### 3a — Harvest a successful run

Validate the curated output and pull stage stats from `run.log`.

```bash
PY=/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python

$PY -m agent_evolve.model.algorithms.nemo_mas.cli data validate    --path "$CURATED"
$PY -m agent_evolve.model.algorithms.nemo_mas.cli data length-dist --path "$CURATED" --field completion
$PY -m agent_evolve.model.algorithms.nemo_mas.cli data sample      --path "$CURATED" -n 5

# Extract stage stats from the log (tolerate missing lines)
STAGE2_LINE=$(grep -m1 'stage_2: rows_with_hit'   "$LOG_PATH" 2>/dev/null || true)
STAGE3_LINE=$(grep -m1 'stage_3: rows_with_hit'   "$LOG_PATH" 2>/dev/null || true)
STAGE4_LINE=$(grep -m1 'stage_4: pass='           "$LOG_PATH" 2>/dev/null || true)
STAGE5_LINE=$(grep -m1 'stage_5: scanned'         "$LOG_PATH" 2>/dev/null || true)
HASH=$(echo "$CURATED" | sed -n 's|.*/curated/\([^/]*\)/.*|\1|p')
KEPT_TOTAL=$(wc -l < "$CURATED")
KEPT_TEACHER=$(grep -c '"source": "teacher"' "$CURATED" 2>/dev/null || echo 0)
KEPT_SELF=$(grep -c '"source": "self"'       "$CURATED" 2>/dev/null || echo 0)
```

Write `/tmp/distill_batch_${PIPELINE_NAME}_body.md`:

```
source: pipeline=<pipeline_name>
        config=<context.config_path basename>
        teacher_endpoint=<from config: stage_2_teacher_distill.endpoint.model>
        self_endpoint=<from config: stage_3_self_distill.endpoint.model or "disabled">
        from_stage=<context.from_stage>  to_stage=<context.to_stage>
domain: <domain>
total_kept: <KEPT_TOTAL>   (teacher=<KEPT_TEACHER>  self=<KEPT_SELF>)
hash: <HASH>
out_paths:
  raw_teacher:  artifacts/generation/<name>/stage2_teacher.jsonl
  raw_self:     artifacts/generation/<name>/stage3_self.jsonl   (if enabled)
  audit:        artifacts/generation/<name>/stage4_audit.jsonl
  curated:      <CURATED relative to NEMO_MAS_WORKSPACE_ROOT>
stage_stats:
  stage_2: <STAGE2_LINE>
  stage_3: <STAGE3_LINE or "disabled">
  stage_4: <STAGE4_LINE>
  stage_5: <STAGE5_LINE>
length (completion): p50 <...> p95 <...> p99 <...>  (from data length-dist)
sample rows (3-5):
  - <id>: <prompt[:60]>…   source=<...>  boxed=<...>
  - …
notes: harvested asynchronously by dw-pipeline-collect.
       <flag p95 > 7600 truncation risk if applicable>
```

Append:

```bash
$PY -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role data_worker --kind distill_batch \
  --title "pipeline-distill <domain>: kept=<KEPT_TOTAL> hash=<HASH>" \
  --body-file "/tmp/distill_batch_${PIPELINE_NAME}_body.md" \
  --tag "$DOMAIN" --tag pipeline-distill \
  $(for ref in $(jq -r '.refs[]?' "$MARKER" 2>/dev/null); do echo --ref "$ref"; done)
```

### 3b — Harvest a failure

Capture the log tail and write `failed_attempt`. Distinguish *crash*
(no exit sentinel; tmux died for system reason) from *halt* (exit
sentinel non-zero; pipeline driver itself returned non-zero, e.g. Stage
2 / Stage 4 threshold halt).

```bash
TAIL_LOG="/tmp/dw_distill_${PIPELINE_NAME}_tail.log"
tail -200 "$LOG_PATH" 2>/dev/null > "$TAIL_LOG" || true

# Try to identify the failure mode from the log
if grep -q 'pass rate .* < threshold' "$LOG_PATH" 2>/dev/null; then
  FAILURE_MODE="threshold_halt: $(grep -m1 'pass rate .* < threshold' "$LOG_PATH")"
elif grep -q 'FATAL'      "$LOG_PATH" 2>/dev/null; then
  FAILURE_MODE="fatal: $(grep -m1 'FATAL' "$LOG_PATH")"
elif grep -q 'UNREACHABLE' "$LOG_PATH" 2>/dev/null; then
  FAILURE_MODE="endpoint_unreachable"
elif [ -z "$EXIT_CODE" ]; then
  FAILURE_MODE="crash (no exit sentinel)"
elif [ "$EXIT_CODE" != "0" ]; then
  FAILURE_MODE="non-zero exit: $EXIT_CODE"
else
  FAILURE_MODE="success_but_no_curated_jsonl"
fi
```

Write `/tmp/failed_attempt_${PIPELINE_NAME}_body.md`:

```
attempted: dw-pipeline-launch / pipeline run
pipeline_name: <pipeline_name>
config: <context.config_path>
session: <session_name>
from_stage: <context.from_stage>  to_stage: <context.to_stage>
exit_code: <EXIT_CODE or "(none)">
failure_mode: <FAILURE_MODE>
log_tail: /tmp/dw_distill_<pipeline_name>_tail.log
notes: harvested asynchronously by dw-pipeline-collect.
```

Append:

```bash
$PY -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role data_worker --kind failed_attempt \
  --title "pipeline_failed: <pipeline_name> (<short failure_mode>)" \
  --body-file "/tmp/failed_attempt_${PIPELINE_NAME}_body.md" \
  $(for ref in $(jq -r '.refs[]?' "$MARKER" 2>/dev/null); do echo --ref "$ref"; done)
```

### 4 — Move the marker to `done/`

After a successful append (success or failure path), move the marker
out of the pending dir so subsequent collect passes skip it:

```bash
mv "$MARKER" "$DONE_DIR/$(basename "$MARKER")"
```

This makes the skill idempotent — already-harvested markers won't be
re-processed.

## Report back

A small table per invocation:

| pipeline | status | record_id | note |
|---|---|---|---|
| bits_solver_hinted_v1 | success | rec_… | kept=N teacher=Nt self=Ns |
| <other> | running | — | left for next pass |
| <other> | failed  | rec_… (failed_attempt) | <failure_mode> |

If everything was already harvested in a prior pass, just report
"no pending pipeline runs" and exit cleanly.

## Anti-patterns

- ❌ Do NOT delete a marker without writing the corresponding ledger
  record. The marker is the only handle the harness has on the run.
- ❌ Do NOT block waiting for a still-running session — leave it for
  the next collect pass.
- ❌ Do NOT fabricate stats. If `run.log` is missing the stage line
  you need, omit that field from the body and note "log truncated."
- ❌ Do NOT write a `distill_batch` for a session that exited non-zero
  even if a curated JSONL exists (it would be partial / inconsistent).
  Treat as failure.
- ❌ Do NOT kill a tmux session that's still alive. If the user wants
  to abort, that's a separate operator action.
