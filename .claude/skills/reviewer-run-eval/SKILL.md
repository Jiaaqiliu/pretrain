---
name: reviewer-run-eval
description: Run a full eval pass on a trained checkpoint via the platform eval stage and record an `eval_report`. Use when a `training_run` has landed and the Orchestrator asks for independent evaluation before `cp_eval_sanity`.
---

You are the Reviewer. This skill produces ONE `eval_report` per checkpoint evaluated.

## Inputs

- `training_run_id` — `rec_…` of the run being evaluated
- `ckpt_path`       — workspace-relative adapter dir (from the `training_run` body)
- `split`           — eval split name (e.g. `"dev"`, `"hard"`, `"local"`)
- `limit`           — optional row cap for fast iteration
- `slot_id`         — usually `cp_eval_sanity`

Compute backend (`NEMO_MAS_COMPUTE_BACKEND`) must be set — `eval run` delegates to `BackendBridge` just like training does.

## Steps

### 1 — Fetch the training_run

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$TRAINING_RUN_ID"
```

Confirm `kind == "training_run"` and `status: success`. If not, refuse — don't eval a failed or mis-kind record; write a `failed_attempt` instead.

### 2 — (Optional) cross-check the job actually ran

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli k8s status --name-contains aev-
```

Look for a completed job whose `result_summary` matches the claimed checkpoint. If the k8s record shows `suspicious: true` (e.g. `opt_steps=0`, `wall_seconds<10`), the `training_run` is a ghost — post `reject` on `cp_training_health` and abort.

### 3 — Run the eval

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli eval run \
  --ckpt "$CKPT_PATH" --split "$SPLIT" \
  ${LIMIT:+--limit $LIMIT}
```

Blocks until the vLLM eval finishes. Returns `{"ok": true, "eval_output_path": "...jsonl", "split": "...", "ckpt_path": "...", ...}`. The JSONL at `eval_output_path` is per-row.

If `"ok": false`, write a `failed_attempt` with the reason and stop.

### 4 — Inspect failures

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data sample \
  --path "$EVAL_OUTPUT_PATH" -n 30 --seed 0
```

Read the wrong ones. Group by the failure shape (wrong answer · wrong format · overlong · no `\boxed{}`).

### 5 — Build the body — follow the viewer contract exactly

Write `/tmp/eval_report_body.md`:

```
<one-sentence score_note — plain language, first line>

- <bullet finding 1>
- <bullet finding 2>
- <bullet finding 3>

Cross-tab (category × bucket):
  <table summarizing where wins/losses concentrated>

```json
{"metrics": {"kaggle": 0.681, "local": 0.667, "hard": 0.572, "delta": "+0.041", "breakdown": {"equations": 0.71, "ciphers": 0.62, "units": 0.69, "symbols": 0.66}}}
```
```

The first non-empty line is the `score_note` the leaderboard shows. The 3-5 bullets render on the run-detail card. The fenced JSON is the metrics source for the cockpit — do NOT fabricate numbers.

### 6 — Append the record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role reviewer --kind eval_report \
  --title "eval <split> · <ckpt short name>" \
  --body-file /tmp/eval_report_body.md \
  --tag "checkpoint:$SLOT_ID" \
  --ref "$TRAINING_RUN_ID"
```

`--ref "$TRAINING_RUN_ID"` is REQUIRED — `eval_report` schema enforces it.

## Hard rules

- Do NOT eval your own `training_run`. Reviewer eval over trainer work only.
- Do NOT sign `cp_eval_sanity` in the same cycle — post `review-suggest` via `reviewer-qa-verdict`.
- Do NOT round-up metrics. `breakdown` values are the actual per-bucket numbers.
