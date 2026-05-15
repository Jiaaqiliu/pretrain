---
name: trainer-kaggle-submit
description: Audit a previously-produced `submission_artifact`, push it to Kaggle (gated by per-run budget), and record the `kaggle_submission_result`. Invoke only when the orchestrator explicitly asks to submit to Kaggle. Do NOT invoke as a general "submit to Kaggle" entry point.
---

You are the Trainer. This skill is the ONLY path that calls the Kaggle CLI. It is budget-gated (default 1 submit per run, enforced by a Claude Code pre-tool hook).

## Inputs

- `submission_artifact_id` — `rec_…` the trainer wrote
- `message`                — Kaggle submission description (e.g. `"cycle 7: sft_v3+grpo_v1"`)

## Steps

### 1 — Audit the artifact

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$SUBMISSION_ARTIFACT_ID"
```

Confirm:
- `kind == "submission_artifact"`
- `adapter_rank <= 32` (Kaggle refuses higher; trainer's `pack` already validates this, but double-check)
- `zip_path` is set and points at a real file (you can sanity-check with `ls -la`)
- `base_model_name_or_path` matches the challenge base model
- At least one `--ref` in the record points to a `training_run` (schema requires this)

If anything is off, write a `failed_attempt` citing the artifact and stop.

### 2 — Check budget (the hook will also enforce this)

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent \
  --kind kaggle_submission_result -k 5
```

Count entries in this run. Default budget is 1 per run (`NEMO_MAS_KAGGLE_MAX_PER_RUN`). If already at the cap, STOP — leave the submit to the human.

### 3 — Submit

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli kaggle submit \
  --zip "$ZIP_PATH" \
  --message "$MESSAGE"
```

Returns `{"ok": true, "submission_id": "...", "status": "pending", "competition": "...", ...}`. Public score arrives 30-60 min later.

On `"ok": false`, the CLI returns the reason (`kaggle` not on PATH, budget exhausted at handler level, upload fail). Write a `failed_attempt` with the reason and the artifact ref. Do NOT retry blindly.

### 4 — Record the result

Write `/tmp/kaggle_submission_result_body.md`:

```
submission_id: <id from step 4>
status: pending
competition: nvidia-nemotron-model-reasoning-challenge
zip_path: <from artifact body>
submitted_at: <ISO timestamp>
message: <the --message you sent>

Scoring window: public LB score arrives ~30-60 min after submit. Call
`kaggle fetch-score` in a later cycle and append the public/private
score to this record (or update it).
```

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind kaggle_submission_result \
  --title "submitted cycle <N>: <short>" \
  --body-file /tmp/kaggle_submission_result_body.md \
  --ref "$SUBMISSION_ARTIFACT_ID"
```

`--ref "$SUBMISSION_ARTIFACT_ID"` is REQUIRED.

## Hard rules

- Do NOT submit an artifact that failed your audit in step 1.
- Do NOT call `kaggle submit` twice without budget. The handler + hook both enforce it; the second call will hard-fail.
- After submit, the `public_score` is not yours yet. Poll with `kaggle fetch-score` in a later cycle.
