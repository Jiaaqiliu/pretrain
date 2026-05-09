---
name: reviewer-kaggle-submit
description: Audit a trainer-produced `submission_artifact`, push it to Kaggle (gated by per-run budget), and record the `kaggle_submission_result`. Only invoke for `cp_submission_ready` tasks. Do NOT invoke as a general "submit to Kaggle" entry point.
---

You are the Reviewer. This skill is the ONLY path that calls the Kaggle CLI. It is budget-gated (default 1 submit per run, enforced by a Claude Code pre-tool hook).

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

If anything is off, post `reject` on `cp_submission_ready` via `reviewer-qa-verdict`, cite the artifact, and stop.

### 2 — Post `ready_to_sign` BEFORE submitting

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli checkpoints review-suggest \
  --slot-id cp_submission_ready --verdict ready_to_sign \
  --reason "adapter_rank=$R, zip=$ZIP_PATH, base=$BASE, training_run=$TR_ID" \
  --ref "$SUBMISSION_ARTIFACT_ID"
```

This is how the cockpit shows a "ready for submit" state BEFORE the Kaggle push. If the budget is exhausted, this is the last thing you do — the human submits manually outside the MAS.

### 3 — Check budget (the hook will also enforce this)

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent \
  --kind kaggle_submission_result -k 5
```

Count entries in this run. Default budget is 1 per run (`NEMO_MAS_KAGGLE_MAX_PER_RUN`). If already at the cap, STOP — post `ready_to_sign` and leave the submit to the human.

### 4 — Submit

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli kaggle submit \
  --zip "$ZIP_PATH" \
  --message "$MESSAGE"
```

Returns `{"ok": true, "submission_id": "...", "status": "pending", "competition": "...", ...}`. Public score arrives 30-60 min later.

On `"ok": false`, the CLI returns the reason (`kaggle` not on PATH, budget exhausted at handler level, upload fail). Write a `failed_attempt` with the reason and the artifact ref. Do NOT retry blindly.

### 5 — Record the result

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
  --role reviewer --kind kaggle_submission_result \
  --title "submitted cycle <N>: <short>" \
  --body-file /tmp/kaggle_submission_result_body.md \
  --ref "$SUBMISSION_ARTIFACT_ID"
```

`--ref "$SUBMISSION_ARTIFACT_ID"` is REQUIRED.

### 6 — (Auto mode only) sign `cp_submission_ready`

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli checkpoints sign \
  --slot-id cp_submission_ready --role reviewer \
  --ref "$SUBMISSION_ARTIFACT_ID"
```

Manual mode: stop here. Human sees the `ready_to_sign` from step 2 + the `kaggle_submission_result` from step 5 and clicks Sign in the viewer.

## Hard rules

- Do NOT skip step 2 (`ready_to_sign`). It is how the cockpit reflects a submission-in-flight vs. a scored one.
- Do NOT submit on a `reject` verdict. That is a bug in your flow.
- Do NOT call `kaggle submit` twice without budget. The handler + hook both enforce it; the second call will hard-fail.
- After submit, the `public_score` is not yours yet. Poll with `kaggle fetch-score` in a later cycle.
