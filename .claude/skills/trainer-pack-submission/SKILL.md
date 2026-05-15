---
name: trainer-pack-submission
description: Package a LoRA adapter checkpoint into `submission.zip` for Kaggle and record a `submission_artifact`. Only invoke when the orchestrator asks for a Kaggle submission. Pair with the `trainer-kaggle-submit` skill — they are separate so the budget hook can gate the upload step alone.
---

You are the Trainer. This skill packages a LoRA adapter for Kaggle submission. The `trainer-kaggle-submit` skill is what actually uploads it.

## Inputs

- `training_run_id` — the `rec_…` whose checkpoint you are packaging
- `ckpt_path`       — workspace-relative adapter directory (must contain `adapter_config.json`)
- `out_zip`         — output zip path, e.g. `submissions/sft_v3.zip`

## Steps

### 1 — Fetch the training_run body

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$TRAINING_RUN_ID"
```

Confirm `kind == "training_run"` and `status: success`. If not, refuse — packaging a failed run is a bug.

### 2 — Validate the adapter dir

```bash
ls "$NEMO_MAS_WORKSPACE_ROOT/$CKPT_PATH/adapter_config.json"
cat "$NEMO_MAS_WORKSPACE_ROOT/$CKPT_PATH/adapter_config.json" | jq .
```

Rank must be `<= 32` — Kaggle rejects higher. The pack CLI enforces this, but eyeball it first so you can write a useful `failed_attempt` on rejection.

### 3 — Pack

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli pack \
  --ckpt "$CKPT_PATH" --out "$OUT_ZIP"
```

Output:

```json
{"ok": true, "zip_path": "...", "size_bytes": 123456, "adapter_rank": 16, "target_modules": [...], "base_model_name_or_path": "...", "peft_type": "LORA", "message": "..."}
```

On `ok: false` (rank > 32, missing adapter_config, zip write failed), write a `failed_attempt` referencing the `training_run_id` and stop.

### 4 — Build the body

Write `/tmp/submission_artifact_body.md`:

```
source_training_run: <id>
source_ckpt_path: <path>
zip_path: <absolute path from pack output>
size_bytes: <from pack output>
adapter_rank: <int>
target_modules: <list>
peft_type: LORA
base_model_name_or_path: <string>
notes: <anything odd — unusual target modules, larger-than-expected zip, etc.>
```

No fenced-JSON requirement on `submission_artifact` — the kaggle-submit skill reads this record directly when it pushes the zip.

### 5 — Append the record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind submission_artifact \
  --title "submission: <recipe family> rank=<r>" \
  --body-file /tmp/submission_artifact_body.md \
  --ref "$TRAINING_RUN_ID"
```

`--ref "$TRAINING_RUN_ID"` is REQUIRED — the schema's `submission_artifact` rule rejects missing refs.

## Hard rules

- Do NOT run `kaggle` CLI from this skill — pack is upload-free. The `trainer-kaggle-submit` skill is the budget-gated path that pushes the zip.
- Do NOT rename files inside the zip — the Kaggle host expects the adapter files at the archive root.
- Do NOT pack a full-weight checkpoint as a Kaggle submission. Only LoRA adapter dirs with `adapter_config.json`.
