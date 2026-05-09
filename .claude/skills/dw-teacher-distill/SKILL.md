---
name: dw-teacher-distill
description: Generate long-CoT training rows by calling a teacher model on a named prompt source, format-validate the output, and record a `distill_batch`. Use when the Orchestrator hands you a `recipe_proposal` or `data_gap` that names source / category / count. Do NOT pick prompts yourself.
---

You are the Data Worker. This skill produces ONE `distill_batch` per teacher call.

## Inputs

- `spec_id`        — `rec_…` of the `recipe_proposal` OR `data_gap` naming this batch
- `prompts_path`   — workspace-relative JSONL of prompt rows (from the spec)
- `prompt_field`   — field name in `prompts_path` holding the prompt text
- `teacher_model`  — explicit teacher id (e.g. `"claude-opus-4-7"`, `"gpt-4-1"`)
- `out_path`       — workspace-relative JSONL for distilled rows (e.g. `data/raw/distill/<batch>.jsonl`)
- `category`       — category tag for the batch (from the spec)
- `count`          — expected row count (from the spec; 0 means "all prompts")
- `slot_id`        — optional; if the batch serves `cp_data_check`, propagate as tag

Required env: `NEMO_MAS_COMPUTE_BACKEND` must be set. The `teacher call` handler delegates through `BackendBridge`.

## Steps

### 1 — Session-start context

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind data_gap -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$SPEC_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "$CATEGORY" --kind distill_batch --top-k 5
```

Read the spec body — it MUST name source / category / count. If it does not, STOP and write a `failed_attempt` saying what is missing. Do NOT pick prompts yourself.

### 2 — Budget check

From your task brief, you have a token budget `$BUDGET`. Estimate this call:

    est_tokens ≈ (n_prompts × max_tokens) + prompt_tokens

If `est_tokens > 5 × BUDGET`, refuse and write a `failed_attempt` citing the overshoot.

### 3 — Write the system prompt (optional)

If the spec includes a system prompt, put it in `/tmp/distill_system.md`. Otherwise skip `--system-prompt-file`.

### 4 — Run the teacher

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli teacher call \
  --model "$TEACHER_MODEL" \
  --prompts "$PROMPTS_PATH" --prompt-field "$PROMPT_FIELD" \
  --max-tokens 8000 --temperature 0.7 \
  ${SYS_PROMPT:+--system-prompt-file "$SYS_PROMPT"} \
  --out "$OUT_PATH"
```

Output: `{"ok": true, "output_path": "...", "n": N, ...}`.

On `"ok": false`, write a `failed_attempt` with the reason (missing backend wiring, API fail, rate limit). Do NOT retry blindly — teacher calls cost.

### 5 — Format-validate the output

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data validate --path "$OUT_PATH"
```

Required fields: `prompt_rendered`, `completion`, `category`, `source`. If missing fields show up, you need to remap: use `Edit`/`Write` on a small post-processing script, or (cleanly) surface this as a `failed_attempt` and let the Planner amend the spec.

### 6 — Sanity-check completion shape

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data length-dist \
  --path "$OUT_PATH" --field completion
```

If `p95 > 7600` tokens, rows will truncate at train / eval time — flag this in the record body.

### 7 — Build and append the `distill_batch`

Write `/tmp/distill_batch_body.md`:

```
source: teacher_model=<name>, prompts=<prompts_path>
category: <category>
count: <n rows written>
cost: <USD or token estimate>
out_path: <absolute out_path>
format_validate: <0 problems, or list>
length p50/p95/p99: <...>
sample rows (3-5):
  - <row 1 one-line>
  - <row 2 one-line>
  - <row 3 one-line>
notes: <anything unusual — truncation risk, refusal rate, etc.>
```

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role data_worker --kind distill_batch \
  --title "distill <category> via <teacher_model>: n=<count>" \
  --body-file /tmp/distill_batch_body.md \
  ${SLOT_ID:+--tag checkpoint:$SLOT_ID} \
  --ref "$SPEC_ID"
```

`--ref "$SPEC_ID"` ties the batch to the spec that authorized it.

## Hard rules

- Do NOT pick prompts. The spec MUST name them.
- Do NOT overshoot the budget. 5× is a hard ceiling (per the role contract).
- Do NOT skip `data validate`. Format-broken rows silently poison downstream training.
- Do NOT write `dataset_snapshot` here — that's for the mix step (`dw-curate-mix`).
- Do NOT write `recipe_proposal` or `data_audit_finding` — wrong role.
