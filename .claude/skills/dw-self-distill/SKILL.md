---
name: dw-self-distill
description: Generate rows from the current best checkpoint (solver self-distill) and keep only rows whose completion matches the gold answer (rejection sampling) — then record a `distill_batch`. Use when the Orchestrator's spec names a checkpoint + prompts-with-gold.
---

You are the Data Worker. This skill produces ONE `distill_batch` via solver self-distill + rejection sampling.

## Inputs

- `spec_id`        — `rec_…` of the `recipe_proposal` or `data_gap` authorizing the batch
- `ckpt_path`      — workspace-relative adapter dir (the current best checkpoint)
- `prompts_path`   — workspace-relative JSONL of prompt rows WITH gold answers
- `prompt_field`   — field in prompts holding prompt text
- `gold_field`     — field in prompts holding the gold answer (for rejection)
- `out_path`       — workspace-relative JSONL for kept rows
- `category`       — batch category (from spec)

Compute always runs on k8s; no backend env var to set.

## Steps

### 1 — Context + spec check

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$SPEC_ID"
```

Confirm the spec names prompts WITH gold. Without gold, rejection sampling isn't possible — refuse and write a `failed_attempt`.

### 2 — Generate from the checkpoint

The CLI combines load + batch-generate in one call (handles can't cross Bash invocations):

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli infer generate \
  --ckpt "$CKPT_PATH" \
  --prompts "$PROMPTS_PATH" --prompt-field "$PROMPT_FIELD" \
  --out /tmp/self_distill_raw.jsonl \
  --temperature 0.7 --top-p 0.95 --max-tokens 7680
```

On `"ok": false`, write a `failed_attempt` with the reason and stop.

### 3 — Rejection sample against gold

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data filter-by-gold \
  --generations /tmp/self_distill_raw.jsonl \
  --golds "$PROMPTS_PATH" --gold-field "$GOLD_FIELD" \
  --out /tmp/self_distill_kept.jsonl
```

Returns `{"ok": true, "n_kept": X, "n_rejected": Y, "yield_": Z, "output_path": ...}`. Yield under 0.2 is a signal that the checkpoint is weak for this category — note it in the record body.

### 4 — Format validate

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data write \
  --from /tmp/self_distill_kept.jsonl --path "$OUT_PATH"
python -m agent_evolve.model.algorithms.nemo_mas.cli data validate --path "$OUT_PATH"
```

Reject rows that don't validate — fix the generator / prompt template, not the data.

### 5 — Append the `distill_batch`

Write `/tmp/distill_batch_body.md`:

```
source: solver_self_distill, ckpt=<ckpt_path>
category: <category>
count: <n_kept>
yield: <yield_ from filter output>
rejected: <n_rejected>
rejection reasons: (reported by filter-by-gold — no_box / wrong extracted value)
out_path: <absolute>
notes: <yield warning if < 0.2, kernel choice, etc.>

Sample rows (3-5):
  - <row 1 one-line>
  - <row 2 one-line>
```

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role data_worker --kind distill_batch \
  --title "self-distill <category> via <ckpt short>: n_kept=<N>" \
  --body-file /tmp/distill_batch_body.md \
  --ref "$SPEC_ID"
```

## Hard rules

- Do NOT rejection-sample against something other than gold. The whole point is to keep only the correct completions.
- Do NOT discard the reject stats. Yield < 0.2 is informative — Planner may want to pick a stronger checkpoint.
- Do NOT self-distill on prompts where the gold is missing or malformed.
- Do NOT skip `data validate`.
