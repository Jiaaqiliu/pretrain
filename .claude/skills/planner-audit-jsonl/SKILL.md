---
name: planner-audit-jsonl
description: Audit a freshly produced JSONL training batch — sample rows, validate required fields, inspect field counts + length distribution — and record a `data_audit_finding`. Use when a data_worker says "batch ready at <path>" and you need to sign off that the content is sane before training.
---

You are the Planner, wearing your data-analyst hat. This skill produces ONE `data_audit_finding` per JSONL audited.

## Inputs

- `batch_id`   — short handle for the batch (e.g. `"sft_mix_v3_150k"`), used as a tag
- `jsonl_path` — workspace-relative path to the batch (e.g. `artifacts/data/<hash>/dataset.jsonl`)
- `dataset_snapshot_id` — `rec_…` the data_worker wrote; refs back to it

## Session-start context

```bash
# Don't re-audit the same batch
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "$BATCH_ID" --kind data_audit_finding --top-k 5

# Global priors
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
```

If a recent `data_audit_finding` already covers this batch, cite it and stop unless the brief says "re-audit".

## Steps

### 1 — Eyeball a reproducible sample

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data sample \
  --path "$JSONL_PATH" -n 50 --seed 0
```

Returns `{"ok": true, "rows": [...], "n": 50, "total": N, "summary": {field: {types, str_len_p50, str_len_p95}}}`. Read 5-10 rows in detail. Look for: wrong fields, truncated completions, unescaped prompt templates, answer leakage.

### 2 — Validate required fields

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data validate --path "$JSONL_PATH"
```

Expects `{prompt_rendered, completion, category, source}` to be present on every row. `problems` counter surfaces missing fields / type issues. Any non-zero count → probably fail the audit.

### 3 — Category + source counts

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data count-by --path "$JSONL_PATH" --field category
python -m agent_evolve.model.algorithms.nemo_mas.cli data count-by --path "$JSONL_PATH" --field source
```

Check for: category skew (one dominating 80%+), missing categories, unknown sources.

### 4 — Completion length distribution

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data length-dist \
  --path "$JSONL_PATH" --field completion
```

Compare against the benchmark's 7680-token cap. If `p99` is near cap, some rows will truncate during eval.

### 5 — Build the finding body

Write `/tmp/audit_body.md`:

```
batch: <batch_id>
path:  <jsonl_path>
total_rows: <from sample.total>

format validation: <summary, e.g. "all 150k rows have required fields, 0 problems">
category distribution: <top-3 categories with %>
source distribution: <summary>
length distribution (completion, tokenizer=approx):
  p50=<int> p95=<int> p99=<int> max=<int>

Sample eyeballed (5 rows read in full):
  - row 12: <one-line note>
  - row 27: <one-line note>
  - ...

verdict: <pass|fail|warn>  # your call
reasons: <what you saw that matters>
```

### 6 — Append the record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role planner --kind data_audit_finding \
  --title "audit <batch_id>: <verdict>" \
  --body-file /tmp/audit_body.md \
  --tag "$BATCH_ID" \
  --ref "$DATASET_SNAPSHOT_ID"
```

## Hard rules

- Do NOT write `data_audit_finding` without a `--ref` to the `dataset_snapshot` being audited.
- A "warn" verdict is valid evidence that the data_worker should address before training; surface the warnings in the body so the Orchestrator can route the fix.
