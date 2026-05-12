---
name: dw-curate-mix
description: Dedup, format-filter, and mix a set of source JSONLs into `artifacts/data/<hash>/dataset.jsonl`, then record a `dataset_snapshot` with per-source counts, hash, and diff vs. the prior snapshot. Use after `dw-teacher-distill` / `dw-self-distill` batches have landed and the Orchestrator asks for a fresh training mix.
---

You are the Data Worker. This skill produces ONE `dataset_snapshot` per mix.

## Inputs

- `distill_batch_ids` — list of `rec_…` for the batches feeding this mix
- `sources`           — list of workspace-relative JSONL paths (one per batch, or curated subsets)
- `weights`           — parallel list of target proportions (sum need not = 1; each weight is a target fraction of that source's rows)
- `curriculum_yaml`   — optional workspace-relative path (provenance only; the mix handler doesn't execute curriculum logic)
- `slot_id`           — usually `cp_data_check`

## Steps

### 1 — Sanity-check each source

For each `src` in `sources`:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data validate --path "$src"
python -m agent_evolve.model.algorithms.nemo_mas.cli data length-dist \
  --path "$src" --field completion
```

If any source has format problems, STOP — fix the upstream distill batch first; do NOT paper over it in the mix.

### 2 — Dedup each source (optional but usually correct)

```bash
for src in $SOURCES; do
  python -m agent_evolve.model.algorithms.nemo_mas.cli data dedup \
    --path "$src" --key-field completion
done
```

Each call writes `<src>.dedup.jsonl`. Use those paths in step 4 if dedup shrank the source meaningfully.

### 3 — Format-filter each source (optional)

```bash
for src in $SOURCES; do
  python -m agent_evolve.model.algorithms.nemo_mas.cli data format-filter --path "$src"
done
```

Writes `<src>.filtered.jsonl`. Good for shaking out rows missing `\boxed{}` or missing `[verify]: PASS` — but do NOT apply blindly; check the `drops` counter in the output to see what was rejected.

### 4 — Mix

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data mix \
  $(for src in $SOURCES; do echo --source $src; done) \
  $(for w   in $WEIGHTS; do echo --weight $w;  done) \
  ${CURRICULUM_YAML:+--curriculum $CURRICULUM_YAML}
```

Output: `{"ok": true, "output": "<ws>/artifacts/data/<hash>/dataset.jsonl", "total": N, "per_source": {...}, "sha256_short": "..."}`. The handler writes the mix to the canonical path `artifacts/data/<hash>/dataset.jsonl`; do NOT override the destination.

### 5 — Final eyeball

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli data sample --path artifacts/data/<hash>/dataset.jsonl -n 10
python -m agent_evolve.model.algorithms.nemo_mas.cli data count-by --path artifacts/data/<hash>/dataset.jsonl --field category
python -m agent_evolve.model.algorithms.nemo_mas.cli data count-by --path artifacts/data/<hash>/dataset.jsonl --field source
python -m agent_evolve.model.algorithms.nemo_mas.cli data length-dist --path artifacts/data/<hash>/dataset.jsonl --field completion
```

### 6 — Diff vs prior snapshot

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind dataset_snapshot -k 1
```

If a prior snapshot exists, note in the body: `total_delta: +X rows`, `per_category_delta: {...}`.

### 7 — Build and append `dataset_snapshot`

Write `/tmp/dataset_snapshot_body.md`:

```
output: artifacts/data/<hash>/dataset.jsonl
total: <from mix>
sha256_short: <from mix>
per_source:
  <src>: <count>
per_category:
  <cat>: <count>
length (completion, approx):
  p50 p95 p99 max

curriculum: <yaml path or "none">
diff vs prior: <lines below or "no prior snapshot">
  - total: +<N> rows
  - category <cat>: <delta>
```

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role data_worker --kind dataset_snapshot \
  --title "mix sha=<sha>: n=<N>" \
  --body-file /tmp/dataset_snapshot_body.md \
  --tag "checkpoint:$SLOT_ID" \
  $(for id in $DISTILL_BATCH_IDS; do echo --ref $id; done)
```

The `checkpoint:<slot_id>` tag is REQUIRED if this snapshot serves a slot (usually `cp_data_check`). Each `distill_batch` id goes in `--ref`.

## Hard rules

- Do NOT overwrite `artifacts/data/<hash>/dataset.jsonl` without first writing this `dataset_snapshot`. Reviewer audits from the snapshot, not the raw file.
- Do NOT pick your own weights. The spec tells you which sources and what target mix; if unclear, refuse and ask the Orchestrator.
- Do NOT silently change the dedup / filter behavior. Those live in `recipes/data/<name>.yaml` and belong to Planner.
- Do NOT include sources that didn't pass `data validate`. Fix upstream first.
