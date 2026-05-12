# Skill: mix_by_curriculum

When to use: after one or more `distill_batch`es have been audited
and accepted, OR when the data recipe changed (per a Planner
`recipe_proposal`). This produces the materialized
`artifacts/data/<hash>/dataset.jsonl` that Trainer trains on.

## Inputs

- List of accepted `distill_batch` record ids (each has a path in
  its body).
- The target data recipe at `recipes/data/<name>.yaml` (declares
  sources, filters, dedup, and mix weights as one YAML document).
- The previous `dataset_snapshot` record id, if any (for diff
  reporting).

## Procedure

1. `read_file("recipes/data/<name>.yaml")` — single source of truth
   for this build: which sources, weights, filters, dedup key.
2. For each batch referenced by the recipe: `mem_get(<batch_id>)`
   to find the path. Confirm the batch was audited and verdict is
   not "reject"/"quarantine" (`mem_search(<batch_id>,
   kind="data_audit_finding")`). If any batch lacks an audit, refuse
   and ask Orchestrator (in your final response text) to spawn a
   Reviewer first.
3. Apply per-source filters from `recipes/data/<name>.yaml::filters`
   (require_verify_pass, max_cot_tokens, schema). Record dropped
   counts.
4. Optionally `minhash_dedup` across sources — use threshold from
   `recipes/data/<name>.yaml::filters::dedup_by`.
5. Mix per the recipe's weights; produce the final JSONL.
   Curriculum policies (shuffle + length-bucket-sort, or staged
   easy-first-then-hard) come from the recipe's `curriculum` block.
6. Compute the snapshot stats:
   - total rows
   - per-source counts (after filter / dedup / sampling)
   - per-category distribution
   - length distribution (p50, p95, p99, max)
   - SHA-256 hash of the materialized file
7. Write to `artifacts/data/<hash>/dataset.jsonl` where `<hash>` is
   the SHA-256 of the file. Sibling `.provenance.json` records the
   input batch ids + recipe path.

## Output

```yaml
kind: dataset_snapshot
title: "Snapshot v<N>: <total> rows, <#categories> cats, hash <short>"
body: |
  Recipe:            recipes/data/<name>.yaml
  Output path:       artifacts/data/<full-hash>/dataset.jsonl
  SHA-256:           <hash>
  Total rows:        <N>

  Sources used (post-filter, post-dedup):
    - <batch_id_1>   path=<path>  rows=<n>  weight=<w>
    - <batch_id_2>   path=<path>  rows=<n>  weight=<w>
    - ...

  Filter drop reasons:
    require_verify_pass=false: <n>
    max_cot_tokens (>7600):    <n>
    schema invalid:            <n>
    near-dup:                  <n>

  Per-category distribution:
    bit_manipulation:    <n>  (<%>)
    cryptarithm:         <n>  (<%>)
    ...

  Length distribution (completion, tokens):
    p50: <>, p95: <>, p99: <>, max: <>

  Diff vs previous snapshot <prev_id>:
    rows:              <prev> → <new>  (Δ <±>)
    per-category Δ:    {<cat>: ±N, ...}
    new batches added: [<batch_ids>]
    batches dropped:   [<batch_ids>]  (with reason: e.g. weight=0)

  Recipe hash:         <short hash of recipes/data/<name>.yaml>
tags: ["snapshot"]
refs: [<all batch ids included>, <previous_snapshot_id>]
```

## Anti-patterns

- Do NOT include batches that haven't been audited.
- Do NOT silently mutate `recipes/data/<name>.yaml` — that is an
  input from Planner's `recipe_proposal`. Read-only here.
- Do NOT skip the diff-vs-previous-snapshot section — that's how
  Planner sees what changed without re-reading the whole snapshot.
- Do NOT emit a snapshot with < 95% of the previous row count UNLESS
  Planner's proposal explicitly authorized the shrink. Otherwise
  refuse with `failed_attempt` and let Orchestrator decide.
