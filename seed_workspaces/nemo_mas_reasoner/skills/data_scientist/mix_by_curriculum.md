# Skill: mix_by_curriculum

When to use: after one or more `distill_batch`es have been audited
and accepted, OR when `data/mix.yaml` weights changed (per a
ResearchScientist `recipe_proposal`). This produces the final
`data/final/train.jsonl` that MachineLearningEngineer trains on.

## Inputs

- List of accepted `distill_batch` record ids (each has a path in
  its body).
- The current `data/mix.yaml` (per-source weights) and
  `data/curriculum.yaml` (ordering / staging policy).
- The previous `dataset_snapshot` record id, if any (for diff
  reporting).

## Procedure

1. `read_file("data/mix.yaml")` and `read_file("data/curriculum.yaml")`.
2. For each batch in the input list: `mem_get(<batch_id>)` to find
   the path. Confirm the batch was audited and verdict is not
   "reject"/"quarantine" (`mem_search(<batch_id>,
   kind="data_audit_finding")`). If any batch lacks an audit, refuse
   and ask Orchestrator (in your final response text) to spawn an
   AppliedScientist first.
3. Apply per-source filters: `apply_format_filter(<path>)` per
   `data/recipes/default.yaml` (require_verify_pass, max_cot_tokens,
   etc.). Record dropped counts.
4. Optionally `minhash_dedup` across sources (see that skill) — use
   threshold from `data/recipes/default.yaml::dedup_by`.
5. `mix_sources(sources=<paths>, weights=<from mix.yaml>,
   curriculum_yaml="data/curriculum.yaml")` — produces the
   final JSONL. Curriculum policies typically: shuffle + length-
   bucket-sort, or staged (easy first then hard).
6. Compute the snapshot stats:
   - total rows
   - per-source counts (after filter / dedup / sampling)
   - per-category distribution
   - length distribution (p50, p95, p99, max)
   - SHA-256 hash of the file
7. `write_jsonl(path="data/final/train.jsonl", rows=<final>)`.
   (The same path is overwritten each time; the snapshot record is
   what gives history.)

## Output

```yaml
kind: dataset_snapshot
title: "Snapshot v<N>: <total> rows, <#categories> cats, hash <short>"
body: |
  Output path:       data/final/train.jsonl
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

  Mix.yaml + curriculum.yaml versions used:
    mix.yaml hash:        <short hash>
    curriculum.yaml hash: <short hash>
tags: ["snapshot"]
refs: [<all batch ids included>, <previous_snapshot_id>]
```

## Anti-patterns

- Do NOT include batches that haven't been audited.
- Do NOT silently mutate `data/mix.yaml` or `data/curriculum.yaml`
  — those are inputs from ResearchScientist's `recipe_proposal`. Read-only
  here.
- Do NOT skip the diff-vs-previous-snapshot section — that's how
  ResearchScientist sees what changed without re-reading the whole snapshot.
- Do NOT overwrite `data/final/train.jsonl` if the new snapshot
  shows < 95% of the previous row count UNLESS ResearchScientist's proposal
  explicitly authorized the shrink. Otherwise refuse with
  `failed_attempt` and let Orchestrator decide.
