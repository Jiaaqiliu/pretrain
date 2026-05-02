# Skill: compute_data_gap

When to use: after you've written an `eval_report` + a few
`error_pattern` records, AND inspecting the dataset distribution
suggests the failures correlate with under-representation. This
skill turns the diagnosis into a concrete `data_gap` that
DataEngineer can directly act on.

## Inputs

- The `eval_report` record id.
- The `error_pattern` record ids from the same eval.
- The current `dataset_snapshot` record id (what we trained on).

## Procedure

1. `mem_get(<eval_report_id>)` — get the per-category accuracy and
   the cross-tab.
2. `mem_get(<dataset_snapshot_id>)` — get per-category training
   counts and length distribution.
3. `compute_data_gap_table(<eval_report_id>)` — auto-cross-tab
   eval errors by (category, length_bucket).
4. For each (category, bucket) cell where errors are concentrated,
   compute the corresponding training-data coverage:
   - per-category training count
   - per-(category × length_bucket) training count
   - ratio of training count to error count
5. A "gap" exists when a (category, length_bucket) cell has:
   - error count > 5% of total eval rows AND
   - training count < the median training-count across cells, AND
   - the cell is over-represented in errors relative to its
     training share.
6. Ignore (category, bucket) cells where the cause is clearly NOT
   a data shortage — e.g. `format_error` is usually a recipe issue
   (boxing discipline), not a data-volume issue.
7. For each gap, propose concrete next-batch params (these go in
   the body so DataEngineer can execute directly).

## Output

Write 1 `data_gap` record per genuine gap (typically 1-3 per
eval). NOT one per (category, bucket) cell — only the actionable
ones.

```yaml
kind: data_gap
title: "Gap: <category> × <length_bucket> — <count> needed"
body: |
  Evidence:
    eval_report:        <eval_report_id>
    error_patterns:     [<error_pattern_ids>]
    dataset_snapshot:   <dataset_snapshot_id>

  Diagnosis:
    Category:           <category>
    Length bucket:      <e.g. "completion 2k-4k tokens">
    Eval errors here:   <N> rows (<%> of total)
    Training rows here: <M>  (<%> of training)
    Imbalance:          eval-error share / training-data share = <ratio>

  Proposed next batch (DataEngineer reads this):
    method:             teacher_distill | solver_self_distill
    teacher_model:      <which one>     (if teacher_distill)
    source_ckpt:        <ckpt path>     (if solver_self_distill)
    prompt_source:      <which file or prompt-set>
    target_count:       <N> rows after filtering
    sampling_config:
      max_tokens:       <X>             (must respect benchmark cap)
      temperature:      <T>
    expected_yield:     <Y>%            (justify with prior similar batches)
    estimated_cost:     <USD or token count>

  Acceptance criteria (Analyst will re-audit):
    - Schema valid: 100%
    - Length p95 in target bucket
    - Format valid (boxing): >90%
tags: ["gap", <category>, <length_bucket>]
refs: [<eval_report_id>, <dataset_snapshot_id>]
```

## Anti-patterns

- Do NOT write a `data_gap` for `format_error` failures — those are
  recipe issues (boxing discipline). Theorist proposes a recipe
  change, not more data.
- Do NOT write a `data_gap` without a concrete proposed-next-batch
  spec. "We need more data" is not actionable — DataEngineer will
  refuse it.
- Do NOT propose target counts > 2x the existing per-category
  training count without explicit justification (avoid
  category-imbalance cascades).
- Do NOT write more than 3 `data_gap`s per eval. If you have more
  candidates, list the top 3 and note the rest in the body — let
  Theorist + Orchestrator prioritize.
