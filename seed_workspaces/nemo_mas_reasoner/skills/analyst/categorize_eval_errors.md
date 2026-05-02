# Skill: categorize_eval_errors

When to use: every time `run_eval` finishes. The raw eval output is a
per-row list of (input, gold, model_output, score); this skill turns
it into structured `eval_report` + `error_pattern` records that
Theorist can act on.

## Inputs

- `training_run_id` — the run that produced the checkpoint just
  evaluated.
- The path to the per-row eval JSONL (returned by `run_eval`).
- The error taxonomy (see `eval/error_taxonomy.yaml`). The current
  buckets: `format_error`, `wrong_rule`, `partial_rule`,
  `answer_extraction_fail`, `overlong_reasoning`, `eval_runtime_error`.

## Procedure

1. Load the per-row JSONL.
2. For each row with score < 1.0, classify into a bucket:
   - **`format_error`**: model_output has no `\boxed{}` OR has
     malformed boxes. Detect with regex `\\boxed\{[^}]*\}`.
   - **`overlong_reasoning`**: model_output token-length ≥
     max_tokens (3584 by default) AND no boxed answer. (If it
     boxed in time but ran out, classify by content not length.)
   - **`answer_extraction_fail`**: a box exists, but the parser
     couldn't extract a normalized form. Look at the eval log for
     extraction errors.
   - **`partial_rule`**: extracted answer is "close" to gold (e.g.
     numeric within 5x relative tolerance, or matches first N
     chars) but not within scoring tolerance.
   - **`wrong_rule`**: extracted answer is a well-formed but
     unrelated value.
   - **`eval_runtime_error`**: tagged in the eval log.
3. Cross-tabulate by `(category, bucket)`. Note the top 3
   (category, bucket) cells by row count.
4. For each top cell, sample 5 rows; quote 2-3 of them in the
   `error_pattern` body so Theorist can see the failure shape.

## Output

Write 1 `eval_report` summarizing the run, plus 1 `error_pattern`
per top-3 (category, bucket) cell.

```yaml
# eval_report
kind: eval_report
title: "Eval <split>: score <S>, top failure (<category>, <bucket>) <N> rows"
body: |
  Training run:  <training_run_id>
  Split:         <split>, <total_rows> rows
  Primary metric (kaggle_nemo_boxed_em): <S>
  Per-category accuracy:
    bit_manipulation:    <acc>
    cryptarithm:         <acc>
    ...
  Error bucket counts:
    format_error:        <N>  (<%>)
    overlong_reasoning:  <N>
    wrong_rule:          <N>
    partial_rule:        <N>
    answer_extraction_fail: <N>
    eval_runtime_error:  <N>
  Cross-tab (category × bucket): <markdown table>
  Top-3 failure cells: [(<cat>,<bucket>,<count>), ...]
  Per-row jsonl: <path>
tags: ["eval", <split>, <recipe_family>]
refs: [<training_run_id>]

# error_pattern (one per top-3 cell)
kind: error_pattern
title: "<category> × <bucket>: <N> rows — <one-line shape>"
body: |
  Cell:        (<category>, <bucket>)
  Count:       <N> of <total_in_category> in this category
  Example rows (3):
    - row_id: <id>
      gold: <gold>
      output (last 200 chars): "<...>"
      diagnosis: <one line>
    - ...
  Plausible cause: <one line>
  Smallest experiment that would test it: <one line>
tags: ["error_pattern", <category>, <bucket>]
refs: [<eval_report_id>]
```

## Anti-patterns

- Do NOT write `recipe_proposal` based on patterns you observed —
  Theorist's job. Your `error_pattern` should INVITE a hypothesis,
  not propose a fix.
- Do NOT lump unrelated failures into one `error_pattern`. One cell
  per record.
- Do NOT skip the example rows — they are the most useful part for
  Theorist (and for future BM25 search).
- Do NOT classify by single keyword ("contains 'overflow'"). Use
  the structural signals: regex on `\boxed`, token length,
  parser-error vs parser-success.
