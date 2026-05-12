# Skill: format_validate

When to use: at the very end of any data-producing flow
(teacher_distill, solver_self_distill, mix_by_curriculum) — before
writing the final JSONL. This catches silent schema breakage that
would only show up later as `eval_runtime_error` or, worse, broken
training.

## Inputs

- A JSONL path to validate.
- A schema spec (defaults to the Nemotron contract, below).

## Default schema (Nemotron training row)

```yaml
required_fields:
  prompt_rendered:  str   # the full prompt as the model will see it
  completion:       str   # what the model should learn to produce
  category:         str   # one of the categories in benchmark_reference.md
  source:           str   # provenance string for dedup_by=prompt_and_source_hash

optional_fields:
  metadata:         dict  # free-form, preserved through training
  difficulty:       str   # e.g. "easy" | "medium" | "hard"
  seed:             int   # generation seed if applicable
```

Plus `recipes/data/<name>.yaml::filters` per-row constraints (if present):

- `completion` MUST contain `[verify]: PASS` somewhere (per
  `require_verify_pass: true`).
- `completion` token-length MUST be ≤ `max_cot_tokens: 7600`.
- `completion` MUST end with a parseable `\boxed{...}`.

## Procedure

1. `read_file(<path>)` — line-count to confirm > 0 rows.
2. For each row:
   - parse JSON; if fail → flag, continue.
   - check required fields present; if missing → flag.
   - type-check (str fields are str, dict fields are dict).
   - check `[verify]: PASS` substring present in `completion`.
   - check `\boxed{...}` regex match at end of `completion`.
   - tokenize `completion` (nemotron-3-nano tokenizer); check
     length ≤ 7600.
   - check `category` is one of the known set (warn if not — may
     be a new category, but flag for human review).
3. Aggregate counts. Bucket failures by reason.
4. If failure rate > 1%, return fail.
5. If failure rate ≤ 1%, return pass + the per-row indices that
   should be dropped.

## Output

This is a procedure, not a record-producing skill. The caller
(teacher_distill_long_cot, solver_self_distill_with_rejection,
mix_by_curriculum) uses the result to decide whether to proceed.

If you discover a contract-level violation that suggests an
upstream bug (e.g., 30% of rows from a "verified" teacher batch are
missing `[verify]: PASS`), write a `failed_attempt`:

```yaml
kind: failed_attempt
title: "Format validate: <path> failed at <reason>"
body: |
  Path:        <path>
  Total rows:  <N>
  Failures:
    json parse:           <n>
    missing required:     <n> (fields: <list>)
    no [verify]: PASS:    <n>
    no parseable \boxed:  <n>
    over-length (>7600):  <n>
    unknown category:     <n>
  Failure rate:  <%>
  Hypothesis:    <e.g. "teacher temperature too high — verify
                  marker dropped">
  Recommendation: <e.g. "rerun teacher_distill with
                   temperature=0.5 instead of 0.9">
tags: ["format_validate", "upstream_bug"]
refs: [<the batch id whose data this validates>]
```

## Anti-patterns

- Do NOT silently fix rows. format_validate only reports — fixing
  is the job of the upstream skill (e.g. teacher_distill should
  reject FAIL rows itself; if they got through to here, that's a
  bug to surface, not paper over).
- Do NOT use a different tokenizer than `nemotron-3-nano` — length
  budget assumptions break.
- Do NOT bypass this check in mix_by_curriculum because "the
  batches were already audited". Audit and validate are different:
  audit samples 50 rows, validate checks all rows.
