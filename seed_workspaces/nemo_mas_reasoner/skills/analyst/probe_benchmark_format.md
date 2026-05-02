# Skill: probe_benchmark_format

When to use: at the start of every campaign on this workspace, OR
whenever an `eval_report` shows an unexpected spike in `format_error`
that can't be explained by the recipe.

## Inputs

- The eval split name (default: `kaggle_dev_local`).
- A small set of "probe completions" — handcrafted strings designed to
  test format edge cases. (See `Output → reference list` below.)

## Procedure

1. `mem_recent(kind="benchmark_rule", k=10)` — see what's already known
   about format. Do not redo confirmed rules.
2. Construct probe completions covering each suspected edge:
   - missing `\boxed{}`
   - `\boxed{}` with leading/trailing whitespace
   - `\boxed{}` with units inside (e.g. `\boxed{42 m/s}`)
   - `\boxed{}` with comma thousands (e.g. `\boxed{1,000}`)
   - `\boxed{}` with LaTeX (e.g. `\boxed{\frac{1}{2}}`)
   - multiple `\boxed{}` in one response
   - `\boxed{}` followed by additional reasoning
   - empty box `\boxed{}`
3. For each probe, construct an eval row where the gold answer is
   known (use existing dev rows; replace the model output with your
   probe; run the host scorer locally via `run_eval` on a 1-row split
   if needed, or use the offline scorer from `eval/kaggle_eval.yaml`).
4. Record which probes scored 1.0 and which scored 0.0.
5. Compare against `mem_recent(kind="benchmark_rule")` — anything
   that contradicts a prior rule is a `breakthrough`.

## Output

Write 1 `benchmark_rule` per confirmed-or-newly-discovered format
behavior. Do NOT write a `benchmark_rule` for things you already
found in prior records.

```yaml
kind: benchmark_rule
title: "Boxing tolerance: <one-line description>"
body: |
  Probe input:    <the probe completion>
  Gold answer:    <the gold>
  Score:          1.0 | 0.0
  Implication:    <what this means for training data / inference>
  Confirmed via:  run_eval on split=<split>, row=<row_id>
tags: ["format", <category>]
refs: [<the probe-eval training_run_id if applicable, otherwise omit>]
```

If you find a contradiction with a prior `benchmark_rule`:

```yaml
kind: breakthrough
title: "Format rule changed: <what>"
body: |
  Prior belief: <quote from old benchmark_rule rec_X>
  New evidence: <your probe + result>
  Decision impact: <which existing skills / recipes may be wrong>
tags: ["format", "supersedes"]
refs: [<old_benchmark_rule_id>, <your_probe_record_id>]
```

## Reference probe list (keep in sync with eval changes)

| # | Probe completion ends with | Expected score |
|---|---|---|
| 1 | `\boxed{42}` | 1.0 if gold=42 |
| 2 | `\boxed{ 42 }` | likely 1.0 (whitespace tolerated) — verify |
| 3 | `\boxed{42 m/s}` | likely 0.0 (units break match) — verify |
| 4 | `\boxed{1,000}` if gold=1000 | likely 0.0 — verify |
| 5 | `\boxed{\frac{1}{2}}` if gold=0.5 | depends on numeric fallback — verify |
| 6 | text without any \boxed{} | 0.0 |
| 7 | `\boxed{42}\boxed{43}` | depends — last one wins? — verify |
| 8 | `\boxed{}` empty | 0.0 |

Update this table in this skill file as you confirm/refute behaviors.

## Anti-patterns

- Do NOT probe with model-generated completions — use handcrafted
  strings so causation is unambiguous.
- Do NOT skip probes you "remember" the answer to — reverify each
  cycle's start (eval can change).
- Do NOT write probe results as `data_audit_finding`. Format facts
  go to `benchmark_rule`.
