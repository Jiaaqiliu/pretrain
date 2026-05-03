# Skill: teacher_distill_long_cot

When to use: a `data_gap` (or a `recipe_proposal` from ResearchScientist)
asks for additional long-CoT reasoning traces in a specific category.
The standard short-completion teacher distill is a different skill
(not yet written — see `data/recipes/default.yaml` for the
`verify_pass_v1` template).

## Inputs

- The `data_gap` or `recipe_proposal` record id naming the request.
- Specifically: `category`, `target_count`, `max_tokens` (per-trace
  upper bound), expected_yield, prompt_source.

## Hard rules (re-stated from data_scientist.md)

- DO NOT pick prompts on your own. The `data_gap` body specifies
  `prompt_source`. If it doesn't, refuse and write a `failed_attempt`.
- DO NOT exceed the budget: if expected total teacher cost > 5× your
  spawn token budget, refuse.
- DO NOT generate beyond ~7000 tokens per trace. The eval cap is
  `max_tokens: 7680`, so traces near or over that will systematically
  truncate at eval and the model learns to imitate output it cannot
  reproduce. Set teacher `max_tokens` to ~6800 with a hard reject
  for anything > 7600. (See `benchmark_reference.md`.)

## Procedure

1. `mem_get(<gap_id>)` — extract `category`, `target_count`,
   `prompt_source`, `teacher_model`, `sampling_config`,
   `expected_yield`.
2. `mem_search(<category>, kind="distill_batch", top_k=5)` — see
   prior batches in this category. Note their actual yield and
   accepted-row count, average completion length.
3. Estimate prompt count needed:
   `n_prompts = ceil(target_count / expected_yield)`. Cap at
   1.5× target_count to avoid runaway.
4. Load prompts: `read_file(<prompt_source>)`. Sample `n_prompts`
   with a fixed seed for reproducibility (record the seed in body).
5. Cost preview:
   `estimated_cost ≈ n_prompts × (avg_prompt_tokens + max_tokens)`.
   If > 5× spawn budget, refuse with `failed_attempt`.
6. Build the teacher system prompt (template):

   ```
   You are solving a <category> reasoning problem. Show your work
   step by step. End with `[verify]: PASS` followed by `\boxed{...}`
   containing only the final answer (no units, no commas in
   numbers). If you cannot solve it confidently, end with
   `[verify]: FAIL` instead — do not guess.
   ```

   Adjust per category (cryptarithm needs different format from
   gravity).
7. `call_teacher_model(model=<teacher>, prompts=<prompts>,
   max_tokens=<max>, temperature=<T>, system_prompt=<above>)`.
8. Filter: keep only rows where the response ends with
   `[verify]: PASS` AND contains a parseable `\boxed{}`. Reject
   `[verify]: FAIL` and unparseable boxes (record reject counts).
9. `format_validate(<filtered jsonl>)` — schema check.
10. `write_jsonl(path="data/generated/teacher/<batch_id>.jsonl",
    rows=<filtered>)`.

## Output

```yaml
kind: distill_batch
title: "Teacher distill: <category>, <model>, <accepted_count> accepted"
body: |
  Source request:    <data_gap_id or recipe_proposal_id>
  Category:          <category>
  Teacher model:     <model>
  Prompts attempted: <n_prompts>
  Sampling:          temperature=<T>, max_tokens=<max>, seed=<seed>
  System prompt:     (paste verbatim here)

  Output:
    Path:            data/generated/teacher/<batch_id>.jsonl
    Total rows produced:  <gross>
    Accepted (PASS + parseable box): <accepted>
    Rejected breakdown:
      - verify=FAIL:     <n>
      - no \boxed{}:     <n>
      - unparseable box: <n>
      - schema fail:     <n>
    Yield:           <accepted/n_prompts>%
    Length p50/p95:  <p50>/<p95> tokens

  Cost:              ~<USD> (or token count)

  Sample rows (3):
    - prompt: "<truncated to 200 chars>"
      completion ends: "[verify]: PASS \boxed{...}"
    - ...

  Caveats / next steps:
    - <e.g. "yield was 22%, below the 30% target — AppliedScientist should
       audit before this is mixed">
tags: [<category>, "teacher_distill", <model>]
refs: [<data_gap_id_or_recipe_proposal_id>]
```

After writing, the Orchestrator (per the standard cycle) will spawn
an AppliedScientist to audit your batch.

## Anti-patterns

- Do NOT silently use a different teacher than the one specified.
  If the requested model is unavailable, refuse with `failed_attempt`.
- Do NOT skip the `[verify]: PASS` filter — the whole point of
  `verify_pass_v1` is preventing the model from learning teacher
  hallucinations.
- Do NOT keep `[verify]: FAIL` rows "for diversity" — they're
  poison.
- Do NOT over-generate "to be safe". Stop at 1.5× target_count.
