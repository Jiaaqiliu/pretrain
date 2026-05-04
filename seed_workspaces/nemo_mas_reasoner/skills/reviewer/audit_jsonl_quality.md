# Skill: audit_jsonl_quality

When to use: auditing a freshly produced JSONL batch (teacher_distill,
solver_distill, or human-uploaded source) before it's allowed into the
training mix.

## Inputs

- `batch_id` (the `distill_batch` record id from DataWorker) OR a
  direct path to a JSONL file.
- Optional: target schema (defaults to the Nemotron contract:
  `prompt_rendered`, `completion`, `category`, `source`).

## Procedure

1. `mem_search(<batch_id>, kind="data_audit_finding")` — has someone
   already audited this batch? If yes, do not duplicate; instead
   refresh ONLY if the batch was modified since.
2. Load the batch (via `mem_get(<batch_id>)` to find the path, or use
   the path directly).
3. `sample_jsonl(path, n=50, seed=0)` — fixed seed so two auditors
   converge.
4. Schema check: every row has `prompt_rendered`, `completion`,
   `category`, `source`. Missing field → flag.
5. `length_distribution(path, field="completion", tokenizer="nemotron-3-nano")`
   — record p50, p95, p99, max. Per `benchmark_reference.md`, anything
   over 7600 tokens contaminates the model (it learns to imitate
   traces it cannot reproduce within the eval's 3584 cap).
6. `count_by_field(path, field="category")` — ensure the batch
   actually delivers what it claimed (a "math_olympiad" batch with
   80% bit_manipulation rows is broken).
7. Format spot-check 10 random `completion`s: do they end with
   `\boxed{...}`? Does the box contain something parseable?
8. Near-dup check vs prior batches in the same domain:
   `mem_search(<domain>, kind="distill_batch", top_k=5)`, get their
   paths from bodies, run `minhash_dedup` (preview-only) to see
   overlap rate. >40% overlap → flag.
9. Yield computation: rows that pass schema + length + format gates,
   divided by total rows. Below 30% is a red flag.

## Output

Write 1 summary `data_audit_finding` always, plus 0–3 issue
`data_audit_finding`s per concrete problem found.

```yaml
# Summary record
kind: data_audit_finding
title: "Audit: <batch_id> — yield <Y>%, length p95 <T>, <verdict>"
body: |
  Path: <path>
  Total rows: N
  Schema check: <pass/fail counts>
  Length distribution (completion, tokens): p50=<>, p95=<>, p99=<>, max=<>
  Category distribution: { ... }
  Format spot-check (10 rows): <pass>/10 had valid \boxed{}
  Near-dup overlap with prior batches: <%>
  Yield (passes all gates): <Y>%
  Verdict: <one of: ship-as-is | minor-issues | quarantine | reject>
tags: [<batch_id>, "audit", <domain>]
refs: [<batch_id>]
```

If yield < 30% OR you saw a systematic issue, ALSO write a `data_gap`
record describing what concrete next-batch params would fix it
(stricter teacher prompt? lower max_tokens? different category mix?).

## Anti-patterns

- Do NOT audit the same batch twice without `mem_search` first.
- Do NOT write a finding that says "looks ok" with no stats — every
  finding must include the numbers.
- Do NOT extrapolate from <50-row samples without saying so in the
  body.
- Do NOT recommend deletion; only "quarantine" or "reject". Deletion
  is DataWorker's call.
