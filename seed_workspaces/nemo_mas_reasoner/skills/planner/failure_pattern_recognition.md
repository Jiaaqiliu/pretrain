# Skill: failure_pattern_recognition

When to use: multiple `error_pattern` records have accumulated
from recent eval_reports and you need to classify the dominant
failure mode before proposing a recipe change.

This is not "find the one pattern" — it's "name the family of
patterns and decide which axis to attack first".

## Procedure

1. `mem_recent(kind="error_pattern", k=20)` — scan recent
   patterns.
2. Cluster by (category, bucket):
   - same category × same bucket → same axis, just amplify the
     fix
   - different categories × same bucket → bucket is the axis
     (e.g. boxing discipline broken across categories)
   - same category × different buckets → category is the axis
     (e.g. cryptarithm has many failure modes)
3. Apply the heuristic priority order. When multiple families
   are present, attack in this order:

   a. **`format_error` across categories** — boxing discipline.
      Fix at SFT (re-distill teacher with stricter format filter)
      or filter (require_verify_pass=true). Highest priority
      because it gates everything else.

   b. **`overlong_reasoning` concentrated in 1-2 categories** —
      length budget. Fix at data-side (lower max_cot_tokens for
      those categories) or recipe (higher gradient_clip).

   c. **`wrong_rule` for a specific category** — model never
      learned the transformation. Fix at data-side
      (teacher_distill more in that category) — see
      compute_data_gap skill.

   d. **`partial_rule` for a category** — model learned rule but
      composition fails. Fix is model-capability-bound; usually
      means we need teacher distill at higher difficulty in that
      category, OR the category is fundamentally hard for the base
      model and we should accept the ceiling.

   e. **`answer_extraction_fail`** — usually an eval-side bug.
      Investigate via `probe_benchmark_format` (Reviewer), not via
      training change.

   f. **`eval_runtime_error`** — surface as `failed_attempt` and
      escalate to Trainer. Not a Planner concern.

4. Pick ONE family to attack this cycle. The heuristic order is
   not absolute — if (b) has 5x more rows than (a), prioritize
   (b). But never let a high-noise low-priority family delay (a)
   if (a) is meaningful (>3% of rows).

## Output

This is an analysis skill — your output is the pick of which
family to attack, which then feeds into one of the other Planner
skills (most often `propose_recipe_from_gap` or
`lr_warmup_for_long_cot`).

If your analysis reveals a cross-cutting failure mode that doesn't
fit the table (e.g. all categories regress simultaneously after a
specific recipe change), write a `hypothesis`:

```yaml
kind: hypothesis
title: "<recipe change> caused cross-category regression"
body: |
  Observation: After <recipe_proposal_id>, <eval_report_id>
    shows regression in <list categories> averaging <Δ>.
  Pattern: cross-category, not isolated to one bucket.
  Most likely cause: <e.g. "the LoRA rank reduction underfit">
  Smallest experiment to test:
    spawn trainer to rerun the prior recipe (rec_X) with LoRA
    rank reverted to 32 — if score recovers, hypothesis confirmed.
tags: ["regression", "cross-category"]
refs: [<eval_report_id>, <recipe_proposal_id>, <prior_eval_report_id>]
```

## Anti-patterns

- Do NOT optimize the lowest-priority family because it's
  "interesting". Boxing discipline failures cap everything else;
  fix them first.
- Do NOT cluster across cycles older than 3 — patterns from 5
  cycles ago may reflect a different recipe and be irrelevant.
- Do NOT diagnose without sample row evidence. Each
  `error_pattern` should already have 2-3 sample rows in body —
  read them, don't just trust the cell counts.
- Do NOT escalate to a `breakthrough` based on patterns alone.
  Pattern → hypothesis → experiment → confirmation → only then a
  breakthrough.
