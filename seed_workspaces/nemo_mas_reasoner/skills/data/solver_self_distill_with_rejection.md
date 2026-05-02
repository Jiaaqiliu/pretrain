# Skill: solver_self_distill_with_rejection

When to use: there's a checkpoint that's competitive but not perfect,
AND we want to amplify the categories it's already good at while
filtering out its mistakes. Standard rejection-sampling self-distill.
Cheaper than teacher distill — useful when teacher cost is
prohibitive.

## Inputs

- `recipe_proposal` or `data_gap` specifying:
  `source_ckpt` (the checkpoint to self-distill from),
  `category`, `target_count`, `prompt_source`,
  `sampling_config` (defaults to Kaggle eval contract:
  temperature=1.0, top_p=1.0, max_tokens=3584).

## Hard rules

- Sampling config MUST default to the Kaggle eval contract (temp=0.0,
  top_p=1.0, max_tokens=7680, max_model_len=8192). Self-distill data
  should match inference-time distribution.
- BUT: because eval is deterministic (temp=0.0), the same ckpt + same
  prompt → same output. Rejection sampling at temp=0.0 produces ONE
  trace per prompt. To grow coverage you MUST use **different prompts**
  (preferred), not different seeds — varying the seed at temp=0 has
  no effect.
- If the request explicitly asks for diverse outputs from the same
  prompt set (rare; usually a Theorist decision), you may raise
  temperature to 0.5–0.7, but document the distribution-shift risk
  in the `distill_batch` body.
- MUST filter by gold answer (rejection sampling) — never accept
  rows without a gold to verify against.
- DO NOT generate from a checkpoint that hasn't passed
  `runner_capability` smoke test (`mem_search(<ckpt>, kind="training_run")`
  to confirm).

## Procedure

1. `mem_get(<request_id>)` — extract source_ckpt, category, target,
   prompt_source, sampling_config.
2. `mem_search(<source_ckpt>, kind="training_run")` — confirm the
   checkpoint's eval score so you have a yield prior. If the ckpt's
   per-category accuracy is X%, expect yield around X% (rejection
   sampling halves the gap to perfect).
3. `load_checkpoint_for_inference(<source_ckpt>)` → handle.
4. Load prompts from `prompt_source`. Compute n_prompts =
   ceil(target_count / expected_yield), capped at 2× target_count.
5. `batch_generate(handle, prompts=<prompts>,
   sampling_config=<config>)` — get completions.
6. `filter_by_gold(<generations>, <golds>)` — rejection sampling.
   Keep rows where extracted boxed answer matches gold per the
   Kaggle metric. Record reject reasons (no_box / wrong / partial /
   overlong).
7. `format_validate(<filtered>)`.
8. `write_jsonl(path="data/generated/solver/<batch_id>.jsonl",
   rows=<filtered>)`.

## Output

```yaml
kind: distill_batch
title: "Solver self-distill: <category> from <source_ckpt>, <accepted> accepted"
body: |
  Source request:    <request_id>
  Source checkpoint: <source_ckpt>
  Category:          <category>
  Prompts attempted: <n_prompts>
  Sampling:          (eval-contract: temp=0.0, top_p=1.0, max_tokens=7680)
                      OR if temperature was raised: temp=<T>, document why

  Output:
    Path:            data/generated/solver/<batch_id>.jsonl
    Total rows:      <gross>
    Accepted (gold-matched): <accepted>
    Reject breakdown:
      - no \boxed{}:     <n>
      - wrong answer:    <n>
      - partial (close but not within tolerance): <n>
      - overlong (truncated, no box): <n>
    Yield:           <accepted/n_prompts>%
    Length p50/p95:  <p50>/<p95> tokens

  Compute cost:      <GPU-hours or token count>

  Sample accepted (2): ...

  Notes:
    - Yield matched / under / over the prior <source_ckpt> per-category
       accuracy of <X%>.
    - Sampling config matches Kaggle eval contract — these traces are
       distributionally appropriate for further SFT.
tags: [<category>, "solver_distill", "self"]
refs: [<request_id>, <source_ckpt_training_run_id>]
```

## Anti-patterns

- Do NOT raise temperature "for higher yield" without explicit
  Theorist authorization — that creates a distribution shift between
  training and eval. If yield is too low at temp=0.0, the checkpoint
  isn't ready for self-distill in this category; tell Theorist via
  your final response text. Yield != trace count; with deterministic
  eval, low yield means the ckpt is wrong, not undersampled.
- Do NOT mix in teacher-distilled traces here. Solver self-distill
  is one batch, teacher distill is another batch — DataCurator
  mixes them per `data/mix.yaml`.
- Do NOT accept rows where the box matches by accident (e.g. the
  prompt has "= 42" and the model echoed it without reasoning).
  If `filter_by_gold` doesn't already detect this, add a
  reasoning-length check (reject completions < 50 tokens for
  problems whose teacher-distill counterparts are 500+ tokens).
- Do NOT self-distill from a divergent / OOM checkpoint. Confirm
  via `mem_search(<ckpt>, kind="training_run")` first.
