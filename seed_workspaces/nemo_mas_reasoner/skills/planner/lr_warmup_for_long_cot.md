# Skill: lr_warmup_for_long_cot

When to use: data is long-CoT-heavy (median completion length >
1500 tokens) AND the latest `profile_run` shows loss decreasing
slowly OR the latest `training_run` shows early-step instability
(grad-norm spikes in the first 200 steps).

This is a known-good warmup pattern, not a generic LR-tuning
playbook. It exists because long-CoT data has a different
gradient distribution from short-completion data and the default
warmup is calibrated for short data.

## Background (why this matters)

Long-CoT samples have many tokens per example, so each step's
gradient is dominated by the in-trace tokens. The model effectively
sees more "tokens per step" than the optimizer's LR was tuned for,
which over-steps in early training and produces grad-norm spikes
that destabilize the LoRA adapter.

The fix is a longer + more conservative warmup that lets the model
adjust to the larger effective gradient before LR ramps up.

## Pattern

For LoRA training on Nemotron-3-Nano with median completion ≥ 1500
tokens:

| Field | Default (short data) | Long-CoT recommendation |
|---|---|---|
| `optimizer.lr` | 2e-5 | 1e-5 |
| `optimizer.warmup_steps` | 100 | 400 |
| `optimizer.warmup_schedule` | linear | linear |
| `optimizer.lr_schedule` | cosine | cosine |
| `optimizer.gradient_clip_norm` | 1.0 | 0.5 |
| `batching.gradient_accum` | 1 | 4 (to stabilize) |

(These numbers came from the cycle-2/cycle-3 sweep — re-derive when
the model size or LoRA rank changes meaningfully.)

## When NOT to use this pattern

- Median completion < 1000 tokens: the standard warmup is fine.
- Full fine-tuning (no LoRA): different optimizer dynamics, this
  pattern is calibrated for LoRA only.
- After SFT is stable and we're switching to RL: RL has its own
  warmup (separate skill, not yet written).

## Procedure to use this skill in a `recipe_proposal`

1. Confirm trigger conditions:
   - `mem_search(<dataset_snapshot_id>, kind="dataset_snapshot")`
     — read the length p50; ≥ 1500 tokens?
   - `mem_recent(kind="profile_run", k=3)` — slow / spiking?
2. Cite this skill in your `recipe_proposal` body and include the
   diff:

   ```yaml
   diff:
     train/optimizer.yaml:
       lr: 1e-5            # was 2e-5
       warmup_steps: 400   # was 100
       gradient_clip_norm: 0.5  # was 1.0
     train/batching.yaml:
       gradient_accum: 4   # was 1
   ```

3. Predicted effect: stabilizes early-step loss; final loss may be
   marginally higher (LR ceiling lowered) but downstream eval
   typically improves +0.003 to +0.010 because the adapter
   converges to a less-overshot weight.
4. Rollback: if eval doesn't improve, revert to defaults. If eval
   regresses > 0.005, the long-CoT data may itself be the issue
   (over-length traces) — DataWorker should re-mix with stricter
   length filter rather than tweaking the recipe further.

## Anti-patterns

- Do NOT apply this pattern to short data — it under-trains.
- Do NOT combine this change with another change in the same
  proposal. (Planner hard rule: one change per proposal.)
- Do NOT update the numbers in the table above without writing a
  `breakthrough` first explaining the new evidence — this skill is
  the canonical reference, treat it like protected configuration.
