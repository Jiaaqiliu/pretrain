# Skill: when_to_skip_sft

When to use: deciding whether to run a fresh SFT from base or continue
from a prior cycle's adapter. Continuing saves ~15h of GPU per cycle but
only works if the prior checkpoint is already in a good basin for the
current data mix.

Note: RL is not part of the active recipe set — the `recipes/train/`
anchor is a pure-SFT recipe (huikang). If you want to introduce RL,
propose a new training recipe YAML with the RL hyperparameters and
flag it as out-of-scope for this skill.

## When it's safe to skip SFT

ALL of the following must be true:

1. There exists a `training_run` with a SFT checkpoint AND a
   `cv_result` showing stable score (std/mean < 0.02).
2. The current `dataset_snapshot` is ≥ 90% the same prompts as
   the snapshot the prior SFT used (per `dataset_snapshot.body
   diff` — measure overlap from the diff section).
3. Planner's proposed change is RL-only (e.g. updating
   `rl/reward.py`, `rl/advantage.py`, or `rl/rollout.yaml`).
   Pure SFT-side changes (mix weights, curriculum, distill mix)
   benefit from SFT.
4. The reference checkpoint is recent — within the last 3 cycles.
   Older checkpoints often need re-SFT because the data
   distribution has drifted.

## When SFT is mandatory

ANY of the following:

- The dataset_snapshot diff vs prior SFT shows new categories or
  > 30% new rows.
- A `breakthrough` since the last SFT run is tagged "data" or
  "format" — the model needs to learn the new convention.
- The latest `eval_report` shows `format_error` rate > 5% — RL
  cannot fix what SFT didn't establish.
- Cycle 0 / 1: cold start always SFTs.

## Cost / benefit framing

Approximate (Nemotron-3-Nano, 8×H200, LoRA r=32):

| Option | Wallclock | GPU-hours | Typical Δ on metric |
|---|---|---|---|
| SFT + RL | ~6h | 48 | baseline |
| RL only (re-use prior SFT ckpt) | ~2h | 16 | -0.0 to +0.005 vs baseline |

So skipping SFT trades ~32 GPU-hours for a near-flat metric IF
conditions above hold. It's a clear win when iterating on RL
hyperparameters.

## Procedure for using this skill in a `recipe_proposal`

1. Check conditions above. `mem_get` the relevant
   `training_run`, `cv_result`, `dataset_snapshot`,
   `breakthrough` records.
2. If safe to continue from a prior adapter, your proposal writes a
   training recipe at `recipes/train/<cycle>_<name>.yaml` that
   diffs against the anchor by adding an `initial_adapter` field:

   ```yaml
   name: cycle12_continue_from_w7_step250
   inherits: huikang
   initial_adapter: <abs path from cited training_run, e.g.
     /fsx/.../runs/<exp>/cycles/NNN1/.fork_target/.../artifacts/sft/w7/step_250/>
   batching:
     num_steps: 100    # short continuation
   ```

   Cite the `training_run`, `cv_result`, and `dataset_snapshot`
   records that justify reuse.

3. If NOT safe to skip, your `recipe_proposal` should still
   include SFT — and your body should mention you considered
   skipping and why you didn't (so future Theorists in the BM25
   record see the reasoning).

## Anti-patterns

- Do NOT skip SFT to save time when conditions don't hold. The
  saved hours are dwarfed by an unstable training_run.
- Do NOT cite a `training_run` whose `cv_result` doesn't exist
  yet. CV stability is the gate.
- Do NOT bundle "skip SFT" with another change. One change per
  proposal — a separate proposal can layer the recipe-side
  tweak.
- Do NOT update the cost numbers above casually. Re-derive from
  recent training_run wallclocks; if they've drifted, write a
  `breakthrough` first.
