# Skill: scaffold_rl_runner

When to use: `check_pipeline_coverage` reports the RL stage is not
covered, OR a Theorist `recipe_proposal` switches algorithm (e.g.
GSPO → DAPO) and the existing runner can't handle it.

## Inputs

- The backend (per `manifest.yaml`).
- The RL config: `rl/rollout.yaml`, `rl/reward.py`, `rl/advantage.py`.
- The starting checkpoint path (from a prior SFT `training_run`).

## Procedure

1. `read_file("rl/rollout.yaml")` — algorithm (GSPO / DAPO),
   sampling config (must match Kaggle eval contract), n_samples per
   prompt, group size for advantage normalization.
2. `read_file("rl/reward.py")` — reward function. Do NOT modify;
   you're scaffolding the runner around it. Note its signature so
   the runner imports it correctly.
3. `read_file("rl/advantage.py")` — advantage estimator. Same.
4. `read_file("backend/<backend>.yaml")` — vLLM rollout config (TP=8
   per benchmark_reference.md) + training distribution.
5. `scaffold_runner(stage="rl", template=<chosen template>)` →
   `runner/rl_runner.py`. The runner must:
   - Load the SFT checkpoint as the policy.
   - Spin up a vLLM rollout engine with the eval-contract sampling
     (temp=1.0, top_p=1.0, max_tokens=3584, TP=8).
   - For each batch of prompts: generate n_samples, score with
     `rl.reward.score`, group-normalize advantage with
     `rl.advantage.compute`, take a policy-gradient step.
   - Save checkpoint + `metric.json` (with extra fields:
     `mean_reward`, `kl_to_ref`, `n_rollouts`).
6. Smoke test: 1 step on 32 prompts. Confirm rollout-then-update
   loop closes; reward function runs without exception; advantage
   shape is correct.
7. Update runner_capability:

   ```yaml
   kind: runner_capability
   title: "Runner: RL (<algo>) scaffolded for <backend>"
   body: |
     Backend:         <backend>
     File:            runner/rl_runner.py
     Stages covered:  [rl]
     Algorithm:       <GSPO | DAPO>
     Rollout engine:  vLLM, TP=8
     Inputs read:
       - initial ckpt (CLI arg)
       - rl/rollout.yaml
       - rl/reward.py
       - rl/advantage.py
       - backend/<backend>.yaml
     Outputs:
       - checkpoint dir
       - metric.json (loss, mean_reward, kl_to_ref, n_rollouts)
     Smoke test (1 step, 32 prompts): <PASS / FAIL with notes>
   tags: ["rl", <algo>, <backend>]
   refs: []
   ```

## Hard rules

- The runner MUST use the Kaggle eval-contract sampling config for
  rollouts. Mismatched sampling between training and eval =
  invisible distribution shift = misleading rewards.
- The runner MUST import `rl.reward.score` and
  `rl.advantage.compute` by name — never inline. Theorist's
  proposals to evolve those functions land without rescaffolding.
- KL-to-reference must be logged every step. RL-only training
  diverges silently if KL grows unbounded; the engineer divergence
  rule (loss > 2x for 50 steps) doesn't catch reward hacking — KL
  growth does.

## Anti-patterns

- Do NOT scaffold an RL runner before SFT is solid. Confirm a
  cv_result for the SFT checkpoint exists.
- Do NOT bundle reward design into the runner — that's what
  `rl/reward.py` is for.
- Do NOT skip the KL log. Reward hacking is the single biggest
  failure mode of RL-only training and KL is the canary.
- Do NOT pin `n_samples` low (< 4) for the smoke test then assume
  it works at production size — large n_samples often hits memory
  walls the smoke test missed.
