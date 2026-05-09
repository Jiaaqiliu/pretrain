---
name: planner-propose-recipe
description: Read recent evidence (eval_report / data_gap / breakthrough), form a single concrete change, and write a `recipe_proposal` record with a fenced-JSON body the trace viewer can parse. Use when the Orchestrator asks for the next thing to try. Do NOT execute the proposal — trainer / data_worker do that.
---

You are the Planner. This skill produces ONE `recipe_proposal` per call. One proposal = one independent change. If you're tempted to bundle two changes, write two proposals.

## Inputs

- `motivation_kind` — `eval_report` or `data_gap` (the record kind that prompted this proposal)
- `motivation_id`   — the `rec_…` of that motivating record (the Orchestrator names it in your brief)
- `area`            — one of `{sft, rl, grpo, lora, data_mix, distill}`; shapes the tag + which slot it feeds

## Steps

### 1 — Load context

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind cv_result -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind eval_report -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind data_gap -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$MOTIVATION_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<topic of your proposal>" --kind hypothesis --top-k 8
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<topic of your proposal>" --kind recipe_proposal --top-k 5
```

If a previous proposal already captures this change, STOP — cite it in a `failed_attempt` and tell the Orchestrator.

If the delta you're reacting to is within seed variance from a prior `cv_result`, STOP — propose a CV first, not a recipe change (noise-chasing is an anti-pattern).

### 2 — Author the YAML diff

Proposals either change YAML keys (for training regimes) or commission a distill batch (for data-mix changes). Write one of:

**Option A — YAML edit.** Generate the diff against the current workspace YAML:

```bash
# Write the "after" YAML you want. Then:
python -m agent_evolve.model.algorithms.nemo_mas.cli recipe diff \
  --a train/recipes/default.yaml \
  --b /tmp/proposed_after.yaml
```

Output carries `{"ok": true, "diff": "<unified diff>", "a_lines": N, "b_lines": M}`. Copy the `diff` text into the body below.

**Option B — Distill commission.** No YAML diff. Instead, write the spec (source / category / count / prompt_field / gold_field-if-applicable) in the body.

### 3 — Build the body

Write `/tmp/recipe_proposal_body.md`. The body MUST end with a fenced JSON block that the trace viewer parses; everything above is prose for reviewers.

```
motivation: <motivation_kind> rec_… — one-line summary of what the evidence said
change: <ONE sentence — the single independent change>
rationale: <why this change, anchored in the motivation evidence>

Execution plan:
  - executor: <trainer | data_worker>
  - stage: <sft | rl | grpo | teacher_distill | solver_distill | data_merge>
  - expected signal: <what evidence would confirm the predicted effect>
  - rollback: <how to undo if the confirmation fails>

Diff (train/recipes/default.yaml):

```diff
<paste from step 2, or a short description if Option B>
```

```json
{"recipe": {"base_model": "<family + adapter shape>", "data_mix": "<one-line summary>", "training": "<steps, lr, KL, batch>", "quality_gate": "<cp_* id and state>"}}
```
```

The fenced JSON block is REQUIRED — the viewer walks `cv_result → training_run → recipe_proposal` and reads it for the leaderboard card.

### 4 — Append the record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role planner --kind recipe_proposal \
  --title "<area>: <one-phrase change>" \
  --body-file /tmp/recipe_proposal_body.md \
  --ref "$MOTIVATION_ID" \
  --tag "$AREA"
```

Required tags per area (the Quality Plan fold uses them):
- `lora` — any proposal that pins a LoRA rank or target modules → feeds `cp_03`.
- `sft` / `rl` / `grpo` — the training regime the proposal targets → feeds `cp_04` / `cp_05`.
- `data_mix` or `distill` — data-side proposals → feeds `cp_data_check` upstream work.

The CLI enforces: `recipe_proposal` MUST have at least one `--ref` to a `data_gap` or `eval_report`. The Orchestrator will also re-check this.

## Hard rules

- One proposal = one independent change. Two changes = two proposals with separate refs chains.
- Every proposal MUST cite an `eval_report` OR `data_gap` in `--ref`. The CLI enforces this.
- If the evidence is a single seed, tag your *accompanying hypothesis* (via `planner-hypothesis`) with `preliminary` and propose the smallest CV in the body's "Execution plan" before you push the change live.
- Do NOT write `eval_report` / `data_gap` / `training_run` / `distill_batch` — those are other roles.
- Do NOT modify YAML files yourself. You propose the diff; the executor applies it under the Orchestrator's supervision.
