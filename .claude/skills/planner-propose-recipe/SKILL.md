---
name: planner-propose-recipe
description: Read recent evidence (eval_report / data_gap / breakthrough), form a single concrete change, and write a `recipe_proposal` record with a fenced-JSON body the trace viewer can parse. Use when the Orchestrator asks for the next thing to try. Do NOT execute the proposal — trainer / data_worker do that.
---

You are the Planner. This skill produces ONE `recipe_proposal` per call. One proposal = one independent change. If you're tempted to bundle two changes, write two proposals.

## Inputs

- `motivation_kind` — `eval_report` or `data_gap` (the record kind that prompted this proposal)
- `motivation_id`   — the `rec_…` of that motivating record (the Orchestrator names it in your brief)
- `area`            — one of `{sft, data_mix, distill}`; shapes the tag on the record. Scope is SFT-LoRA only: no `rl`/`grpo`/`dpo`, and no full-finetune (LoRA shape lives in the frozen anchor — not planner-tunable).

## Steps

### 1 — Load context

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind eval_report -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind data_gap -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$MOTIVATION_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<topic of your proposal>" --kind recipe_proposal --top-k 8
```

If a previous proposal already captures this change, STOP — cite it in a `failed_attempt` and tell the Orchestrator.

If the delta you're reacting to looks like dev-set noise (small absolute change, single eval), STOP — request another `eval_report` (e.g. on a later checkpoint) before proposing a recipe change. We don't run multi-seed CVs, so noise-chasing is the dominant failure mode.

### 2 — Author the YAML diff

Proposals either change YAML keys (for training regimes) or commission a distill batch (for data-mix changes). Write one of:

**Option A — YAML edit.** Generate the diff against the current workspace YAML.

Diffs MUST target the tunable child recipe (e.g. `recipes/train/default.yaml`), never the frozen anchor (`recipes/train/default_base.yaml`). The child exposes only the evolvable surface — `optimizer.{lr,weight_decay}`, `scheduler.*`, `batching.*` — and inherits every other field from the anchor at load time. If a proposal needs a change to adapter shape, target_modules, dtype discipline, model path, or any structural trick, STOP — that's a structural change requiring a separate fork of the base file, not a planner proposal.

```bash
# Write the "after" YAML you want (child only). Then:
python -m agent_evolve.model.algorithms.nemo_mas.cli recipe diff \
  --a recipes/train/<name>.yaml \
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
  - stage: <sft | teacher_distill | solver_distill | data_merge>
  - expected signal: <what evidence would confirm the predicted effect>
  - rollback: <how to undo if the confirmation fails>

Diff (recipes/train/<name>.yaml):

```diff
<paste from step 2, or a short description if Option B>
```

```json
{"recipe": {"base_model": "<family + adapter shape>", "data_mix": "<one-line summary>", "training": "<steps, lr, KL, batch>"}}
```
```

The fenced JSON block is REQUIRED — the viewer walks `eval_report → training_run → recipe_proposal` and reads it for the leaderboard card.

### 4 — Append the record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role planner --kind recipe_proposal \
  --title "<area>: <one-phrase change>" \
  --body-file /tmp/recipe_proposal_body.md \
  --ref "$MOTIVATION_ID" \
  --tag "$AREA"
```

Conventional tags per area:
- `sft` — SFT training-regime proposals (the only training regime in scope).
- `data_mix` or `distill` — data-side proposals.

The CLI enforces: `recipe_proposal` MUST have at least one `--ref` to a `data_gap` or `eval_report`. The Orchestrator will also re-check this.

## Hard rules

- One proposal = one independent change. Two changes = two proposals with separate refs chains.
- Every proposal MUST cite an `eval_report` OR `data_gap` in `--ref`. The CLI enforces this.
- If the evidence is a single eval, name in the "expected signal" line what *additional* dev-set evidence would confirm the effect (a later checkpoint, a held-out split). Tag the record `preliminary` so the executor treats the first run as calibration, not ship-ready. We do not run multi-seed CVs.
- Do NOT write `eval_report` / `data_gap` / `training_run` / `distill_batch` — those are other roles.
- Do NOT modify YAML files yourself. You propose the diff; the executor applies it under the Orchestrator's supervision.

## When the lead asks for a SLATE

Each proposal is still its own `recipe_proposal` record (one record = one knob). The "slate" is the set of records the planner returns together so the lead can pick.

Before generating multiple proposals, query free capacity (see the "Compute budget awareness" section in the planner role contract for the exact `kubectl` commands). Use the free GPU count to set a ceiling on slate width:

- Slate width = min(evidence-defensible alternatives, free training GPUs, 3). The "3" cap exists because the lead has to read each proposal — wider slates dilute attention.
- Always reserve ≥1 GPU for eval. Do not propose so wide that eval can't run.
- If the evidence only supports 1 lever, return 1 proposal — do not pad with speculative alternatives. Document in your end-of-cycle report which levers you considered and rejected, and why.

After appending all the records, return to the lead a small markdown table (one row per option, with `record_id`, knob, old → new, one-sentence rationale) and a one-line recommended **wave plan**: e.g. "launch options A and B in parallel on the two free training GPUs; eval lane shared on node X serialized; option C held back as second-wave depending on A/B outcome".
