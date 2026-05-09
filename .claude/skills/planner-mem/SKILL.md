---
name: planner-mem
description: Read / search / append records in the nemo_mas shared ledger as the Planner. Thin wrapper around `python -m agent_evolve.model.algorithms.nemo_mas.cli mem ...`. Use when you need context from prior cycles or to append `breakthrough` / `failed_attempt` outside the structured propose / hypothesis flows.
---

Thin reference for reading and writing the shared ledger as the Planner. All commands print one JSON object on stdout.

## Standard session-start queries

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind cv_result -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind eval_report -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind data_gap -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<topic>" --kind hypothesis --top-k 8
```

## Append

Your role (`--role planner`) may write: `hypothesis`, `recipe_proposal`, `directive_response`, plus cross-cutting `breakthrough` / `failed_attempt` / `checkpoint_event`. The CLI rejects anything else.

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role planner --kind <kind> \
  --title "<headline>" \
  --body-file /tmp/body.md \
  --ref <rec_…> \
  --tag <tag>
```

- `--ref` is repeatable. `recipe_proposal` MUST `--ref` at least one `eval_report` or `data_gap` (CLI enforces). `hypothesis` MUST `--ref` at least one evidence record (role protocol, enforced by convention and reviewed).
- `--tag` is repeatable. For `recipe_proposal`: `lora` / `sft` / `rl` / `grpo` / `data_mix` / `distill` shape which Quality Plan slot the proposal feeds. For `hypothesis`: `preliminary` marks single-seed evidence; `supersedes:<old_id>` flags a contradiction of a prior hypothesis.

## Reference lookups

```bash
# Look up the current Quality Plan state before planning
python -m agent_evolve.model.algorithms.nemo_mas.cli checkpoints list

# Look up a specific slot's state
python -m agent_evolve.model.algorithms.nemo_mas.cli checkpoints state --slot-id cp_eval_sanity

# Diff two recipe YAMLs (a or b can be inline text OR workspace-relative path)
python -m agent_evolve.model.algorithms.nemo_mas.cli recipe diff --a train/a.yaml --b train/b.yaml
```

You may NOT call `checkpoints review-suggest` or `checkpoints sign` — those are Reviewer-only (and the CLI's mode guard will refuse). Read the slot state and let your proposals be shaped by it.

## What NOT to use this skill for

- Writing a `recipe_proposal` → use `planner-propose-recipe`. Don't hand-roll the body.
- Writing a `hypothesis` → use `planner-hypothesis`.
- Writing a `training_run` / `distill_batch` / `eval_report` / `checkpoint_review` → wrong role.
