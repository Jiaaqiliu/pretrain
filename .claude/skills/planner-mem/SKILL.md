---
name: planner-mem
description: Read / search / append records in the nemo_mas shared ledger as the Planner. Thin wrapper around `python -m agent_evolve.model.algorithms.nemo_mas.cli mem ...`. Use when you need historical context or to append `breakthrough` / `failed_attempt` outside the structured propose flow.
---

Thin reference for reading and writing the shared ledger as the Planner. All commands print one JSON object on stdout.

## Standard session-start queries

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind eval_report -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind data_gap -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<topic>" --kind recipe_proposal --top-k 8
```

## Append

Your role (`--role planner`) may write: `recipe_proposal`, `data_audit_finding`, `data_gap`, `error_pattern`, `benchmark_rule`, plus cross-cutting `breakthrough` / `failed_attempt`. The CLI rejects anything else.

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role planner --kind <kind> \
  --title "<headline>" \
  --body-file /tmp/body.md \
  --ref <rec_…> \
  --tag <tag>
```

- `--ref` is repeatable. `recipe_proposal` MUST `--ref` at least one `eval_report` or `data_gap` (CLI enforces).
- `--tag` is repeatable. For `recipe_proposal`: `sft` / `data_mix` / `distill` describe the area (scope is SFT-LoRA only — no other training regime, no full-finetune); `preliminary` marks single-seed evidence; `supersedes:<old_id>` flags a contradiction of a prior proposal.

## Reference lookups

```bash
# Diff two recipe YAMLs. Diffs target the tunable child recipe
# (e.g. recipes/train/default.yaml), never the frozen `_base` anchor.
# See planner-propose-recipe for the full rule.
python -m agent_evolve.model.algorithms.nemo_mas.cli recipe diff \
  --a recipes/train/default.yaml --b /tmp/proposed_after.yaml
```

## What NOT to use this skill for

- Writing a `recipe_proposal` → use `planner-propose-recipe`. Don't hand-roll the body.
- Writing a `data_audit_finding` → use `planner-audit-jsonl`. Don't hand-roll the body.
- Writing a `training_run` / `eval_report` / `kaggle_submission_result` → Trainer's job.
- Writing a `distill_batch` / `dataset_snapshot` → DataWorker's job.
