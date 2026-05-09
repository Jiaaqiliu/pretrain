---
name: dw-mem
description: Read / search / append records in the nemo_mas shared ledger as the Data Worker. Thin wrapper around `python -m agent_evolve.model.algorithms.nemo_mas.cli mem ...`. Use when you need context from prior cycles or to append `breakthrough` / `failed_attempt` outside the structured distill / curate flows.
---

Thin reference for reading and writing the shared ledger as the Data Worker. All commands print one JSON object on stdout.

## Standard session-start queries

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind data_gap -k 3
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$SPEC_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<domain>" --kind distill_batch --top-k 5
```

## Append

Your role (`--role data_worker`) may write: `distill_batch`, `dataset_snapshot`, `directive_response`, and the cross-cutting `breakthrough` / `failed_attempt` / `checkpoint_event`. Other kinds will be rejected by the CLI.

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role data_worker --kind <kind> \
  --title "<headline>" \
  --body-file /tmp/body.md \
  --ref <rec_…> \
  --tag <tag>
```

- `--ref` is repeatable. `dataset_snapshot` does not require refs at the schema level, but cite the contributing `distill_batch` ids anyway — Reviewer needs them.
- `--tag` is repeatable. Use `checkpoint:<slot_id>` on records that serve a Quality Plan slot; the fold ignores un-tagged evidence.

## What NOT to use this skill for

- Writing a `distill_batch` → use `dw-teacher-distill` or `dw-self-distill`.
- Writing a `dataset_snapshot` → use `dw-curate-mix`.
- Writing a `recipe_proposal` / `hypothesis` → Planner's job.
- Writing a `data_audit_finding` / `eval_report` / `checkpoint_review` → Reviewer's job.
