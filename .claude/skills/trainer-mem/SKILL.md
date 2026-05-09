---
name: trainer-mem
description: Read / search / append records in the nemo_mas shared ledger (`memory/records.jsonl`). Thin wrapper around `python -m agent_evolve.model.algorithms.nemo_mas.cli mem ...`. Use any time you need context from prior cycles, or to append a record outside the structured training/cv/submission flows.
---

Thin reference for reading and writing the shared ledger from Bash. All commands print a single JSON object on stdout.

## Read

```bash
# Fetch one record by id
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id rec_abcdef

# BM25 search (body + title + tags; top-k=8 by default)
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "sft mixture v3" --kind training_run --top-k 5

# N most recent by kind (newest first)
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
```

## Standard session-start queries for the Trainer

Do this at the start of every training task before anything else:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$RECIPE_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$DATASET_ID"
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<recipe family>" --kind training_run --top-k 5
```

The last search tells you how prior runs of similar configs performed or broke.

## Append

```bash
# Body must be a file path. Write it with Edit/Write first.
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind <training_run|cv_result|submission_artifact|breakthrough|failed_attempt> \
  --title "<one-line headline>" \
  --body-file /tmp/my_body.md \
  --ref rec_abc --ref rec_def \
  --tag sft --tag stable
```

- `--role` MUST be `trainer`. The CLI enforces kind whitelist per role; breakthrough/failed_attempt are allowed because they're cross-cutting.
- `--ref` is repeatable. Some kinds have required refs (`training_run` needs both a `recipe_proposal` and a `dataset_snapshot`; `cv_result` needs `training_run`; `submission_artifact` needs `training_run`). The CLI will reject missing refs with `ok: false`.
- `--tag` is repeatable. Tags are free-form but some conventions matter:
  * `sft` / `rl` / `grpo` on `cv_result` gates the Quality Plan's `cp_04` / `cp_05` slots.
  * `checkpoint:<slot_id>` on evidence records is how the Reviewer finds them during review.

## What NOT to use this skill for

- Writing a `training_run` → use `trainer-launch-stage`. Don't hand-write the body.
- Writing a `cv_result` → use `trainer-cross-validate`.
- Writing a `submission_artifact` → use `trainer-pack-submission`.
- Writing a `recipe_proposal` or `dataset_snapshot` → you're the wrong role.
- Writing a `checkpoint_review` — Reviewer-only kind.
