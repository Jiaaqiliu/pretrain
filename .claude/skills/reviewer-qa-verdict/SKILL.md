---
name: reviewer-qa-verdict
description: Post a Quality Plan verdict on a checkpoint slot after reviewing its evidence — and, in auto mode only, sign the slot yourself once ready. Use when the Orchestrator assigns a `qa_checkpoint_review` task. Do NOT use for data audit or eval work (those produce the evidence; this skill judges it).
---

You are the Reviewer wearing your QA-officer hat. This skill does not create new evidence — it reads existing evidence and posts a verdict.

## Inputs

- `slot_id` — e.g. `cp_data_check`, `cp_training_health`, `cp_eval_sanity`, `cp_submission_ready`

## Steps

### 1 — Read the slot's declaration + current fold

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli checkpoints state --slot-id "$SLOT_ID"
```

Pay attention to:
- `requires_evidence` — the kinds of records that must exist, slot-tagged
- `depends_on`        — upstream slots that must be signed first
- `evidence_counts`   — what's already attached
- `last_review_verdict` — whether there's already a verdict you're replacing

If `depends_on` is non-empty and any upstream is not `signed` / `reopened`, STOP. Post `insufficient` with a reason citing the unmet dependency, or wait.

### 2 — Find the slot-tagged evidence

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "checkpoint:$SLOT_ID" --top-k 20
```

Also look at what the Orchestrator cited in your task brief (those are the high-priority candidates).

### 3 — Read each candidate record in full

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$REC_ID"
```

For each candidate, check:
- Does the evidence match the slot's intent? e.g. `profile_run` for `cp_training_health` should cover forward-shape + overfit batch, not just a lucky train loss.
- Is it recent? Evidence older than 2 cycles is probably stale — note it, but lean toward `insufficient`.
- Are the numbers healthy? Loss monotone decreasing, no NaN, length distribution reasonable, eval breakdown balanced.

### 4 — Pick a verdict

| verdict              | when                                      |
|----------------------|-------------------------------------------|
| `evidence_attached`  | some evidence present, not complete       |
| `ready_to_sign`      | evidence complete, numbers healthy        |
| `insufficient`       | some evidence, not enough to judge        |
| `reject`             | evidence looks wrong / training unhealthy |

### 5 — Post the verdict

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli checkpoints review-suggest \
  --slot-id "$SLOT_ID" \
  --verdict "$VERDICT" \
  --reason "<concrete one-line reason with specific numbers or artifact paths>" \
  --ref "$REC_ID_1" --ref "$REC_ID_2" ...
```

The `--reason` is what the cockpit shows next to the slot in the ledger. Make it specific: "eval_report rec_abc shows kaggle=0.681 on dev split, 3/4 breakdown buckets > baseline". Don't write "looks good".

### 6 — Auto-mode sign (optional)

If the env is in **auto mode** AND your verdict was `ready_to_sign`:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli checkpoints sign \
  --slot-id "$SLOT_ID" --role reviewer \
  --ref "$REC_ID_1" --ref "$REC_ID_2" ...
```

The sign command re-checks evidence coverage + dependency state — the CLI will refuse if anything's off.

In **manual mode** the CLI will reject `--role reviewer`. Stop after `review-suggest`; the human sees your verdict in the viewer and clicks Sign.

## Hard rules

- Do NOT sign a slot whose evidence you produced in the same cycle. Self-signing bypasses the two-eyes check the Quality Plan is built on.
- Do NOT post `ready_to_sign` when `requires_evidence` kinds are missing — that is what `insufficient` is for.
- Do NOT invent evidence. If a required kind isn't present, the Orchestrator needs to spawn a worker to produce it — your verdict is `insufficient`, not a promise.
- Do NOT hand-roll `checkpoint_review` with `mem append`. Use `checkpoints review-suggest` — it enforces slot+verdict+ref shape.
