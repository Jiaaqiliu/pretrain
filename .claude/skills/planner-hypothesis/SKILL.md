---
name: planner-hypothesis
description: Write a falsifiable `hypothesis` record — a predicted-effect claim paired with the smallest experiment that would test it. Use when evidence is suggestive but not yet strong enough to justify a `recipe_proposal`, or when a recipe proposal needs a separate CV confirmation plan.
---

You are the Planner. This skill produces ONE `hypothesis` record per call. A hypothesis without a falsification plan is not a hypothesis — it's a guess. This skill forces the plan.

## Inputs

- `evidence_ids`  — one or more `rec_…` that motivate the claim (typically `eval_report`, `cv_result`, `error_pattern`, `breakthrough`)
- `effect`        — one-line predicted-effect direction (e.g. `"raising warmup from 0 to 200 steps should remove the early-step loss spike on long-CoT"`)
- `magnitude`     — optional quantitative prediction (`"+0.5-1.5% on hard split"`)
- `test_kind`     — `profile_run` | `cv_result` | `eval_report` — the shape of the experiment that would confirm or reject

## Steps

### 1 — Load context

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem recent --kind breakthrough -k 5
python -m agent_evolve.model.algorithms.nemo_mas.cli mem search \
  --query "<topic>" --kind hypothesis --top-k 8
for id in $EVIDENCE_IDS; do
  python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id $id
done
```

If a live hypothesis already covers the same claim, either:
- **Agree**: cite it as a ref and tag `corroborates`. You still get one record.
- **Contradict**: cite it as a ref AND tag `supersedes:<old_id>`. Name exactly how your prediction differs.
- **No delta**: STOP and write nothing. Hypothesis spam pollutes `mem_search`.

### 2 — Design the smallest falsifying experiment

This is the whole point. "We should try X" is rejected. The experiment must have:
- **Exact executor**: `trainer` or `data_worker` with a concrete stage (e.g. `profile_run 200 steps`, `run_eval --split dev --limit 500`, `teacher_distill math 1000 rows`).
- **Pre-declared pass/fail threshold**: e.g. "pass if dev kaggle ≥ 0.62 AND hard ≥ 0.55; otherwise reject".
- **Budget**: token/GPU-hour estimate.

If you cannot name a concrete threshold, the hypothesis is not falsifiable — rework the claim or write a `failed_attempt` instead.

### 3 — Build the body

Write `/tmp/hypothesis_body.md`:

```
claim: <one-line predicted-effect direction>
magnitude: <quantitative prediction OR "directional only">
evidence: <summary referencing the motivating records>

Smallest falsifying experiment:
  executor: <trainer | data_worker | reviewer>
  test_kind: <profile_run | cv_result | eval_report>
  setup: <exact stage + args, e.g. "sft warmup=200, 1 seed, 500 steps">
  budget: <tokens or GPU-hours>
  pass if: <concrete threshold>
  reject if: <concrete threshold>
  ambiguous if: <what would make the experiment inconclusive — call that out>

Scope limits:
  - applies to: <regime / category / checkpoint range>
  - does NOT apply to: <what would make this claim irrelevant>

Prior art in memory:
  - rec_… (corroborates / contradicts / ships as-is)
```

### 4 — Append the record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role planner --kind hypothesis \
  --title "<one-phrase claim>" \
  --body-file /tmp/hypothesis_body.md \
  $(for id in $EVIDENCE_IDS; do echo --ref $id; done) \
  ${SUPERSEDES:+--tag supersedes:$SUPERSEDES} \
  ${PRELIMINARY:+--tag preliminary}
```

Tag `preliminary` when the motivating evidence is a single seed — the Planner protocol says so and the Orchestrator routes preliminary hypotheses to `cv_result`-gating before they become proposals.

## Hard rules

- A hypothesis MUST name the smallest experiment that would falsify it. No exceptions.
- At least one `--ref` to a motivating evidence record is required. A hypothesis from thin air is a guess.
- Do NOT pair a hypothesis with an immediate `recipe_proposal` in the same skill call. If you know the change to ship, skip straight to `planner-propose-recipe`. Hypotheses are for "I suspect X; let's confirm first".
- Do NOT write multiple orthogonal claims in one record. One claim = one hypothesis = one refs chain.
