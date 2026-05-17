---
name: trainer-ablation-collect
description: Harvest pending per-category ablation markers (`ablation-*.json` under `<work_dir>/.pending_jobs/`). Two-phase finalizer — phase 1 waits for both train arms to finish then submits both eval Jobs; phase 2 waits for both eval Jobs then writes the `ablation_report` with delta on `breakdown.<category>.acc`. Idempotent — safe to invoke any time, between cycles, or from a Stop hook.
---

You are the Trainer. This skill is the harvester half of the
ablation-launch pattern. Mirrors `trainer-collect-results` shape but
specialized: each ablation marker drives **two** training_runs and
**two** eval_reports through the harness, and the final
`ablation_report` summarizes the delta.

## When to invoke

- The lead asks "any ablations done?" or "collect ablation results."
- After `trainer-ablation-launch` returns, periodically (every 10–30
  minutes; both phases are wall-clock-bound by k8s Job duration).
- From a Stop hook (optional automation).

## Watcher / LLM split

Phase 1 (train-done → submit eval) is **automated** by
`agent_evolve/backends/nemo_reasoner/k8s/ablation_watcher.py` running
in tmux session `ablation_watcher`. It polls `ablation-*.json` every
30s; when both arms succeed it writes the `training_run`s, dispatches
both eval Jobs, and flips the marker to `phase: awaiting_evals`. This
skill's phase-1 logic (step 2 below) only fires when the watcher is
down — it's a manual fallback.

Phase 2 (evals-done → write `ablation_report`) is **always LLM-driven**
through this skill. Reading `breakdown.<cat>.acc` from each metrics.json,
computing the delta, picking the verdict, and narrating side-effects
needs synthesis the watcher can't do. Once the watcher logs
`awaiting_evals_ready_for_LLM=1` (or when the lead asks), invoke this
skill and run step 3.

## Environment

- `NEMO_MAS_WORK_DIR`         — run root (markers under
  `<work_dir>/.pending_jobs/`)
- `NEMO_MAS_WORKSPACE_ROOT`   — forked workspace
- `NEMO_MAS_MEMORY_PATH`      — ledger
- `KUBECTL_CTX="arn:aws:eks:ap-southeast-3:801953956576:cluster/p5-llm"`

## Steps

### 1 — Enumerate ablation markers

```bash
PENDING_DIR="$NEMO_MAS_WORK_DIR/.pending_jobs"
DONE_DIR="$PENDING_DIR/done"
mkdir -p "$DONE_DIR"
ls -1 "$PENDING_DIR"/ablation-*.json 2>/dev/null | grep -v '/done/' \
  || echo "no pending ablations"
```

For each marker, parse JSON and branch on `context.phase`:

- `phase` absent or `"awaiting_train"` → step 2.
- `phase == "awaiting_evals"`            → step 3.

### 2 — Phase 1: train-done → submit eval

For each ablation marker in `awaiting_train`:

```bash
ARM_A_REUSED=$(jq -r '.context.arm_a.reused // false' "$MARKER")
JOB_A="$(jq -r '.context.arm_a.job_name // "null"' "$MARKER")"
JOB_B="$(jq -r '.context.arm_b.job_name'           "$MARKER")"
```

If `arm_a.reused == true`, **skip arm A's job status check entirely** —
its `training_run_id` is already on the marker (cache hit from launch
step 3.5). Otherwise check both arms:

```bash
if [ "$ARM_A_REUSED" != "true" ]; then
  SUCC_A=$(kubectl --context "$KUBECTL_CTX" get job "$JOB_A" -o jsonpath='{.status.succeeded}' 2>/dev/null)
  FAIL_A=$(kubectl --context "$KUBECTL_CTX" get job "$JOB_A" -o jsonpath='{.status.failed}' 2>/dev/null)
fi
SUCC_B=$(kubectl --context "$KUBECTL_CTX" get job "$JOB_B" -o jsonpath='{.status.succeeded}' 2>/dev/null)
FAIL_B=$(kubectl --context "$KUBECTL_CTX" get job "$JOB_B" -o jsonpath='{.status.failed}' 2>/dev/null)
```

Outcomes (treat `arm_a.reused=true` as "succeeded for arm A"):

- Either submitted arm still running → skip; next pass.
- Either submitted arm failed → write a single `failed_attempt` (refing
  `distill_batch_id` from marker.refs[0]) describing which arm fell
  over, delete the failing job (and the other if it's also a real
  job — not the reused one), move marker to `done/`. Stop on this marker.
- All submitted arms succeeded → harvest each as `training_run`,
  then submit eval jobs (only for arms that don't already have an
  `eval_report_id` from the launch cache). See sub-steps below.

#### 2a — Harvest each arm as `training_run`

If `arm_a.reused == true`, **skip 2a for arm A** — `arm_a.training_run_id`
is already on the marker. Reuse the body shape from
[`trainer-collect-results` step 3a](.claude/skills/trainer-collect-results/SKILL.md#L52-L94).
Per arm that actually trained:

```bash
PY=/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python
$PY -m agent_evolve.model.algorithms.nemo_mas.cli metric read \
  --ckpt "<arm.ckpt_out>/final"
```

Write `/tmp/training_run_body_<run_name>.md` with `recipe_path`,
`data_path`, `final_ckpt_path`, the metric line, and any
`train_metric_trajectory` you can scrape from the pod log.

```bash
$PY -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind training_run \
  --title "sft (ablation $LABEL): $RUN_NAME" \
  --body-file "/tmp/training_run_body_${RUN_NAME}.md" \
  --ref "<distill_batch_id from marker.refs>" \
  --ref "<a recipe_proposal id IF you have one — otherwise see note>"
```

**Note on `training_run` ref constraints**: schema requires
`recipe_proposal` + `dataset_snapshot` refs (see
`agent_evolve/model/algorithms/nemo_mas/schema.py`). For ablation arms
launched ad-hoc (no formal proposal/snapshot), there's a tension. Two
options, in order of preference:

1. If the marker's `distill_batch_id` is set, use that as a `dataset_snapshot`-substitute
   ref AND have the launch caller register a one-line `recipe_proposal`
   stub before launch (out of scope for THIS skill — orchestrator's job).
2. If neither is available, write a `failed_attempt` instead noting
   "ablation arm trained successfully but training_run ref constraint
   not satisfied; please register a recipe_proposal stub or relax the
   schema." This is a known gap; surface it instead of fabricating.

Capture both arms' `training_run` record ids — call them
`TRAIN_RUN_ID_A` and `TRAIN_RUN_ID_B`.

#### 2b — Submit eval Jobs (skip arms that already have eval_report_id)

Use the existing eval flow from
[`trainer-run-eval` SKILL.md](.claude/skills/trainer-run-eval/SKILL.md):
one `submit.sh eval` per arm — **except arms whose marker already
carries an `eval_report_id`** (set by the launch skill on a baseline
cache double-hit: same checkpoint, same eval already on file).

```bash
BACKEND=/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner
ARMS_TO_EVAL=""
for ARM in a b; do
  HAVE_EVAL=$(jq -r ".context.arm_${ARM}.eval_report_id // empty" "$MARKER")
  [ -z "$HAVE_EVAL" ] && ARMS_TO_EVAL="$ARMS_TO_EVAL $ARM"
done

for ARM in $ARMS_TO_EVAL; do
  RUN_NAME="<run_name_prefix>-${ARM}-eval"
  CKPT="<arm.ckpt_out>/final"
  OUT_DIR="$NEMO_MAS_WORKSPACE_ROOT/artifacts/eval/<run_name_prefix>_${ARM}/"
  mkdir -p "$OUT_DIR"
  "$BACKEND/k8s/submit.sh" eval \
    --adapter "$CKPT" --out "$OUT_DIR" --name "$RUN_NAME" --tp 1 &
done
wait
```

Then write per-arm eval pending markers (kind=`eval_report`,
refs=[`<train_run_id>`]) only for arms that actually launched. The
next call to `trainer-collect-results` will harvest them. Cached
eval_reports stay as-is.

#### 2c — Update the ablation marker in place

Rewrite the JSON to flip `phase` and inject the new ids:

```json
{
  "kind": "ablation_report",
  "category": "<…>",
  "run_name_prefix": "<…>",
  "submitted_at": "<original>",
  "refs": ["<distill_batch_id>"],
  "context": {
    "category": "<…>",
    "phase": "awaiting_evals",
    "num_steps": <int>,
    "expected_eval_split": "balanced_dev726",
    "arm_a": {
      "label": "<…>",
      "job_name": "<original train job>",
      "ckpt_out": "<…>",
      "data_path": "<…>",
      "rows": <int>,
      "training_run_id": "<TRAIN_RUN_ID_A>",
      "eval_job_name": "ne-eval-<run_name_prefix>-a-eval",
      "eval_out_dir": "<abs eval out_dir>"
    },
    "arm_b": {
      "...": "...",
      "training_run_id": "<TRAIN_RUN_ID_B>",
      "eval_job_name": "ne-eval-<run_name_prefix>-b-eval",
      "eval_out_dir": "<abs eval out_dir>"
    }
  }
}
```

DO NOT move to `done/` yet.

### 3 — Phase 2: evals-done → write ablation_report

For each ablation marker in `awaiting_evals`:

```bash
EVAL_JOB_A="<context.arm_a.eval_job_name>"
EVAL_JOB_B="<context.arm_b.eval_job_name>"
# Same kubectl status checks as step 2.
```

Outcomes:

- Either eval still running → skip; next pass.
- Either eval failed → `failed_attempt`, move marker to `done/`.
- Both succeeded:
  1. Read `metrics.json` from each arm's `eval_out_dir`.
  2. Find each arm's `eval_report` record id (the regular
     `trainer-collect-results` flow should have written these by now;
     if it hasn't, run that skill first or read the JSON directly).
  3. Pull `breakdown.<category>.acc` from both metrics blobs.
  4. Compute `delta = arm_b - arm_a`.
  5. Verdict:
     - `delta >= +0.05` → `improved`
     - `-0.02 < delta < +0.05` → `flat`
     - `delta <= -0.02` → `regressed`
  6. Write `/tmp/ablation_report_body_<prefix>.md`:

```
category: <category>
num_steps: <int>
eval_split: balanced_dev726

arm_a (baseline-<category>):
  training_run_id: <…>
  eval_report_id:  <…>
  data: <abs baseline subset>  (<rows> rows)
  <category>.acc: 0.????
  full breakdown: cipher=… equations=… gravity=… numerals=… units=… bits=…

arm_b (curated-<category>):
  training_run_id: <…>
  eval_report_id:  <…>
  data: <abs decontaminated curated>  (<rows> rows)
        decontam_dropped: <int> of <int> curated rows overlapped balanced_dev726 by row_id
  <category>.acc: 0.????
  full breakdown: …

delta_<category>_acc = arm_b - arm_a = <signed>
verdict: <improved | flat | regressed>

side-effects (other domains, sanity check that arm_b doesn't regress them):
  cipher delta: …   equations delta: …   gravity delta: …   …

notes:
  - harvested asynchronously by trainer-ablation-collect
  - <anything notable about the run>
```

  7. Append the record:

```bash
$PY -m agent_evolve.model.algorithms.nemo_mas.cli mem append \
  --role trainer --kind ablation_report \
  --title "ablation <category>: arm_b $VERDICT (Δ=<signed>)" \
  --body-file "/tmp/ablation_report_body_<prefix>.md" \
  --ref "$TRAIN_RUN_ID_A" --ref "$TRAIN_RUN_ID_B" \
  --ref "$EVAL_REPORT_ID_A" --ref "$EVAL_REPORT_ID_B" \
  --ref "<distill_batch_id from marker.refs>" \
  --tag "category:<category>" --tag ablation
```

  Schema requires ≥2 `training_run` refs — both arms satisfy that. The
  extra `eval_report` and `distill_batch` refs are recommended but not
  required.

  8. Move marker to `done/`:

```bash
mv "$PENDING_DIR/<marker>.json" "$DONE_DIR/<marker>.json"
```

## Report back

A small table per invocation:

| ablation | phase before | action taken | record_id | note |
|---|---|---|---|---|
| abl-bits-20260517 | awaiting_train | submitted evals | — | training_runs rec_…, rec_… |
| abl-bits-20260517 | awaiting_evals | wrote ablation_report | rec_… | bits delta=+0.??? verdict=improved |
| abl-cipher-…     | awaiting_train | left for next pass — train still running | — | — |

If everything was already harvested, just report "no pending ablations"
and exit cleanly.

## Hard rules

- ❌ Do NOT delete a marker without writing the corresponding ledger
  record (or `failed_attempt`). The marker is the only handle the
  harness has on the run.
- ❌ Do NOT block waiting for still-running Jobs — leave them for the
  next collect pass.
- ❌ Do NOT proceed to phase 2 if EITHER arm's eval is missing
  metrics.json. That's a backend bug — write a `failed_attempt` and
  surface it.
- ❌ Do NOT fabricate metrics. Numbers come from `metrics.json`
  produced by the eval Job — never round, never invent.
- ❌ Do NOT skip the `--tag category:<cat>` tag — the leaderboard
  helper filters on it.

## Anti-patterns

- Do NOT eval an ablation arm's checkpoint manually with
  `trainer-run-eval` and write the `ablation_report` separately. The
  marker's `phase` state is the contract; bypassing it loses the
  symmetric harvesting.
- Do NOT compute `delta` from a different metric than
  `breakdown.<category>.acc`. The whole leaderboard depends on this
  one field.
- Do NOT include cross-domain regressions in the verdict. The verdict
  is a single-axis call on `breakdown.<category>.acc`. Cross-domain
  effects go in the body's `side-effects` section for the planner to
  weigh, not in the verdict line.
