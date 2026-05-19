---
name: trainer-ablation-launch
description: Submit a per-category data-efficiency ablation — train two SFT-LoRA arms in parallel (baseline-<cat> from default_14718 vs curated-<cat> from a dw-pipeline-launch curated JSONL, decontaminated against balanced_dev726), watch ~60s for fast-fail per arm, then drop a single combined marker and return. The companion `trainer-ablation-collect` skill drives both arms through eval and writes one `ablation_report` once both arms + both evals finish.
---

You are the Trainer. This skill submits TWO training Jobs (one per
ablation arm) and returns fast — it does NOT wait for completion. The
companion `trainer-ablation-collect` harvests results in two phases
(train-done → submit eval; eval-done → write report). One ablation =
one category = one combined marker.

## When to invoke

The Orchestrator hands you a `recipe_proposal` / `data_gap` whose body
names a `curated_jsonl_path` (output of `dw-pipeline-launch`) and asks
"is this curated set worth promoting into the recipe?" That's an
ablation question, not a recipe-tuning question — use this skill.

For all other "train one config" requests, use `trainer-launch-stage`.

## Inputs

- `category`           — one of: bits, cipher, equations, gravity,
                         numerals, units (the categories
                         `DOMAIN_HEURISTIC` covers in
                         `agent_evolve/model/data/pipelines/legacy/shared/stages/witness_search.py`).
- `curated_jsonl_path` — abs path to the curated JSONL produced by
                         `dw-pipeline-launch`, e.g.
                         `<workspace>/artifacts/generation/<pipeline_name>/curated/<hash>/<cat>_distilled.jsonl`
- `distill_batch_id`   — `rec_…` of the corresponding `distill_batch`
                         (provenance for arm B)
- `num_steps`          — same value used for both arms (default 460).
                         For per-domain ablations the smaller arm is
                         typically <2k rows, so a 60–200 step budget is
                         the right scope. Match `num_steps` between
                         arms — do NOT vary it across the comparison.
- `run_name_prefix`    — short id, e.g. `abl-bits-20260517` (used as
                         marker filename + the prefix for both arm
                         job names + the prefix for both arm child
                         recipes)

**Required: save at least two checkpoints per arm.** Set
`save_every = num_steps / 2` (rounded down) so each arm saves at the
midpoint and at the end. Two checkpoints per arm is the minimum the
ablation needs:
- midpoint catches the fast-learner regime (curated arm often peaks
  early on small data, then overfits);
- end-of-training is the apples-to-apples comparator vs every other
  recipe in the ledger.
If you save only the final checkpoint you cannot tell whether arm B
peaked-and-collapsed or genuinely underperformed. The collect skill
evals both checkpoints per arm.

If anything is missing, STOP and write a `failed_attempt`.

## Environment

Harness sets:
- `NEMO_MAS_WORK_DIR`         — run root (markers under
  `<work_dir>/.pending_jobs/`)
- `NEMO_MAS_WORKSPACE_ROOT`   — forked workspace
- `NEMO_MAS_MEMORY_PATH`      — ledger

You also need:
- `WANDB_API_KEY` — required for `--wandb`; without it pass `--no-wandb`
- `KUBECTL_CTX="arn:aws:eks:ap-southeast-3:801953956576:cluster/p5-llm"`

## Steps

### 1 — Verify the spec record

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli mem get --id "$DISTILL_BATCH_ID"
ls "$CURATED_JSONL_PATH"
```

Both must succeed. If not → `failed_attempt` and stop.

### 2 — Reject if a marker for this prefix already exists

```bash
MARKER="$NEMO_MAS_WORK_DIR/.pending_jobs/ablation-${RUN_NAME_PREFIX}.json"
mkdir -p "$NEMO_MAS_WORK_DIR/.pending_jobs"
[ -f "$MARKER" ] && { echo "marker exists for $RUN_NAME_PREFIX — collect first"; exit 2; }
```

### 3 — Prep both arm JSONLs

The helper handles content-addressed output dirs + provenance; we
just call it twice.

```bash
PY=/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python
DEV_CSV=/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/dev/balanced_dev726.csv
DEFAULT_14718=/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/train/default_14718.jsonl

# Arm A — baseline subset of default_14718 (no decontamination by user's call)
ARM_A_JSON=$($PY -m agent_evolve.model.data.pipelines.legacy.shared.extract_subset baseline \
  --src "$DEFAULT_14718" \
  --domain "$CATEGORY" \
  --out-dir "$NEMO_MAS_WORKSPACE_ROOT/artifacts/data/baseline_${CATEGORY}")
ARM_A_PATH=$(echo "$ARM_A_JSON" | $PY -c "import sys,json; print(json.loads(sys.stdin.read())['out'])")

# Arm B — curated, decontaminated against the dev split
ARM_B_JSON=$($PY -m agent_evolve.model.data.pipelines.legacy.shared.extract_subset decontaminate \
  --src "$CURATED_JSONL_PATH" \
  --dev "$DEV_CSV" \
  --out-dir "$NEMO_MAS_WORKSPACE_ROOT/artifacts/data/curated_clean_${CATEGORY}")
ARM_B_PATH=$(echo "$ARM_B_JSON" | $PY -c "import sys,json; print(json.loads(sys.stdin.read())['out'])")
ARM_B_ROWS_OUT=$(echo "$ARM_B_JSON" | $PY -c "import sys,json; print(json.loads(sys.stdin.read())['rows_out'])")
ARM_B_DROPPED=$(echo "$ARM_B_JSON" | $PY -c "import sys,json; print(json.loads(sys.stdin.read())['decontam_dropped'])")
ARM_A_ROWS_OUT=$(echo "$ARM_A_JSON" | $PY -c "import sys,json; print(json.loads(sys.stdin.read())['rows_out'])")
```

Sanity gates (refuse + `failed_attempt` if either trips):
- `arm_a_rows_out == 0` — domain heuristic doesn't match anything in
  default_14718; check the category name.
- `arm_b_rows_out == 0` — curated set is entirely contaminated. Halt
  hard; the curated set is unusable as-is.
- `arm_b_rows_out < 50` — too few clean rows to train on. Halt.

NOTE: high `decontam_dropped` (e.g. >10%) is **expected** when the dev
split is drawn from the same Kaggle train CSV the distillation pipeline
ran over — record the count in the marker, do not halt.

### 3.5 — Check baseline cache (arm A)

Arm A is **deterministic per (category, num_steps)** — same data subset
of `default_14718.jsonl`, same hyperparams from `default.yaml`. Once
trained, we should reuse it across every ablation that asks the same
question. Cache key:

```
ARM_A_CACHE_KEY="baseline_${CATEGORY}_${ARM_A_DATA_HASH}_${NUM_STEPS}steps"
ARM_A_CKPT_DIR="$NEMO_MAS_WORKSPACE_ROOT/artifacts/sft/${ARM_A_CACHE_KEY}"
```

where `ARM_A_DATA_HASH` is the 12-char sha read from the baseline
provenance.json (the dir name `extract_subset.py baseline` already wrote
the data into).

```bash
ARM_A_REUSED=0
ARM_A_TRAINING_RUN_ID=""
ARM_A_EVAL_REPORT_ID=""

if [ -f "$ARM_A_CKPT_DIR/final/adapter_config.json" ]; then
  # Checkpoint exists. Look up its training_run record so we can ref it
  # later. The record's body line `final_ckpt_path:` carries the dir.
  ARM_A_TRAINING_RUN_ID=$(python -m agent_evolve.model.algorithms.nemo_mas.cli \
    mem search --query "$ARM_A_CACHE_KEY" --kind training_run --top-k 5 \
    | python -c "
import json, sys
data = json.loads(sys.stdin.read().strip().split('\n')[-1])
for r in data.get('hits', []):
    if '$ARM_A_CKPT_DIR' in r.get('body', ''):
        print(r['id']); break
")
  if [ -n "$ARM_A_TRAINING_RUN_ID" ]; then
    ARM_A_REUSED=1
    echo "arm_a cache hit: $ARM_A_TRAINING_RUN_ID at $ARM_A_CKPT_DIR"
    # Bonus: look for an eval_report on this checkpoint to skip arm A's
    # eval too. Match on parent_id == ARM_A_TRAINING_RUN_ID.
    ARM_A_EVAL_REPORT_ID=$(python -m agent_evolve.model.algorithms.nemo_mas.cli \
      mem search --query "$ARM_A_TRAINING_RUN_ID" --kind eval_report --top-k 5 \
      | python -c "
import json, sys
data = json.loads(sys.stdin.read().strip().split('\n')[-1])
for r in data.get('hits', []):
    if '$ARM_A_TRAINING_RUN_ID' in (r.get('refs') or []):
        print(r['id']); break
")
  else
    echo "arm_a ckpt on disk but no training_run record found — will retrain to keep ledger consistent"
  fi
fi
```

If `ARM_A_REUSED=1`, the launch flow skips arm A's recipe write,
`submit.sh train`, and 60s watch entirely. The marker (step 7) records
the cache hit so the collect skill knows to skip arm A's training and
(if `ARM_A_EVAL_REPORT_ID` is also set) arm A's eval.

If you need to **invalidate** a cache hit (e.g. recipe defaults
changed), just delete `$ARM_A_CKPT_DIR/final/adapter_config.json` —
the next launch retrains.

### 4 — Build child train recipes

Two siblings of `default.yaml`, one per arm. `data.train_jsonl` is the
field [`train_unsloth.py:82`](agent_evolve/backends/nemo_reasoner/k8s/entries/train_unsloth.py#L82)
reads — overriding it here pins the arm to its own JSONL.

```bash
DEFAULT_YAML="$NEMO_MAS_WORKSPACE_ROOT/recipes/train/default.yaml"

# Skip arm A entirely if cache hit. Always write arm B (its data is
# curated-set-specific; no shared cache).
ARMS_TO_BUILD=""
[ "$ARM_A_REUSED" = "0" ] && ARMS_TO_BUILD="a"
ARMS_TO_BUILD="$ARMS_TO_BUILD b"

for ARM in $ARMS_TO_BUILD; do
  if [ "$ARM" = "a" ]; then DATA_PATH="$ARM_A_PATH"; LABEL="baseline-${CATEGORY}"
  else                       DATA_PATH="$ARM_B_PATH"; LABEL="curated-${CATEGORY}"
  fi
  NEW_YAML="$NEMO_MAS_WORKSPACE_ROOT/recipes/train/default_${RUN_NAME_PREFIX}_${ARM}.yaml"
  cp "$DEFAULT_YAML" "$NEW_YAML"
  # Append the two-key override block. The base inherits 'data' from
  # default_base.yaml, which is what we want to override.
  # CRITICAL: a copy of `default.yaml` already contains a `batching:`
  # block. PyYAML's safe_load keeps only the LAST occurrence of a
  # top-level key, so appending a second `batching:` block silently
  # drops batch_size / micro_batch_size / seed. Build one merged block.
  SAVE_EVERY=$(( NUM_STEPS / 2 ))   # required: two checkpoints per arm
  python - <<PY
import yaml, sys
p = "${NEW_YAML}"
cfg = yaml.safe_load(open(p))
cfg.setdefault("batching", {})
cfg["batching"]["num_steps"]  = ${NUM_STEPS}
cfg["batching"]["save_every"] = ${SAVE_EVERY}
cfg.setdefault("data", {})
cfg["data"]["train_jsonl"] = "${DATA_PATH}"
cfg["name"] = "default_${RUN_NAME_PREFIX}_${ARM}"
open(p, "w").write(yaml.safe_dump(cfg, sort_keys=False))
PY
done
```

Sanity-check the diff:

```bash
python -m agent_evolve.model.algorithms.nemo_mas.cli recipe diff \
  --a "$DEFAULT_YAML" \
  --b "$NEMO_MAS_WORKSPACE_ROOT/recipes/train/default_${RUN_NAME_PREFIX}_a.yaml"
```

### 5 — Submit train Jobs in parallel

If arm A was a cache hit (§3.5), only arm B is submitted. Otherwise
both arms fire in one message via `Bash(run_in_background: true)`.

The arm A `--out` is the **content-addressed cache dir**
`artifacts/sft/${ARM_A_CACHE_KEY}` so the next ablation that asks the
same `(category, num_steps)` question hits the cache. Arm B keeps the
per-run dir `artifacts/sft/${RUN_NAME_PREFIX}_b` (its data is
curated-set-specific; no shared cache).

```bash
BACKEND=/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner
WANDB_FLAG=$([ -n "${WANDB_API_KEY:-}" ] && echo --wandb || echo --no-wandb)

for ARM in $ARMS_TO_BUILD; do
  RUN_NAME="${RUN_NAME_PREFIX}-${ARM}"
  if [ "$ARM" = "a" ]; then CKPT_OUT="$ARM_A_CKPT_DIR"
  else                       CKPT_OUT="$NEMO_MAS_WORKSPACE_ROOT/artifacts/sft/${RUN_NAME_PREFIX}_${ARM}"
  fi
  TRAIN_RECIPE="$NEMO_MAS_WORKSPACE_ROOT/recipes/train/default_${RUN_NAME_PREFIX}_${ARM}.yaml"
  mkdir -p "$CKPT_OUT" /tmp/trainer
  "$BACKEND/k8s/submit.sh" train \
    --train-recipe "$TRAIN_RECIPE" \
    --data-recipe  "$NEMO_MAS_WORKSPACE_ROOT/recipes/data/default_data.yaml" \
    --out          "$CKPT_OUT" \
    --name         "$RUN_NAME" \
    $WANDB_FLAG  2>&1 | tee "/tmp/trainer/${RUN_NAME}_submit.log" &
done
wait
```

### 6 — 60s fast-fail watch per submitted arm

Same pattern as
[`trainer-launch-stage`](.claude/skills/trainer-launch-stage/SKILL.md#L97-L112).
Run the watch for each arm in `$ARMS_TO_BUILD` in parallel — if arm A
was a cache hit, this is just arm B. If EITHER submitted arm fast-fails:

- delete BOTH Jobs (we want symmetric arms — if one is broken the
  comparison is meaningless),
- write a `failed_attempt` ref'ing `$DISTILL_BATCH_ID`,
- exit. Do NOT drop a marker.

### 7 — Drop one combined marker

```json
{
  "kind": "ablation_report",
  "category": "<category>",
  "run_name_prefix": "<run_name_prefix>",
  "submitted_at": "<ISO-8601>",
  "refs": ["<distill_batch_id>"],
  "context": {
    "category": "<category>",
    "phase": "awaiting_train",
    "num_steps": <int>,
    "save_every": <int>,                       // = num_steps / 2 (per skill rule)
    "checkpoint_steps_to_eval": [<mid>, <num_steps>],  // e.g. [30, 60]
    "expected_eval_split": "balanced_dev726",
    "arm_a": {
      "label": "baseline-<category>",
      "job_name": "ne-train-<prefix>-a",        // null if reused
      "ckpt_out": "<abs ARM_A_CKPT_DIR — content-addressed cache dir>",
      "data_path": "<abs baseline subset>",
      "rows": <int>,
      "recipe_path": "recipes/train/default_<prefix>_a.yaml",   // null if reused
      "cache_key": "baseline_<cat>_<data_hash>_<num_steps>steps",
      "reused": false,                          // true on cache hit
      "training_run_id": null,                  // set to ARM_A_TRAINING_RUN_ID on cache hit
      "eval_report_id": null                    // set to ARM_A_EVAL_REPORT_ID on cache hit
    },
    "arm_b": {
      "label": "curated-<category>",
      "job_name": "ne-train-<prefix>-b",
      "ckpt_out": "<abs>",
      "data_path": "<abs decontaminated curated>",
      "rows": <int>,
      "decontam_dropped": <int>,
      "recipe_path": "recipes/train/default_<prefix>_b.yaml"
    }
  }
}
```

`phase: "awaiting_train"` tells the collect skill which branch to take.

### 8 — Return

Brief one-paragraph report: marker path, both job names, both data
paths, both arm row counts, the decontam drop count, and "next step:
trainer-ablation-collect once both Jobs finish."

## Hard rules

- ❌ Do NOT mix baseline and curated rows into one training set. The
  whole point of the ablation is two clean arms.
- ❌ Do NOT decontaminate the baseline arm. The user's call was
  "trust default_14718"; revisit only if a baseline shows surprisingly
  high `<cat>.acc`.
- ❌ Do NOT lower `num_steps` between arms — they must match for the
  comparison to mean anything. The skill enforces this by writing one
  `NUM_STEPS` into both child recipes.
- ❌ Do NOT save only the final checkpoint. `save_every` MUST be set
  to `num_steps / 2` so each arm produces at least two checkpoints
  (midpoint + final). The collect skill evaluates both — without a
  midpoint we cannot tell early-peak-then-collapse from genuine
  underperformance.
- ❌ Do NOT block past the 60s fast-fail watch. The async pattern is
  the whole point of the launch/collect split.
- ❌ Do NOT proceed when only one arm passes fast-fail. Cancel both;
  asymmetric arms = no comparison.

## Anti-patterns

- Do NOT call `trainer-launch-stage` and `trainer-collect-results`
  twice manually with custom data paths to fake an ablation. This skill
  exists so the marker carries `phase` state and the collect skill
  knows to drive eval automatically.
- Do NOT write `ablation_report` from this skill. That happens in
  `trainer-ablation-collect` after both evals land.
- Do NOT skip the decontamination step. The point is honest data
  efficiency — without it the curated arm trains on dev rows.
