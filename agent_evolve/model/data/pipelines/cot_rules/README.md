# cot_rules pipeline

Programmatic, LLM-free training-data pipeline. Replicates huikang's
public 0.85-LB Kaggle recipe (the same code path that produced the
`default_14718.jsonl` baseline) — three steps:

```
step_1_reasoning      → reasoning/<id>.txt              (deterministic CoT)
step_2_augmentation   → augmentations/<id>.txt          (synthetic tasks)
step_3_corpus         → corpus.jsonl + corpus/<id>/...  (tokenized SFT)
```

Coverage on the 1602 Kaggle bits problems: ~85% land in `rule_found`,
the rest fall through to `hypothesis_formed` / `rule_unknown` and are
excluded from the training set.

This pipeline is **complementary**, not a replacement for, the
solver-as-hint + LLM-teacher pipelines under [../legacy/](../legacy/)
(`legacy/bits/`, `legacy/equations/`). Different SFT recipes can mix
either, neither, or both.

## Files

| File                    | Role                                           |
|-------------------------|------------------------------------------------|
| `pipeline.yaml`         | the single source of truth for paths + knobs   |
| `run.py`                | reads the YAML, dispatches the 3 steps         |
| `run_reasoning.py`      | step 1 — programmatic CoT generation           |
| `run_augmentation.py`   | step 2 — synthetic-task generation             |
| `run_corpus.py`         | step 3 — tokenize + assemble SFT corpus        |
| `augmenters/`           | 5 synthetic-task generators                    |
| `README.md`             | this file                                      |

Domain solvers live one level up at [../../reasoners/](../../reasoners/)
and are shared with [../../verifiers/](../../verifiers/) and the
[legacy/](../legacy/) solver-as-hint pipelines.

## Running

```bash
cd /fsx/zzsamshi/a-evolve

# All three steps end-to-end:
python -m agent_evolve.model.data.pipelines.cot_rules.run \
    --config agent_evolve/model/data/pipelines/cot_rules/pipeline.yaml

# Single step (e.g. only step 1):
python -m agent_evolve.model.data.pipelines.cot_rules.run \
    --config agent_evolve/model/data/pipelines/cot_rules/pipeline.yaml \
    --from-step 1 --to-step 1
```

## Required workspace inputs

The `paths.work_dir` directory must contain (or its referenced paths
must point to) before step 1 runs:

| Key                | What                                                    |
|--------------------|---------------------------------------------------------|
| `problems.jsonl`   | one JSON object per line: `{id, category, …}` index    |
| `problems/<id>.jsonl` | per-problem payloads (loaded by `Problem.load_from_json`) |
| `train.csv`        | Kaggle `train.csv` with `id, prompt, answer` columns   |
| `tokenizer.json`   | BPE vocab; matches the inference / chat-template setup |

The driver creates `work_dir` if it doesn't exist and writes `reasoning/`,
`augmentations/`, `corpus/`, and `corpus.jsonl` under it.

## Knobs in pipeline.yaml

* `step_1_reasoning.categories` — whitelist a subset of Kaggle categories
  (e.g. only `bit_manipulation`). `null` runs every supported category.
* `step_2_augmentation.augmenters.*.enabled` — toggle each of the 5
  synthetic-task generators independently.
* `step_2_augmentation.augmenters.*.n_problems` — override the
  module-level default count (e.g. shrink `concatenation` from 1500 to
  100 for a smoke run).
* `step_3_corpus.token_limit` — truncate over-long sequences (default 8192).
* `step_3_corpus.chat_tokenizer` — HF tokenizer name for the chat template
  (default `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).

## Required Python deps for each step

| Step | Needs                                       |
|------|---------------------------------------------|
| 1    | stdlib only                                 |
| 2    | `tokenizers` (only if spelling enabled)     |
| 3    | `tokenizers` + `transformers` (mandatory)   |

Step 1 alone is enough to validate the `reasoners/` port without
having a tokenizer environment available.
