# INTEGRATION.md — plug your training job into a-evolve

This guide tells you **where to wire in** a new training job, data-generation
method, inference path, fine-tuning surface, or benchmark without editing
core files (`training/loop.py`, `training/trial.py`, `training/algorithms/`).

Every extension point is a Protocol + a small registry. Implement the
Protocol, register your class, set a YAML field — done.

---

## 0. Decision tree — which Protocol do I need?

| You want to … | Implement | Registry |
|---|---|---|
| Swap the entire training framework (sklearn, JAX, PyTorch-from-scratch, tabular GBM) | [`TrainingJobRunner`](agent_evolve/training/runner_protocol.py) | `TRAINING_JOB_RUNNERS` (dotted-path) |
| Add a new pipeline stage type (new `stage.type` in `train/pipeline.yaml`) | `@register_stage("<name>")` on a function taking `StageContext → StageResult` | [`stage_registry.py`](agent_evolve/training/stage_registry.py) |
| Add a new fine-tuning surface (DoRA, IA³, full fine-tune, QLoRA, prefix-tuning, custom head) | [`ModelAdapter`](agent_evolve/backends/tinkerlite/adapters/base.py) | `@register_adapter("<kind>")` |
| Add a new way to generate training rows (self-instruct, OOD augment, rule-perturb, scraper) | [`DataGenerator`](agent_evolve/training/data/generator.py) | `@register_data_generator("<name>")` |
| Add a new benchmark (eval protocol, metrics, error taxonomy) | [`TrainingBenchmarkAdapter`](agent_evolve/benchmarks/training_base.py) | `TRAINING_BENCHMARKS` (dotted-path) |
| Add a new compute target (ECS, AWS Batch, Slurm) | [`ComputeTarget`](agent_evolve/backends/tinkerlite/elastic/compute_target.py) | passed explicitly into `K8sTinkerLiteBackend(targets=[...])` |
| Add a new search algorithm | class with `run_cycle(ctx) → MCGSCycleReport` | `TRAINING_ALGORITHMS` (dotted-path) |

Most integrations only need **one** of these.

---

## 1. `TrainingJobRunner` — completely non-LLM job

Minimum viable sklearn runner:

```python
# myproj/sklearn_runner.py
from pathlib import Path
from agent_evolve.training.runner_protocol import TrainingJobRunner
from agent_evolve.training.types import (
    CheckpointRef, EvalMetrics, TrainingTrialResult, ValidityReport,
)


class SklearnJobRunner:
    name = "sklearn_tabular"

    def run_trial(self, workspace, node, budget, benchmark):
        import joblib, sklearn.ensemble, pandas as pd

        # 1. Read hyperparameters from the (forked, per-candidate) workspace.
        cfg = workspace.read_yaml("train/sklearn.yaml")
        train_df = pd.read_csv(Path(workspace.root) / cfg["train_csv"])

        # 2. Fit.
        model = sklearn.ensemble.GradientBoostingRegressor(**cfg["params"])
        model.fit(train_df[cfg["features"]], train_df[cfg["target"]])

        # 3. Save a CheckpointRef — anything serializable is fine.
        outdir = Path(workspace.root) / "checkpoints" / node.node_id
        outdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, outdir / "model.pkl")
        ckpt = CheckpointRef(
            name=node.node_id, path=str(outdir),
            kind="full_weights", metadata={"framework": "sklearn"},
        )

        # 4. Eval — call your benchmark (contract in §5 below).
        metrics, buckets = benchmark.score_sklearn(workspace, model)

        return TrainingTrialResult(
            node_id=node.node_id,
            workspace_path=str(workspace.root),
            status="success",
            checkpoint=ckpt,
            eval_metrics=metrics,
            error_buckets=buckets,
            validity=ValidityReport(is_valid=True),
        )
```

Register the dotted path in [`training/registries.py`](agent_evolve/training/registries.py):

```python
TRAINING_JOB_RUNNERS["sklearn_tabular"] = "myproj.sklearn_runner.SklearnJobRunner"
```

Use it:

```python
import agent_evolve as ae
evolver = ae.TrainingEvolver(
    workspace="seed_workspaces/my_tabular_task",
    benchmark="my_tabular_benchmark",
    algorithm="mcgs",
    backend="sklearn_tabular",    # <- your runner
)
evolver.run(cycles=10)
```

**What you do NOT need to touch:** `training/loop.py`, `training/trial.py`,
`training/algorithms/`, `training/workspace.py`, `training/types.py`,
`backends/tinkerlite/`.

---

## 2. `StageWorker` — new pipeline stage

Use this when you're extending the LLM pipeline (`train/pipeline.yaml`) with
a new `stage.type`. If your work is one-shot (sklearn-style), use §1
instead — stages are for composable, reorderable steps.

```python
# myproj/stages/rule_perturb.py
from agent_evolve.training.stage_registry import (
    register_stage, StageContext, StageResult,
)


@register_stage("rule_perturb")
def _drive_rule_perturb(ctx: StageContext) -> StageResult:
    # ctx fields you can read:
    #   ctx.workspace, ctx.stage (dict from pipeline.yaml),
    #   ctx.benchmark, ctx.budget_seconds, ctx.smoke, ctx.last_ckpt,
    #   ctx.optimizer, ctx.training_client_fn, ctx.sampling_client_fn,
    #   ctx.close_training_client_fn
    rows = do_the_work(ctx.workspace, ctx.stage, ctx.benchmark, ctx.smoke)
    out_path = write_rows(ctx.workspace, rows, ctx.stage["name"])
    return StageResult(
        checkpoint=None,                               # no new checkpoint produced
        metrics={"out_path": str(out_path), "rows_written": len(rows)},
    )
```

Wire it into `pipeline.yaml`:

```yaml
stages:
  - name: perturb_v1
    type: rule_perturb
    enabled: true
    # any other keys are passed through as ctx.stage
```

**Plugin registration** — the decorator only runs when the module is
imported. Put the import somewhere that fires before `backend.run_trial`:
either in your benchmark's `__init__.py` or in the seed workspace's top
of `data/renderer.py`, or call `importlib.import_module("myproj.stages.rule_perturb")`
from your driver.

---

## 3. `ModelAdapter` — new fine-tuning surface

Use this when your training job is still LLM-shaped but with a different
trainable surface than LoRA. Ships with `LoRAAdapter` registered under
`"lora"`. Examples of things you might add:

- `"full"` — unfreeze all base params (no adapter).
- `"dora"` — PEFT LoRA with `use_dora=True`.
- `"qlora"` — `BitsAndBytesConfig(load_in_4bit=True)` + LoRA.
- `"ia3"`, `"prefix"`, `"soft_prompt"` — other PEFT surfaces.
- `"custom_head"` — freeze base, train a classification/regression head.

```python
# myproj/adapters/dora.py
from pathlib import Path
from agent_evolve.backends.tinkerlite.adapters import (
    register_adapter, ATTACH_MODE_WRAP,
)
from agent_evolve.training.types import CheckpointRef


@register_adapter("dora")
class DoRAAdapter:
    kind = "dora"
    attach_mode = ATTACH_MODE_WRAP

    def attach(self, base_model, cfg: dict):
        from peft import LoraConfig, get_peft_model
        return get_peft_model(
            base_model,
            LoraConfig(
                r=int(cfg.get("rank", 16)),
                lora_alpha=int(cfg.get("alpha", 32)),
                lora_dropout=float(cfg.get("dropout", 0.05)),
                use_dora=True,
                target_modules=list(cfg.get("target_modules", ["q_proj", "k_proj"])),
                bias="none", task_type="CAUSAL_LM",
            ),
        )

    def save(self, model, tokenizer, outdir: Path) -> CheckpointRef:
        outdir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(outdir))
        tokenizer.save_pretrained(str(outdir))
        return CheckpointRef(
            name=outdir.name, path=str(outdir), kind="adapter",
        )

    def vllm_lora_request(self, checkpoint: CheckpointRef):
        from vllm.lora.request import LoRARequest
        return LoRARequest(checkpoint.name or "candidate", 1, checkpoint.path)
```

Set `model/adapter.yaml::type: dora` in your seed workspace.

> **Current status:** today's `HFTrainingClient` and `ddp_worker.py` still
> inline the LoRA call directly rather than delegating through
> `resolve_adapter(cfg.type)`. The Protocol + `LoRAAdapter` are shipped as
> the reserved extension point; a follow-up PR will make those two files
> call `resolve_adapter(...)` so your custom adapter actually gets invoked.

---

## 4. `DataGenerator` — new data-generation method

Use this when you want a new way to produce training rows. The stage
worker `runners/stages/generate.py` drives any registered `DataGenerator`
uniformly — you don't write a new stage worker, just the generator class.

```python
# myproj/generators/rule_perturb.py
from agent_evolve.training.data import (
    register_data_generator, GeneratedRow, DataRecipe,
)


@register_data_generator("rule_perturb")
class RulePerturbGenerator:
    name = "rule_perturb"

    def generate(self, workspace, recipe: DataRecipe, *,
                 benchmark=None, budget_seconds=None, smoke=False):
        for row in benchmark.iter_training_rows(workspace):
            for perturbation in self._perturb(row):
                yield GeneratedRow(
                    id=f"{row.id}-p{perturbation.idx}",
                    prompt=perturbation.prompt,
                    answer=row.answer,
                    category=row.category,
                    cot=perturbation.cot,
                    source="rule_perturb",
                )
```

Pipeline YAML:

```yaml
stages:
  - name: perturb_v1
    type: generate              # unified dispatcher
    generator: rule_perturb     # ← your registered name
    enabled: true
```

The stage worker writes `rows.jsonl` + `stats.json` under
`data/generated/<stage-name>/`. Downstream `data_merge` picks it up and
appends to `data/sources.yaml`.

---

## 5. `TrainingBenchmarkAdapter` — new benchmark

A benchmark owns: primary metric, eval protocol, metric parsing, error
taxonomy, validity. It does NOT own reward or incumbent selection (that's
the algorithm). See [`benchmarks/training_base.py`](agent_evolve/benchmarks/training_base.py).

### Required methods

| Method | Purpose |
|---|---|
| `name: str` | class attribute; matches registry key |
| `primary_metric() → MetricSpec` | primary metric name + direction |
| `build_eval_plan(workspace, checkpoint, split) → EvalPlan` | how to set up eval |
| `evaluate(workspace, checkpoint, backend, split)` | run eval, return result dir |
| `parse_metrics(result_dir) → EvalMetrics` | read metrics from disk |
| `analyze_errors(result_dir, metrics) → ErrorBuckets` | categorize failures |
| `check_validity(workspace, trial_result) → ValidityReport` | hard-fail trials |

### Optional methods (only implement if you use the corresponding stage)

| Method | Used by | What it does |
|---|---|---|
| `iter_training_rows(workspace)` | any data-gen stage | yields `TrainingExample` |
| `classify_category(prompt)` | `solver_distill`, mutators | regex/keyword bucketing |
| `solvers()`, `verifiers()`, `cot_renderers()` | `solver_distill` | per-category implementations |
| `data_synth_generators()` | future `ood_augment` | per-category OOD gens |
| `build_eval_prompt(row, tokenizer)` | `eval`, `teacher_distill`, `rl` | chat-templated prompt string |
| `extract_final_answer(text)` | `teacher_distill`, `rl` | parse answer from completion |
| `verify(pred, gt)` | `teacher_distill`, `rl` | correctness check |
| `load_distill_prompts(workspace, stage)` | `teacher_distill` | replaces hard-coded CSV paths |
| `load_rl_prompts(workspace, stage)` | `rl` | replaces hard-coded CSV paths |

Legacy fallback: today `teacher_distill` and `rl` still `import ... from
benchmarks.nemo_reasoner` directly. If you're adding a new benchmark and
you want to use these stages, route through
[`benchmarks/helpers.py`](agent_evolve/benchmarks/helpers.py) — it prefers
`benchmark.<method>` and falls back to `nemo_reasoner` only when absent.
Over time those stages will stop importing `nemo_reasoner` entirely.

### For a completely non-LLM benchmark

Implement whatever your `TrainingJobRunner` calls (e.g. `score_sklearn`).
The Protocol is structural — just define the methods your runner uses and
ignore the LLM-specific ones.

Register:

```python
TRAINING_BENCHMARKS["my_tabular_benchmark"] = "myproj.benchmark.MyTabularBenchmark"
```

---

## 6. Seed workspace layout

A seed workspace is the evolvable "training DNA" on disk. MCGS forks it
per candidate, mutators patch fields, and the backend reads it.

### Required files (validator enforces — see [`training/schema.py`](agent_evolve/training/schema.py))

```
seed_workspaces/<name>/
├── manifest.yaml                contract_version: train-1.0; name; layer lists
├── model/
│   ├── base.yaml                name + absolute path + dtype + architecture
│   └── adapter.yaml             type: lora|full|qlora|dora|... + knobs
├── data/
│   ├── sources.yaml             list of {path, split, format} entries
│   └── mix.yaml                 buckets + ratios
├── train/
│   └── pipeline.yaml            ordered [{name, type, enabled, ...}]
└── eval/
    ├── local_splits.yaml        protected split config
    └── error_taxonomy.yaml      error-bucket ids + descriptions
```

### Recommended (read by workers; not validator-enforced)

```
├── train/
│   ├── optimizer.yaml           lr, warmup_ratio, weight_decay
│   ├── batching.yaml            per_device_bs, grad_accum, max_seq_len
│   └── loss.yaml                informational: sft + rl loss names
├── eval/
│   ├── local_holdout_small.jsonl  smoke-path eval rows
│   └── kaggle_eval.yaml         presence → real vLLM path (LLM benchmarks)
├── data/
│   ├── recipes/<name>.yaml      data-recipe (evolvable)
│   ├── generators/*.yaml        per-generator configs (e.g. teacher_llm.yaml)
│   ├── filters/                 per-category filters
│   ├── curriculum.yaml          optional schedule
│   └── renderer.py              custom dataset renderer (else fallback)
├── rl/                          only if pipeline uses rl stages
│   ├── rollout.yaml
│   ├── reward.py
│   └── advantage.py
├── memory/runs.jsonl            bootstrap empty
├── checkpoints/registry.jsonl   bootstrap empty
└── evolution/.gitkeep
```

### `manifest.yaml` template (copy from
[nemotron_reasoner](seed_workspaces/nemotron_reasoner/manifest.yaml))

```yaml
name: <your_task>
contract_version: train-1.0

defaults:
  benchmark: <your_benchmark_registry_key>
  algorithm: mcgs
  backend: h200_single_node             # or your TrainingJobRunner key

evolvable_layers:     # MCGS mutators may patch any of these
  - data/mix.yaml
  - data/recipes
  - train/pipeline.yaml
  - train/optimizer.yaml
  - train/batching.yaml
  # add any workspace-specific evolvable files

protected_layers:     # workspace.fork refuses patches touching these
  - model/base.yaml
  - eval/local_splits.yaml
  - eval/private_holdout.jsonl
  - data/raw

artifact_layers:      # excluded from fingerprint; workers write freely
  - memory
  - checkpoints
  - evolution
  - data/generated
```

### Non-LLM workspaces

The validator only checks *file existence*, not content. A sklearn /
tabular workspace can stub:

```yaml
# model/base.yaml — content is opaque to non-LLM runners
name: sklearn-placeholder
path: /dev/null
dtype: float64
architecture: sklearn

# model/adapter.yaml — irrelevant for sklearn; your runner ignores this
type: full
```

And swap `train/pipeline.yaml` to match your stages:

```yaml
stages:
  - name: sklearn_fit
    type: sklearn_fit     # ← your @register_stage("sklearn_fit")
    enabled: true
```

---

## 7. Registries — where to list your component

| Component | Where to register | Method |
|---|---|---|
| Benchmark | [`training/registries.py::TRAINING_BENCHMARKS`](agent_evolve/training/registries.py) | add dotted-path entry |
| Algorithm | `training/registries.py::TRAINING_ALGORITHMS` | add dotted-path entry |
| `TrainingJobRunner` (backend) | `training/registries.py::TRAINING_JOB_RUNNERS` | add dotted-path entry (alias `TRAINING_BACKENDS` works too) |
| Stage worker | `@register_stage(stype)` decorator | import the module to trigger registration |
| Model adapter | `@register_adapter(kind)` decorator | import to trigger |
| Data generator | `@register_data_generator(name)` decorator | import to trigger |
| Compute target | `K8sTinkerLiteBackend(targets=[...])` | passed explicitly, no global registry |

Decorator-based registrations happen **at import time**. If your plugin
lives outside `agent_evolve/`, import it from:
- your benchmark's `__init__.py` (fires when `TRAINING_BENCHMARKS` resolves it), or
- your driver script before constructing `TrainingEvolver`, or
- a `conftest.py` if it's only needed for tests.

---

## 8. What NOT to edit

Core training-loop files are stable — don't fork them:

- `training/api.py`, `training/loop.py`, `training/trial.py`, `training/observer.py`
- `training/algorithms/mcgs/*.py`
- `training/types.py`, `training/workspace.py`, `training/schema.py`

If you think you need to edit one of these, **file an issue first** — it
usually means a new Protocol or registry is missing, and we should add it
instead of forking.

---

## 9. Testing your integration

```bash
# 1. Protocol conformance — copy+adapt the relevant test file:
pytest tests/training/test_job_runner_protocol.py    # for TrainingJobRunner
pytest tests/training/test_stage_registry.py         # for new stage types
pytest tests/backends/tinkerlite/test_adapter_registry.py   # for ModelAdapter
pytest tests/training/data/test_data_generator_registry.py  # for DataGenerator
pytest tests/training/test_benchmark_adapter_hooks.py       # for benchmark hooks

# 2. End-to-end smoke — copy from:
pytest tests/training/test_seed_nemotron_smoke_cycle.py

# 3. CLI drive — smoke run:
python -m agent_evolve.training.run \
    --workspace seed_workspaces/<your_workspace> \
    --benchmark <your_benchmark_key> \
    --backend <your_runner_key> \
    --smoke --cycles 1
```

---

## 10. File references (cheat sheet)

- **TrainingJobRunner Protocol:** [agent_evolve/training/runner_protocol.py](agent_evolve/training/runner_protocol.py)
- **Stage registry:** [agent_evolve/training/stage_registry.py](agent_evolve/training/stage_registry.py)
- **ModelAdapter Protocol:** [agent_evolve/backends/tinkerlite/adapters/base.py](agent_evolve/backends/tinkerlite/adapters/base.py) + [lora.py](agent_evolve/backends/tinkerlite/adapters/lora.py)
- **DataGenerator Protocol:** [agent_evolve/training/data/generator.py](agent_evolve/training/data/generator.py) + [generators/](agent_evolve/training/data/generators/)
- **Benchmark Protocol:** [agent_evolve/benchmarks/training_base.py](agent_evolve/benchmarks/training_base.py) + cross-benchmark [helpers.py](agent_evolve/benchmarks/helpers.py)
- **Registries (dotted-path):** [agent_evolve/training/registries.py](agent_evolve/training/registries.py)
- **Workspace schema validator:** [agent_evolve/training/schema.py](agent_evolve/training/schema.py)
- **Reference seed workspace:** [seed_workspaces/nemotron_reasoner/](seed_workspaces/nemotron_reasoner/)
- **Reference LLM backend:** [agent_evolve/backends/tinkerlite/single_node/backend.py](agent_evolve/backends/tinkerlite/single_node/backend.py)

---

## 11. FAQ

**Q: I want my TrainingJobRunner to also use MCGS fan-out across a k8s cluster. Do I need to implement `TinkerLiteBackend`?**
A: No. `TinkerLiteBackend` is the LLM-specific Protocol that adds HF/vLLM
client factories. For a generic runner on k8s, either (a) subclass
`K8sTinkerLiteBackend` and override `run_trial` to call your sklearn
logic, or (b) implement `TrainingJobRunner` and write a custom
`ComputeTarget` that runs your job. Option (b) is cleaner if you don't
reuse any LLM plumbing.

**Q: Can I mix legacy `type: solver_distill` and new `type: generate, generator: solver_distill` in the same pipeline?**
A: Yes. Both are registered as separate stage types. Use whichever suits
the YAML you're authoring; the new form is cheaper to extend to new
generators.

**Q: I added `@register_stage("my_stage")` but the backend raises `Unknown stage type`.**
A: The decorator only fires when the module is imported. Import it from
your benchmark's `__init__.py` or from your driver, before `evolver.run()`.

**Q: Can I use a non-PEFT adapter library (e.g. unsloth)?**
A: Yes — implement `ModelAdapter.attach` to return whatever wrapped model
your library produces. `save` and `vllm_lora_request` tell a-evolve how
to persist + load it.

**Q: My benchmark has no training rows (e.g. synthetic rule generation only). Do I have to implement `iter_training_rows`?**
A: No. Methods in the "Optional" table in §5 are only consulted when the
corresponding stage runs. If you skip those stages in `pipeline.yaml`,
skip the methods.
