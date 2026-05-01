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
| Swap the entire training framework (sklearn, JAX, PyTorch-from-scratch, tabular GBM) | [`TrainingJobRunner`](agent_evolve/model/runner_protocol.py) | `TRAINING_JOB_RUNNERS` (dotted-path) |
| Add a new pipeline stage type (new `stage.type` in `train/pipeline.yaml`) | `@register_stage("<name>")` on a function taking `StageContext → StageResult` | [`stage_registry.py`](agent_evolve/model/stage_registry.py) |
| Add a new fine-tuning surface (DoRA, IA³, full fine-tune, QLoRA, prefix-tuning, custom head) | [`ModelAdapter`](agent_evolve/backends/tinkerlite/adapters/base.py) | `@register_adapter("<kind>")` |
| Add a new way to generate training rows (self-instruct, OOD augment, rule-perturb, scraper) | [`DataGenerator`](agent_evolve/model/data/generator.py) | `@register_data_generator("<name>")` |
| Add a new benchmark (eval protocol, metrics, error taxonomy) | [`TrainingBenchmarkAdapter`](agent_evolve/benchmarks/training_base.py) | `TRAINING_BENCHMARKS` (dotted-path) |
| Add a new compute target (ECS, AWS Batch, Slurm) | [`ComputeTarget`](agent_evolve/backends/tinkerlite/elastic/compute_target.py) | passed explicitly into `K8sTinkerLiteBackend(targets=[...])` |
| Add a new search algorithm | class with `run_cycle(ctx) → MCGSCycleReport` | `TRAINING_ALGORITHMS` (dotted-path) |

Most integrations only need **one** of these.

---

## 0.5 Where does `myproj/` live?

`myproj/` in the examples below is **just a placeholder name** for
wherever your integration code lives. There's no hard rule — the
Protocol + registry design means your class is resolved by dotted path,
and Python doesn't care whether that path is inside `agent_evolve/`, in
a sibling package, or in a separate pip-installable repo.

The project is still actively filling in built-in backends / benchmarks
/ adapters, so **directly adding your code under `agent_evolve/` is
normal and often the right call** — especially for things that will
eventually live in-tree anyway (new backend variants, new benchmarks,
new adapter kinds).

### In-tree (during active development)

Natural homes alongside their peers:

| Extension | Natural in-tree home |
|---|---|
| `TrainingJobRunner` / backend | `agent_evolve/backends/<name>/` (same shape as `tinkerlite/`) |
| Pipeline stage worker | `agent_evolve/model/runners/stages/<name>.py` |
| `ModelAdapter` | `agent_evolve/backends/tinkerlite/adapters/<name>.py` |
| `DataGenerator` | `agent_evolve/model/data/generators/<name>.py` |
| Benchmark | `agent_evolve/benchmarks/<name>.py` |
| Compute target | `agent_evolve/backends/tinkerlite/elastic/targets/<name>.py` |

If you go this route, also add the dotted-path entry to
[`training/registries.py`](agent_evolve/model/registries.py) so the
string key resolves.

### Out-of-tree (when you want isolation)

Works identically — just drop a package at the repo root (sibling to
`agent_evolve/`) or install it via pip. Because `agent_evolve` only
knows about your class via a dotted path string, the location is
orthogonal:

```text
a-evolve/
├── agent_evolve/              # core
├── seed_workspaces/
└── myproj/                    # your package (sibling, importable as myproj)
    ├── __init__.py
    ├── sklearn_runner.py
    └── ...
```

Then in `myproj/__init__.py`, mutate the registry at import time so no
edits to `agent_evolve/` are needed:

```python
# myproj/__init__.py
from agent_evolve.model.registries import (
    TRAINING_JOB_RUNNERS, TRAINING_BENCHMARKS,
)
TRAINING_JOB_RUNNERS["sklearn_tabular"] = "myproj.sklearn_runner.SklearnJobRunner"
TRAINING_BENCHMARKS["my_tabular_benchmark"] = "myproj.benchmark.MyTabularBenchmark"

# Decorator-based registries (stages, adapters, generators):
from .stages import rule_perturb       # noqa: F401
from .adapters import dora             # noqa: F401
from .generators import rule_perturb   # noqa: F401
```

Import `myproj` once before `evolver.run()` and every registry is
populated.

### Rule of thumb

- Will this eventually land in-tree (new backend / benchmark / adapter
  that belongs to the project)? → put it under `agent_evolve/`.
- Is it truly personal / experimental / project-specific? → sibling
  package or installable.
- Either way, the Protocol shape and registry call are the same. You
  can start in-tree and factor out later with no code changes.

The only directory that's **off-limits** for integrations is
`agent_evolve/model/algorithms/mcgs/` and the core training-loop
files (§7) — those are stable contracts.

---

## 1. `TrainingJobRunner` — completely non-LLM job

For a non-LLM job (sklearn, JAX, XGBoost, tabular GBM, …) you bypass the
`tinkerlite` backend entirely. A runner just has to take a forked
workspace, train something, evaluate it, and return a `TrainingTrialResult`.

### What to create, where, and why

Concrete sklearn example. Four layers, four file sets — nothing else in
the core needs to change.

#### (a) Runner — your training logic

```text
myproj/
├── __init__.py
└── sklearn_runner.py          ← implements TrainingJobRunner
```

```python
# myproj/sklearn_runner.py
from pathlib import Path
from agent_evolve.model.types import (
    CheckpointRef, TrainingTrialResult, ValidityReport,
)


class SklearnJobRunner:
    name = "sklearn_tabular"

    def run_trial(self, workspace, node, budget, benchmark):
        import joblib, sklearn.ensemble, pandas as pd

        cfg = workspace.read_yaml("train/sklearn.yaml")
        train_df = pd.read_csv(Path(workspace.root) / cfg["train_csv"])

        model = sklearn.ensemble.GradientBoostingRegressor(**cfg["params"])
        model.fit(train_df[cfg["features"]], train_df[cfg["target"]])

        outdir = Path(workspace.root) / "checkpoints" / node.node_id
        outdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, outdir / "model.pkl")
        ckpt = CheckpointRef(
            name=node.node_id, path=str(outdir),
            kind="full_weights", metadata={"framework": "sklearn"},
        )

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

#### (b) Benchmark — eval semantics for your task

```text
myproj/
└── benchmark.py               ← implements TrainingBenchmarkAdapter
```

Only the methods your runner actually calls need to exist (the Protocol
is structural). For the sklearn example above, `score_sklearn` is enough:

```python
# myproj/benchmark.py
from agent_evolve.types import EvalMetrics, ErrorBuckets, ValidityReport

class MyTabularBenchmark:
    name = "my_tabular_benchmark"

    def primary_metric(self):                         # used by MCGS
        from agent_evolve.model.types import MetricSpec
        return MetricSpec(name="rmse", direction="minimize")

    def score_sklearn(self, workspace, model):        # your runner calls this
        # run model on held-out set, return (EvalMetrics, ErrorBuckets)
        ...

    def check_validity(self, workspace, trial_result):
        return ValidityReport(is_valid=True)
```

#### (c) Registry — make the strings resolvable

Edit [agent_evolve/model/registries.py](agent_evolve/model/registries.py)
(or, if you're keeping everything out-of-tree, add the entries at import
time in your `myproj/__init__.py`):

```python
from agent_evolve.model.registries import (
    TRAINING_JOB_RUNNERS, TRAINING_BENCHMARKS,
)

TRAINING_JOB_RUNNERS["sklearn_tabular"] = "myproj.sklearn_runner.SklearnJobRunner"
TRAINING_BENCHMARKS["my_tabular_benchmark"] = "myproj.benchmark.MyTabularBenchmark"
```

Algorithm — no change; `"mcgs"` already registered. You only add a row
to `TRAINING_ALGORITHMS` if you write a custom search algorithm (§0).

#### (d) Workspace — the evolvable DNA on disk

```text
seed_workspaces/my_tabular_task/
├── manifest.yaml              ← contract_version + layer lists (required)
├── model/
│   ├── base.yaml              ← required by schema; placeholder OK
│   └── adapter.yaml           ← required by schema; placeholder OK
├── data/
│   ├── sources.yaml           ← required; list of {path, split, format}
│   ├── mix.yaml               ← required; can be empty dict for tabular
│   └── train.csv              ← your actual data
├── train/
│   ├── pipeline.yaml          ← required; even if a single fit step
│   └── sklearn.yaml           ← your runner's hyperparameters (evolvable)
└── eval/
    ├── local_splits.yaml      ← required (protected)
    ├── error_taxonomy.yaml    ← required; bucket ids used by benchmark
    └── holdout.csv            ← your eval split
```

Minimum `manifest.yaml`:

```yaml
name: my_tabular_task
contract_version: train-1.0

defaults:
  benchmark: my_tabular_benchmark
  algorithm: mcgs
  backend: sklearn_tabular

evolvable_layers:
  - train/sklearn.yaml         # MCGS mutates hyperparameters here
  - data/mix.yaml

protected_layers:
  - model/base.yaml
  - eval/local_splits.yaml
  - eval/holdout.csv

artifact_layers:
  - memory
  - checkpoints
  - evolution
```

`model/base.yaml` + `model/adapter.yaml` can be stubs (`name: placeholder`,
`type: full`) — the schema validator only checks file existence, not
content. Your runner ignores them.

#### Run it

```python
import agent_evolve as ae
import myproj   # fires the registry inserts in (c)

evolver = ae.TrainingEvolver(
    workspace="seed_workspaces/my_tabular_task",
    benchmark="my_tabular_benchmark",
    algorithm="mcgs",
    backend="sklearn_tabular",
)
evolver.run(cycles=10)
```

### Summary — per-layer responsibilities

| Layer | File(s) | Owns |
|---|---|---|
| **Runner** | `myproj/sklearn_runner.py` | `run_trial(workspace, node, budget, benchmark) → TrainingTrialResult`. Trains, writes checkpoint, asks benchmark to score |
| **Benchmark** | `myproj/benchmark.py` | `primary_metric()`, `check_validity()`, plus whatever scoring method your runner calls |
| **Algorithm** | — (reuse `"mcgs"`) | Only write one if you need custom search; default MCGS is task-agnostic |
| **Workspace** | `seed_workspaces/my_tabular_task/` | On-disk, forked per candidate. `train/sklearn.yaml` is the evolvable surface MCGS mutates |
| **Registry** | `agent_evolve/model/registries.py` *or* `myproj/__init__.py` | String-key → dotted-path for runner + benchmark |

---

## 2. `StageWorker` — new pipeline stage

Use this when you're extending the LLM pipeline (`train/pipeline.yaml`)
with a new `stage.type`. Existing runner + backend + benchmark are reused
— your stage is just a new composable step inside the pipeline
dispatcher.

If your work is one-shot (sklearn-style), use §1 instead — stages are
for composable, reorderable steps inside the tinkerlite pipeline.

#### (a) Stage worker — your code

```text
myproj/
└── stages/
    └── rule_perturb.py        ← implements StageWorker
```

```python
# myproj/stages/rule_perturb.py
from agent_evolve.model.stage_registry import (
    register_stage, StageContext, StageResult,
)


@register_stage("rule_perturb")
def _drive_rule_perturb(ctx: StageContext) -> StageResult:
    rows = do_the_work(ctx.workspace, ctx.stage, ctx.benchmark, ctx.smoke)
    out_path = write_rows(ctx.workspace, rows, ctx.stage["name"])
    return StageResult(
        checkpoint=None,
        metrics={"out_path": str(out_path), "rows_written": len(rows)},
    )
```

`StageContext` fields (workspace, benchmark, last_ckpt, training_client_fn,
sampling_client_fn, budget_seconds, smoke, …) are defined in
[stage_registry.py](agent_evolve/model/stage_registry.py).

#### (b) Benchmark — only if your stage needs task-specific semantics

Untouched for generic stages. If your stage calls e.g.
`ctx.benchmark.iter_training_rows(...)` or a custom method on the
benchmark, either reuse an existing method or add one to your benchmark
(per §5).

#### (c) Registry — `@register_stage` runs at import time

`@register_stage("rule_perturb")` mutates the global `STAGE_WORKERS` dict.
**The decorator fires only when the module is imported.** Trigger the
import in one of:

- `myproj/__init__.py` → `from .stages import rule_perturb  # noqa`
- your driver script, before `evolver.run()`
- your benchmark's `__init__.py` (if the stage is benchmark-specific)

#### (d) Workspace — add the stage to `train/pipeline.yaml`

```yaml
# seed_workspaces/<your_workspace>/train/pipeline.yaml
stages:
  - name: perturb_v1
    type: rule_perturb          # ← your registered name
    enabled: true
    # any other keys are passed through as ctx.stage[...]
```

Nothing else in the workspace changes. The stage reads `ctx.stage[...]`
for its own config and writes outputs under `data/generated/<stage-name>/`
(artifact layer, auto-created).

### Summary — per-layer responsibilities

| Layer | File(s) | Changes |
|---|---|---|
| **Runner (stage worker)** | `myproj/stages/rule_perturb.py` | New function + `@register_stage` |
| **Benchmark** | — | Untouched (unless your stage calls a custom benchmark method) |
| **Registry** | `myproj/__init__.py` or driver | Must import the stage module before `evolver.run()` |
| **Workspace** | `train/pipeline.yaml` | One new entry under `stages:` |

---

## 3. `ModelAdapter` — new fine-tuning surface

Use this when your training job is still LLM-shaped but with a different
trainable surface than LoRA (DoRA, IA³, QLoRA, full fine-tune, prefix,
custom classification head). Ships with `LoRAAdapter` registered under
`"lora"`.

#### (a) Adapter — your wrap/save/load logic

```text
myproj/
└── adapters/
    └── dora.py                ← implements ModelAdapter
```

```python
# myproj/adapters/dora.py
from pathlib import Path
from agent_evolve.backends.tinkerlite.adapters import (
    register_adapter, ATTACH_MODE_WRAP,
)
from agent_evolve.model.types import CheckpointRef


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

#### (b) Backend — untouched

`HFTrainingClient` / `ddp_worker.py` resolve the adapter by `kind` at
runtime via `resolve_adapter(cfg.type)`. No backend code changes.

#### (c) Registry — `@register_adapter` runs at import time

Trigger the import from `myproj/__init__.py`:

```python
# myproj/__init__.py
from .adapters import dora  # noqa: F401  — fires @register_adapter("dora")
```

#### (d) Workspace — flip `model/adapter.yaml`

```yaml
# seed_workspaces/<your_workspace>/model/adapter.yaml
type: dora                    # ← matches @register_adapter("dora")
rank: 16
alpha: 32
dropout: 0.05
target_modules: [q_proj, k_proj]
```

Everything else in the workspace is unchanged — this is a pure swap of
the trainable surface.

### Summary — per-layer responsibilities

| Layer | File(s) | Changes |
|---|---|---|
| **Runner** | — | Untouched (stage workers resolve adapter by kind) |
| **Backend** | — | Untouched (`resolve_adapter` dispatches) |
| **Adapter** | `myproj/adapters/dora.py` | New `@register_adapter` class |
| **Registry** | `myproj/__init__.py` | Import the adapter module |
| **Workspace** | `model/adapter.yaml` | `type: <kind>` + any adapter-specific knobs |

---

## 4. `DataGenerator` — new data-generation method

Use this when you want a new way to produce training rows (self-instruct,
OOD augment, rule-perturb, scraper, …). The stage worker
`runners/stages/generate.py` drives any registered `DataGenerator`
uniformly — you write the generator class, not a new stage worker.

#### (a) Generator — your row-producing logic

```text
myproj/
└── generators/
    └── rule_perturb.py        ← implements DataGenerator
```

```python
# myproj/generators/rule_perturb.py
from agent_evolve.model.data import (
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

#### (b) Benchmark — `iter_training_rows` must exist

The generator above pulls seed rows via `benchmark.iter_training_rows(workspace)`.
Your benchmark (§5) must implement that method. Nothing else on the
benchmark needs to change.

#### (c) Registry — `@register_data_generator` runs at import time

```python
# myproj/__init__.py
from .generators import rule_perturb  # noqa: F401
```

#### (d) Workspace — a `type: generate` stage in the pipeline

```yaml
# seed_workspaces/<your_workspace>/train/pipeline.yaml
stages:
  - name: perturb_v1
    type: generate              # unified dispatcher (not a new stage type)
    generator: rule_perturb     # ← your registered DataGenerator name
    enabled: true
  - name: merge_all
    type: data_merge
    inputs: [data/generated/perturb_v1]
    enabled: true
```

The `generate` stage worker writes `rows.jsonl` + `stats.json` under
`data/generated/<stage-name>/` (artifact layer). Downstream `data_merge`
picks it up and appends to `data/sources.yaml`, which the `sft` stage
then consumes.

### Summary — per-layer responsibilities

| Layer | File(s) | Changes |
|---|---|---|
| **Runner (stage worker)** | — | Reuses the built-in `generate` dispatcher |
| **Generator** | `myproj/generators/rule_perturb.py` | New `@register_data_generator` class |
| **Benchmark** | your `benchmark.py` | Must implement `iter_training_rows` (if not already) |
| **Registry** | `myproj/__init__.py` | Import the generator module |
| **Workspace** | `train/pipeline.yaml` | Add a `type: generate, generator: <name>` stage |

---

## 5. `TrainingBenchmarkAdapter` — new benchmark

A benchmark owns: primary metric, eval protocol, metric parsing, error
taxonomy, validity. It does NOT own reward or incumbent selection (that's
the algorithm). See [`benchmarks/training_base.py`](agent_evolve/benchmarks/training_base.py).

#### (a) Benchmark class — eval semantics

```text
myproj/
└── benchmark.py               ← implements TrainingBenchmarkAdapter
```

**Required methods** (all benchmarks):

| Method | Purpose |
|---|---|
| `name: str` | class attribute; matches registry key |
| `primary_metric() → MetricSpec` | primary metric name + direction |
| `build_eval_plan(workspace, checkpoint, split) → EvalPlan` | how to set up eval |
| `evaluate(workspace, checkpoint, backend, split)` | run eval, return result dir |
| `parse_metrics(result_dir) → EvalMetrics` | read metrics from disk |
| `analyze_errors(result_dir, metrics) → ErrorBuckets` | categorize failures |
| `check_validity(workspace, trial_result) → ValidityReport` | hard-fail trials |

**Optional methods** (only if you use the corresponding stage):

| Method | Used by |
|---|---|
| `iter_training_rows(workspace)` | any data-gen stage |
| `classify_category(prompt)` | `solver_distill`, mutators |
| `solvers()`, `verifiers()`, `cot_renderers()` | `solver_distill` |
| `data_synth_generators()` | future `ood_augment` |
| `build_eval_prompt(row, tokenizer)` | `eval`, `teacher_distill`, `rl` |
| `extract_final_answer(text)`, `verify(pred, gt)` | `teacher_distill`, `rl` |
| `load_distill_prompts(ws, stage)` / `load_rl_prompts(ws, stage)` | `teacher_distill` / `rl` |

The Protocol is structural — for a non-LLM benchmark, define only the
methods your runner actually calls (e.g. `score_sklearn` in §1).

#### (b) Runner / backend — untouched

Built-in runners call benchmark methods by name; as long as the methods
they invoke exist, nothing downstream changes.

For a non-`nemo_reasoner` benchmark that uses the built-in
`teacher_distill` / `rl` stages, go through
[`benchmarks/helpers.py`](agent_evolve/benchmarks/helpers.py) — it prefers
`benchmark.<method>` and falls back to `nemo_reasoner` only when absent.

#### (c) Registry — dotted-path entry

```python
# agent_evolve/model/registries.py   (or myproj/__init__.py at import time)
TRAINING_BENCHMARKS["my_benchmark"] = "myproj.benchmark.MyBenchmark"
```

#### (d) Workspace — point to your benchmark

```yaml
# seed_workspaces/<your_workspace>/manifest.yaml
defaults:
  benchmark: my_benchmark       # ← matches registry key

# eval/error_taxonomy.yaml — required; bucket ids your analyze_errors returns
# eval/local_splits.yaml    — required; maps split names to paths
```

Workspace contents beyond these two eval files are driven by what your
stages need (e.g. `data/sources.yaml` for SFT, `eval/kaggle_eval.yaml`
for Kaggle-mode eval).

### Summary — per-layer responsibilities

| Layer | File(s) | Changes |
|---|---|---|
| **Benchmark** | `myproj/benchmark.py` | New class implementing required methods (+ optional per stage) |
| **Runner / Backend** | — | Untouched |
| **Registry** | `registries.py::TRAINING_BENCHMARKS` | Add `"<key>": "<dotted.path>"` |
| **Workspace** | `manifest.yaml::defaults.benchmark` + `eval/{error_taxonomy,local_splits}.yaml` | Point to new benchmark; define eval splits + error buckets |

---

## 6. Seed workspace layout

A seed workspace is the evolvable "training DNA" on disk. MCGS forks it
per candidate, mutators patch fields, and the backend reads it.

### Required files (validator enforces — see [`training/schema.py`](agent_evolve/model/schema.py))

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

### Optional (read by specific stages; not validator-enforced)

- `train/{optimizer,batching,loss}.yaml`
- `eval/{local_holdout_small.jsonl, kaggle_eval.yaml}` (the latter's
  presence activates the real vLLM path for LLM benchmarks)
- `data/{recipes,generators,filters,curriculum.yaml,renderer.py}`
- `rl/{rollout.yaml, reward.py, advantage.py}` — only if pipeline uses rl
- `memory/`, `checkpoints/`, `evolution/` — artifact layers (auto-created)

Copy `manifest.yaml` from
[seed_workspaces/nemotron_reasoner/manifest.yaml](seed_workspaces/nemotron_reasoner/manifest.yaml)
and edit `defaults`, `evolvable_layers`, `protected_layers`,
`artifact_layers` for your task.

The validator only checks *file existence*, not content — a non-LLM
workspace can stub `model/base.yaml` and `model/adapter.yaml` with
placeholder values.

---

## 7. What NOT to edit

Core training-loop files are stable — don't fork them:

- `training/api.py`, `training/loop.py`, `training/trial.py`, `training/observer.py`
- `training/algorithms/mcgs/*.py`
- `training/types.py`, `training/workspace.py`, `training/schema.py`

If you think you need to edit one of these, **file an issue first** — it
usually means a new Protocol or registry is missing, and we should add it
instead of forking.

---

## 8. FAQ

**Q: My TrainingJobRunner needs MCGS fan-out across k8s. Must I implement `TinkerLiteBackend`?**
A: No. `TinkerLiteBackend` is LLM-specific (HF/vLLM factories). Either
subclass `K8sTinkerLiteBackend` and override `run_trial`, or implement
`TrainingJobRunner` + a custom `ComputeTarget`. The latter is cleaner if
you don't reuse any LLM plumbing.

**Q: Can I mix `type: solver_distill` and `type: generate, generator: solver_distill` in the same pipeline?**
A: Yes — both are registered as separate stage types.

**Q: `@register_stage("my_stage")` but backend raises `Unknown stage type`.**
A: The decorator only fires on import. Import the module from your
benchmark's `__init__.py` or driver before `evolver.run()`.

**Q: Non-PEFT adapter library (e.g. unsloth)?**
A: Yes — `ModelAdapter.attach` can return any wrapped model; `save` +
`vllm_lora_request` tell a-evolve how to persist + load it.

**Q: My benchmark has no training rows. Do I need `iter_training_rows`?**
A: No — optional methods in §5 are only called by the stages that use
them. Skip the stage, skip the method.
