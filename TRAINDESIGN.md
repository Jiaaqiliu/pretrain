# TrainingEvolver — Design & Kaggle Run Log

## 1. What this subsystem is

`ae.TrainingEvolver(...)` is a first-class, additive path alongside `ae.Evolver(...)`. Where `Evolver` evolves **agent workspaces** (prompts / skills / tools / memory), `TrainingEvolver` evolves **training recipes** — data mix, curriculum, pipeline, optimizer, loss, reward, rollout, adapter config.

```python
import agent_evolve as ae

evolver = ae.TrainingEvolver(
    workspace="seed_workspaces/nemotron_reasoner",
    benchmark="nemo_reasoner",
    algorithm="mcgs",
    backend="h200_single_node",
)
result = evolver.run(cycles=4)
```

Each cycle: the algorithm selects a parent node → proposes a mutation (e.g. `train/optimizer.yaml::lr`) → forks the candidate workspace → backend runs an SFT training stage → backend evaluates the resulting adapter → benchmark returns metrics + error buckets + a validity report → algorithm computes reward, backprops, updates the top-k, decides whether to promote incumbent. The existing `ae.Evolver(...)` is untouched.

Four axes, one constructor kwarg each. The table below is the same four objects shown against the constructor above — same order, same names:

| Constructor kwarg | Role | Registered value in the example | Owns |
|---|---|---|---|
| `workspace=` | Training DNA on disk | `seed_workspaces/nemotron_reasoner` | Path to a directory with `manifest.yaml`, `train/`, `data/`, `model/`, `eval/`, `rl/` (see §6) |
| `algorithm=` | Search | `"mcgs"` → `MCGSSearch` | Selection, mutation, reward, backprop, promotion, memory, fusion — everything that reads metrics and proposes next moves |
| `backend=` | Execution | `"h200_single_node"` → `SingleNodeTinkerLiteBackend` (or `"k8s_h200"` → `K8sTinkerLiteBackend`) | Runs SFT/RL stages, saves adapter, loads it for eval, samples. Never scores a trajectory |
| `benchmark=` | Evaluation semantics | `"nemo_reasoner"` → `NemoReasonerBenchmark` | Primary metric, split loading, error taxonomy, validity rules, (optionally) per-category solvers/verifiers/renderers |

The algorithm never executes a forward pass. The benchmark never computes a reward or picks an incumbent. The backend never scores a trajectory. These are enforced with tests (`test_backend_no_reward.py`, `test_benchmark_no_reward.py`, `test_loop_no_gate.py`).

String values on the right are *registry keys* — `ae.TrainingEvolver` resolves them via `TRAINING_ALGORITHMS` / `TRAINING_BACKENDS` / `TRAINING_BENCHMARKS` (see [registries.py](agent_evolve/training/registries.py)). You can also pass instances directly (`algorithm=MCGSSearch(...)`, `backend=MyBackend()`) to bypass the registry.

## 2. Package layout

```text
agent_evolve/
├── training/
│   ├── api.py                    TrainingEvolver facade
│   ├── types.py                  All shared dataclasses (single source of truth)
│   ├── registries.py             Dotted-path registry for benchmarks / algorithms / job runners
│   ├── runner_protocol.py        TrainingJobRunner Protocol — root extension point
│   ├── stage_registry.py         @register_stage decorator + StageContext/StageResult
│   ├── schema.py                 Workspace structural validation
│   ├── workspace.py              Fork / fingerprint / materialize_incumbent
│   ├── loop.py                   TrainingEvolutionLoop — no accept/reject gate
│   ├── trial.py                  TrialRunner wraps backend.run_trial + validity
│   ├── observer.py               Writes evolution/reports + evolution/observations
│   ├── run.py                    CLI: python -m agent_evolve.training.run
│   ├── algorithms/
│   │   ├── null.py               No-op algorithm for scaffolding
│   │   └── mcgs/
│   │       ├── search.py         MCGSSearch.run_cycle
│   │       ├── graph.py          TrainingSearchGraph (persistent DAG)
│   │       ├── node.py           NodeStatus + summaries
│   │       ├── selection.py      UCTSelector + TopKStore (branch-diversity cap)
│   │       ├── mutation.py       BaselineMutationProposer + LRBagMutationProposer
│   │       ├── reward.py         DefaultRewardPolicy
│   │       ├── promotion.py      PromotionPolicy (incumbent + tie-breakers)
│   │       ├── memory.py         NodeMemoryStore (BM25-lite retrieval, JSONL on disk)
│   │       └── fusion.py         FusionPolicy (cross-branch recombination on stagnation)
│   ├── data/                     Benchmark-agnostic data-generation primitives
│   │   ├── base.py               Solver / Verifier / CoTRenderer / DataSynthGenerator Protocols
│   │   ├── generator.py          DataGenerator Protocol + registry
│   │   ├── generators/           Built-in DataGenerator impls (solver_distill, teacher_llm)
│   │   ├── recipe.py             DataRecipe YAML loader + validator (evolvable surface)
│   │   ├── dedup.py              Deterministic dedup key + per-source upsample
│   │   ├── cot_template.py       Force \boxed{answer} + inject [verify]: PASS, idempotent
│   │   └── verifier_gate.py     Shared correctness filter (answer + marker + token cap)
│   └── runners/
│       ├── ddp_worker.py         torchrun subprocess entrypoint (SFT + GSPO update)
│       ├── stages/               one module per pipeline stage.type
│       │   ├── sft.py            MockTrainingClient (smoke) OR HF+PEFT Trainer (real)
│       │   ├── rl.py             GSPO / DAPO rollout + clipped-IS update
│       │   ├── teacher_distill.py  Teacher-LLM distillation (stage.type=synth_generate)
│       │   ├── solver_distill.py   Deterministic per-category solver → CoT JSONL
│       │   ├── data_merge.py     Dedup + upsample + register in data/sources.yaml
│       │   ├── generate.py       Unified type: generate, generator: <name> dispatcher
│       │   └── eval.py           run_eval_plan: smoke stub OR vLLM + LoRA
│       └── helpers/
│           ├── dataset.py        render_datums (smoke) + render_hf_dataset (real)
│           └── pack_adapter.py   Only used by smoke path
├── backends/tinkerlite/
│   ├── base.py                   TinkerLiteBackend Protocol + Datum/ModelInput/SamplingParams
│   ├── common_cfg.py             Pure .ddp_config.json builder (single source of truth)
│   ├── adapters/                 ModelAdapter Protocol + registry
│   │   ├── base.py               Protocol, register_adapter, resolve_adapter, ADAPTERS
│   │   └── lora.py               LoRAAdapter (today's PEFT behavior; kind="lora")
│   ├── clients/
│   │   ├── mock.py               MockTrainingClient / MockSamplingClient
│   │   ├── hf.py                 HF+PEFT TrainingClient (real single-GPU path)
│   │   └── vllm.py               VLLMSamplingClient (real eval path)
│   ├── single_node/
│   │   ├── backend.py            SingleNodeTinkerLiteBackend (mock & real paths)
│   │   └── ddp_launcher.py       torchrun spawner + override_stage_runner ContextVar hook
│   └── elastic/                  k8s-first elastic backend + target-agnostic scheduler
│       ├── backend.py            K8sTinkerLiteBackend (extends SingleNode; elastic scheduler)
│       ├── scheduler.py          ElasticScheduler + FanoutCapacity (k8s-first, local fallback)
│       ├── compute_target.py     ComputeTarget Protocol + CapacityReport
│       ├── targets/
│       │   ├── k8s.py            K8sComputeTarget (batch/v1 Job submit / poll / log tail)
│       │   ├── local.py          LocalComputeTarget (torchrun subprocess + GPU lock)
│       │   └── gpu_lock.py       flock-based GPU reservation, stale-PID cleanup
│       └── k8s/                  k8s-specific assets
│           ├── job_manifest.py   batch/v1 Job manifest builder
│           ├── image/            Docker image build (thin trainer; code mounted via FSx PVC)
│           └── smoke/            host-side smoke drivers + manual kubectl manifests
└── benchmarks/
    ├── training_base.py          TrainingBenchmarkAdapter Protocol (+ optional LLM hooks)
    ├── helpers.py                Cross-benchmark shim: prefer benchmark.*, fall back to nemo_reasoner
    └── nemo_reasoner.py          NemoReasonerBenchmark — smoke + Kaggle modes
```

## 3. How data flows through one cycle

```text
TrainingEvolutionLoop.run(cycles=N)
 └─ for each cycle:
    ├─ algorithm.run_cycle(ctx)                                 MCGS owns this
    │   ├─ selector.select(graph, cycle=c)                      → parent node
    │   ├─ mutator.propose(parent, graph)                       → WorkspaceMutation
    │   ├─ workspace.fork(node_id, mutation, work_dir)          → candidate workspace
    │   ├─ trial.run(candidate, node, budget)                   → TrainingTrialResult
    │   │   └─ backend.run_trial(workspace, node, budget, benchmark)
    │   │       ├─ _run_pipeline(workspace, pipeline, budget)   → CheckpointRef
    │   │       │   └─ for each enabled stage in pipeline.yaml::stages  (in declared order;
    │   │       │       StageRegistry.resolve(stage.type) → adapter)
    │   │       │       Available stage types (each trial only runs the subset flipped on):
    │   │       │         data-gen:  solver_distill | synth_generate | data_merge | generate
    │   │       │         train:     sft | rl
    │   │       └─ _run_evaluation(workspace, ckpt, benchmark, backend)
    │   │           └─ benchmark.evaluate(...)                  → result_dir
    │   │               └─ backend.run_eval_plan(plan)          (real: vLLM + LoRA)
    │   ├─ benchmark.check_validity(workspace, trial_result)    → ValidityReport
    │   ├─ reward_policy.compute(trial, validity, parent, graph) → float
    │   ├─ _backpropagate(node, reward)                          (updates mean_reward)
    │   ├─ promotion_policy.update_incumbent(graph, node)        → (id, changed)
    │   ├─ topk.update(graph.nodes)                              (branch-diversity cap)
    │   ├─ fusion.update_streak / maybe_fuse                     (stagnation detector)
    │   └─ memory.record(node)                                   (successful/failed jsonl)
    ├─ observer.record_cycle(report)                            → evolution/reports/cycle_N.json
    └─ if report.incumbent_changed:
        workspace.materialize_incumbent(node)                    → evolution/incumbent/
```

## 3.5 Data level — how a single training row flows through the pipeline

Three distinct lenses on "what happens to a row" during one cycle. Reading
top-to-bottom, a row is: **produced** by a data-gen stage → **merged + dedup'd**
→ **tokenized** → **consumed** as a training batch → **evaluated** (eval uses
a different, eval-specific data path).

### (a) Data-generation lane — `solver_distill` / `synth_generate` → `data_merge`

```
benchmark.iter_training_rows(workspace)                              ─┐
  └─ yields TrainingExample(id, prompt, answer, category, metadata)  │
                                                                     │
  ┌───────────── solver_distill stage (CPU, per category) ───────────┤
  │  1. solver = benchmark.solvers()[row.category]                   │
  │  2. sr = solver.solve(row.prompt)          → SolverResult        │
  │  3. verifier = benchmark.verifiers()[row.category]               │
  │     if not verifier.check(sr.predicted_answer, row.answer):      │
  │         drop; increment drop_reasons["wrong_answer"]             │
  │  4. renderer = benchmark.cot_renderers()[row.category]           │
  │     cot = renderer.render(row.prompt, row.answer, sr.trace_dict) │
  │  5. cot = cot_template.postprocess_cot(cot, row.answer)          │
  │         ↳ forces \boxed{row.answer}                              │
  │         ↳ injects [verify]: PASS if not already present          │
  │  6. emit GeneratedRow(id, prompt, answer, category,              │
  │                       cot, source="solver", metadata={})         │
  │                                                                  │
  ├───────────── synth_generate (teacher LLM, separate subprocess) ──┤
  │  Same shape of output: GeneratedRow(..., source="teacher_llm")   │
  │  (verifier_gate is an intent flag today — gate not yet enforced) │
  │                                                                  │
  ▼                                                                  ▼
data/generated/solver_distill/rows.jsonl       data/synth/teacher_traces.jsonl
  stats.json                                     stats.json
                                                                     │
data_merge stage                                                     │
  1. for src in stage.inputs (in order):                             ◀
       read each rows.jsonl; parse GeneratedRow.from_dict
  2. dedup.dedup(rows, filters.dedup_by)     first-seen wins,
                                             stable order (solver > teacher
                                             when both supplied)
  3. dedup.upsample(rows, recipe)            ×recipe.categories[cat].solver_upsample
                                             ×recipe.categories[cat].teacher_upsample
  4. write merged.jsonl  +  stats.json
  5. append {path: merged.jsonl, split: "train", format: "jsonl_cot"}
     into data/sources.yaml     ◀── the SFT loader reads this
```

**Key contract on every GeneratedRow at this point:** the final `\boxed{...}`
in `row.cot` equals `row.answer` exactly. No generator — solver, teacher,
future OOD — can violate this; `postprocess_cot` rewrites the box
unconditionally (enforced by `test_cot_template.py`).

### (b) SFT lane — `sft` stage consumes `data/sources.yaml`

`run_sft_stage` reads every `split=train` source in `data/sources.yaml`,
tokenizes each row into `input_ids = prompt_ids + completion_ids` with
`labels` masked to `-100` over the prompt (loss only on completion tokens),
then hands the HF `Dataset` to HF `Trainer` (DDP-sharded via
`DistributedSampler` when `AE_TRAIN_DDP=1`). Output: a `CheckpointRef`
(`kind="adapter"` for LoRA today) written under `train/<stage_name>/`.

Details — tokenization shape, batching, adapter save path — live in
**Appendix A1**.

### (c) RL rollout lane — `rl` stage

Two-phase per step: **rollout** (vLLM samples `n_samples` completions per
prompt; records `logprobs_old` + group-normalized `advantage`) → **update**
(HF training client loads from `last_ckpt`, consumes each rollout as a
`Datum` with `tokens = prompt_ids + completion_ids`, applies GSPO or
DAPO-token-level loss with importance-ratio clipping). Output: new LoRA
`CheckpointRef`.

Details — `Datum` payload shape, GSPO slicing at `prompt_len`, the
SFT→vLLM GPU handoff — live in **Appendix A2**.

### (d) Eval lane — `benchmark.evaluate(workspace, checkpoint, backend, split)`

Independent of (a)–(c); does **not** consume `data/sources.yaml`. Calls
`benchmark.build_eval_plan` → `backend.run_eval_plan` → vLLM generate (with
LoRA request for the adapter) → `extract_final_answer` + `verify` per row →
writes `metrics.json` + `predictions.jsonl` + `error_buckets.json`. Back to
MCGS: `metrics.primary_metric_value` (one float) + `ErrorBuckets.counts`.

Details — prompt rendering, sampling params, score_predictions contract —
live in **Appendix A3**.

## 3.6 Module level — what each module owns (responsibility map)

### Why `backend` and `runners` are two layers, not one

**`backend` fixes what one training step looks like. `runners` fix how SFT /
RL / data-gen stages assemble those steps into a stage.** Splitting them
means you can swap hardware (local torchrun ↔ k8s ↔ future Ray/Slurm/TPU)
or swap loss functions (CE ↔ GSPO ↔ future DPO) without touching the other
side.

What each layer owns:

| | `backend/` | `runners/` |
|---|---|---|
| Protocols | `TrainingClient` / `SamplingClient` / `ModelAdapter` / `TinkerLiteBackend` (in [base.py](agent_evolve/backends/tinkerlite/base.py)) | — (consumes Protocols) |
| Concrete impls | `MockTrainingClient` / `HFTrainingClient` / `VLLMSamplingClient` / `LoRAAdapter` | — |
| Instantiation strategy | `SingleNodeTinkerLiteBackend.create_training_client()` / `create_sampling_client()` — mock vs real, DDP vs single, local vs k8s | — (receives factories via `ctx.training_client_fn` / `ctx.sampling_client_fn`) |
| Stage dispatcher | `_run_pipeline` loop + `StageContext` packing ([backend.py](agent_evolve/backends/tinkerlite/single_node/backend.py)) | — |
| Execution topology | `single_node/` (local torchrun subprocess) + `elastic/` (same calls, routed through k8s via `override_stage_runner` ContextVar hook) | — |
| Per-stage flow | — | "SFT = for epoch, for batch: forward_backward('cross_entropy') + optim_step". "RL = for prompt, for g: sample → record → forward_backward('gspo') + optim_step." |

Resulting extension matrix:

| You want to add | Touch | Leave alone |
|---|---|---|
| A new compute target (Ray / Slurm-native / TPU) | new `TrainingClient` / `SamplingClient` impl + new `ComputeTarget` | all of `runners/` |
| A new pipeline stage (`distillation_v2`) | `runners/stages/distillation_v2.py` + `@register_stage("distillation_v2")` | `backend/` |
| A new adapter (DoRA / QLoRA / full) | `backends/tinkerlite/adapters/` + `@register_adapter("<kind>")` | `runners/` (resolves via `resolve_adapter` at runtime) |
| A new loss fn (e.g. `dpo`) | one branch in `HFTrainingClient.forward_backward` | `runners/` (passes a different string) |

### Module map

One diagram, five axes: algorithm (scoring), backend (execution), benchmark
(eval semantics), runners (per-stage workers), workspace (on-disk DNA).
Training-subsystem glue (`api/loop/trial/observer/workspace/types/registries`)
is shared by all five and isn't branched here.

```
algorithm  →  training/algorithms/<name>/               ← reads metrics, proposes moves
  + training/algorithms/null.py                           (reference impl: 45 LoC)
  + training/algorithms/mcgs/{search,graph,node,
       selection,mutation,reward,promotion,               (UCT + mutation + reward
       memory,fusion}.py                                   + promotion + memory + fusion)

runners    →  training/runners/                         ← per-stage workers, dispatched
  + runners/stages/sft.py                                 (@register_stage("sft"))
  + runners/stages/rl.py                                  (@register_stage("rl"))
  + runners/stages/solver_distill.py                      (@register_stage("solver_distill"))
  + runners/stages/teacher_distill.py                     (@register_stage("synth_generate"))
  + runners/stages/data_merge.py                          (@register_stage("data_merge"))
  + runners/stages/generate.py                            (@register_stage("generate"))
  + runners/stages/eval.py                                (run_eval_plan; not a pipeline stage)
  + runners/ddp_worker.py                                 (torchrun subprocess entry)
  + runners/helpers/{dataset,pack_adapter}.py             (tokenize + collate + pack)

backend    →  backends/tinkerlite/<name>/               ← runs the pipeline, no scoring
  + backends/tinkerlite/single_node/                      (local — mock/real; stage dispatcher)
  + backends/tinkerlite/elastic/                          (k8s-first + local fallback)
  + backends/tinkerlite/clients/{mock,hf,vllm}.py         (TrainingClient + SamplingClient impls)
  + backends/tinkerlite/adapters/{base,lora}.py           (ModelAdapter registry)
  + backends/tinkerlite/common_cfg.py                     (byte-identical .ddp_config.json)

benchmark  →  benchmarks/<name>.py                      ← eval semantics, error buckets
  + benchmarks/training_base.py                           (TrainingBenchmarkAdapter Protocol)
  + benchmarks/helpers.py                                 (routes teacher_distill/rl through
                                                           benchmark.* with nemo_reasoner fallback)
  + benchmarks/nemo_reasoner.py                           (reference impl — smoke + Kaggle)

data       →  training/data/                            ← benchmark-agnostic primitives
  + data/base.py                                          (Solver/Verifier/CoTRenderer Protocols)
  + data/generator.py                                     (DataGenerator Protocol + registry)
  + data/generators/{solver_distill,teacher_llm}.py       (registered wrappers)
  + data/{recipe,dedup,cot_template,verifier_gate}.py     (recipe schema + dedup + CoT + gate)

workspace  →  seed_workspaces/<name>/                   ← on-disk training DNA, evolvable
```

**Rule of thumb:** need to add a new search algo / stage type / adapter / data-gen / benchmark / compute target? Don't edit core files — register one Protocol impl. See [INTEGRATION.md](INTEGRATION.md) for the hands-on recipe.

## 4. What MCGS can see (score-driven, by design)

From each `TrainingTrialResult`, MCGS only looks at:

| Field | Use |
|---|---|
| `status` | → `node.trial_status`, marks terminal on failure |
| `checkpoint` | → `node.checkpoint`, used by promotion to materialize |
| `eval_metrics.primary_metric_value` | → `node.metric`, the UCT + promotion comparison key |
| `error_buckets.counts` | → reward formula's `format_error_rate` / `overlong_rate` |
| `validity.is_valid` + `hard_fail_reason` | → invalid ⇒ reward = −1.0, is_terminal = True |
| `cost` | → reward's `normalized_cost` penalty |

MCGS does **not** currently read per-prediction text, raw stdout, or benchmark secondary metrics (e.g. per-domain accuracy). `ErrorBuckets.examples` is written to `evolution/observations/<node_id>.json` but not consumed by mutator or selector today. `mutation.py::BaselineMutationProposer.propose` ignores parent entirely (rotates a hardcoded bag); `LRBagMutationProposer` does the same.

Example-driven mutation (Claude-in-the-loop reading bucket examples and writing the next patch) is an obvious next step — hooks exist (`memory.retrieve(query, k)`, `NodeMemoryStore` already persists `examples`). Not done yet.

## 5. Tinker parallel

The backend's shape mirrors Tinker's training API:

| Tinker primitive | TinkerLite equivalent | Used in |
|---|---|---|
| `forward_backward_async(data, loss_fn)` | `TrainingClient.forward_backward(batch, loss_fn)` | `stages/sft.py` (smoke: `MockTrainingClient`; real: `HFTrainingClient` w/ `cross_entropy`) + `stages/rl.py` (real: `HFTrainingClient` w/ `gspo`/`dapo_token_level`) |
| `optim_step_async(AdamParams)` | `TrainingClient.optim_step(AdamParams)` | Same call sites as above |
| `save_state(name)` / `save_weights_and_get_sampling_client(name)` | `save_state` / `save_weights_for_sampler` | `stages/sft.py` end-of-stage checkpoint |
| `sample_async(prompt, ...)` | `SamplingClient.sample(prompts, params)` | `stages/rl.py` rollout (real: `VLLMSamplingClient`); smoke uses `MockSamplingClient` |
| `Datum(model_input, loss_fn_inputs={target_tokens, weights, logprobs, advantages})` | Same shape, sync | `backends/tinkerlite/base.py` |

Implementation note: the full 30B SFT path still runs HF `Trainer`'s inner loop under the hood, but that loop is wrapped by `HFTrainingClient.forward_backward` so stages see one uniform surface (smoke and real both drive `forward_backward + optim_step`). RL doesn't need the wrap — it calls the client directly step-by-step.

## 6. Seed workspace contract

`seed_workspaces/nemotron_reasoner/` is the working example:

```text
manifest.yaml                      evolvable/protected/artifact layers
model/
  base.yaml                        path: /fsx/models/Nemotron-3-Nano-30B-A3B-BF16
  adapter.yaml                     rank/alpha/dropout/target_modules; optional seed_adapter_path
data/
  sources.yaml                     list of {path, split, format} — absolute paths allowed
  mix.yaml / curriculum.yaml       evolvable
  renderer.py                      pass-through stub
train/
  pipeline.yaml                    stages: [{name, type, epochs, max_steps, enabled}], top-level override_seed_adapter
  optimizer.yaml                   lr, warmup_ratio, betas, eps, weight_decay  ← mutated by LRBagMutationProposer
  batching.yaml                    per_device_bs, grad_accum, max_seq_len, log_every
  loss.yaml                        (evolvable placeholder)
rl/                                reward.py / advantage.py / rollout.yaml (disabled in v1)
eval/
  local_splits.yaml                protected; maps split names to paths (absolute OK)
  kaggle_eval.yaml                 protected; presence activates Kaggle mode
  local_holdout_small.jsonl        tiny smoke split
  error_taxonomy.yaml              evolvable
memory/ / checkpoints/ / evolution/   artifact layers
```

`manifest.yaml::evolvable_layers` lists what MCGS is allowed to mutate. `protected_layers` is enforced in `workspace._assert_not_protected` — attempting to patch a protected file raises `TrainingWorkspaceValidationError` inside `fork`, so an illegal mutation surfaces as `train_failed` not a corrupted workspace.

## 7. TODO

- **Runner → backend name leakage.** [stages/sft.py:24-25](agent_evolve/training/runners/stages/sft.py#L24-L25) imports `HFTrainingClient` / `MockTrainingClient` by name — a fallback for the (currently untaken) path where the dispatcher doesn't supply a `training_client`. Ideally even those references disappear and runners only see the `ctx.training_client_fn()` factory. Low priority; doesn't affect the "swap backend, runners don't budge" guarantee because the dispatcher always supplies the client today.

---

## Appendix A — Detailed data-lane diagrams

Expanded diagrams for §3.5(b)(c)(d). The prose in §3.5 is sufficient for a
mental model; read these when you need the exact shape of a tensor, the
exact call into a stage worker, or the exact file that consumes a
checkpoint.

### A1. SFT lane — full detail for §3.5(b)

```
train/pipeline.yaml::stages[sft]                                  (stage config)
  ├─ epochs, max_steps, loss, batching.yaml, optimizer.yaml
  ▼
sft stage adapter  (StageRegistry dispatch — stage_registry.py:resolve_stage)
  ▼
run_sft_stage(workspace, stage, ...)
  ├─ smoke: render_datums(workspace)   →  Iterable[Datum]  (mock ints)
  │         └─ MockTrainingClient.forward_backward(batch, "cross_entropy")
  │
  ▼ real:
  render_hf_dataset(workspace, tokenizer, max_len)
    │  for src in data/sources.yaml (split="train"):
    │    for row in open(src.path):
    │      prompt_ids     = tokenizer.encode(row.prompt_rendered)     ┐
    │      completion_ids = tokenizer.encode(row.completion)          │ tokenize
    │      input_ids  = prompt_ids + completion_ids                   │ per row
    │      labels     = [-100] * len(prompt_ids) + completion_ids     │ (prompt
    │      attention_mask = [1] * len(input_ids)                      │  mask'd)
    │      if len(input_ids) > max_len: left-truncate prompt          ┘
    ▼
  HF Dataset {input_ids, attention_mask, labels}  (one row per JSONL line)
    │
    ▼  shuffle(seed)  →  batched per_device_bs
    │
    ▼  DDP sharded via DistributedSampler (one slice per rank) — OR
    │  single-process path via DataLoader(..., collate_fn=PadToLongest)
    │
    ▼  for each optim step (epoch × batches // grad_accum):
    │    for micro_batch in range(grad_accum):
    │      out = model(input_ids, attention_mask, labels)             (HF computes
    │      (out.loss / grad_accum).backward()                          CE per token
    │                                                                  w/ -100 mask)
    │    optimizer.step(); optimizer.zero_grad()                      ← lr = cosine
    ▼
  resolve_adapter(adapter_cfg.type).save(model, tokenizer, outdir)   ← LoRA today;
                                                                       DoRA / full /
                                                                       QLoRA via @register_adapter
    │
    ▼  CheckpointRef(name=stage_name, path=outdir, kind="adapter" | "full_weights")
       last_ckpt := CheckpointRef            ◀── consumed by next stage (rl/eval)
```

**Data reshape per row:** `{prompt_rendered: str, completion: str}` → tokenize
with prompt-masking → model trains only on completion tokens. The `-100` in
`labels` is what tells HF loss to ignore the prompt positions.

### A2. RL rollout lane — full detail for §3.5(c)

```
last_ckpt (from prior SFT)   +   benchmark.load_rl_prompts(workspace, stage)
      │                                        │
      │                                        ▼
      │                        for prompt_row in prompts:                         ┐
      ├──▶ sampling_client_fn(last_ckpt)       for g in range(n_samples):         │
      │    └─ VLLMSamplingClient (LoRA adapter │   completion, logprobs_old =     │ rollout
      │                           via          │       sampler.sample(prompt)     │  (vLLM)
      │                           LoRARequest)  ▼   advantage = group_normalize(...) │
      │                             rollout record {                              │
      │                               prompt_ids, completion_tokens,              │
      │                               logprobs_old, advantage, prompt_len,        │
      │                             }                                             ┘
      │                                        │
      │                                        ▼
      │                             rollouts.jsonl  (one record per completion)
      │
      ├──▶ close_training_client_fn()    ◀── release SFT client GPU
      │                                        ▼
      ├──▶ training_client_fn()               HFTrainingClient (fresh,
      │                                        loaded from last_ckpt)
      │                                        │
      │                                        ▼
      │    for rec in rollouts.jsonl:
      │      x = Datum(
      │        model_input.tokens = prompt_ids + completion_tokens,
      │        loss_fn_inputs = {
      │          logprobs_old, advantage, prompt_len,
      │        },
      │      )
      │      forward_backward(x, "gspo"  OR  "dapo_token_level")      ← clipped
      │      if step % grad_accum == 0: optim_step(AdamParams(lr=...))   importance
      │                                                                  sampling
      ▼
  new CheckpointRef — kind="adapter", path=rl_gspo outdir
```

**Key shape twist:** RL `Datum.model_input.tokens` contains prompt+completion
*concatenated*. The training client's GSPO loss slices at `prompt_len` to
recover per-completion-token logits, then clips the importance ratio
`exp(logprob_new − logprob_old)` against `eps_low/eps_high`.

### A3. Eval lane — full detail for §3.5(d)

Entirely independent of (a)–(c). Consumes the checkpoint produced by (b) or
(c), does **not** consume `data/sources.yaml`.

```
benchmark.build_eval_plan(workspace, checkpoint, split)  →  EvalPlan
  │   generation_config: {model_path, tp, max_model_len, max_tokens, limit, ...}
  │   checkpoint: the adapter path
  │
  ▼
backend.run_eval_plan(plan)  (SingleNodeTinkerLiteBackend.run_eval_plan
                              → stages.eval.run_eval_plan)
  ├─ smoke: read eval/local_holdout_small.jsonl; score is_correct flags
  │         write metrics.json + predictions.jsonl
  │
  ▼ real:
  benchmark.load_dev_rows(workspace, split)  →  list[DevRow]
     (DevRow = {id, prompt, answer, domain, ...} — shape is benchmark-specific)
  │
  ▼  for row in dev_rows:
  │    prompt_str = benchmark.build_eval_prompt(row, tokenizer)
  │    ↳ for Nemotron: chat_template + "\nPlease put your final answer inside
  │                                      \boxed{}..." suffix
  │
  ▼  vllm.LLM.generate(prompt_strs, SamplingParams(T,top_p,max_tokens),
                       lora_request=adapter.vllm_lora_request(checkpoint))
  │                                              ← resolve_adapter(cfg.type);
  │                                                LoRA=LoRARequest, full=None
  │
  ▼  for row, out in zip(dev_rows, outputs):
  │    pred = benchmark.extract_final_answer(out.text)         → canonical str
  │    is_correct = benchmark.verify(pred, row.answer)          → bool
  │    record row-level prediction
  │
  ▼  metrics, buckets, records = benchmark.score_predictions(dev_rows, raw_texts)
  │
  ▼  write metrics.json + predictions.jsonl + error_buckets.json
       + raw_outputs.jsonl under evolution/eval/<ckpt>/<split>/
```

**What flows back to MCGS:** `metrics.primary_metric_value` (one float) plus
`ErrorBuckets.counts` (dict[str,int]). Per-prediction text is on disk in
`predictions.jsonl` but MCGS doesn't read it today (§4 — example-driven
mutator is deliberately not wired yet).
