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
| `workspace=` | Training DNA on disk | `seed_workspaces/nemotron_reasoner` | Path to a directory with `manifest.yaml`, `train/`, `data/`, `model/`, `eval/`, `rl/` (see §8) |
| `algorithm=` | Search | `"mcgs"` → `MCGSSearch` | Selection, mutation, reward, backprop, promotion, memory, fusion — everything that reads metrics and proposes next moves |
| `backend=` | Execution | `"h200_single_node"` → `SingleNodeTinkerLiteBackend` (or `"k8s_h200"` → `K8sTinkerLiteBackend`, §14) | Runs SFT/RL stages, saves adapter, loads it for eval, samples. Never scores a trajectory |
| `benchmark=` | Evaluation semantics | `"nemo_reasoner"` → `NemoReasonerBenchmark` | Primary metric, split loading, error taxonomy, validity rules, (optionally) per-category solvers/verifiers/renderers for §15 |

The algorithm never executes a forward pass. The benchmark never computes a reward or picks an incumbent. The backend never scores a trajectory. These are enforced with tests (`test_backend_no_reward.py`, `test_benchmark_no_reward.py`, `test_loop_no_gate.py`).

String values on the right are *registry keys* — `ae.TrainingEvolver` resolves them via `TRAINING_ALGORITHMS` / `TRAINING_BACKENDS` / `TRAINING_BENCHMARKS` (see §10). You can also pass instances directly (`algorithm=MCGSSearch(...)`, `backend=MyBackend()`) to bypass the registry.

## 2. Package layout

```text
agent_evolve/
├── training/
│   ├── api.py                    TrainingEvolver facade
│   ├── types.py                  All shared dataclasses (single source of truth)
│   ├── registries.py             Dotted-path registry for benchmarks / algorithms / job runners
│   ├── runner_protocol.py        TrainingJobRunner Protocol (§16) — root extension point
│   ├── stage_registry.py         @register_stage decorator + StageContext/StageResult (§16)
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
│   │       ├── reward.py         DefaultRewardPolicy (spec §8 formula)
│   │       ├── promotion.py      PromotionPolicy (incumbent + tie-breakers)
│   │       ├── memory.py         NodeMemoryStore (BM25-lite retrieval, JSONL on disk)
│   │       └── fusion.py         FusionPolicy (cross-branch recombination on stagnation)
│   ├── data/                     Benchmark-agnostic data-generation primitives (§15)
│   │   ├── base.py               Solver / Verifier / CoTRenderer / DataSynthGenerator Protocols
│   │   ├── generator.py          DataGenerator Protocol + registry (§16) — new 4th axis
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
│       │   ├── generate.py       Unified type: generate, generator: <name> dispatcher (§16)
│       │   └── eval.py           run_eval_plan: smoke stub OR vLLM + LoRA
│       └── helpers/
│           ├── dataset.py        render_datums (smoke) + render_hf_dataset (real)
│           └── pack_adapter.py   Only used by smoke path
├── backends/tinkerlite/
│   ├── base.py                   TinkerLiteBackend Protocol + Datum/ModelInput/SamplingParams
│   ├── common_cfg.py             Pure .ddp_config.json builder (single source of truth)
│   ├── adapters/                 ModelAdapter Protocol + registry (§16)
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
    ├── training_base.py          TrainingBenchmarkAdapter Protocol (+ optional LLM hooks §16)
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
    │   │       │   └─ run_sft_stage(workspace, stage, ...)     (real: HF Trainer)
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
unconditionally (see §13 invariant 10).

### (b) SFT lane — `sft` stage consumes `data/sources.yaml`

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
                                                                       QLoRA via §16
    │
    ▼  CheckpointRef(name=stage_name, path=outdir, kind="adapter" | "full_weights")
       last_ckpt := CheckpointRef            ◀── consumed by next stage (rl/eval)
```

**Data reshape per row:** `{prompt_rendered: str, completion: str}` → tokenize
with prompt-masking → model trains only on completion tokens. The `-100` in
`labels` is what tells HF loss to ignore the prompt positions.

### (c) RL rollout lane — `rl` stage

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

### (d) Eval lane — `benchmark.evaluate(workspace, checkpoint, backend, split)`

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
  │                                              ← resolve_adapter(...) in §16;
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

### Sources of truth at each transition

| Transition | Schema | Enforced by |
|---|---|---|
| benchmark → generator | `TrainingExample` | [training/data/base.py](agent_evolve/training/data/base.py) |
| generator → JSONL | `GeneratedRow` dataclass round-trip | `to_dict/from_dict` + `test_base_protocols.py` |
| JSONL → SFT row | `{prompt_rendered, completion}` keys required | `render_hf_dataset` raises if missing |
| SFT row → Datum | `Datum(model_input.tokens, loss_fn_inputs={attention_mask, labels})` | [backends/tinkerlite/base.py](agent_evolve/backends/tinkerlite/base.py) |
| SFT checkpoint | `CheckpointRef.kind ∈ {"adapter","full_weights","sampler_weights","full_state"}` | `training/types.py` Literal (§16 widened) |
| Eval result_dir | `metrics.json` + `predictions.jsonl` | `benchmark.parse_metrics` parses |
| Cycle record | `MCGSCycleReport` | [training/types.py](agent_evolve/training/types.py) |

## 3.6 Module level — what each module owns (responsibility map)

Everything organized by "if I need to change X, which file do I touch".

### The four-axes package layout (§1 recap + where each axis lives)

```
algorithm  →  training/algorithms/<name>/             ← reads metrics, proposes moves
  + training/algorithms/null.py                         (reference impl: 45 LoC)
  + training/algorithms/mcgs/{search,graph,node,         (UCT + mutation + reward
       selection,mutation,reward,promotion,               + promotion + memory
       memory,fusion}.py                                  + fusion — one file each)

backend    →  backends/tinkerlite/<name>/             ← runs the pipeline, no scoring
  + backends/tinkerlite/single_node/                    (local — mock/real)
  + backends/tinkerlite/elastic/                        (k8s-first + local fallback §14)
  + backends/tinkerlite/clients/{mock,hf,vllm}.py       (per-Protocol concrete impls)
  + backends/tinkerlite/adapters/{base,lora}.py         (ModelAdapter registry §16)
  + backends/tinkerlite/common_cfg.py                   (byte-identical .ddp_config.json)

benchmark  →  benchmarks/<name>.py                    ← eval semantics, error buckets
  + benchmarks/training_base.py                         (TrainingBenchmarkAdapter Protocol)
  + benchmarks/helpers.py                               (routes teacher_distill/rl through
                                                         benchmark.* with nemo_reasoner fallback)

workspace  →  seed_workspaces/<name>/                 ← on-disk training DNA, evolvable
```

### Training subsystem (orchestration — shared by all algorithms/backends/benchmarks)

| File | Owns | Depended on by |
|---|---|---|
| [training/api.py](agent_evolve/training/api.py) | `TrainingEvolver` facade — resolves string keys, validates workspace, builds loop. | End-user code |
| [training/loop.py](agent_evolve/training/loop.py) | `TrainingEvolutionLoop` — per-cycle driver. Owns the `LoopContext` dataclass that algorithms read. No accept/reject gate here. | api.py |
| [training/trial.py](agent_evolve/training/trial.py) | `TrialRunner` — thin wrapper around `backend.run_trial` that normalizes exceptions into `TrainingTrialResult(status="train_failed")` and applies `benchmark.check_validity`. | loop.py (via `LoopContext`) |
| [training/observer.py](agent_evolve/training/observer.py) | Persists `MCGSCycleReport` to `evolution/reports/cycle_NNNN.json` and `TrainingTrialResult` to `evolution/observations/<node_id>.json`. Dumb serializer — no business logic. | loop.py |
| [training/workspace.py](agent_evolve/training/workspace.py) | `TrainingWorkspace.fork(node_id, mutation, work_dir)` copies the root to a candidate dir + applies the `WorkspacePatch` + validates. Also `materialize_incumbent`, `fingerprint`. | algorithm + loop + tests |
| [training/schema.py](agent_evolve/training/schema.py) | `validate_training_workspace(path)` — checks required files exist. Called inside `TrainingWorkspace.validate()`. | workspace.py |
| [training/types.py](agent_evolve/training/types.py) | **Single source of truth** for every shared dataclass: `TrainingTrialResult`, `CheckpointRef`, `EvalMetrics`, `ErrorBuckets`, `ValidityReport`, `EvalPlan`, `TrainingSearchNode`, `MCGSCycleReport`, `WorkspaceMutation`, `WorkspacePatch`, `TrialBudget`. No other module re-defines these (§13 invariant 5). | everything |
| [training/registries.py](agent_evolve/training/registries.py) | `TRAINING_BENCHMARKS / ALGORITHMS / JOB_RUNNERS` dicts + `resolve_*` functions. Dotted-path strings → classes. | api.py |
| [training/runner_protocol.py](agent_evolve/training/runner_protocol.py) | `TrainingJobRunner` Protocol (§16) — root extension point. Even a sklearn runner implements just this. | trial.py, backends |
| [training/stage_registry.py](agent_evolve/training/stage_registry.py) | `@register_stage(stype)` decorator + `StageContext` / `StageResult` dataclasses + `resolve_stage`. Replaces the old `if stype == "sft":` ladder. | single_node/backend.py dispatcher, stage modules |
| [training/run.py](agent_evolve/training/run.py) | CLI entrypoint `python -m agent_evolve.training.run`. Just argparse + `TrainingEvolver().run()`. | end-users, CI |

**Rule of thumb:** if a file here needs a new method, the extension is probably missing a registry — add one instead of editing (§13 invariant 5, INTEGRATION.md §8).

### Algorithm subsystem (search logic)

| File | Owns |
|---|---|
| [algorithms/null.py](agent_evolve/training/algorithms/null.py) | `NullSearchAlgorithm` — 45-line reference impl. Increments a counter, returns an empty `MCGSCycleReport`. Used to test loop/backend plumbing without dragging in MCGS. |
| [algorithms/mcgs/search.py](agent_evolve/training/algorithms/mcgs/search.py) | `MCGSSearch.run_cycle(ctx)` — the whole cycle: select → mutate → fork → trial → reward → backprop → promotion → topk → memory → fusion → persist graph. |
| [algorithms/mcgs/graph.py](agent_evolve/training/algorithms/mcgs/graph.py) | `TrainingSearchGraph` — persistent DAG. JSON save/load round-trips (§13 invariant 7). |
| [algorithms/mcgs/node.py](agent_evolve/training/algorithms/mcgs/node.py) | `NodeStatus` enum + `node_summary(n) → TrainingSearchNodeSummary`. |
| [algorithms/mcgs/selection.py](agent_evolve/training/algorithms/mcgs/selection.py) | `UCTSelector` (piecewise exploration decay + branch-diversity guard) + `TopKStore` (per-branch cap). |
| [algorithms/mcgs/mutation.py](agent_evolve/training/algorithms/mcgs/mutation.py) | `BaselineMutationProposer` (rotating data-mix bag) + `LRBagMutationProposer` (lr sweep). |
| [algorithms/mcgs/reward.py](agent_evolve/training/algorithms/mcgs/reward.py) | `DefaultRewardPolicy.compute(trial, validity, parent, graph) → float` — spec §8 formula. |
| [algorithms/mcgs/promotion.py](agent_evolve/training/algorithms/mcgs/promotion.py) | `PromotionPolicy` — incumbent decision + tie-breakers (instability, seconds, tokens, buckets). |
| [algorithms/mcgs/memory.py](agent_evolve/training/algorithms/mcgs/memory.py) | `NodeMemoryStore` — JSONL persistence + BM25-lite retrieval. |
| [algorithms/mcgs/fusion.py](agent_evolve/training/algorithms/mcgs/fusion.py) | `FusionPolicy` — cross-branch recombination when stagnating. |

### Runners subsystem (per-stage workers)

Everything dispatchable from `train/pipeline.yaml::stages[i].type`. See INTEGRATION.md §2 for how to add a new stage type.

| File | Owns | Stage type |
|---|---|---|
| [runners/stages/sft.py](agent_evolve/training/runners/stages/sft.py) | SFT orchestration: smoke (Mock) + real (in-process HF+PEFT) + real-DDP dispatch (`AE_TRAIN_DDP=1`). | `sft` |
| [runners/stages/rl.py](agent_evolve/training/runners/stages/rl.py) | GSPO/DAPO: rollout via `SamplingClient` → advantage normalization → update via `TrainingClient.forward_backward("gspo")`. Includes the GPU-lifecycle dance (close SFT client before vLLM, rebuild after). | `rl` |
| [runners/stages/teacher_distill.py](agent_evolve/training/runners/stages/teacher_distill.py) | 120B teacher distillation. Subprocess-isolated so teacher CUDA state is reclaimed on exit. | `synth_generate` |
| [runners/stages/solver_distill.py](agent_evolve/training/runners/stages/solver_distill.py) | Deterministic per-category solver → verifier → CoT renderer → postprocess → JSONL. | `solver_distill` |
| [runners/stages/data_merge.py](agent_evolve/training/runners/stages/data_merge.py) | Load each input JSONL → dedup (first-seen wins) → upsample per recipe → register in `data/sources.yaml`. | `data_merge` |
| [runners/stages/generate.py](agent_evolve/training/runners/stages/generate.py) | Unified dispatcher for `type: generate, generator: <name>` — delegates to any `DataGenerator` from the registry (§16). | `generate` |
| [runners/stages/eval.py](agent_evolve/training/runners/stages/eval.py) | `run_eval_plan(plan)` — smoke stub or real vLLM + LoRA loop. Invoked via `backend.run_eval_plan`, not through stage registry. | (not a pipeline stage) |
| [runners/ddp_worker.py](agent_evolve/training/runners/ddp_worker.py) | `torchrun --nproc_per_node=N -m agent_evolve.training.runners.ddp_worker` subprocess entry. Handles SFT + GSPO update. Byte-for-byte the same under local or k8s. | (subprocess; not a stage) |
| [runners/helpers/dataset.py](agent_evolve/training/runners/helpers/dataset.py) | `render_datums` (smoke — mock ints) + `render_hf_dataset` (real tokenization, prompt masking) + `PadToLongest` collator. | — |
| [runners/helpers/pack_adapter.py](agent_evolve/training/runners/helpers/pack_adapter.py) | Smoke-only: wraps a checkpoint dir into a `CheckpointRef`. | — |

### Data subsystem (benchmark-agnostic data primitives)

| File | Owns |
|---|---|
| [data/base.py](agent_evolve/training/data/base.py) | 4 Protocols: `Solver / Verifier / CoTRenderer / DataSynthGenerator` (per-category). Dataclasses: `SolverResult`, `TrainingExample`, `GeneratedRow` (JSONL wire format). |
| [data/generator.py](agent_evolve/training/data/generator.py) | `DataGenerator` Protocol (§16) — stage-level generation abstraction. `@register_data_generator` + `resolve_data_generator` + `DATA_GENERATORS` dict. |
| [data/generators/solver_distill.py](agent_evolve/training/data/generators/solver_distill.py) | `SolverDistillGenerator` — registered wrapper. |
| [data/generators/teacher_llm.py](agent_evolve/training/data/generators/teacher_llm.py) | `TeacherLLMGenerator` — registered wrapper. |
| [data/recipe.py](agent_evolve/training/data/recipe.py) | `DataRecipe` + `CategoryRecipe` + `RecipeFilters` dataclasses + `load_recipe(path)`. Evolvable YAML — MCGS mutators target this. |
| [data/dedup.py](agent_evolve/training/data/dedup.py) | `dedup_key(row, mode)` ("prompt_hash" or "prompt_and_source_hash") + `dedup(rows, filters)` (first-seen wins, stable order) + `upsample(rows, recipe)`. |
| [data/cot_template.py](agent_evolve/training/data/cot_template.py) | `postprocess_cot(cot, answer)` — the load-bearing guarantee (§13 invariant 10): forces `\boxed{answer}` + injects `[verify]: PASS`. Idempotent. |
| [data/verifier_gate.py](agent_evolve/training/data/verifier_gate.py) | `apply_verifier_gate(rows, verifiers, filters) → (kept, GateStats)` — shared correctness filter across any generator that produces first, verifies later. |

### Backends subsystem (execution, no scoring)

| File | Owns |
|---|---|
| [backends/tinkerlite/base.py](agent_evolve/backends/tinkerlite/base.py) | `TinkerLiteBackend` Protocol (LLM-specific extension of `TrainingJobRunner`) + `TrainingClient / SamplingClient` Protocols + all LLM dataclasses (`Datum, ModelInput, Prompt, SamplingParams, Logprobs, SampleResponse`). |
| [backends/tinkerlite/common_cfg.py](agent_evolve/backends/tinkerlite/common_cfg.py) | Pure `.ddp_config.json` builders (`build_sft_cfg`, `build_gspo_cfg`). Single source of truth consumed by both local torchrun and k8s pod — byte-identical output guaranteed (§13 invariant 8). |
| [backends/tinkerlite/adapters/base.py](agent_evolve/backends/tinkerlite/adapters/base.py) | `ModelAdapter` Protocol + `@register_adapter` + `resolve_adapter` + `ADAPTERS` dict + `ATTACH_MODE_WRAP`/`ATTACH_MODE_INPLACE` constants. §16 extension point — widens beyond LoRA. |
| [backends/tinkerlite/adapters/lora.py](agent_evolve/backends/tinkerlite/adapters/lora.py) | `LoRAAdapter` — the PEFT LoRA implementation. Today's `HFTrainingClient` behavior, captured. |
| [backends/tinkerlite/clients/mock.py](agent_evolve/backends/tinkerlite/clients/mock.py) | `MockTrainingClient` / `MockSamplingClient` — in-memory fakes. Geometric loss decay so smoke tests can assert progress. |
| [backends/tinkerlite/clients/hf.py](agent_evolve/backends/tinkerlite/clients/hf.py) | `HFTrainingClient` — eager HF + PEFT training client. Supports `cross_entropy`, `gspo`, `dapo_token_level` loss functions. Called from `sft.py` (in-process path) and `rl.py` (update phase). |
| [backends/tinkerlite/clients/vllm.py](agent_evolve/backends/tinkerlite/clients/vllm.py) | `VLLMSamplingClient` — vLLM engine + LoRA request. Used by RL rollout + eval. |
| [backends/tinkerlite/single_node/backend.py](agent_evolve/backends/tinkerlite/single_node/backend.py) | `SingleNodeTinkerLiteBackend.run_trial` + `_run_pipeline` (StageRegistry-driven dispatcher — §16). Owns the shared-training-client lifecycle across stages. |
| [backends/tinkerlite/single_node/ddp_launcher.py](agent_evolve/backends/tinkerlite/single_node/ddp_launcher.py) | `run_sft_ddp` / `run_gspo_ddp` — spawns torchrun subprocess. `override_stage_runner(fn)` ContextVar hook lets the k8s backend substitute a different runner. |
| [backends/tinkerlite/elastic/backend.py](agent_evolve/backends/tinkerlite/elastic/backend.py) | `K8sTinkerLiteBackend` — inherits SingleNode, installs an `ElasticScheduler`-backed `override_stage_runner`. Async extras: `submit_stage_async / wait_any / cancel_stage`. |
| [backends/tinkerlite/elastic/scheduler.py](agent_evolve/backends/tinkerlite/elastic/scheduler.py) | `ElasticScheduler.run_stage` — priority-first k8s → local fallback (§14). `probe_capacity` returns `FanoutCapacity`. |
| [backends/tinkerlite/elastic/compute_target.py](agent_evolve/backends/tinkerlite/elastic/compute_target.py) | `ComputeTarget` Protocol + `CapacityReport` + `TargetHandle` + `PendingTimeout`. |
| [backends/tinkerlite/elastic/targets/k8s.py](agent_evolve/backends/tinkerlite/elastic/targets/k8s.py) | `K8sComputeTarget` — kubernetes client wrapper. Submits `batch/v1` Jobs, polls, streams logs. |
| [backends/tinkerlite/elastic/targets/local.py](agent_evolve/backends/tinkerlite/elastic/targets/local.py) | `LocalComputeTarget` — torchrun subprocess + GPU lock. Used by k8s backend as local fallback. |
| [backends/tinkerlite/elastic/targets/gpu_lock.py](agent_evolve/backends/tinkerlite/elastic/targets/gpu_lock.py) | `acquire_gpus` — flock-based multi-process GPU coordination on `/fsx/.ae_locks/`. |
| [backends/tinkerlite/elastic/k8s/job_manifest.py](agent_evolve/backends/tinkerlite/elastic/k8s/job_manifest.py) | `build_job_manifest(...)` — `batch/v1` Job body with PVC mount + GPU resources + ttlSecondsAfterFinished. |

### Benchmarks subsystem

| File | Owns |
|---|---|
| [benchmarks/training_base.py](agent_evolve/benchmarks/training_base.py) | `TrainingBenchmarkAdapter` Protocol — required (eval) + optional (data pipeline §15, LLM hooks §16). Structural; any class with the right methods conforms. |
| [benchmarks/helpers.py](agent_evolve/benchmarks/helpers.py) | Cross-benchmark shim (§16 PR-E): `build_eval_prompt / extract_final_answer / verify` prefer `benchmark.<method>` when available, fall back to `nemo_reasoner` otherwise. New benchmarks should implement the methods; the shim keeps the legacy Nemotron path green. |
| [benchmarks/nemo_reasoner.py](agent_evolve/benchmarks/nemo_reasoner.py) | `NemoReasonerBenchmark` — reference implementation covering smoke + Kaggle modes. Implements every optional Protocol method. |

### Quick lookup table — "I need to change X"

| I want to | Change |
|---|---|
| Add a new search algorithm | New package under `training/algorithms/<name>/` + dict entry in `registries.py::TRAINING_ALGORITHMS` |
| Add a new training framework (non-LLM) | Implement `TrainingJobRunner`; register in `TRAINING_JOB_RUNNERS` (§16 PR-A) |
| Add a new pipeline stage type | `@register_stage("<type>")` on a function taking `StageContext` (§16 PR-B) |
| Add a new adapter (DoRA, full, QLoRA) | `@register_adapter("<kind>")` under `backends/tinkerlite/adapters/` (§16 PR-C) |
| Add a new data-gen method | `@register_data_generator("<name>")` under `training/data/generators/` (§16 PR-D) |
| Add a new benchmark | New class in `benchmarks/<name>.py` conforming to `TrainingBenchmarkAdapter`; register in `TRAINING_BENCHMARKS` |
| Add a new compute target (Slurm, Batch) | Implement `ComputeTarget`; pass into `K8sTinkerLiteBackend(targets=[...])` |
| Change the reward formula | Subclass/swap `DefaultRewardPolicy`; pass into `MCGSSearch(reward_policy=...)` |
| Change what's persisted per cycle | `TrainingObserver` — it's the only writer of evolution artifacts |
| Change workspace validation | `training/schema.py` REQUIRED_FILES / REQUIRED_DIRS |

See [INTEGRATION.md](INTEGRATION.md) for the hands-on recipe for each row above.

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
| `forward_backward_async(data, loss_fn)` | `TrainingClient.forward_backward(batch, loss_fn)` | `train_worker` (smoke) |
| `optim_step_async(AdamParams)` | `TrainingClient.optim_step(AdamParams)` | `train_worker` (smoke) |
| `save_state(name)` / `save_weights_and_get_sampling_client(name)` | `save_state` / `save_weights_for_sampler` | `train_worker` (smoke) |
| `sample_async(prompt, ...)` | `SamplingClient.sample(prompts, params)` | `MockSamplingClient` |
| `Datum(model_input, loss_fn_inputs={target_tokens, weights, logprobs, advantages})` | Same shape, sync | `backends/tinkerlite/base.py` |

Non-smoke training at 30B scale bypasses the `TrainingClient` abstraction and goes directly through HF `Trainer` + PEFT `LoraConfig` — because the Tinker abstractions are designed for in-process step-by-step control whereas HF `Trainer` owns the inner loop. The shape stays useful for RL (future work).

## 6. The backends: smoke vs real, local vs cluster

Two registry keys, three execution modes:

| Key | Class | Execution |
|---|---|---|
| `h200_single_node` | `SingleNodeTinkerLiteBackend` | Always local: mock (CPU) or real (1×8-GPU DDP via torchrun subprocess) |
| `k8s_h200` | `K8sTinkerLiteBackend` (extends SingleNode) | Elastic: k8s cluster first, local fallback |

`SingleNodeTinkerLiteBackend(mock: bool)`:

- `mock=True` → `MockTrainingClient` for training, deterministic metrics.json stub for eval. Used by all unit tests and the `--smoke` CLI path.
- `mock=False` → real HF Trainer + LoRA load from `/fsx/models/Nemotron-3-Nano-30B-A3B-BF16`, real vLLM + LoRA eval on 951-row Kaggle dev CSV. Multi-GPU DDP kicks in when `AE_TRAIN_DDP=1` or when `ddp_launcher.run_sft_ddp` is called directly — a `torchrun --nproc_per_node=N` subprocess spawns the DDP worker.

`K8sTinkerLiteBackend` **inherits** from `SingleNodeTinkerLiteBackend` and reuses the full pipeline orchestration (§3). The only seam is the DDP spawn boundary: `ddp_launcher.override_stage_runner(fn)` is a ContextVar-scoped hook that replaces the local `torchrun` subprocess with a caller-supplied runner. The k8s backend installs a runner that routes each stage through an `ElasticScheduler` (§14) which picks between k8s and local targets per availability. Pod and local subprocess both read the same byte-identical `.ddp_config.json` and run the same `train_worker_ddp.py`.

The `mock` flag propagates from `TrainingEvolveConfig.smoke` through `TrainingEvolver._resolve_backend` to `backend.mock` so `--smoke` on the CLI actually controls the real/mock split.

Two escape hatches for real cycles:

- **`model/adapter.yaml::seed_adapter_path`**: when set, `run_trial` skips training and evaluates that adapter directly. Used to reproduce E-28's 49.63% dev (see §9).
- **`train/pipeline.yaml::override_seed_adapter: true`**: when set, forces `_run_pipeline` to train even if `seed_adapter_path` exists. Used for real from-base SFT cycles.

## 7. The two benchmark modes

`NemoReasonerBenchmark` auto-switches based on presence of `<workspace>/eval/kaggle_eval.yaml`:

| Mode | Primary metric | Eval worker | Split |
|---|---|---|---|
| Smoke (default) | `local_holdout_pass_at_1` | Deterministic stub reading `local_holdout_small.jsonl` | In-workspace |
| Kaggle (when `eval/kaggle_eval.yaml` exists) | `kaggle_nemo_boxed_em` | vLLM + LoRA, verbatim host `extract_final_answer` + `verify` | CSV pointed to by `local_splits.yaml::kaggle_dev_local` |

In Kaggle mode:
- Prompt wrapper = chat-template + `\nPlease put your final answer inside \boxed{}` suffix. Byte-for-byte match with the host kernel at `../nemotron-auto-research/scripts/eval_metric.py`.
- Sampling: T=1.0, top_p=1.0, max_tokens=3584, max_model_len=4096, max_lora_rank=32.
- Scoring: boxed-EM with relative-tolerance (rel_tol=1e-2) fallback for numeric answers, binary-string exact for bit patterns.
- Per-domain breakdown (`bits/cipher/equations/gravity/units/numerals`) is stored in `EvalMetrics.secondary` and `metrics.json::per_domain`.
- `check_validity` enforces the Kaggle-host rule: LoRA rank ≤ 32 (reads `adapter_config.json::r`).

## 8. Seed workspace contract

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

## 9. What ran end-to-end

### Eval-only cycle (commit f4f8009)
Set `seed_adapter_path` to `../nemotron-auto-research/experiments/E-28-iter3-noprm/adapter`, ran `TrainingEvolver(cycles=1)`. Result: **50.05% dev on 951 rows in 7.9 min** (E-28 reported 49.63% — within noise floor). Proved the vLLM + LoRA + verbatim-host-metric pipeline.

### 4-cycle real-SFT LR sweep (this commit)
Each cycle forks a fresh sibling from root, runs 8 optimizer steps of SFT at one of `[1e-4, 5e-5, 3e-5, 1e-5]` on 476-row `short_correct.jsonl` (E-05's data), saves a rank-16 LoRA adapter, evaluates on 951-row Kaggle dev.

| Cycle | Branch | LR | Node | Dev Acc | Incumbent? |
|---|---|---|---|---|---|
| 1 | 0 | 1e-4 | node-34674e10b7 | 26.08% | ← |
| 2 | 1 | 5e-5 | node-e417bd2b8b | **30.49%** | **← promoted** |
| 3 | 2 | 3e-5 | node-4ca27d34f7 | 26.29% | no-op |
| 4 | 3 | 1e-5 | node-e41245c00c | 18.09% | no-op |

Wallclock 2h 54m total on a single H200 (each cycle ≈ 27 min train + 12 min eval).

MCGS correctly promoted lr=5e-5 as incumbent after cycle 2, stayed put through cycles 3 and 4 as later scores came in below. These are not state-of-the-art numbers (E-05 with 28 steps hit 43.22% at lr=5e-5; we only ran 8 steps per cycle to stay under the 3-hour budget) — but the search signal is clean: the scores form a unimodal curve with peak at 5e-5 and decay toward both extremes, exactly what you want an lr sweep to show.

Custom `RootFanoutSelector` in the driver (not in the library): forces the first 4 cycles to pick root as parent so we get 4 independent siblings instead of a depth-1 chain. Library default `UCTSelector` would have chained after cycle 1.

### Pitfalls caught along the way
- **`grad_accum=8` → 30 steps/cycle**: initial run would've taken 7h. Changed to `grad_accum=32` + `max_steps=8` → 27 min/cycle training. Matches E-05's effective-batch-32 recipe.
- **Train-eval memory ownership**: HF `Trainer.save_model` + `del trainer, model` leaves ~62 GiB in PyTorch's allocator cache. vLLM at `gpu_memory_utilization=0.85` on the same GPU asks for 118 GiB and explodes with a `Free memory on device cuda:0 (76.69/139.8 GiB) on startup is less than desired` ValueError. Fix: aggressive `gc.collect() + torch.cuda.empty_cache() + torch.cuda.synchronize()` in `train_worker._run_real_stage` AND lower `gpu_memory_utilization` to 0.50 in `kaggle_eval.yaml`. This leaves vLLM with 70 GB (60 GB for the model, 10 GB KV cache — plenty for the 951 × ≤4096-token prompts).
- **`Mamba fast path not available`**: `selective_state_update` / `causal_conv1d_fn` missing in the venv, Nemotron-H falls back to the naïve PyTorch implementation. Costs throughput but not correctness. Per-step time on one H200 is ~3.3 min; installing `mamba-ssm` kernels would cut this but wasn't required for this validation.
- **Spurious "completed" notifications from the harness**: background-command status is driven by the wrapping shell exiting, not the `python ... &` child. Polled `pgrep -f drive.py` + `grep step markers` directly to track real progress.

## 10. Registry keys

```python
TRAINING_BENCHMARKS = {
    "nemo_reasoner": "agent_evolve.benchmarks.nemo_reasoner.NemoReasonerBenchmark",
}

TRAINING_ALGORITHMS = {
    "mcgs": "agent_evolve.training.algorithms.mcgs.search.MCGSSearch",
}

TRAINING_BACKENDS = {
    "h200_single_node": "agent_evolve.backends.tinkerlite.single_node.SingleNodeTinkerLiteBackend",
    "k8s_h200":         "agent_evolve.backends.tinkerlite.elastic.K8sTinkerLiteBackend",
}
```

## 11. How to run

**Smoke (CPU, no GPU):**
```bash
pytest tests/training/ -q                               # 66 tests, < 1 s
python -m agent_evolve.training.run --smoke --cycles 1 \
    --workspace seed_workspaces/nemotron_reasoner \
    --benchmark nemo_reasoner --algorithm mcgs --backend h200_single_node \
    --work-dir runs/smoke
```

**Eval-only against a known adapter (1× H200, ~10 min):**
- Uncomment `seed_adapter_path` in `seed_workspaces/nemotron_reasoner/model/adapter.yaml`.
- Uncomment `limit: 16` in `eval/kaggle_eval.yaml` to do a quick shakeout first, then set `limit: null` for the full 951 rows.
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/fsx/zzsamshi/a-evolve /fsx/zzsamshi/nemotron-auto-research/.venv/bin/python -m agent_evolve.training.run --workspace seed_workspaces/nemotron_reasoner --benchmark nemo_reasoner --algorithm mcgs --backend h200_single_node --cycles 1 --work-dir runs/eval-only`

**Real 4-cycle LR sweep (1× H200, ~3 h):**
- Make sure `seed_adapter_path` is commented out.
- Ensure `train/pipeline.yaml::stages[0].enabled = true`, `max_steps: 8`.
- Run `runs/lr-sweep-4cycle/drive.py` (ad-hoc driver with custom `RootFanoutSelector` + `LRBagMutationProposer`):
  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/fsx/zzsamshi/a-evolve \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true \
    /fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
    runs/lr-sweep-4cycle/drive.py > runs/lr-sweep-4cycle/logs/run.log 2>&1 &
  ```

**Where the logs are:**
- `runs/<name>/logs/run.log` — stdout+stderr (vLLM progress, HF Trainer loss, cycle markers)
- `runs/<name>/<workspace>/evolution/reports/cycle_NNNN.json` — per-cycle MCGSCycleReport
- `runs/<name>/<workspace>/evolution/mcgs_graph.json` — persistent DAG (JSON, reloadable)
- `runs/<name>/<workspace>/evolution/observations/node-*.json` — per-trial TrainingTrialResult with error examples
- `runs/<name>/<workspace>/evolution/memory/{successful,failed}_mutations.jsonl` — BM25-retrievable
- `runs/<name>/nodes/node-*/workspace/checkpoints/adapters/<stage>/adapter_config.json` — the trained LoRA
- `runs/<name>/nodes/node-*/workspace/evolution/eval/<stage>/<split>/metrics.json` — per-cycle scores

## 12. What's deliberately not done yet

1. **Example-driven mutator** — `mutator.propose(parent, graph)` ignores `parent.error_buckets` and `memory.retrieve(...)`. All that data is on disk; the rule-based mutators just don't consume it. §15 ships the *data-side* infrastructure a mutator needs (per-category recipe YAML, solver_distill stage, verifier_gate filter); what's missing is the proposer class that reads `error_buckets[cat]` and emits a `WorkspacePatch` on `data/recipes/default.yaml`.
2. **LLM-driven mutator** — sibling of (1); reads bucket examples + past successes, writes the next YAML patch. Would plug in as a drop-in `propose()` implementation.
3. **Per-category solvers for nemo_reasoner** — §15 defines the Protocol but no `Solver` / `Verifier` / `CoTRenderer` instances ship yet. Prior art (Kaggle public solutions we've read) has drop-in algorithms for all six categories: bit_manipulation per-bit boolean enumeration (konbu17, ~75% solve rate), cryptarithm brute-force digit/op assignment (kimberleyduran, +0.01 LB), gravity/unit/roman/cipher as 4-10 line regex+algebra (nicbarthelemy1). Until these land, `data/recipes/default.yaml` ships every category `solver: disabled` and `solver_distill` is a no-op.
4. **Real RL stages** — the `rl_gspo` stage in `pipeline.yaml` is currently `enabled: false`. The Tinker-style `sample / compute_logprobs / forward_backward("importance_sampling")` protocol is in `backends/tinkerlite/base.py`; the GSPO loss implementation lives in `../nemotron-auto-research/scripts/gspo_update.py` and could be ported.
5. **k8s eval stage** — the k8s backend runs training on cluster pods but eval (`run_eval_plan` → vLLM + LoRA) still executes on the local host. Cloudifying eval would need a second `ComputeTarget` variant that routes vLLM Jobs through k8s, plus a way for the driver to pull `metrics.json` back.
6. **Cross-node DDP (>8-GPU trials)** — `K8sComputeTarget` submits single-pod DDP (`nproc_per_node=world_size`). Training a single trial across 2+ H200 nodes needs a different orchestrator (PyTorchJob / MPIJob with gang scheduling). Orthogonal to the current "elastic fan-out" need.
7. **Persistent vLLM engine across cycles** — currently torn down and reloaded every cycle (~2 min overhead / cycle). A module-level singleton + `load_lora` per cycle would save ~6 min across 4 cycles.
8. **Auto-submit to Kaggle** — the repo sits next to `../nemotron-auto-research/scripts/auto_submit.py` but nothing wires MCGS's incumbent into a submission. Submissions are gated on user judgment because the 5/day quota is tight and the CLAUDE.md-documented dev→LB correlation break (E-33 regressed despite +0.89 dev) means auto-submission would burn quota on noise.
9. **MCGS-aware concurrency** — the fan-out fast path (§14) is a driver-level pattern (`submit_stage_async` + `wait_any`). MCGS's `run_cycle` is still serial. A future `run_cycle_parallel(ctx, max_inflight=N)` hooked to `backend.probe_fanout_capacity` would let the search itself expand siblings concurrently. Algorithm-layer change, decoupled from backend.
10. **OOD synth + teacher verifier gate** — §15 Phase 3. `DataSynthGenerator` Protocol exists but no `ood_augment` stage worker yet. Teacher `synth_generate` stage records `verifier_gate: true` intent in stats but doesn't yet re-read its own rollout JSONL to drop wrong answers (the teacher output schema is `prompt_rendered` / `completion`, distinct from `GeneratedRow` — deferred to a follow-up to keep the proven E-28 distill path untouched).

## 13. Invariants (enforced in tests)

1. `ae.Evolver` is not touched by the new code — same import path, same signature, existing behavior unchanged.
2. Backend never reads or computes reward; `TrainingTrialResult` has no `reward` / `incumbent` fields (`test_backend_no_reward.py`).
3. Benchmark never picks an incumbent; `NemoReasonerBenchmark` has no `compute_reward` / `promote_incumbent` methods (`test_benchmark_no_reward.py`).
4. Loop has no accept/reject gate surface; materialization only happens when MCGS says `incumbent_changed` (`test_loop_no_gate.py`).
5. Every shared dataclass lives in `training/types.py` and is imported elsewhere — no duplicate definitions.
6. Protected workspace layers cannot be mutated by MCGS (`test_workspace_mutation.py`).
7. Graph survives a JSON round-trip (`test_mcgs_save_reload.py`).
8. `.ddp_config.json` produced by `ddp_launcher` (local) and by the k8s backend is byte-identical for the same inputs — both paths import `common_cfg.build_sft_cfg` / `build_gspo_cfg` (`test_common_cfg.py`, `test_override_stage_runner.py`).
9. `K8sTinkerLiteBackend` is a subclass of `SingleNodeTinkerLiteBackend`; the `k8s_h200` registry entry resolves to a class whose `name` attribute equals `"k8s_h200"` (`test_registry.py`).
10. Every row emitted by `solver_distill` contains `\boxed{gt_answer}` exactly (CoT post-processor overrides whatever the renderer wrote — model never trains on a CoT whose final answer drifts from the ground truth) and, when the recipe requires it, exactly one `[verify]: PASS` marker (`test_cot_template.py`, `test_workers_end_to_end.py`).
11. `data_merge` is deterministic under the stable-order dedup contract: rows from input N are evaluated before input N+1, first-seen wins. This preserves upstream priority — solver rows beat teacher rows for the same prompt when both are supplied (`test_dedup.py::test_dedup_preserves_input_order_for_stable_upstream_priority`).
12. A benchmark without `solvers()` / `verifiers()` / `cot_renderers()` still works — `solver_distill` emits an empty JSONL + a diagnostic `drop_reasons={"no_solvers_registered": 1}` rather than crashing. This guarantees the data pipeline is opt-in per benchmark (`test_workers_end_to_end.py::test_solver_distill_empty_benchmark_emits_empty_with_warning`).

## 14. Elastic execution: k8s-first with local fallback

### Why

The local H200 box is one machine; the shared k8s cluster is N unknown. Single-trial deep training fits the box; batch sweeps and parallel MCGS want the cluster. But the cluster is shared — capacity is unpredictable, and sometimes zero. `K8sTinkerLiteBackend` handles all of this in one backend.

### Layer cake

```
MCGS / driver
     │
     ▼
K8sTinkerLiteBackend  (extends SingleNode; re-exposes run_trial + async extras)
     │
     ▼  override_stage_runner(fn) injects fn at each DDP stage
ElasticScheduler                     ← FanoutCapacity.probe_capacity(world_size)
     │
     ├──▶ K8sComputeTarget   (priority 0: batch/v1 Job, PVC-mounted code)
     └──▶ LocalComputeTarget (priority 10: torchrun subprocess + GPU lock)
                                       ← both write the same .ddp_result.json
                                       ← both run the same train_worker_ddp.py
```

### Scheduling policy (ElasticScheduler.run_stage)

Per DDP stage, the scheduler is strict priority-first:

1. Probe every target. If **primary (k8s)** `can_run_now` → submit + wait.
2. Else if **primary** `can_queue` → submit, tolerate `Pending` up to `queue_timeout_secs` (default 600s), cancel + fall back on `PendingTimeout`.
3. Else walk fallback targets in priority order; submit to the first one that can run-now or queue.
4. Else raise `CapacityExhausted`. MCGS surfaces this as a trial failure; the driver can decide to retry later.

Once a Job transitions to `Running`, no stage-level timeout is imposed — `TrialBudget` already governs wall-clock bounds at the MCGS level.

Key design point: `can_queue=False` when the cluster has **zero matching H200 nodes**. The scheduler then skips the queue wait entirely and goes straight to local, so a dead cluster doesn't waste 10 minutes of every cycle on hope.

### Fan-out fast path

Drivers that want to dispatch multiple trials concurrently (LR sweep, MCGS root expansion) use the backend's async API:

```python
cap = backend.probe_fanout_capacity(world_size=8)
# FanoutCapacity(recommended=N, breakdown={...}, reason=...)
max_inflight = min(len(trials), max(1, cap.recommended))

# Submit up to max_inflight non-blocking
inflight = [backend.submit_stage_async(cfg_path=..., world_size=8, log_dir=...)
            for _ in range(max_inflight)]

# Drain + refill
while inflight or pending:
    handle, result = backend.wait_any([h for h in inflight])
    inflight.remove(handle)
    if pending:
        inflight.append(backend.submit_stage_async(...))
```

The fan-out formula (`ElasticScheduler.probe_capacity`):

```
recommended = k8s_run_now + (k8s_queue_budget if k8s.can_queue else 0) + local_run_now
            where *_run_now = available_gpus // world_size
            and k8s_queue_budget defaults to 4 (constructor kwarg)
```

Example capacities at `world_size=8`, `k8s_queue_budget=4`:

| Situation | k8s_run_now | queue_budget | local_run_now | recommended |
|---|---|---|---|---|
| Cluster idle (16 GPU free) + local idle | 2 | 4 | 1 | **7** |
| Cluster saturated but queueing + local idle | 0 | 4 | 1 | **5** |
| Zero H200 nodes + local idle | 0 | 0 | 1 | **1** |
| Cluster saturated + local locked | 0 | 4 | 0 | **4** |
| Nothing free, but primary can queue | 0 | 0 | 0 | **1** (fallback) |
| Nothing available anywhere | 0 | 0 | 0 | **0** (driver should back off) |

`k8s_queue_budget=0` is the friendly-to-other-tenants mode — the backend only dispatches what can actually run immediately and never holds Pending pods.

### Local GPU bookkeeping

Multiple driver processes (or MCGS + an interactive shell) can coexist on one node. `LocalComputeTarget` coordinates via `flock` on `/fsx/.ae_locks/gpu_{0..7}.lock`:

- `acquire_gpus(count, pool)` grabs N locks non-blockingly, writes `pid=<ours>` into each file, returns a `GpuLease`. On subprocess exit (or lease `.release()`), flocks drop and files unlink.
- `capacity_probe` unions OS flock state with `nvidia-smi --query-gpu=memory.free`. If `nvidia-smi` is missing (CI), falls back to lock state alone.
- Stale-PID sweep: lock files left by a crashed process (flock released but file lingers) are reclaimed on the next probe via `os.kill(pid, 0)` check.

Not a replacement for SLURM — single-node coordination only.

### Deployment prereqs

- PVC named `fsx-zzsamshi` that maps to the same FSx filesystem accessible at `/fsx` on the host (code, models, checkpoints all via the shared mount).
- Image built from [`elastic/k8s/image/Dockerfile`](agent_evolve/backends/tinkerlite/elastic/k8s/image/Dockerfile) and pushed to a registry the cluster can pull from. The image is deliberately **thin** — it contains torch+transformers+peft+vllm+kubernetes and nothing else. Application code is mounted via the PVC so iteration is "save file → resubmit Job" with no rebuild.
- `pip install 'a-evolve[k8s]'` (adds `kubernetes>=31.0`).
- Optional: node label `nvidia.com/gpu.product=H200` so `node_selector` can target the right nodes. Construct with `node_selector={"nvidia.com/gpu.product": "H200"}`.

If `kubernetes` is missing or no kubeconfig is reachable, `K8sComputeTarget` fails to construct and the backend logs a warning, falling back to local-only. This means `backend="k8s_h200"` is always safe to specify — it degrades gracefully.

### Graceful degradation matrix

| Local H200 | K8s kubeconfig | Behavior |
|---|---|---|
| Available | Available | Elastic: k8s priority, local fallback on `PendingTimeout` |
| Available | Missing / unreachable | Local-only (k8s target fails to construct; backend logs warning) |
| Disabled via `local_enabled=False` | Available | K8s-only (block on queue indefinitely) |
| Disabled | Missing | `RuntimeError` at backend construction |

### Test coverage

44 unit tests under `tests/backends/tinkerlite/elastic/`:
- `test_gpu_lock.py` — flock correctness, stale-PID reclaim, partial-allocation rollback
- `test_local_target_probe.py` — capacity probe under nvidia-smi/lock state combinations
- `test_job_manifest.py` — Job manifest structure (GPU limits match world_size, PVC mount, env vars, no restart)
- `test_scheduler.py` — priority-first routing, queue fallback, zero-node skip, capacity exhaustion
- `test_fanout_capacity.py` — the `recommended` formula under 7 scenarios
- `test_override_stage_runner.py` — the ContextVar hook actually intercepts `run_sft_ddp` and releases on scope exit
- `test_common_cfg.py` — k8s path and local path generate byte-identical `.ddp_config.json`
- `test_registry.py` — `k8s_h200` dotted path resolves; subclass relationship preserved

All tests run CPU-only — no k8s cluster, no kubeconfig, no GPUs required. Existing 117 training/backends tests continue to pass (zero regressions).

## 15. Data-generation pipeline

### Why

Teacher-LLM distillation (the existing `synth_generate` stage) is the easy path to CoT training data but has two known failure modes:

1. **Label noise.** The teacher hallucinates a wrong answer; the CoT that ships as "gold" diverges from the training-row `answer` column. Kaggle public code shows the resulting LoRA learning to ignore its own CoT and guess.
2. **LB overfit.** Public notebook report: a synthetic-CoT pipeline hit 98.9% on bit_manipulation locally but *dropped* on the leaderboard — adding single-category volume without structural diversity over-specializes the adapter.

The cheapest fix shipped repeatedly across public notebooks (konbu17, kimberleyduran, nicbarthelemy1): **use deterministic per-category solvers to produce verified gold CoT, with ground truth baked in as the only label source**. Rank-16 LoRA on ~6k verified rows beats the teacher-only path on LB while costing zero GPU-hours for data generation.

This section documents the *infrastructure* for that path. Concrete solver implementations for nemo_reasoner are a follow-up (§12 item 3).

### Layered architecture

```
benchmark-agnostic (training/data/)                       benchmark-specific
                                                          (benchmarks/<name>/)
  Solver     ─────────────┐                              ┌─ BitManipSolver
  Verifier   ─ Protocols ─┤                              ├─ CryptarithmSolver
  CoTRenderer ────────────┤   ← benchmark wires          ├─ BitManipVerifier
  DataSynthGenerator ─────┘     implementations via      ├─ BitManipRenderer
                                 solvers() / verifiers() ├─ ... (per category)
                                 / cot_renderers()       └─ iter_training_rows
  GeneratedRow (JSONL wire format: id, prompt, answer, category, cot, source, metadata)
  DataRecipe   (YAML: categories × solver on/off × upsample × filters)

  solver_distill_worker ─▶ runs solver, verifies, renders, post-processes CoT
  data_merge_worker     ─▶ dedup + upsample + register in data/sources.yaml

  cot_template          ─▶ forces \boxed{answer}, injects [verify]: PASS
  verifier_gate         ─▶ drops wrong answer / missing marker / over-long CoT
  dedup                 ─▶ prompt_hash | prompt_and_source_hash
```

### Pipeline shape

Two new stages in `train/pipeline.yaml`, fully independent of `synth_generate`:

```yaml
stages:
  - name: solver_distill
    type: solver_distill
    enabled: false                 # flip when solvers land
    recipe: data/recipes/default.yaml
    out_subdir: data/generated/solver_distill

  - name: teacher_distill
    type: synth_generate           # existing; unchanged
    enabled: false
    # verifier_gate: true          # Phase 2 — intent field, not yet enforced

  - name: data_merge
    type: data_merge
    enabled: false
    recipe: data/recipes/default.yaml
    inputs:
      - data/generated/solver_distill
      # - data/synth   # teacher output (when enabled)
    out_subdir: data/generated/data_merge
    upsample_from_recipe: true
    register_in_sources: true
    source_format: jsonl_cot

  - name: sft_warmup
    type: sft
    enabled: true                  # unchanged — consumes data/sources.yaml
    ...
```

Per cycle, `single_node._run_pipeline` dispatches `solver_distill` → `data_merge` → `sft`. The data-stage output subdirs live under `data/generated/`, which `manifest.yaml` lists as an artifact layer — they're regenerated each trial, never committed.

### Recipe schema (evolvable)

`data/recipes/default.yaml` is added to `evolvable_layers` — MCGS mutators target it:

```yaml
recipe_name: solver_first_v1
categories:
  bit_manipulation:
    solver: enabled               # enabled | disabled
    solver_upsample: 3            # per-source upsample at merge time
    teacher_upsample: 1
    cot_template: verify_pass_v1
  cryptarithm:
    solver: enabled
    solver_upsample: 12           # +0.01 LB ratio per kimberleyduran
filters:
  require_verify_pass: true       # drop rows missing [verify]: PASS
  max_cot_tokens: 7600            # Nemotron-3-Nano budget per konbu17
  dedup_by: prompt_and_source_hash
```

Unknown fields on a category are preserved under `extra` so benchmark-specific knobs (`ood_ratio`, `ood_operators`, ...) can ride along without a schema bump.

### CoT post-processing — the load-bearing guarantee

Every row emitted by `solver_distill` (and every row a future `ood_augment` stage emits) passes through `cot_template.postprocess_cot`:

1. Replace any `\boxed{...}` in the CoT with `\boxed{<ground-truth answer>}`. If none exists, append one.
2. Inject `[verify]: PASS` immediately before the final box (unless one is already present).
3. Strip trailing whitespace.

The function is idempotent — running it twice is the same as running it once (`test_cot_template.py::test_postprocess_idempotent`). It also sidesteps the `re.sub` backslash-escape trap (`\b` in a replacement becomes `\x08`, silently corrupting data) via a lambda replacement. SFT invariant: the label the model is trained against is exactly the ground truth from `train.csv`, regardless of what the generator hallucinated.

### Verifier gate (shared across generators)

`verifier_gate.apply_verifier_gate(rows, verifiers, filters)` is the single checkpoint every `GeneratedRow` source will pass through before `data_merge`:

- extracts `\boxed{...}` from the CoT,
- calls `verifier[row.category].check(pred, row.answer)`,
- drops rows missing `[verify]: PASS` when `filters.require_verify_pass`,
- drops rows whose CoT exceeds `filters.max_cot_tokens` (whitespace token count by default; benchmarks can inject a real tokenizer counter).

Stats report `kept`, `dropped_wrong_answer`, `dropped_missing_verify_mark`, `dropped_cot_too_long`, plus `per_category_kept` / `per_category_dropped` counters so a driver can attribute dataset loss per category without re-deriving it.

`solver_distill_worker` applies its own inline correctness check (solver + verifier, pre-render) — the gate function exists for `synth_generate` and future stages that generate first and filter second.

### Dedup + upsample

`data_merge` loads every input directory's `rows.jsonl` in order, runs `dedup.dedup(rows, filters)` (first-seen wins, stable iteration), then `dedup.upsample(deduped, recipe)` which multiplies by `{source}_upsample` per-category. The merged JSONL registers in `data/sources.yaml` as one more entry with `format: jsonl_cot` — the SFT loader dispatches on format.

Dedup mode is a recipe knob, not a hardcoded choice:
- `prompt_hash`: the same prompt produced by any source collapses to one row. Use when the teacher has nothing to add over the solver.
- `prompt_and_source_hash` (default): same prompt from different sources survives. Use when training on *both* solver-gold and teacher-gold gives the adapter a richer signal.

Input order matters: a duplicate prompt present in both `solver_distill` and `teacher_distill` outputs is kept from whichever appears first in the `inputs:` list. Convention is solver first — solver rows are more trustworthy.

### Protocol extensions on `TrainingBenchmarkAdapter`

Five new *optional* methods were added to the benchmark Protocol (`benchmarks/training_base.py`). Presence is checked at runtime by the stage workers — a benchmark without them keeps working (teacher-only via `synth_generate`):

| Method | Purpose |
|---|---|
| `iter_training_rows(workspace)` | Yield `TrainingExample` over the benchmark's training corpus |
| `classify_category(prompt)` | Deterministic prompt → category key (regex / keyword) |
| `solvers()` | `dict[cat, Solver]` — per-category deterministic solver |
| `verifiers()` | `dict[cat, Verifier]` — per-category answer checker |
| `cot_renderers()` | `dict[cat, CoTRenderer]` — per-category trace → CoT |
| `data_synth_generators()` | `dict[cat, DataSynthGenerator]` — Phase 3 OOD generation |

### MCGS interface — the evolvable axes

The recipe YAML is *the* surface for data-mix mutators:

```python
# Example mutator (follow-up PR)
class SolverUpsampleMutationProposer:
    bag = (1, 2, 3, 8, 12)
    def propose(self, parent, graph):
        weakest = self._weakest_category(parent.error_buckets)
        new_val = self._bag_rotate(parent, weakest)
        return WorkspaceMutation(patch=WorkspacePatch(operations=[
            PatchOperation("replace", "data/recipes/default.yaml",
                           key_path=["categories", weakest, "solver_upsample"],
                           value=new_val),
        ]))
```

Axes a mutator can target without schema churn:
- `categories.<cat>.solver: enabled|disabled` — on/off per category
- `categories.<cat>.solver_upsample` / `teacher_upsample` — per-source weight
- `categories.<cat>.cot_template` — switch template version
- `categories.<cat>.extra.*` — benchmark-specific (`ood_ratio`, `ood_operators`, ...)
- `filters.require_verify_pass` / `max_cot_tokens` / `dedup_by` — filter strictness

Error-bucket connection: `analyze_errors` returns a `dict[bucket_key, count]`. For the data pipeline to be usefully steered by MCGS, the key convention is `<category>__<error_mode>` (e.g. `bit_manipulation__wrong_shift`, `cryptarithm__unsolved`). The mutator reads `parent.error_buckets`, finds the largest category contribution, and patches that category's recipe entry. Recipe-change → fork → new trial → new buckets: a closed loop between evaluation signal and data recipe.

### Test coverage

46 unit tests under `tests/training/data/`:

- `test_base_protocols.py` — Protocol conformance + `GeneratedRow.to_dict` / `from_dict` round-trips
- `test_recipe.py` — schema validation (bad solver value, non-int upsample, unknown dedup mode) + seed recipe parses + extras preserved
- `test_dedup.py` — key determinism, first-seen semantics, per-source upsample, unknown-source default + extras fallback
- `test_cot_template.py` — force-box + inject-verify + idempotence + re.sub backslash trap regression guard
- `test_verifier_gate.py` — drops on wrong answer / missing marker / over-long CoT, stats accuracy
- `test_workers_end_to_end.py` — solver_distill happy path + wrong-answer drop + empty-benchmark graceful degradation + data_merge dedup+upsample+sources.yaml registration + missing-input resilience
- `test_end_to_end_smoke.py` — full pipeline dispatcher round-trip through `SingleNodeTinkerLiteBackend._run_pipeline` in mock mode

## 16. Reserved abstractions for future training jobs

The training subsystem exposes **four plugin points** so new frameworks
(sklearn, JAX, tabular GBM), new adapters (DoRA, full fine-tune, QLoRA),
and new data-gen methods can be added without editing the core loop,
algorithm, or workspace code. Each is a Protocol + registry with a
decorator-based registration API.

| Protocol | File | Registry / decorator | Answers the question |
|---|---|---|---|
| `TrainingJobRunner` | [training/runner_protocol.py](agent_evolve/training/runner_protocol.py) | `TRAINING_JOB_RUNNERS` dotted-path | "How is one candidate trained + evaluated end-to-end?" (may be entirely non-LLM) |
| *(stage worker)* | [training/stage_registry.py](agent_evolve/training/stage_registry.py) | `@register_stage("<type>")` + `StageContext`/`StageResult` | "What happens when `pipeline.yaml` has `stage.type: <new>`?" |
| `ModelAdapter` | [backends/tinkerlite/adapters/base.py](agent_evolve/backends/tinkerlite/adapters/base.py) | `@register_adapter("<kind>")` | "How does a trainable surface attach to a base model, save, load?" |
| `DataGenerator` | [training/data/generator.py](agent_evolve/training/data/generator.py) | `@register_data_generator("<name>")` | "How do I produce training rows from (workspace, recipe, benchmark)?" |

Pre-existing Protocols — `TrainingBenchmarkAdapter`, `ComputeTarget`,
`TrainingClient` / `SamplingClient` — continue to serve the same axes they
always did. `TinkerLiteBackend` is now a documented refinement of
`TrainingJobRunner` (adds LLM client factories); non-LLM runners implement
only the root Protocol.

Transitional shim — [benchmarks/helpers.py](agent_evolve/benchmarks/helpers.py) —
routes `teacher_distill.py` and `rl.py` through `benchmark.build_eval_prompt
/ extract_final_answer / verify` when available, falling back to the legacy
`nemo_reasoner` imports otherwise. New benchmarks implement the optional
Protocol methods; Nemotron-Reasoner pipelines stay green during the
migration.

**Hands-on integration recipes live in [INTEGRATION.md](INTEGRATION.md)** —
6 sections covering a sklearn runner example, a DoRA adapter example, a
RulePerturb generator example, the benchmark Protocol table, seed-workspace
layout for non-LLM jobs, a registries cheat sheet, and a FAQ.

### Test coverage

24 unit tests under the new namespaces:

- `tests/training/test_job_runner_protocol.py` (6) — Protocol conformance + registry resolution + `TinkerLiteBackend` refines `TrainingJobRunner`.
- `tests/training/test_stage_registry.py` (5) — built-in stages registered on import + collision detection + idempotent re-registration.
- `tests/backends/tinkerlite/test_adapter_registry.py` (5) — `LoRAAdapter` Protocol conformance + `CheckpointRef.kind` widened + new-adapter registration round-trip.
- `tests/training/data/test_data_generator_registry.py` (5) — `solver_distill` / `teacher_llm` registered + unified `type: generate` dispatcher end-to-end.
- `tests/training/test_benchmark_adapter_hooks.py` (3) — helper prefers benchmark method; falls back to `nemo_reasoner`; `NemoReasonerBenchmark` conforms.

Total suite: **223 tests passing, zero regressions.**

### Design stance

- **No behavior change today.** Every new Protocol ships with an implementation that captures the existing LoRA/Nemotron pipeline exactly (`LoRAAdapter`, `SolverDistillGenerator`, `TeacherLLMGenerator`). The reserved extension points are additive.
- **Plugin registration is import-time.** Decorators fire when the module loads; plugins outside `agent_evolve/` must be imported from the benchmark's `__init__.py` or a driver script before `evolver.run()`.
- **Hot-path refactor deferred.** `HFTrainingClient` + `ddp_worker._build_model_and_optim` + `VLLMSamplingClient` still inline the LoRA path directly; a follow-up PR will route them through `resolve_adapter(cfg.type)` so registered non-LoRA adapters actually get invoked. The Protocol + `LoRAAdapter` are shipped as the *reserved* extension point now — low-risk, no GPU-lifecycle churn.

All CPU-only, no GPU, no network. Full suite: 242 pass, 2 skipped, zero regression.
