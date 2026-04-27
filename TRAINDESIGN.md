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

Each cycle: MCGS selects a parent node → proposes a mutation (e.g. `train/optimizer.yaml::lr`) → forks the candidate workspace → backend runs an SFT training stage → backend evaluates the resulting adapter → benchmark returns metrics + error buckets + a validity report → MCGS computes reward, backprops, updates the top-k, decides whether to promote incumbent. The existing `ae.Evolver(...)` is untouched.

Four axes, four owners:

- **Workspace** = training DNA on disk (`seed_workspaces/<name>/`)
- **MCGS** = search (selection, mutation, reward, backprop, promotion, memory, fusion)
- **Backend** = execution only (train, save adapter, load for eval, sample)
- **Benchmark** = evaluation semantics (primary metric, split loading, error taxonomy, validity)

MCGS never executes a forward pass. Benchmark never computes a reward or picks an incumbent. Backend never scores a trajectory. These are enforced with tests (`test_backend_no_reward.py`, `test_benchmark_no_reward.py`, `test_loop_no_gate.py`).

## 2. Package layout

```text
agent_evolve/
├── training/
│   ├── api.py                    TrainingEvolver facade
│   ├── types.py                  All shared dataclasses (single source of truth)
│   ├── registries.py             Dotted-path registry for benchmarks / algorithms / backends
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
│   └── runners/
│       ├── data_worker.py        render_datums (smoke) + render_hf_dataset (real)
│       ├── train_worker.py       MockTrainingClient (smoke) OR HF+PEFT Trainer (real)
│       ├── eval_worker.py        run_eval_plan: smoke stub OR vLLM + LoRA
│       └── pack_adapter_worker.py  Only used by smoke path
├── backends/tinkerlite/
│   ├── base.py                   TinkerLiteBackend Protocol + Datum/ModelInput/SamplingParams
│   ├── single_node.py            SingleNodeTinkerLiteBackend (mock & real paths)
│   └── mock_clients.py           MockTrainingClient / MockSamplingClient
└── benchmarks/
    ├── training_base.py          TrainingBenchmarkAdapter Protocol
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

## 6. The two backends: smoke vs real

`SingleNodeTinkerLiteBackend(mock: bool)`:

- `mock=True` → `MockTrainingClient` for training, deterministic metrics.json stub for eval. Used by all 66 unit tests and the `--smoke` CLI path.
- `mock=False` → real HF Trainer + LoRA load from `/fsx/models/Nemotron-3-Nano-30B-A3B-BF16`, real vLLM + LoRA eval on 951-row Kaggle dev CSV.

The flag propagates from `TrainingEvolveConfig.smoke` through `TrainingEvolver._resolve_backend` to `backend.mock` so `--smoke` on the CLI actually controls the real/mock split.

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
    "k8s_h200": "agent_evolve.backends.tinkerlite.k8s.K8sTinkerLiteBackend",   # not shipped
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

1. **Example-driven mutator** — `mutator.propose(parent, graph)` ignores `parent.error_buckets` and `memory.retrieve(...)`. All that data is on disk; the rule-based mutators just don't consume it.
2. **LLM-driven mutator** — sibling of (1); reads bucket examples + past successes, writes the next YAML patch. Would plug in as a drop-in `propose()` implementation.
3. **Real RL stages** — the `rl_gspo` stage in `pipeline.yaml` is currently `enabled: false`. The Tinker-style `sample / compute_logprobs / forward_backward("importance_sampling")` protocol is in `backends/tinkerlite/base.py`; the GSPO loss implementation lives in `../nemotron-auto-research/scripts/gspo_update.py` and could be ported.
4. **k8s backend** — PR9 in the original plan. Skeleton file `k8s.py` is not yet shipped.
5. **Persistent vLLM engine across cycles** — currently torn down and reloaded every cycle (~2 min overhead / cycle). A module-level singleton + `load_lora` per cycle would save ~6 min across 4 cycles.
6. **Auto-submit to Kaggle** — the repo sits next to `../nemotron-auto-research/scripts/auto_submit.py` but nothing wires MCGS's incumbent into a submission. Submissions are gated on user judgment because the 5/day quota is tight and the CLAUDE.md-documented dev→LB correlation break (E-33 regressed despite +0.89 dev) means auto-submission would burn quota on noise.

## 13. Invariants (enforced in tests)

1. `ae.Evolver` is not touched by the new code — same import path, same signature, existing behavior unchanged.
2. Backend never reads or computes reward; `TrainingTrialResult` has no `reward` / `incumbent` fields (`test_backend_no_reward.py`).
3. Benchmark never picks an incumbent; `NemoReasonerBenchmark` has no `compute_reward` / `promote_incumbent` methods (`test_benchmark_no_reward.py`).
4. Loop has no accept/reject gate surface; materialization only happens when MCGS says `incumbent_changed` (`test_loop_no_gate.py`).
5. Every shared dataclass lives in `training/types.py` and is imported elsewhere — no duplicate definitions.
6. Protected workspace layers cannot be mutated by MCGS (`test_workspace_mutation.py`).
7. Graph survives a JSON round-trip (`test_mcgs_save_reload.py`).
