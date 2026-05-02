# nemo_mas — Orchestrator-Worker MAS Training Algorithm

Parallel alternative to `mcgs` in `TRAINING_ALGORITHMS`. Same workspace
contract and `run_cycle(ctx) -> MCGSCycleReport` signature; different
search: an LLM **orchestrator** spawns four specialist **workers** that
share a typed-record **memory** (BM25, JSONL, append-only, ref-validated).

- Design: `seed_workspaces/nemo_mas_reasoner/DESIGN.md`
- Workspace: `seed_workspaces/nemo_mas_reasoner/`
- Driver: `examples/nemo_mas_reasoning_example/drive_nemo_mas.py`
- Tests: `tests/model/algorithms/nemo_mas/` (98 tests, ~0.1s, no AWS/GPU)

## Architecture

```
  Orchestrator (no writes, no exec)
      │  spawn_and_run_subagent(role, task)
      ▼
  [Analyst] [DataEngineer] [Theorist] [Engineer]
      │            │            │          │
      └────────────┴──► RecipeMemory (BM25, typed records, refs DAG)
                              │
                              ▼
                     Backend tools (run_eval, launch_training, ...)
                     supplied via `backend_registry`
```

Workers are stateless across spawns; only channel is the memory store.
`mem_write` rejects out-of-whitelist kinds and ref-rule violations.

## Agents

Roles: **Orchestrator** (plans, spawns, read-only) + four workers.
Role tools in `seed_workspaces/nemo_mas_reasoner/tools/<role>.yaml`;
skills in `seed_workspaces/nemo_mas_reasoner/skills/<role>/`.
Platform tools (`mem_*`, `skill_*`, `read_file`, `list_dir`, `spawn_*`)
are inherited from `seed_workspaces/_common_model/tools/`.

All workers share: `mem_write/search/recent/get/link`, `skill_index`,
`skill_load`, `read_file`, `list_dir`.

### Orchestrator — `prompts/system.md`

- **Extra tools:** `spawn_and_run_subagent`, `call_existing_agent`, `mem_search`, `mem_recent`, `mem_get`, `read_file`, `list_dir`
- **Writes:** none · **Skills:** none

### Analyst — `prompts/analyst.md`, `skills/analyst/`

- **Backend tools:** `sample_jsonl`, `count_by_field`, `length_distribution`, `run_eval`, `run_short_training`, `plot_loss_curve`, `compute_data_gap_table`
- **Writes:** `data_audit_finding`, `benchmark_rule`, `profile_run`, `eval_report`, `error_pattern`, `data_gap` (+ `breakthrough`, `failed_attempt`)
- **Skills:** `audit_jsonl_quality`, `categorize_eval_errors`, `compute_data_gap`, `probe_benchmark_format`, `profile_lr_sweep`

### Data Engineer — `prompts/data_engineer.md`, `skills/data/`

- **Backend tools:** `call_teacher_model`, `load_checkpoint_for_inference`, `batch_generate`, `filter_by_gold`, `minhash_dedup`, `apply_format_filter`, `format_validate`, `mix_sources`, `write_jsonl`
- **Writes:** `distill_batch`, `dataset_snapshot` (+ cross-cutting)
- **Skills:** `format_validate`, `minhash_dedup`, `mix_by_curriculum`, `solver_self_distill_with_rejection`, `teacher_distill_long_cot`

### Theorist — `prompts/theorist.md`, `skills/theorist/`

Reasoning + records only, no side effects.

- **Backend tools:** `diff_yaml`, `render_recipe_diff`
- **Writes:** `hypothesis`, `recipe_proposal` (+ cross-cutting)
- **Skills:** `failure_pattern_recognition`, `lr_warmup_for_long_cot`, `propose_recipe_from_gap`, `when_to_skip_sft`

### Engineer — `prompts/engineer.md`, `skills/engineer/`

Training always routes through the platform `StageRegistry`
(`agent_evolve/model/runners/stages/*.py`). Engineer **never** scaffolds
runner scripts.

- **Backend tools:** `launch_training`, `read_training_log`, `read_checkpoint_metric`, `rerun_recipe_with_seeds`, `compute_stability`
- **Writes:** `training_run`, `cv_result` (+ cross-cutting)
- **Skills:** `cross_validate_recipe`, `run_training_stage`

## Record kinds (schema.py)

Enforced on every `mem_write`; violations return a structured error.

| kind | author | refs required |
|---|---|---|
| `data_audit_finding`, `benchmark_rule`, `profile_run`, `error_pattern`, `data_gap` | analyst | — |
| `eval_report` | analyst | ≥1 `training_run` |
| `distill_batch`, `dataset_snapshot` | data_engineer | — |
| `hypothesis` | theorist | — |
| `recipe_proposal` | theorist | ≥1 `eval_report` or `data_gap` |
| `training_run` | engineer | ≥1 `recipe_proposal` **and** ≥1 `dataset_snapshot` |
| `cv_result` | engineer | ≥1 `training_run` |
| `breakthrough` | any | ≥1 (any kind) |
| `failed_attempt` | any | — |

The refs DAG is the audit trail behind every promotion:
`cv_result → training_run → {recipe_proposal, dataset_snapshot} → {eval_report|data_gap, distill_batch*}`.

## Files

```
__init__.py     public API
schema.py       MemoryRecord, KIND_WHITELIST, REF_RULES, validate_record
memory.py       RecipeMemory — JSONL + vendored BM25
tools.py        per-role tool factories
spawner.py      SpawnHandler — wraps BedrockAgent for workers
orchestrator.py NemoMASAlgorithm — TRAINING_ALGORITHMS entry
backends.py     local_handlers + BackendBridge + demo_compute_handlers
```

Stdlib only. BM25 vendored; MinHash uses content fingerprints.

## Quickstart

```bash
# Tests
PYTHONPATH=. pytest tests/model/algorithms/nemo_mas/ -v

# Dry-run: stub BedrockAgent, no compute
PYTHONPATH=. .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 3 --mode dry-run --print-prompts

# Demo: real Bedrock, mocked compute
PYTHONPATH=. .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 5 --mode demo

# Real: Bedrock + GPUs + SingleNodeTinkerLiteBackend
PYTHONPATH=. .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 10 --mode real \
    --workspace seed_workspaces/nemo_mas_reasoner \
    --work-dir runs/nemo-mas-10
```

## Wiring a backend

`NemoMASAlgorithm(backend_registry=...)` takes a `Mapping[str, Callable]`.
Compose from:

- `local_handlers(workspace_root)` — stdlib-only tools (`sample_jsonl`, `format_validate`, `minhash_dedup`, `diff_yaml`, `read_training_log`, `read_checkpoint_metric`, …)
- `BackendBridge(workspace_root, benchmark, backend).as_registry()` — delegates compute-bound tools (`run_eval`, `launch_training`, `rerun_recipe_with_seeds`, `load_checkpoint_for_inference`, `batch_generate`) to the backend
- `demo_compute_handlers()` — plausible mock outputs for dry-run/tests
- Your own `Callable[..., str]` returning a JSON string

```python
from agent_evolve.model.algorithms.nemo_mas import NemoMASAlgorithm
from agent_evolve.model.algorithms.nemo_mas.backends import (
    BackendBridge, local_handlers,
)

bridge = BackendBridge(
    workspace_root=workspace,
    benchmark=NemoReasonerBenchmark(),
    backend=SingleNodeTinkerLiteBackend(mock=False),
)
algo = NemoMASAlgorithm(backend_registry={
    **local_handlers(workspace),
    **bridge.as_registry(),
    "call_teacher_model": my_teacher_call,   # not bridged by default
})
```

## vs. `mcgs`

| | `mcgs` | `nemo_mas` |
|---|---|---|
| Search | UCT graph + branches + top-k | LLM orchestrator + 4 workers |
| Mutation source | `BaselineMutationProposer` | Theorist's `recipe_proposal` records |
| Promotion | `PromotionPolicy` | `cv_result` tagged "stable" |
| Memory | `NodeMemoryStore` (per-node) | `RecipeMemory` (typed, ref DAG) |
| Comms | Implicit (graph + patches) | Explicit (`mem_write` + refs, auditable) |
| Best for | Clean hyperparameter sweeps | Multi-axis exploration (data / recipe / RL) |

Both register to `TRAINING_ALGORITHMS`; switch via `TrainingEvolver(algorithm=...)`.

## Extending

- **New role:** extend `KIND_WHITELIST` in `schema.py`; add
  `tools.py::_BACKEND_TOOL_CATALOGUE` entry; drop `prompts/<role>.md`,
  `tools/<role>.yaml`, `skills/<role>/`.
- **New record kind:** add to the role in `KIND_WHITELIST`; optionally
  add a `REF_RULES` entry; update prompts.
- **New backend tool:** append to `_BACKEND_TOOL_CATALOGUE[<role>]`;
  implement in `backends.py::local_handlers` or `BackendBridge`; add a
  stub in `demo_compute_handlers()`.

## What's yours, not the algorithm's

The Python here is a runtime. Your edge lives in the four artifacts it
serves:

1. `prompts/benchmark_reference.md` — first-hand eval observations
2. `prompts/system.md` — task-crafting / manager style
3. `skills/<role>/*.md` — every failed cycle should yield a new skill
4. `KIND_WHITELIST` + `REF_RULES` — the epistemic contract

Replace one and the system behaves differently. That's the point.
