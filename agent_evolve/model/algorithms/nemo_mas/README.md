# nemo_mas — Orchestrator-Worker MAS Training Algorithm

Independent, parallel alternative to `mcgs` in `TRAINING_ALGORITHMS`.
Same workspace contract, same `run_cycle(ctx) -> MCGSCycleReport`
signature, same registry. Different search strategy: instead of UCT
over a graph of mutations, an LLM **orchestrator** spawns four
specialist **worker** roles that read and write a shared typed-record
**memory** with **BM25** search.

> **Design doc**: `seed_workspaces/nemo_mas_reasoner/DESIGN.md`
> **Workspace**: `seed_workspaces/nemo_mas_reasoner/`
> **Driver**: `examples/nemo_mas_reasoning_example/drive_nemo_mas.py`
> **Tests**: `tests/model/algorithms/nemo_mas/`

---

## Architecture

```
                       ┌──────────────────────────────┐
                       │       Orchestrator           │
                       │  (BedrockAgent, no execute)  │
                       │  Tools: spawn / mem-read     │
                       └──────────┬───────────────────┘
                                  │ spawn_and_run_subagent(role, task)
            ┌────────┬────────────┼──────────────┬──────────┐
            ▼        ▼            ▼              ▼          ▼
        ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Analyst │ │Data      │ │Theorist  │ │Engineer  │
        │        │ │Engineer  │ │(no exec) │ │          │
        └───┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
            │           │            │            │
            └───────────┴──────┬─────┴────────────┘
                               ▼
              ┌─────────────────────────────────────┐
              │       RecipeMemory (BM25)           │
              │  records.jsonl — typed, append-only │
              │  whitelist + ref rules enforced     │
              └─────────────────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────┐
              │   Backend tools (caller-supplied)   │
              │  run_eval / launch_training / ...   │
              │  Stubbed by default; wire in real   │
              │  backend via backend_registry kwarg │
              └─────────────────────────────────────┘
```

The orchestrator's only knobs are which worker to spawn and what task
message to draft. Workers are stateless across spawns; their only
side-channel is the memory store. Roles are constrained at the schema
layer — `mem_write(kind=...)` rejects out-of-whitelist kinds and
violations of per-kind ref rules.

---

## File map

```
agent_evolve/model/algorithms/nemo_mas/
├── __init__.py            (35 lines)  Public API
├── schema.py              (236 lines) MemoryRecord, KIND_WHITELIST, REF_RULES, validate_record
├── memory.py              (362 lines) RecipeMemory — JSONL store + vendored BM25
├── tools.py               (478 lines) Per-role tool factories (mem_*, skill_*, file_*, backend stubs)
├── spawner.py             (262 lines) SpawnHandler — wraps BedrockAgent for workers
├── orchestrator.py        (388 lines) NemoMASAlgorithm — TRAINING_ALGORITHMS entry
├── backends.py            (~470 lines) Tier-1 local handlers + Tier-2 BackendBridge + demo handlers
└── README.md              (this file)
```

No external dependencies beyond the standard library. BM25 is vendored
(no `rank_bm25`); MinHash dedup uses content fingerprints (no
`datasketch`).

---

## Quickstart

### Run the test suite

```bash
PYTHONPATH=. pytest tests/model/algorithms/nemo_mas/ -v
```

98 tests, runs in ~0.1s. No AWS / GPU required.

### Run the driver in dry-run mode

```bash
PYTHONPATH=. .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 3 --mode dry-run --print-prompts
```

`--mode dry-run` injects a stub BedrockAgent so no Bedrock calls are
made. The orchestrator's prompt is built and dumped; cycle reports are
printed; nothing trains.

### Run with the demo backend (still needs Bedrock)

```bash
PYTHONPATH=. .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 5 --mode demo
```

Compute-bound tools (`run_eval`, `launch_training`, etc.) return
plausible mock outputs; the orchestrator + workers are real LLM
agents reading and writing the workspace memory.

### Run with the real backend

```bash
PYTHONPATH=. .venv/bin/python examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 10 --mode real \
    --workspace seed_workspaces/nemo_mas_reasoner \
    --work-dir runs/nemo-mas-10
```

This wires in `SingleNodeTinkerLiteBackend(mock=False)` and
`NemoReasonerBenchmark` — expects configured GPUs, AWS Bedrock access,
and the model files referenced in `model/base.yaml`.

---

## Wiring a real / custom backend

The algorithm exposes one knob: `backend_registry: Mapping[str, Callable]`.
Compose it from:

* **`local_handlers(workspace_root)`** — pure-Python tools that work
  anywhere (sample_jsonl, format_validate, minhash_dedup, diff_yaml,
  read_training_log, read_checkpoint_metric, etc.).
* **`BackendBridge(workspace_root=, benchmark=, backend=).as_registry()`** —
  delegates compute-bound tools (`run_eval`, `launch_training`,
  `rerun_recipe_with_seeds`, `load_checkpoint_for_inference`,
  `batch_generate`) to the supplied backend + benchmark.
* **`demo_compute_handlers()`** — fallback that returns plausible mock
  outputs for the same compute-bound tools. Use for tests + dry runs.
* **Your own** — drop in any `Callable[..., str]`. The handler is
  invoked with the tool's keyword arguments and must return a JSON
  string (stable contract; no exception propagation).

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
algo = NemoMASAlgorithm(
    backend_registry={
        **local_handlers(workspace),
        **bridge.as_registry(),
        # Override one tool with a custom handler:
        "call_teacher_model": my_teacher_call,
    },
)
```

`call_teacher_model` is intentionally NOT bridged by default — teacher
distill needs whatever LLM client you're using. Wire it explicitly.

---

## How it differs from `mcgs`

| | `mcgs` | `nemo_mas` |
|---|---|---|
| Search topology | UCT graph + branches + top-k + fusion | LLM orchestrator + 4 worker roles |
| Mutation source | `BaselineMutationProposer` | Theorist's `recipe_proposal` records |
| Promotion | `PromotionPolicy` over node metric | `cv_result` tagged "stable" with new-best mean |
| Memory | `NodeMemoryStore` (BM25-lite, per-node) | `RecipeMemory` (BM25Okapi, typed records, ref DAG) |
| Inter-step communication | Implicit via graph + workspace patches | Explicit via mem_write + refs (auditable) |
| Adds new dirs to workspace | No | `backend/`, `skills/`, `memory/records.jsonl` (training always runs through `agent_evolve/model/runners/stages/*.py` — no workspace-local runners) |
| External deps | None new | None new |
| Best for | Hyperparameter sweeps where the reward signal is clean | Multi-axis exploration (data / recipe / RL) where evidence trail matters |

Both register to the same `TRAINING_ALGORITHMS` dict — switching is
one string change in `TrainingEvolver(algorithm="...")`.

---

## Schema enforcement (the "moat" that's hard to replicate)

`schema.py` is the contract that makes the memory trustworthy without
trusting the LLM. Every `mem_write` validates:

* **ID format** (`rec_<hex>`).
* **Title and body non-empty**.
* **Kind in this role's whitelist** (Analyst can't write
  `recipe_proposal`; Engineer can't write `data_audit_finding`).
* **Per-kind ref rules**:
  * `breakthrough` → `len(refs) >= 1`
  * `recipe_proposal` → ref to ≥1 `eval_report` or `data_gap`
  * `training_run` → refs to ≥1 `recipe_proposal` AND ≥1 `dataset_snapshot`
  * `cv_result` → ref to ≥1 `training_run`
  * `eval_report` → ref to ≥1 `training_run`

A worker that violates the contract sees a structured error and either
adapts or writes a `failed_attempt`. The DAG of refs is the audit
trail you can walk back from any submission.

Edit these rules to fit your benchmark's epistemics — that's where
your domain knowledge encodes into the system.

---

## Extending

Adding a new role:

1. Add an entry to `KIND_WHITELIST` in `schema.py`.
2. Add the role's allowed kinds to `tools.py::_BACKEND_TOOL_CATALOGUE`
   if it gets backend tools.
3. Drop a `prompts/<role>.md` and `tools/<role>.yaml` (LLM-readable
   doc) in the workspace.
4. Drop skills under `skills/<role>/`.

Adding a new record kind:

1. Add it to the relevant role(s) in `KIND_WHITELIST`.
2. Optionally add an entry to `REF_RULES` if it has provenance
   constraints.
3. Update prompts so workers know when to write it.

Adding a new backend tool:

1. Append a tuple to `tools.py::_BACKEND_TOOL_CATALOGUE[<role>]`.
2. Implement the handler in `backends.py::local_handlers` (if
   GPU/Bedrock-free) or in `backends.py::BackendBridge` (if it
   delegates to the backend).
3. Add a stub in `demo_compute_handlers()` so dry-run / demo mode
   keeps working.

---

## Why this is yours, not the algorithm's

Per the architectural discussion in `DESIGN.md` §9: the BM25 store,
the role spawn pattern, the JSON tool format — those are commodity.
The pieces only you can maintain across cycles are:

1. `prompts/benchmark_reference.md` — your first-hand observations
   about the eval. Don't let an LLM rewrite it.
2. `prompts/system.md` task-crafting guideline — your "manager
   style" encoded.
3. `skills/<role>/*.md` — every failed cycle should yield a new
   skill or amend one. After 30 cycles your skill library *is* your
   edge.
4. `mem_write` whitelist + ref-required constraints — the schema is
   where your epistemic discipline lives. Tightening or loosening
   these rules changes what conclusions the system can draw.

The Python in this directory is a faithful runtime for those four
artifacts. If you replace one of them, the system behaves
differently — that's the point.
