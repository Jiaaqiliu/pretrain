# nemo_mas — Orchestrator-Worker MAS Training Algorithm

Parallel alternative to `mcgs` in `TRAINING_ALGORITHMS`. An LLM
**orchestrator** spawns four specialist **workers** that share a typed-record
**memory** (BM25, JSONL, append-only, ref-validated). Quality-Plan
checkpoints gate the run at critical milestones.

Two runtimes share the same memory / schema / checkpoints / backends:

| Runtime | Front-end | Entry point | Use when |
|---|---|---|---|
| **Bedrock (headless)** | `examples/nemo_mas_reasoning_example/drive_nemo_mas.py` | `NemoMASAlgorithm.run_cycle(ctx) -> MCGSCycleReport` | Marathon, CI, unattended multi-cycle runs |
| **Agent Teams (interactive)** | `.claude/agents/nemo_mas_*.md` + MCP server | `claude` CLI with `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` | Conversational driving with human signoff |

- Design: `seed_workspaces/nemo_mas_reasoner/DESIGN.md`
- Workspace: `seed_workspaces/nemo_mas_reasoner/`
- Headless driver: `examples/nemo_mas_reasoning_example/drive_nemo_mas.py`
- Agent Teams runbook: `examples/nemo_mas_reasoning_example/start_team.md`
- Tests: `tests/model/test_nemo_mas_*.py`

## Architecture

```
  Orchestrator (no writes, no exec)
      │  spawn_and_run_subagent(role, task)        # Bedrock runtime
      │  (or) shared task list + SendMessage       # Agent Teams runtime
      ▼
  [Reviewer] [DataWorker] [Planner] [Trainer]
      │            │            │          │
      └────────────┴──► RecipeMemory (BM25, typed records, refs DAG)
                              │
                              ▼
                     Backend tools (run_eval, launch_training, ...)
                     supplied via `backend_registry` (Bedrock)
                     or the `nemo_mas` MCP server (Agent Teams)
```

Workers are stateless across spawns; only channel is the memory store.
`mem_write` rejects out-of-whitelist kinds and ref-rule violations in
both runtimes.

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

### Reviewer — `prompts/reviewer.md`, `skills/reviewer/`

The QA officer and the analyst, one role. Audits data/eval AND posts
verdicts on Quality Plan checkpoint slots.

- **Backend tools:** `sample_jsonl`, `count_by_field`, `length_distribution`, `run_eval`, `run_short_training`, `plot_loss_curve`, `compute_data_gap_table`, `checkpoint_sign` (auto mode), `checkpoint_review_suggest`, `kaggle_submit`, `kaggle_fetch_score`
- **Writes:** `data_audit_finding`, `benchmark_rule`, `profile_run`, `eval_report`, `error_pattern`, `data_gap`, `checkpoint_review`, `kaggle_submission_result` (+ `breakthrough`, `failed_attempt`, `checkpoint_event` via `checkpoint_sign`)
- **Skills:** `audit_jsonl_quality`, `categorize_eval_errors`, `compute_data_gap`, `probe_benchmark_format`, `profile_lr_sweep`, `qa_checkpoint_review`

### Data Worker — `prompts/data_worker.md`, `skills/data_worker/`

- **Backend tools:** `call_teacher_model`, `load_checkpoint_for_inference`, `batch_generate`, `filter_by_gold`, `minhash_dedup`, `apply_format_filter`, `format_validate`, `mix_sources`, `write_jsonl`
- **Writes:** `distill_batch`, `dataset_snapshot` (+ cross-cutting)
- **Skills:** `format_validate`, `minhash_dedup`, `mix_by_curriculum`, `solver_self_distill_with_rejection`, `teacher_distill_long_cot`

### Planner — `prompts/planner.md`, `skills/planner/`

Reasoning + records only, no side effects.

- **Backend tools:** `diff_yaml`, `render_recipe_diff`
- **Writes:** `hypothesis`, `recipe_proposal` (+ cross-cutting)
- **Skills:** `failure_pattern_recognition`, `lr_warmup_for_long_cot`, `propose_recipe_from_gap`, `when_to_skip_sft`

### Trainer — `prompts/trainer.md`, `skills/trainer/`

Training always routes through the platform `StageRegistry`
(`agent_evolve/model/runners/stages/*.py`). Trainer **never** scaffolds
runner scripts.

- **Backend tools:** `launch_training`, `read_training_log`, `read_checkpoint_metric`, `rerun_recipe_with_seeds`, `compute_stability`, `pack_submission`
- **Writes:** `training_run`, `cv_result`, `submission_artifact` (+ cross-cutting)
- **Skills:** `cross_validate_recipe`, `run_training_stage`

## Record kinds (schema.py)

Enforced on every `mem_write`; violations return a structured error.

| kind | author | refs required |
|---|---|---|
| `data_audit_finding`, `benchmark_rule`, `profile_run`, `error_pattern`, `data_gap` | reviewer | — |
| `eval_report` | reviewer | ≥1 `training_run` |
| `checkpoint_review` | reviewer | ≥1 (evidence being judged) |
| `kaggle_submission_result` | reviewer | ≥1 `submission_artifact` |
| `distill_batch`, `dataset_snapshot` | data_worker | — |
| `hypothesis` | planner | — |
| `recipe_proposal` | planner | ≥1 `eval_report` or `data_gap` |
| `training_run` | trainer | ≥1 `recipe_proposal` **and** ≥1 `dataset_snapshot` |
| `cv_result` | trainer | ≥1 `training_run` |
| `submission_artifact` | trainer | ≥1 `training_run` |
| `checkpoint_event` | any role (via `checkpoint_sign`) or human | ≥1 |
| `breakthrough` | any | ≥1 (any kind) |
| `failed_attempt` | any | — |

The refs DAG is the audit trail behind every promotion:
`cv_result → training_run → {recipe_proposal, dataset_snapshot} → {eval_report|data_gap, distill_batch*}`.

## Quality Plan checkpoints

Gates in the training lifecycle (Plan → Data → Model → … → Submit). The
slot declarations live in the workspace at
`seed_workspaces/<workspace>/checkpoints.yaml` — swap the file for a
different benchmark; missing file ⇒ no gates (pure search mode).

State machine per slot:
`pending → pending_evidence → pending_human → signed` (+ `reopened`).
Transitions come from two kinds of record:

- `checkpoint_review` — verdict from the **reviewer** role. Carries one
  of `{evidence_attached, ready_to_sign, insufficient, reject}`. The
  fold promotes slot state based on the latest verdict.
- `checkpoint_event` — the actual signoff. Emitted by the reviewer /
  orchestrator via `checkpoint_sign` (auto mode only) or by the viewer /
  lead when a human signs (manual mode).

**Protocol per cycle:**

1. Worker produces evidence, tagging each record with
   `checkpoint:<slot_id>`.
2. Reviewer reads the evidence and calls
   `checkpoint_review_suggest(slot_id, verdict, reason, refs)`.
3. Slot advances; `ready_to_sign` in manual mode halts progress until a
   human signs. The halt fires at the next `run_cycle` entry.

## Files

```
__init__.py       public API (NemoMASAlgorithm, RecipeMemory, schema)
schema.py         MemoryRecord, KIND_WHITELIST, REF_RULES, validate_record
memory.py         RecipeMemory — JSONL + vendored BM25
tools.py          per-role tool factories (YAML → (specs, handlers))
spawner.py        SpawnHandler — wraps BedrockAgent for workers
orchestrator.py   NemoMASAlgorithm — TRAINING_ALGORITHMS entry
backends.py       local_handlers + BackendBridge + demo_compute_handlers
agent_teams/      Interactive CC Agent Teams adapter
  __init__.py     public API (re-exports)
  server.py       FastMCP stdio server (29 tools)
  role_guard.py   per-caller role validation
  hook_utils.py   shared helpers for .claude/hooks/nemo_mas_*.py
```

Stdlib + `mcp` (FastMCP) only. BM25 vendored; MinHash uses content
fingerprints.

## Quickstart (Bedrock headless)

```bash
# Tests (23 tests, ~1.5s, no AWS/GPU)
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python -m pytest \
    tests/model/test_nemo_mas_cycle_outcome.py \
    tests/model/test_nemo_mas_mcp_server.py -q

# Dry-run: stub BedrockAgent, no compute
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
    examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 3 --mode dry-run

# Demo: real Bedrock, mocked compute
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
    examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 5 --mode demo

# Real: Bedrock + GPUs + SingleNodeTinkerLiteBackend
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
    examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
    --cycles 10 --mode real \
    --workspace seed_workspaces/nemo_mas_reasoner \
    --work-dir runs/nemo-mas-10
```

## Quickstart (Agent Teams interactive)

```bash
export NEMO_MAS_WORK_DIR=runs/nemo-mas-teams-v1
export NEMO_MAS_CHECKPOINT_MODE=manual   # or: auto
claude                                    # repo root; settings wires the rest
```

Then in the CC session:

> Create a nemo_mas team. Spawn teammates using the subagent types
> `nemo_mas_orchestrator`, `nemo_mas_planner`, `nemo_mas_data_worker`,
> `nemo_mas_trainer`, `nemo_mas_reviewer`. Call
> `mcp__nemo_mas__start_iteration` to fork the workspace and open
> cycle 1.

Full operator runbook: `examples/nemo_mas_reasoning_example/start_team.md`.

## Wiring a backend (headless)

`NemoMASAlgorithm(backend_registry=...)` takes a `Mapping[str, Callable]`.
Compose from:

- `local_handlers(workspace_root_resolver)` — stdlib-only tools. Pass a
  zero-arg callable that returns the active cycle's forked workspace
  root; `NemoMASAlgorithm` publishes it via
  `algo.current_workspace_root` each cycle.
- `BackendBridge(workspace_root_resolver, benchmark, backend).as_registry()`
  — delegates compute-bound tools (`run_eval`, `launch_training`,
  `rerun_recipe_with_seeds`, `load_checkpoint_for_inference`,
  `batch_generate`) to the backend.
- `demo_compute_handlers()` — plausible mock outputs for dry-run/tests.
- Your own `Callable[..., str]` returning a JSON string.

```python
from agent_evolve.model.algorithms.nemo_mas import NemoMASAlgorithm
from agent_evolve.model.algorithms.nemo_mas.backends import (
    BackendBridge, local_handlers,
)

algo_ref: dict = {}
def resolve_ws():
    a = algo_ref.get("algo")
    return getattr(a, "current_workspace_root", None) or seed_workspace

bridge = BackendBridge(
    workspace_root=resolve_ws,
    benchmark=NemoReasonerBenchmark(),
    backend=SingleNodeTinkerLiteBackend(mock=False),
)
algo = NemoMASAlgorithm(backend_registry={
    **local_handlers(resolve_ws),
    **bridge.as_registry(),
})
algo_ref["algo"] = algo
```

The resolver closure ensures each cycle's tool writes land under
`<work_dir>/cycles/<NNNN>/.fork_target/...` instead of the seed. See
`drive_nemo_mas.build_algorithm` for the reference implementation.

## MCP tool surface (Agent Teams)

`agent_teams/server.py` registers 29 tools:

| Group | Tools |
|---|---|
| Memory | `mem_write`, `mem_get`, `mem_search`, `mem_recent` |
| Checkpoints | `list_slots`, `checkpoint_state`, `checkpoint_review_suggest`, `checkpoint_sign` |
| Iteration | `start_iteration`, `current_iteration` |
| Backend (19) | Everything `local_handlers()` returns: `sample_jsonl`, `write_jsonl`, `mix_sources`, `pack_submission`, `kaggle_submit`, `kaggle_fetch_score`, `minhash_dedup`, `format_validate`, `filter_by_gold`, `apply_format_filter`, `length_distribution`, `count_by_field`, `diff_yaml`, `render_recipe_diff`, `read_training_log`, `read_checkpoint_metric`, `compute_stability`, `compute_data_gap_table`, `plot_loss_curve` |

Role enforcement is in `agent_teams/role_guard.py`: teammates pass
`role="<worker-name>"` on every call, and `checkpoint_sign` is
additionally restricted to `role ∈ {human, reviewer (auto only),
orchestrator_auto}`.

One hook runs at the CC boundary:

- `PreToolUse(Agent)` → `.claude/hooks/nemo_mas_agent_spawn.py` logs
  `nemo_mas_*` subagent spawns into the active ledger as
  `task_assignment` records (never blocks; soft-fails to stderr).

Kaggle's per-run cap is enforced inside the `kaggle_submit` handler
itself (see `backends.py`), not at the hook layer. The cap defaults to
1; override via `NEMO_MAS_KAGGLE_MAX_PER_RUN`.

## vs. `mcgs`

| | `mcgs` | `nemo_mas` |
|---|---|---|
| Search | UCT graph + branches + top-k | LLM orchestrator + 4 workers |
| Mutation source | `BaselineMutationProposer` | Planner's `recipe_proposal` records |
| Promotion | `PromotionPolicy` | `cv_result` tagged "stable" |
| Memory | `NodeMemoryStore` (per-node) | `RecipeMemory` (typed, ref DAG) |
| Comms | Implicit (graph + patches) | Explicit (`mem_write` + refs, auditable) |
| Best for | Clean hyperparameter sweeps | Multi-axis exploration (data / recipe / RL) |

Both register to `TRAINING_ALGORITHMS`; switch via `TrainingEvolver(algorithm=...)`.

## Extending

- **New role:** extend `KIND_WHITELIST` in `schema.py`; add
  `tools.py::_BACKEND_TOOL_CATALOGUE` entry; drop `prompts/<role>.md`,
  `tools/<role>.yaml`, `skills/<role>/`. For Agent Teams, add a
  matching `.claude/agents/nemo_mas_<role>.md`.
- **New record kind:** add to the role in `KIND_WHITELIST`; optionally
  add a `REF_RULES` entry; update prompts.
- **New backend tool:** append to `_BACKEND_TOOL_CATALOGUE[<role>]`;
  implement in `backends.py::local_handlers` or `BackendBridge`; add a
  stub in `demo_compute_handlers()`. Agent Teams picks it up
  automatically via `_register_backend_tools()` at server import.

## What's yours, not the algorithm's

The Python here is a runtime. Your edge lives in the four artifacts it
serves:

1. `prompts/benchmark_reference.md` — first-hand eval observations
2. `prompts/system.md` — task-crafting / manager style
3. `skills/<role>/*.md` — every failed cycle should yield a new skill
4. `KIND_WHITELIST` + `REF_RULES` — the epistemic contract

Replace one and the system behaves differently. That's the point.
