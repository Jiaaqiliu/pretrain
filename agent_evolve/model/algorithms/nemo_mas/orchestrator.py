"""NemoMASAlgorithm — entry point that plugs into TRAINING_ALGORITHMS.

Contract (matches MCGSSearch):
  - No-arg constructible: ``cls()`` works, all params have defaults.
  - ``run_cycle(ctx) -> MCGSCycleReport`` is the per-cycle entry.
  - ``ctx`` is a ``LoopContext`` with ``cycle, workspace, benchmark,
    backend, config, work_dir, trial, observer, budget``.

What this algorithm does per cycle:
  1. Load the workspace's ``RecipeMemory`` (creating ``memory/records.jsonl``
     if absent) and stamp it with ``ctx.cycle``.
  2. Build a ``SpawnHandler`` configured with the workspace's prompts,
     skills, and any backend tool registry the caller passed in.
  3. Read ``prompts/system.md`` (orchestrator base) +
     ``prompts/benchmark_reference.md`` and prepend any breakthroughs.
  4. Construct a Bedrock orchestrator agent with read-only memory tools
     plus ``spawn_and_run_subagent`` / ``call_existing_agent``.
  5. Compose the cycle brief from ``ctx`` and ``call`` the orchestrator.
  6. Return an ``MCGSCycleReport`` summarizing what happened.

This algorithm DOES NOT execute training, eval, or distill itself —
those are backend tools the caller wires in. By default they are stubs
that return a structured "not implemented" message; the LLM workers
will see this and either skip or write a ``failed_attempt`` (per their
prompts), so the system degrades gracefully even with no backend.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from pathlib import Path
from typing import Any

from ...types import (
    MCGSCycleReport,
    TrainingSearchNodeSummary,
    WorkspaceMutation,
    WorkspacePatch,
)
from .memory import RecipeMemory
from .spawner import SpawnHandler
from .tools import BackendToolRegistry, build_role_tools

logger = logging.getLogger(__name__)


# ── Cycle contract (proposal d) ────────────────────────────────────────

# Load-bearing kinds that count toward the promotion / trained / partial
# classification. Ordered so the "strongest" verdict wins.
_STRONG_KINDS = frozenset({"cv_result"})
_TRAINED_KINDS = frozenset({"cv_result", "training_run"})
_PARTIAL_KINDS = frozenset({"eval_report", "recipe_proposal", "data_gap",
                             "dataset_snapshot", "distill_batch",
                             "breakthrough"})


def _classify_outcome(
    new_records: list,
    promoted: bool,
    *,
    budget_exhausted: bool,
) -> str:
    """Map the records written this cycle onto the 5-state outcome enum.

    Order of checks matters: budget_exhausted wins over everything else
    (so we don't claim success on a truncated run); promotion then trumps
    plain training; partial only applies when nothing load-bearing ran.
    """
    if budget_exhausted:
        return "budget_exhausted"
    if promoted:
        return "promoted"
    kinds = {r.kind for r in new_records}
    if kinds & _TRAINED_KINDS:
        return "trained"
    if kinds & _PARTIAL_KINDS:
        return "partial"
    return "null"


def _count_kinds(records: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r.kind] = out.get(r.kind, 0) + 1
    return out


class NemoMASAlgorithm:
    """Orchestrator-worker MAS for Nemotron Reasoning training-recipe search.

    Parameters
    ----------
    model_id:
        Bedrock model id for orchestrator + workers. Defaults to
        Claude Opus 4.6 (``us.anthropic.claude-opus-4-6-v1``).
    thinking_effort:
        Extended-thinking budget passed to BedrockAgent. ``""`` (off),
        ``"low" | "medium" | "high" | "max"``. When non-empty, max_tokens
        is bumped to 65536.
    backend_registry:
        Mapping of backend tool name -> handler. Tools not in the
        registry get a stub handler that returns a structured "not wired
        in" error. See ``tools.BackendToolRegistry``.
    workspace_subdir:
        Where in ``ctx.workspace`` the algorithm reads prompts/skills
        from. Defaults to no subdir (workspace root). Use this if you
        want to keep multiple algorithm configurations in one workspace.
    """

    name = "nemo_mas"

    def __init__(
        self,
        *,
        model_id: str = "us.anthropic.claude-opus-4-6-v1",
        thinking_effort: str = "",
        backend_registry: BackendToolRegistry | None = None,
        workspace_subdir: str = "",
        # Cycle contract (proposal d). Null values disable the respective
        # guard. Tuned for the marathon workload on 3 H200 nodes — override
        # for smaller / larger runs.
        cycle_wall_seconds: float | None = 3600.0,
        cycle_orchestrator_turn_cap: int | None = 80,
        fork_per_cycle: bool = True,
    ) -> None:
        self.model_id = model_id
        self.thinking_effort = thinking_effort
        self.backend_registry = backend_registry
        self.workspace_subdir = workspace_subdir.strip("/")
        self.cycle_wall_seconds = cycle_wall_seconds
        self.cycle_orchestrator_turn_cap = cycle_orchestrator_turn_cap
        self.fork_per_cycle = fork_per_cycle

        # Last-cycle artifacts (for topk_summary / debugging).
        self._last_cycle_records: list[str] = []
        self._last_orchestrator_response: str = ""
        self._last_cycle_outcome: str = "null"

    # ── Loop entry point ───────────────────────────────────────

    def run_cycle(self, ctx: Any) -> MCGSCycleReport:
        cycle_id = f"{int(ctx.cycle):04d}"
        t0 = time.time()

        # Per-cycle forked workspace. Falls back to seed-in-place when the
        # caller's ctx doesn't give us a TrainingWorkspace (tests, dry runs).
        seed_root, cycle_root = self._resolve_cycle_root(ctx, cycle_id)

        # ``_common_model/`` is a sibling of the SEED workspace
        # (``seed_workspaces/_common_model/tools/``). When we fork into
        # ``work_dir/cycles/<id>/``, that sibling is gone. Pin the lookup
        # at the seed's sibling so fork + non-fork modes resolve tools the
        # same way. tools.py honours this env var (see
        # ``_common_model_tools_dir``). Restore the prior value after the
        # cycle so we don't leak into other algorithms in the same process.
        prior_common = os.environ.get("NEMO_MAS_COMMON_MODEL")
        os.environ["NEMO_MAS_COMMON_MODEL"] = str(
            seed_root.parent / "_common_model" / "tools"
        )
        try:
            return self._run_cycle_body(
                ctx=ctx, cycle_id=cycle_id, t0=t0,
                seed_root=seed_root, cycle_root=cycle_root,
            )
        finally:
            if prior_common is None:
                os.environ.pop("NEMO_MAS_COMMON_MODEL", None)
            else:
                os.environ["NEMO_MAS_COMMON_MODEL"] = prior_common

    def _run_cycle_body(
        self,
        *,
        ctx: Any,
        cycle_id: str,
        t0: float,
        seed_root: Path,
        cycle_root: Path,
    ) -> MCGSCycleReport:
        # Memory lives one level up from the per-cycle forks so state
        # accumulates across cycles (MCGS's forks keep their own memory
        # per node; we want a single shared store). Placement:
        #   <work_dir>/memory/records.jsonl  (preferred, cross-cycle state)
        #   <seed_root>/memory/records.jsonl (backward compat for dry runs)
        records_path = self._resolve_records_path(ctx, seed_root)
        memory = RecipeMemory(records_path)
        memory.set_cycle_id(cycle_id)

        spawner = SpawnHandler(
            workspace_root=cycle_root,
            memory=memory,
            backend_registry=self.backend_registry,
            model_id=self.model_id,
            thinking_effort=self.thinking_effort,
        )

        # Build the orchestrator's tool set: spawn + memory-read + file-read.
        orchestrator_specs, orchestrator_handlers = self._build_orchestrator_tools(
            spawner=spawner, memory=memory, workspace_root=cycle_root,
        )

        system_prompt = self._build_orchestrator_system_prompt(
            workspace_root=cycle_root, memory=memory,
        )

        cycle_brief = self._compose_cycle_brief(ctx, memory)

        before_ids = {r.id for r in memory.all_records()}

        # Run the orchestrator under wall / turn caps. The caps are soft:
        # BedrockAgent doesn't expose a hook to abort mid-turn, so we
        # instead cap the conversation length up front and catch over-budget
        # after the fact. Wall-time check here bounds "how long we'll wait";
        # finer-grained interruption is a follow-up.
        result_text, orchestrator_turns, budget_exhausted = (
            self._run_orchestrator_with_budget(
                system_prompt=system_prompt,
                tools=orchestrator_specs,
                tool_handlers=orchestrator_handlers,
                task=cycle_brief,
                wall_seconds=self.cycle_wall_seconds,
                turn_cap=self.cycle_orchestrator_turn_cap,
                t0=t0,
            )
        )

        all_records = memory.all_records()
        new_records = [r for r in all_records if r.id not in before_ids]
        new_ids = [r.id for r in new_records]
        self._last_cycle_records = new_ids
        self._last_orchestrator_response = result_text

        incumbent_id, changed, best_metric = self._decide_promotion(
            memory=memory, this_cycle_ids=new_ids,
        )

        outcome = _classify_outcome(
            new_records, promoted=changed, budget_exhausted=budget_exhausted,
        )
        self._last_cycle_outcome = outcome
        wall = time.time() - t0
        kind_counts = _count_kinds(new_records)

        logger.info(
            "[nemo_mas] cycle=%s outcome=%s wall=%.1fs turns=%d "
            "records=%d kinds=%s promoted=%s",
            cycle_id, outcome, wall, orchestrator_turns, len(new_records),
            kind_counts, changed,
        )

        return MCGSCycleReport(
            cycle=int(ctx.cycle),
            selected_parent_id=None,
            trial_node_ids=new_ids,                   # nemo_mas's "trial" = records written
            incumbent_node_id=incumbent_id,
            incumbent_changed=changed,
            best_metric=best_metric,
            graph_path=str(records_path),
            report_path="",
            cycle_outcome=outcome,                    # type: ignore[arg-type]
            wall_seconds=wall,
            orchestrator_turns=orchestrator_turns,
            record_counts=kind_counts,
        )

    def topk_summary(self) -> list[TrainingSearchNodeSummary]:
        # nemo_mas doesn't maintain a graph of nodes; topk is meaningless
        # in the same sense. Return empty so the loop's hasattr-guarded
        # call is a no-op.
        return []

    # ── Helpers ────────────────────────────────────────────────

    def _resolve_cycle_root(
        self, ctx: Any, cycle_id: str,
    ) -> tuple[Path, Path]:
        """Return (seed_root, cycle_root).

        ``seed_root`` is the canonical, read-only seed workspace path.
        ``cycle_root`` is where this cycle reads + mutates — a forked copy
        under ``ctx.work_dir/cycles/<cycle_id>/workspace/`` when possible.

        Fork requires a ``TrainingWorkspace``-shaped ``ctx.workspace`` and
        a ``ctx.work_dir``. When either is missing (tests, dry runs) we
        fall back to the seed root itself — the caller is responsible for
        keeping the seed clean in that mode.
        """
        seed_root = self._resolve_workspace_root(ctx)
        if not self.fork_per_cycle:
            return seed_root, seed_root

        work_dir = getattr(ctx, "work_dir", None)
        ws = getattr(ctx, "workspace", None)
        can_fork = (
            work_dir is not None
            and ws is not None
            and hasattr(ws, "fork")
            and callable(getattr(ws, "fork", None))
        )
        if not can_fork:
            return seed_root, seed_root

        cycle_root_parent = Path(work_dir) / "cycles" / cycle_id
        # `fork()` places the copy under `<work_dir>/nodes/<node_id>/workspace`.
        # Pass a virtual work_dir whose `nodes/` dir points at our cycle
        # slot so the resulting path is `<work_dir>/cycles/<id>/workspace`.
        virtual_work_dir = cycle_root_parent / ".fork_target"
        empty_mutation = WorkspaceMutation(
            mutation_id=f"nemo_mas:cycle-{cycle_id}",
            parent_node_id="seed",
            description="nemo_mas per-cycle fork (no mutation)",
            patch=WorkspacePatch(operations=[]),
        )
        try:
            forked = ws.fork(
                node_id="workspace",           # → nodes/workspace/workspace
                mutation=empty_mutation,
                work_dir=virtual_work_dir,
            )
        except Exception:                        # noqa: BLE001
            logger.exception(
                "[nemo_mas] per-cycle fork failed; falling back to seed root"
            )
            return seed_root, seed_root

        cycle_root = Path(forked.root)
        logger.info("[nemo_mas] cycle=%s forked workspace to %s",
                    cycle_id, cycle_root)
        return seed_root, cycle_root

    def _resolve_records_path(self, ctx: Any, seed_root: Path) -> Path:
        """Memory file path. Shared across cycles when ``work_dir`` is set."""
        work_dir = getattr(ctx, "work_dir", None)
        if work_dir is not None:
            return Path(work_dir) / "memory" / "records.jsonl"
        return seed_root / "memory" / "records.jsonl"

    def _run_orchestrator_with_budget(
        self,
        *,
        system_prompt: str,
        tools: list[dict],
        tool_handlers: dict,
        task: str,
        wall_seconds: float | None,
        turn_cap: int | None,
        t0: float,
    ) -> tuple[str, int, bool]:
        """Drive the Bedrock orchestrator with soft wall/turn caps.

        Returns ``(result_text, turns_used, budget_exhausted)``.

        "Soft" = we check the caps *between* converse turns by wrapping
        ``agent._run_converse_loop`` similar to the trace harness. If the
        cap trips, we signal end-of-conversation via ``agent._stop`` style
        flagging (BedrockAgent supports cooperative termination by
        returning early from the loop). The agent's final assistant text
        is returned as-is; ``budget_exhausted=True`` tells the caller
        that the cycle should be classified as ``budget_exhausted``
        regardless of records written.
        """
        try:
            from agent_evolve.harness.agents.arc.bedrock_agent import BedrockAgent
        except ImportError as e:
            logger.warning(
                "BedrockAgent unavailable (%s); orchestrator will not run. "
                "Returning a no-op response.", e,
            )
            return ("(orchestrator did not run — BedrockAgent unavailable)",
                    0, False)

        max_tokens = 65536 if self.thinking_effort else 16384
        budget_exhausted = {"value": False}
        turns = {"count": 0}

        try:
            agent = BedrockAgent(
                model_id=self.model_id,
                system_prompt=system_prompt,
                tools=tools,
                tool_handlers=tool_handlers,
                agent_id=0,
                max_tokens=max_tokens,
                thinking_effort=self.thinking_effort,
            )
        except Exception as e:                   # noqa: BLE001
            logger.exception("Orchestrator agent construction failed")
            return (f"(orchestrator failed: {e})", 0, False)

        real_converse = getattr(agent, "_converse_with_retry", None)
        if real_converse is not None and (wall_seconds or turn_cap):
            def guarded(tool_config):
                # Pre-turn budget check. Return a synthetic "stop" result
                # that BedrockAgent will treat as an end_turn when it
                # sees ``stopReason=end_turn``.
                elapsed = time.time() - t0
                turns["count"] += 1
                over_wall = wall_seconds is not None and elapsed >= wall_seconds
                over_turns = turn_cap is not None and turns["count"] > turn_cap
                if over_wall or over_turns:
                    budget_exhausted["value"] = True
                    reason = []
                    if over_wall:
                        reason.append(f"wall {elapsed:.1f}s >= {wall_seconds}s")
                    if over_turns:
                        reason.append(f"turns {turns['count']} > {turn_cap}")
                    logger.warning(
                        "[nemo_mas] orchestrator budget exhausted (%s); "
                        "injecting synthetic end_turn", "; ".join(reason),
                    )
                    return {
                        "stopReason": "end_turn",
                        "output": {"message": {"role": "assistant",
                                               "content": [{"text":
                            "(Cycle budget exhausted — terminating cycle.)"}]}},
                        "usage": {"inputTokens": 0, "outputTokens": 0,
                                  "totalTokens": 0,
                                  "cacheReadInputTokens": 0,
                                  "cacheWriteInputTokens": 0},
                    }
                return real_converse(tool_config)
            agent._converse_with_retry = guarded  # type: ignore[method-assign]

        try:
            result = agent.call(task)
        except Exception as e:                   # noqa: BLE001
            logger.exception("Orchestrator agent failed")
            return (f"(orchestrator failed: {e})", turns["count"], False)
        return result, turns["count"], budget_exhausted["value"]

    def _resolve_workspace_root(self, ctx: Any) -> Path:
        ws = getattr(ctx, "workspace", None)
        if ws is None:
            raise RuntimeError("LoopContext.workspace did not yield a path")
        # Path-like (str / Path) → use directly. Note: Path objects DO have a
        # ``.root`` attribute meaning "/", which is NOT what we want — so
        # check string/Path before checking wrapper attrs.
        if isinstance(ws, (str, Path)):
            root = Path(ws)
        else:
            root = None
            for attr in ("path", "workspace_path", "root"):
                if hasattr(ws, attr):
                    candidate = getattr(ws, attr)
                    if isinstance(candidate, (str, Path)):
                        root = Path(candidate)
                        break
            if root is None:
                raise RuntimeError(
                    f"LoopContext.workspace ({type(ws).__name__}) did not "
                    "yield a path-like via .path / .workspace_path / .root"
                )
        if self.workspace_subdir:
            root = root / self.workspace_subdir
        return root

    def _build_orchestrator_tools(
        self,
        *,
        spawner: SpawnHandler,
        memory: RecipeMemory,
        workspace_root: Path,
    ) -> tuple[list[dict], dict]:
        """Build the orchestrator's (specs, handlers) bundle from YAML.

        Tool declarations come from ``<workspace>/tools/orchestrator.yaml``
        + ``_common_model/tools/``; spawn handlers are supplied by the
        caller's ``SpawnHandler`` instance (there is one per cycle) and
        passed in via the ``backend_registry`` kwarg so they land on the
        named tools from YAML.

        The orchestrator role YAML intentionally omits ``mem_write``,
        so the orchestrator cannot write records — only read.
        """
        spawn_handlers = spawner.spawn_specs_and_handlers[1]
        return build_role_tools(
            "orchestrator",
            memory=memory,
            skills_root=workspace_root / "skills",
            workspace_root=workspace_root,
            backend_registry=spawn_handlers,
        )

    def _build_orchestrator_system_prompt(
        self, *, workspace_root: Path, memory: RecipeMemory,
    ) -> str:
        prompts_dir = workspace_root / "prompts"
        sys_md = prompts_dir / "system.md"
        ref_md = prompts_dir / "benchmark_reference.md"
        parts: list[str] = []
        if sys_md.exists():
            parts.append(sys_md.read_text(encoding="utf-8").strip())
        if ref_md.exists():
            parts.append(ref_md.read_text(encoding="utf-8").strip())
        bt = memory.breakthroughs_block()
        if bt.strip():
            parts.append(bt.strip())
        if not parts:
            parts.append(
                "You are the Orchestrator. (No system.md found in workspace; "
                "you are running on bare defaults.)"
            )
        return "\n\n---\n\n".join(parts)

    def _compose_cycle_brief(self, ctx: Any, memory: RecipeMemory) -> str:
        cycle = int(ctx.cycle)
        budget = getattr(ctx, "budget", None)
        budget_str = repr(budget) if budget is not None else "(no budget object provided)"
        recent_break = memory.recent(kind="breakthrough", k=3)
        recent_cv = memory.recent(kind="cv_result", k=3)
        recent_eval = memory.recent(kind="eval_report", k=3)
        recent_gap = memory.recent(kind="data_gap", k=3)

        def _format(items, label):
            if not items:
                return f"  {label}: (none)"
            lines = [f"  {label}:"]
            for r in items:
                lines.append(f"    - {r.id} ({r.cycle_id}): {r.title}")
            return "\n".join(lines)

        brief = textwrap.dedent(f"""
            Cycle {cycle:04d}.

            Budget: {budget_str}.

            State of memory at the start of this cycle:
{_format(recent_break, "recent breakthroughs")}
{_format(recent_cv, "recent cv_results")}
{_format(recent_eval, "recent eval_reports")}
{_format(recent_gap, "recent data_gaps")}

            Run this cycle per the structure in your system prompt. Spawn
            workers as needed. Stop when the termination criteria you were
            given are met. Your final text response should:
              (a) name the recipe id you'd promote (if any),
              (b) cite supporting record ids,
              (c) list what's blocked / would benefit from a future cycle.
        """).strip()
        return brief

    def _decide_promotion(
        self,
        *,
        memory: RecipeMemory,
        this_cycle_ids: list[str],
    ) -> tuple[str | None, bool, float | None]:
        """Promotion rule: a cv_result tagged 'stable' with mean >= prior best mean."""
        all_cv = [r for r in memory.all_records() if r.kind == "cv_result"]
        if not all_cv:
            return (None, False, None)

        def _parse_mean(rec) -> float | None:
            for line in rec.body.splitlines():
                line = line.strip()
                if line.lower().startswith("mean:"):
                    try:
                        return float(line.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        return None
            return None

        stable = [r for r in all_cv if "stable" in (t.lower() for t in r.tags)]
        if not stable:
            best_metric = max(filter(lambda x: x is not None,
                                     (_parse_mean(r) for r in all_cv)),
                              default=None)
            return (None, False, best_metric)

        scored = [(r, _parse_mean(r)) for r in stable]
        scored = [(r, m) for r, m in scored if m is not None]
        if not scored:
            return (None, False, None)
        scored.sort(key=lambda x: x[1], reverse=True)
        best_rec, best_metric = scored[0]
        # incumbent_changed iff the best record was written THIS cycle.
        changed = best_rec.id in this_cycle_ids
        return (best_rec.id, changed, best_metric)
