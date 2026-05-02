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
import textwrap
from pathlib import Path
from typing import Any

from ...types import MCGSCycleReport, TrainingSearchNodeSummary
from .memory import RecipeMemory
from .spawner import SpawnHandler
from .tools import BackendToolRegistry, build_role_tools

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.model_id = model_id
        self.thinking_effort = thinking_effort
        self.backend_registry = backend_registry
        self.workspace_subdir = workspace_subdir.strip("/")

        # Last-cycle artifacts (for topk_summary / debugging).
        self._last_cycle_records: list[str] = []
        self._last_orchestrator_response: str = ""

    # ── Loop entry point ───────────────────────────────────────

    def run_cycle(self, ctx: Any) -> MCGSCycleReport:
        cycle_id = f"{int(ctx.cycle):04d}"
        workspace_root = self._resolve_workspace_root(ctx)

        records_path = workspace_root / "memory" / "records.jsonl"
        memory = RecipeMemory(records_path)
        memory.set_cycle_id(cycle_id)

        spawner = SpawnHandler(
            workspace_root=workspace_root,
            memory=memory,
            backend_registry=self.backend_registry,
            model_id=self.model_id,
            thinking_effort=self.thinking_effort,
        )

        # Build the orchestrator's tool set: spawn + memory-read + file-read.
        orchestrator_specs, orchestrator_handlers = self._build_orchestrator_tools(
            spawner=spawner, memory=memory, workspace_root=workspace_root,
        )

        system_prompt = self._build_orchestrator_system_prompt(
            workspace_root=workspace_root, memory=memory,
        )

        cycle_brief = self._compose_cycle_brief(ctx, memory)

        before_ids = {r.id for r in memory.all_records()}

        result_text = self._run_orchestrator(
            system_prompt=system_prompt,
            tools=orchestrator_specs,
            tool_handlers=orchestrator_handlers,
            task=cycle_brief,
        )

        new_ids = [r.id for r in memory.all_records() if r.id not in before_ids]
        self._last_cycle_records = new_ids
        self._last_orchestrator_response = result_text

        # Decide promotion: if any cv_result this cycle is tagged "stable"
        # AND its mean is the new best across all cv_results in memory,
        # mark incumbent_changed=True. Otherwise no promotion this cycle.
        incumbent_id, changed, best_metric = self._decide_promotion(
            memory=memory, this_cycle_ids=new_ids,
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
        )

    def topk_summary(self) -> list[TrainingSearchNodeSummary]:
        # nemo_mas doesn't maintain a graph of nodes; topk is meaningless
        # in the same sense. Return empty so the loop's hasattr-guarded
        # call is a no-op.
        return []

    # ── Helpers ────────────────────────────────────────────────

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

    def _run_orchestrator(
        self,
        *,
        system_prompt: str,
        tools: list[dict],
        tool_handlers: dict,
        task: str,
    ) -> str:
        try:
            from agent_evolve.harness.agents.arc.bedrock_agent import BedrockAgent
        except ImportError as e:
            logger.warning(
                "BedrockAgent unavailable (%s); orchestrator will not run. "
                "Returning a no-op response.", e,
            )
            return "(orchestrator did not run — BedrockAgent unavailable)"

        max_tokens = 65536 if self.thinking_effort else 16384
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
            return agent.call(task)
        except Exception as e:                # noqa: BLE001
            logger.exception("Orchestrator agent failed")
            return f"(orchestrator failed: {e})"

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
