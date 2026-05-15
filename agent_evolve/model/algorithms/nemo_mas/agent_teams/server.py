"""FastMCP server exposing nemo_mas tools to Claude Code teammates.

Launched once per ``claude`` session via the ``mcpServers`` entry in
``.claude/settings.json``. Teammates call tools over stdio. The server
stays alive for the whole session and mutates the live ``RecipeMemory``
instance bound to the current iteration's workspace.

Tool surface (all tools namespaced ``mcp__nemo_mas__<name>`` from the
teammate's perspective; named bare here):

  Memory (teammates):
    mem_write, mem_get, mem_search, mem_recent

  Iteration control (lead):
    start_iteration, current_iteration

  Backend (data_worker + trainer + planner):
    all 20+ handlers from ``local_handlers()``, auto-registered.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from ..backends import BackendBridge, local_handlers
from ..memory import RecipeMemory
from ..schema import RecordValidationError
from .hook_utils import (
    current_memory_path,
    current_work_dir,
    current_workspace_root,
)
from .role_guard import (
    RoleGuardError,
    check_worker_role,
)

logger = logging.getLogger("nemo_mas.agent_teams.server")


# ── Per-process state ───────────────────────────────────────────────
#
# The MCP server is launched once and serves many tool calls. We lazily
# build a ``RecipeMemory`` bound to the active ledger path and cache it;
# ``start_iteration`` invalidates the cache so the next call re-binds
# against the newly forked workspace.


class _State:
    memory: RecipeMemory | None = None
    memory_path: Path | None = None
    workspace_root: Path | None = None
    backend_handlers: dict[str, Callable[..., str]] = {}

    @classmethod
    def invalidate(cls) -> None:
        cls.memory = None
        cls.memory_path = None
        cls.workspace_root = None
        cls.backend_handlers = {}


def _get_memory() -> RecipeMemory:
    """Resolve the live memory handle, rebinding on workspace change."""
    mpath = current_memory_path()
    if mpath is None:
        raise RuntimeError(
            "NEMO_MAS_MEMORY_PATH is unset. Call start_iteration first, "
            "or export NEMO_MAS_WORK_DIR before launching the team."
        )
    if _State.memory is None or _State.memory_path != mpath:
        _State.memory_path = mpath
        _State.memory = RecipeMemory(mpath)
        # Cycle id comes from meta.json; the algorithm stamps it on writes
        # so records land with the right cycle tag.
        from .hook_utils import read_meta
        cycle = read_meta().get("current_cycle")
        if cycle is not None:
            _State.memory.set_cycle_id(str(cycle))
    return _State.memory


def _get_backend_handlers() -> dict[str, Callable[..., str]]:
    """Resolve the backend-tool handler dict, rebuilding on workspace change."""
    ws = current_workspace_root()
    if ws is None:
        raise RuntimeError(
            "NEMO_MAS_WORKSPACE_ROOT is unset. Call start_iteration first."
        )
    if not _State.backend_handlers or _State.workspace_root != ws:
        _State.workspace_root = ws
        # The resolver-callable contract (landed in 4abfd93) means any
        # later change to NEMO_MAS_WORKSPACE_ROOT gets picked up without
        # rebuilding the dict — but we rebuild anyway to honour the
        # explicit invalidate() from start_iteration.
        _State.backend_handlers = local_handlers(lambda: Path(
            os.environ.get("NEMO_MAS_WORKSPACE_ROOT") or str(ws)
        ))
    return _State.backend_handlers


def _ok(**kw: Any) -> str:
    return json.dumps({"ok": True, **kw})


def _err(reason: str, **kw: Any) -> str:
    return json.dumps({"ok": False, "error": reason, **kw})


# ── FastMCP server ─────────────────────────────────────────────────


mcp = FastMCP("nemo_mas")


# ── Memory tools ─────────────────────────────────────────────────


@mcp.tool()
def mem_write(
    role: str,
    kind: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
    refs: list[str] | None = None,
) -> str:
    """Append a typed record to the ledger.

    ``role`` must be one of ``planner``, ``data_worker``, ``trainer`` —
    the kind must be allowed for that role (see ``schema.KIND_WHITELIST``).
    ``refs`` must resolve to existing records and satisfy any per-kind
    ref constraints (``schema.REF_RULES``).
    """
    try:
        check_worker_role(role)
    except RoleGuardError as e:
        return _err(str(e))
    try:
        rec = _get_memory().write(
            role=role,
            kind=kind,
            title=title,
            body=body,
            tags=tuple(tags or ()),
            refs=tuple(refs or ()),
        )
    except RecordValidationError as e:
        return _err(str(e))
    return _ok(id=rec.id, kind=rec.kind, cycle_id=rec.cycle_id, ts=rec.ts)


@mcp.tool()
def mem_get(rec_id: str) -> str:
    """Fetch one record by id."""
    rec = _get_memory().get(rec_id)
    if rec is None:
        return _err(f"no record with id={rec_id!r}")
    return _ok(record=rec.to_dict())


@mcp.tool()
def mem_search(
    query: str,
    kind: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
    top_k: int = 8,
) -> str:
    """BM25 search over the ledger. Returns up to ``top_k`` hits."""
    hits = _get_memory().search(
        query,
        kind=kind,
        author=author,
        tags=tags,
        top_k=top_k,
    )
    return _ok(hits=[
        {"score": round(s, 4), **rec.to_dict()} for rec, s in hits
    ])


@mcp.tool()
def mem_recent(
    kind: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
    k: int = 10,
) -> str:
    """Return the most recent ``k`` records matching the filters."""
    recs = _get_memory().recent(kind=kind, author=author, tags=tags, k=k)
    return _ok(records=[r.to_dict() for r in recs])


# ── Iteration control ───────────────────────────────────────────


@mcp.tool()
def current_iteration() -> str:
    """Report the active cycle id + fork path + ledger path."""
    from .hook_utils import read_meta
    meta = read_meta()
    return _ok(
        work_dir=str(current_work_dir() or ""),
        workspace_root=str(current_workspace_root() or ""),
        memory_path=str(current_memory_path() or ""),
        current_cycle=meta.get("current_cycle"),
        cycles_completed=meta.get("cycles_completed", 0),
    )


@mcp.tool()
def start_iteration() -> str:
    """Start a new cycle: bump counter, fork the seed workspace, update env.

    Must be called by the lead between cycles.

    Side effects:
      1. ``<work_dir>/meta.json`` gets ``current_cycle`` bumped.
      2. Seed workspace is forked to ``<work_dir>/cycles/<NNNN>/
         .fork_target/nodes/workspace/workspace/`` via
         ``TrainingWorkspace.fork()``.
      3. ``NEMO_MAS_WORKSPACE_ROOT`` is updated in the server process so
         subsequent tool calls target the fork.
      4. In-process caches are invalidated so the next
         ``mem_*`` / backend tool call rebinds.
    """
    wd = current_work_dir()
    if wd is None:
        return _err(
            "NEMO_MAS_WORK_DIR is unset. Export it before launching "
            "`claude`, e.g. `export NEMO_MAS_WORK_DIR=runs/nemo-mas-teams-v1`."
        )
    wd.mkdir(parents=True, exist_ok=True)

    # Resolve the seed workspace. Convention: the seed sits at
    # seed_workspaces/nemo_mas_reasoner unless ``NEMO_MAS_SEED_WORKSPACE``
    # overrides it.
    seed_raw = os.environ.get("NEMO_MAS_SEED_WORKSPACE")
    if seed_raw:
        seed = Path(seed_raw)
    else:
        seed = (wd.parent.parent / "seed_workspaces" / "nemo_mas_reasoner"
                if wd.parent.name == "runs" else
                Path("seed_workspaces/nemo_mas_reasoner").resolve())
    if not seed.is_dir():
        return _err(
            f"seed workspace not found at {seed}. Set "
            "NEMO_MAS_SEED_WORKSPACE to the correct path."
        )

    # Bump cycle counter.
    from .hook_utils import meta_path
    mpath = meta_path()
    assert mpath is not None  # work_dir existed
    meta: dict[str, Any] = {}
    if mpath.is_file():
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    next_cycle = int(meta.get("cycles_completed", 0)) + 1
    cycle_id = f"{next_cycle:04d}"
    meta["current_cycle"] = cycle_id
    meta["cycles_completed"] = next_cycle
    meta.setdefault("seed_workspace", str(seed))
    meta.setdefault("runtime", "agent_teams")
    mpath.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Create the empty ``trace/cycle_<NNNN>/`` directory so the trace
    # viewer classifies this work_dir as a valid run (``_is_run_dir``
    # checks for ``trace/``). The Agent Teams runtime doesn't emit
    # per-agent JSONL traces — those came from the monkey-patched
    # BedrockAgent wrapper in the headless driver, which has no analog
    # here because LLM turns happen inside Claude Code. Leave a README
    # in the cycle dir so a curious human isn't confused by the empty
    # folder.
    trace_cycle_dir = wd / "trace" / f"cycle_{cycle_id}"
    trace_cycle_dir.mkdir(parents=True, exist_ok=True)
    readme = trace_cycle_dir / "README.md"
    if not readme.is_file():
        readme.write_text(
            "This run uses the Agent Teams runtime; per-agent JSONL "
            "traces are not produced. The trace viewer's Quality Plan "
            "cockpit, leaderboard, chat thread, and record detail pages "
            "still work — they all derive from "
            "``<work_dir>/memory/records.jsonl``. Only the per-cycle "
            "agent conversation view is empty.\n",
            encoding="utf-8",
        )

    # Fork via TrainingWorkspace. Path convention is
    # ``<work_dir>/cycles/<cycle_id>/.fork_target/nodes/workspace/workspace``
    # — see ``hook_utils.cycle_workspace_path``.
    try:
        from agent_evolve.model.types import WorkspaceMutation, WorkspacePatch
        from agent_evolve.model.workspace import TrainingWorkspace
    except ImportError as e:
        return _err(f"workspace imports failed: {e}")

    ws_seed = TrainingWorkspace(seed)
    virtual_work_dir = wd / "cycles" / cycle_id / ".fork_target"
    empty_mutation = WorkspaceMutation(
        mutation_id=f"nemo_mas:cycle-{cycle_id}",
        parent_node_id="seed",
        description="nemo_mas per-cycle fork (Agent Teams)",
        patch=WorkspacePatch(operations=[]),
    )
    try:
        forked = ws_seed.fork(
            node_id="workspace",
            mutation=empty_mutation,
            work_dir=virtual_work_dir,
        )
    except Exception as e:  # noqa: BLE001
        return _err(f"fork failed: {e}")

    cycle_root = Path(forked.root)

    # Update env so downstream tool calls target the fork. Memory path
    # stays at <work_dir>/memory/records.jsonl so state accumulates
    # across cycles.
    os.environ["NEMO_MAS_WORKSPACE_ROOT"] = str(cycle_root)
    os.environ.setdefault("NEMO_MAS_MEMORY_PATH",
                          str(wd / "memory" / "records.jsonl"))

    _State.invalidate()
    # Warm up the ledger so the cycle stamp lands on the next write.
    _get_memory()

    return _ok(
        current_cycle=cycle_id,
        workspace_root=str(cycle_root),
        memory_path=os.environ["NEMO_MAS_MEMORY_PATH"],
        seed=str(seed),
    )


# ── Backend tool registration ──────────────────────────────────
#
# ``local_handlers`` returns a dict of plain callables keyed by tool name;
# FastMCP's ``@mcp.tool()`` wraps a function, so we can't register them
# in a loop via the decorator. Instead we register each via the server's
# low-level ``add_tool`` API on first access. This keeps the signatures
# (type hints, docstring) intact for schema derivation.
#
# We register against a placeholder seed path — the resolver callable
# inside each handler re-reads ``NEMO_MAS_WORKSPACE_ROOT`` at invocation
# time, so post-``start_iteration`` calls land in the fork.


def _register_backend_tools() -> None:
    """Register every ``local_handlers`` entry as an MCP tool.

    Called on module import so tools appear in the MCP handshake. The
    handlers themselves resolve the workspace root via the
    resolver-callable pattern, so no per-call rebinding is needed.

    When ``NEMO_MAS_COMPUTE_BACKEND`` is set the compute-bound bridge
    (``launch_training``, ``run_eval``, ``rerun_recipe_with_seeds``,
    ``batch_generate``, ``call_teacher_model``, ``load_checkpoint_for_inference``,
    ``run_short_training``) is also registered — so the trainer teammate
    can actually reach the platform's StageRegistry instead of hitting a
    "no such MCP tool" wall and writing a stub ``training_run``.
    """
    ws_resolver = lambda: Path(os.environ.get("NEMO_MAS_WORKSPACE_ROOT")  # noqa: E731
                               or os.environ.get("NEMO_MAS_SEED_WORKSPACE")
                               or "seed_workspaces/nemo_mas_reasoner")
    handlers = dict(local_handlers(ws_resolver))

    # Compute-bound bridge. Construction failure (missing kubeconfig,
    # benchmark import error) is logged but non-fatal — the server still
    # serves local tools so memory + QA flow keeps working.
    try:
        bridge = BackendBridge.from_env(ws_resolver)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "BackendBridge.from_env failed (%s); compute-bound tools "
            "(launch_training, run_eval, …) will NOT be registered. "
            "Trainer teammate will refuse to train.", exc,
        )
        bridge = None
    if bridge is not None:
        # Bridge entries win on name collision — local_handlers has no
        # launch_training, but we make the ordering explicit.
        handlers.update(bridge.as_registry())
        logger.info(
            "BackendBridge registered: %s", sorted(bridge.as_registry()),
        )

    for name, handler in handlers.items():
        # Skip any handler whose name collides with a memory/iteration
        # tool we've already decorated — none exist today but be safe.
        if name in ("mem_write", "mem_get", "mem_search", "mem_recent",
                    "start_iteration", "current_iteration"):
            logger.warning("skipping backend handler %s (name collision)", name)
            continue
        mcp.add_tool(
            handler,
            name=name,
            description=(handler.__doc__ or f"nemo_mas backend tool: {name}").strip(),
        )


_register_backend_tools()


def main() -> None:
    """Launch the stdio server. Used as the ``mcpServers`` command."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
