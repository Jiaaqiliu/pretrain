"""FastMCP server exposing nemo_mas tools to Claude Code teammates.

Launched once per ``claude`` session via the ``mcpServers`` entry in
``.claude/settings.json``. Teammates call tools over stdio. The server
stays alive for the whole session and mutates the live ``RecipeMemory``
instance bound to the current iteration's workspace.

Tool surface (all tools namespaced ``mcp__nemo_mas__<name>`` from the
teammate's perspective; named bare here):

  Memory (teammates):
    mem_write, mem_get, mem_search, mem_recent

  Checkpoints (reviewer + lead):
    checkpoint_state, list_slots, checkpoint_review_suggest, checkpoint_sign

  Iteration control (lead):
    start_iteration, current_iteration

  Backend (data_worker + trainer + reviewer):
    all 20+ handlers from ``local_handlers()``, auto-registered.

The lead talks to this server just like any teammate. In manual mode,
only the lead is permitted to call ``checkpoint_sign`` (via
``role="human"``); the reviewer is permitted in auto mode (via
``role="reviewer"``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from .backends import local_handlers
from .checkpoints import (
    FoldedSlot,
    fold_checkpoints,
    load_slot_decls,
)
from .hook_utils import (
    current_checkpoint_mode,
    current_memory_path,
    current_work_dir,
    current_workspace_root,
)
from .mcp_role_guard import (
    ROLE_HUMAN,
    ROLE_ORCHESTRATOR_AUTO,
    RoleGuardError,
    check_worker_role,
    resolve_signer_role,
)
from .memory import RecipeMemory
from .schema import RecordValidationError

logger = logging.getLogger("nemo_mas.mcp_server")


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

    ``role`` must be one of ``planner``, ``data_worker``, ``trainer``,
    ``reviewer`` — the kind must be allowed for that role (see
    ``schema.KIND_WHITELIST``). ``refs`` must resolve to existing records
    and satisfy any per-kind ref constraints (``schema.REF_RULES``).
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


# ── Checkpoint tools ─────────────────────────────────────────────


def _fold_now() -> tuple[list[FoldedSlot], str, list[dict]]:
    ws = current_workspace_root()
    mode = current_checkpoint_mode()
    slots = load_slot_decls(ws) if ws else []
    if not slots:
        return ([], mode, [])
    folded = fold_checkpoints(_get_memory().all_records(), mode, slots=slots)
    return (folded, mode, slots)


@mcp.tool()
def list_slots() -> str:
    """List declared Quality Plan slots + their current folded state."""
    folded, mode, _ = _fold_now()
    if not folded:
        return _ok(slots=[], mode=mode,
                   note="no checkpoints.yaml in active workspace")
    return _ok(mode=mode, slots=[{
        "id": s.id, "title": s.title, "state": s.state,
        "required": s.required,
        "requires_evidence": list(s.requires_evidence),
        "depends_on": list(s.depends_on),
        "evidence_counts": s.evidence_counts,
        "last_review_verdict": s.last_review_verdict,
        "last_review_reason": s.last_review_reason,
        "can_sign": s.can_sign,
    } for s in folded])


@mcp.tool()
def checkpoint_state(slot_id: str) -> str:
    """Return the folded state of one slot."""
    folded, mode, _ = _fold_now()
    for s in folded:
        if s.id == slot_id:
            return _ok(mode=mode, slot={
                "id": s.id, "title": s.title, "state": s.state,
                "required": s.required,
                "requires_evidence": list(s.requires_evidence),
                "depends_on": list(s.depends_on),
                "evidence_counts": s.evidence_counts,
                "last_review_verdict": s.last_review_verdict,
                "last_review_reason": s.last_review_reason,
                "can_sign": s.can_sign,
            })
    return _err(f"unknown slot_id={slot_id!r}")


@mcp.tool()
def checkpoint_review_suggest(
    slot_id: str,
    verdict: str,
    reason: str,
    refs: list[str],
    tags: list[str] | None = None,
) -> str:
    """Reviewer-only: attach a QA verdict to a checkpoint slot.

    ``verdict`` ∈ ``{evidence_attached, ready_to_sign, insufficient, reject}``.
    ``refs`` must cite at least one evidence record being judged.
    """
    from .checkpoints import VALID_VERDICTS
    _, _, slots = _fold_now()
    slot_ids = {s["id"] for s in slots}
    if slot_id not in slot_ids:
        return _err(f"unknown slot_id={slot_id!r}; "
                    f"expected one of {sorted(slot_ids)}")
    if verdict not in VALID_VERDICTS:
        return _err(f"verdict must be one of {sorted(VALID_VERDICTS)}; "
                    f"got {verdict!r}")
    if not refs:
        return _err("checkpoint_review_suggest requires ≥1 ref to "
                    "the evidence being judged")
    if not reason.strip():
        return _err("reason must be a non-empty one-line summary")

    combined_tags = [
        f"checkpoint:{slot_id}",
        f"verdict:{verdict}",
        "channel:qa_review",
    ]
    if tags:
        combined_tags.extend(t for t in tags if t not in combined_tags)
    try:
        rec = _get_memory().write(
            role="reviewer",
            kind="checkpoint_review",
            title=f"{verdict} · {slot_id}",
            body=reason,
            tags=tuple(combined_tags),
            refs=tuple(refs),
        )
    except RecordValidationError as e:
        return _err(str(e))
    return _ok(id=rec.id, slot_id=slot_id, verdict=verdict)


@mcp.tool()
def checkpoint_sign(
    slot_id: str,
    refs: list[str],
    role: str = ROLE_HUMAN,
    note: str = "",
) -> str:
    """Sign a checkpoint slot, closing it.

    Caller roles:
      * ``human`` (default, manual or auto mode) — the lead signs on
        behalf of the user.
      * ``orchestrator_auto`` (auto mode only) — the lead acting as the
        orchestrator in auto mode.
      * ``reviewer`` (auto mode only) — reviewer closes the slot it just
        approved.

    Evidence check: every kind in ``slot.requires_evidence`` must appear
    among the refs. Dependency check: all ``slot.depends_on`` must be in
    a terminal state. Both mirror the in-process ``checkpoint_sign``
    handler exactly.
    """
    folded, mode, slots = _fold_now()
    slots_by_id = {s["id"]: s for s in slots}
    slot = slots_by_id.get(slot_id)
    if slot is None:
        return _err(f"unknown slot_id={slot_id!r}")

    try:
        signer_role, actor_label = resolve_signer_role(role)
    except RoleGuardError as e:
        return _err(str(e))

    # Manual mode: only ``human`` can sign. Auto mode: human, reviewer,
    # and orchestrator_auto can all sign.
    from .checkpoints import CHECKPOINT_MODE_AUTO, CHECKPOINT_MODE_MANUAL
    if mode == CHECKPOINT_MODE_MANUAL and role != ROLE_HUMAN:
        return _err(
            f"mode={mode!r}: only role='human' may sign in manual mode; "
            f"got role={role!r}. The reviewer should post "
            "verdict=ready_to_sign and wait for the lead to sign."
        )

    if not refs:
        return _err(
            f"checkpoint_sign requires evidence refs covering "
            f"{slot['requires_evidence']}"
        )

    # Evidence coverage — every required kind must appear in refs.
    mem = _get_memory()
    ref_kinds: list[str] = []
    for rid in refs:
        rec = mem.get(rid)
        if rec is None:
            return _err(f"ref {rid!r} does not resolve")
        ref_kinds.append(rec.kind)
    missing = [k for k in slot["requires_evidence"] if k not in ref_kinds]
    if missing:
        return _err(
            f"slot {slot_id!r} requires evidence of kinds "
            f"{slot['requires_evidence']}; refs cover {ref_kinds}; "
            f"missing {missing}"
        )

    # Dependency: all depends_on must be terminal.
    folded_by_id = {s.id: s for s in folded}
    unmet = [d for d in slot["depends_on"]
             if folded_by_id.get(d) is None
             or folded_by_id[d].state not in {"signed", "reopened"}]
    if unmet:
        return _err(
            f"slot {slot_id!r} depends on {unmet} which are not yet "
            "signed; produce evidence + sign those first"
        )

    body = json.dumps({
        "checkpoint_id": slot_id,
        "event": "signoff",
        "actor": actor_label,
        "note": note,
    })
    try:
        rec = mem.write(
            role=signer_role,
            kind="checkpoint_event",
            title=f"signoff {slot_id}",
            body=body,
            tags=(f"checkpoint:{slot_id}", "event:signoff",
                  f"actor:{actor_label}"),
            refs=tuple(refs),
        )
    except RecordValidationError as e:
        return _err(str(e))
    return _ok(id=rec.id, slot_id=slot_id, actor=actor_label, mode=mode)


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
        checkpoint_mode=current_checkpoint_mode(),
        current_cycle=meta.get("current_cycle"),
        cycles_completed=meta.get("cycles_completed", 0),
    )


@mcp.tool()
def start_iteration() -> str:
    """Start a new cycle: bump counter, fork the seed workspace, update env.

    Replaces the implicit per-invocation fork that ``NemoMASAlgorithm.run_cycle``
    did in the headless path. Must be called by the lead between cycles.

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
    mpath.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Fork via TrainingWorkspace. Matches the path convention used by
    # the headless runtime (orchestrator.cycle_workspace_path).
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
    # ``_common_model/tools`` is a sibling of the seed; pin the lookup so
    # tool-YAML resolution works inside the fork (matches the headless
    # algorithm's ``NEMO_MAS_COMMON_MODEL`` swap).
    os.environ.setdefault(
        "NEMO_MAS_COMMON_MODEL",
        str(seed.parent / "_common_model" / "tools"),
    )

    _State.invalidate()
    # Warm up the ledger so the cycle stamp lands on the next write.
    _get_memory()

    return _ok(
        current_cycle=cycle_id,
        workspace_root=str(cycle_root),
        memory_path=os.environ["NEMO_MAS_MEMORY_PATH"],
        checkpoint_mode=current_checkpoint_mode(),
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
    """
    ws_resolver = lambda: Path(os.environ.get("NEMO_MAS_WORKSPACE_ROOT")  # noqa: E731
                               or os.environ.get("NEMO_MAS_SEED_WORKSPACE")
                               or "seed_workspaces/nemo_mas_reasoner")
    handlers = local_handlers(ws_resolver)

    for name, handler in handlers.items():
        # Skip any handler whose name collides with a memory/checkpoint
        # tool we've already decorated — none exist today but be safe.
        if name in ("mem_write", "mem_get", "mem_search", "mem_recent",
                    "checkpoint_state", "checkpoint_sign",
                    "checkpoint_review_suggest", "list_slots",
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
