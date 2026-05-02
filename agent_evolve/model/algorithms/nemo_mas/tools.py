"""Per-role tool builders for nemo_mas workers.

Tool declarations (name + description + JSON schema + required) live in
workspace YAML — both the role-specific YAML under
``<workspace>/tools/<role>.yaml`` and a shared ``_common_model`` directory
peer to the workspace:

    seed_workspaces/
      _common_model/tools/{memory,skills,filesystem,orchestrator}.yaml
      <workspace>/tools/{analyst,data_engineer,theorist,engineer,orchestrator}.yaml

Resolution per tool name inside a role YAML:
  1. If the role YAML provides ``schema:`` inline, use it (override).
  2. Else fall back to ``_common_model/tools/*.yaml`` for the same name.
  3. Else error.

Descriptions resolve the same way (role YAML description wins, falls
back to ``_common_model/``).

Handler wiring (how the tool actually runs when the LLM calls it):
  * **Platform tools** — ``mem_*``, ``skill_*``, ``read_file``,
    ``list_dir``, ``spawn_*`` — get built-in handlers constructed here.
  * **Backend tools** — everything else — look up handlers from the
    caller-supplied ``backend_registry``. Missing registry entry →
    structured "not wired in" stub.

Per-role ``mem_write_kinds:`` in the role YAML is authoritative for the
write-whitelist. If absent, platform falls back to
``schema.KIND_WHITELIST[role]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .memory import RecipeMemory
from .schema import (
    INTERNAL_KINDS,
    KIND_WHITELIST,
    RecordValidationError,
    kinds_for_role,
)


# Platform tool names — these have built-in handlers and come with
# schemas from ``_common_model/tools/`` by default. The set is also used
# to decide which handler to wire when a name appears in a role YAML.
_MEM_TOOL_NAMES = frozenset({
    "mem_write", "mem_search", "mem_recent", "mem_get", "mem_link",
})
_SKILL_TOOL_NAMES = frozenset({"skill_index", "skill_load"})
_FILE_TOOL_NAMES = frozenset({"read_file", "list_dir"})
_ORCHESTRATOR_TOOL_NAMES = frozenset({
    "spawn_and_run_subagent", "call_existing_agent",
})
_PLATFORM_TOOL_NAMES = (
    _MEM_TOOL_NAMES | _SKILL_TOOL_NAMES | _FILE_TOOL_NAMES
    | _ORCHESTRATOR_TOOL_NAMES
)

# Block reads under these subdirs (safety) — checkpoints can be huge,
# memory/ should be accessed via mem_*.
_BLOCKED_FILE_PREFIXES = ("memory/", "checkpoints/", ".git/")


# ── Bedrock tool spec helper ─────────────────────────────────────────


def _spec(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return {
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {"json": schema},
        }
    }


# ── YAML loader (single source of truth for tool declarations) ─────


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _common_model_tools_dir(workspace_root: Path) -> Path:
    """Where shared platform tool YAMLs live.

    Convention: ``_common_model`` is a peer of the workspace root, under
    the same ``seed_workspaces/`` parent. Override with env var
    ``NEMO_MAS_COMMON_MODEL`` if needed.
    """
    import os
    override = os.environ.get("NEMO_MAS_COMMON_MODEL")
    if override:
        return Path(override)
    return workspace_root.parent / "_common_model" / "tools"


def _load_common_model_tools(workspace_root: Path) -> dict[str, dict]:
    """Union of ``tools:`` maps from every YAML under ``_common_model/tools/``.

    Later files DO NOT override earlier ones (each tool name should be
    unique across the _common_model files). Missing dir → empty dict
    (the workspace must then inline schemas for every tool it uses).
    """
    root = _common_model_tools_dir(workspace_root)
    out: dict[str, dict] = {}
    if not root.exists():
        return out
    for p in sorted(root.glob("*.yaml")):
        data = _load_yaml(p)
        tools = data.get("tools", {}) or {}
        for name, entry in tools.items():
            if name in out:
                # Duplicate declaration across _common_model files.
                # First wins; log via a raised error to surface loudly.
                raise ValueError(
                    f"_common_model: tool {name!r} declared twice "
                    f"(second occurrence in {p})"
                )
            out[name] = entry or {}
    return out


def _resolve_tool_spec(
    name: str,
    role_entry: dict | None,
    common_entry: dict | None,
) -> dict:
    """Build a Bedrock toolSpec for ``name`` from role YAML + _common_model YAML.

    Role YAML wins for ``description`` and ``schema`` when non-empty;
    otherwise falls back to _common_model.
    """
    role_entry = role_entry or {}
    common_entry = common_entry or {}

    description = role_entry.get("description") or common_entry.get("description")
    schema = role_entry.get("schema") or common_entry.get("schema")

    if not description:
        raise ValueError(
            f"tool {name!r} has no description "
            f"(neither in role YAML nor in _common_model/)"
        )
    if not schema:
        raise ValueError(
            f"tool {name!r} has no schema "
            f"(neither in role YAML nor in _common_model/); "
            f"add a `schema:` block in the role YAML or register the tool "
            f"in seed_workspaces/_common_model/tools/"
        )

    properties = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    return _spec(name.strip(), description.strip(), properties, required)


# ── Role YAML loader ─────────────────────────────────────────────────


def load_role_yaml(workspace_root: Path, role: str) -> dict:
    """Load ``<workspace>/tools/<role>.yaml``. Returns {} if absent."""
    return _load_yaml(workspace_root / "tools" / f"{role}.yaml")


def mem_write_kinds_for_role(workspace_root: Path, role: str) -> list[str]:
    """Authoritative per-role write whitelist.

    Resolution: role YAML ``mem_write_kinds:`` if present, else
    fallback to platform default ``KIND_WHITELIST[role]`` — empty list
    if neither exists (e.g. orchestrator role — no write permissions).
    """
    y = load_role_yaml(workspace_root, role)
    kinds = y.get("mem_write_kinds")
    if kinds:
        return sorted(set(kinds))
    default = KIND_WHITELIST.get(role)
    if default is None:
        return []
    return sorted(default)


def kinds_for_role_text(role: str) -> str:
    """Helper for prompt assembly: comma-separated allowed kinds.

    Retained for backward compatibility. Callers that want the
    workspace-YAML override should use :func:`mem_write_kinds_for_role`.
    """
    return ", ".join(kinds_for_role(role))


# ── Stub handler ─────────────────────────────────────────────────────


def _stub_handler(tool_name: str) -> Callable[..., str]:
    def handler(**_kwargs) -> str:
        return json.dumps({
            "ok": False,
            "tool": tool_name,
            "reason": (
                f"Backend tool {tool_name!r} is not wired in for this run. "
                "Skip this step, or write a failed_attempt explaining the "
                "missing capability so Theorist can adjust."
            ),
        })
    return handler


# ── Platform handlers (built-ins — memory / skill / file) ──────────


def _build_mem_handlers(
    memory: RecipeMemory,
    role: str,
    allowed_kinds: list[str],
) -> dict[str, Callable[..., str]]:

    def mem_write(*, kind: str, title: str, body: str,
                  tags: list[str] | None = None,
                  refs: list[str] | None = None) -> str:
        if kind not in allowed_kinds and kind not in INTERNAL_KINDS:
            return json.dumps({
                "ok": False,
                "error": (
                    f"role={role!r} is not allowed to write kind={kind!r}; "
                    f"allowed: {allowed_kinds}"
                ),
            })
        try:
            rec = memory.write(
                role=role,
                kind=kind,
                title=title,
                body=body,
                tags=tuple(tags or ()),
                refs=tuple(refs or ()),
            )
            return json.dumps({"ok": True, "id": rec.id, "kind": rec.kind,
                               "ts": rec.ts})
        except RecordValidationError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def mem_search(*, query: str, kind: str | None = None,
                   author: str | None = None, tags: list[str] | None = None,
                   cycle_range: list[str] | None = None,
                   top_k: int = 8) -> str:
        cr = tuple(cycle_range) if cycle_range and len(cycle_range) == 2 else None
        hits = memory.search(query=query, kind=kind, author=author,
                             tags=tags, cycle_range=cr, top_k=top_k)  # type: ignore[arg-type]
        return json.dumps([
            {"id": r.id, "kind": r.kind, "author": r.author,
             "cycle_id": r.cycle_id, "title": r.title, "score": round(s, 4),
             "snippet": r.body[:200]}
            for r, s in hits
        ])

    def mem_recent(*, kind: str | None = None, author: str | None = None,
                   tags: list[str] | None = None, k: int = 10) -> str:
        recs = memory.recent(kind=kind, author=author, tags=tags, k=k)
        return json.dumps([
            {"id": r.id, "kind": r.kind, "author": r.author,
             "cycle_id": r.cycle_id, "title": r.title, "tags": list(r.tags),
             "refs": list(r.refs), "ts": r.ts}
            for r in recs
        ])

    def mem_get(*, id: str) -> str:
        rec = memory.get(id)
        if rec is None:
            return json.dumps({"ok": False, "error": f"id {id!r} not found"})
        return json.dumps({"ok": True, **rec.to_dict()})

    def mem_link(*, child_id: str, parent_id: str,
                 relation: str = "refs") -> str:
        try:
            rec = memory.link(child_id, parent_id, role=role, relation=relation)
            return json.dumps({"ok": True, "link_id": rec.id})
        except RecordValidationError as e:
            return json.dumps({"ok": False, "error": str(e)})

    return {
        "mem_write":  mem_write,
        "mem_search": mem_search,
        "mem_recent": mem_recent,
        "mem_get":    mem_get,
        "mem_link":   mem_link,
    }


def _build_skill_handlers(skills_root: Path) -> dict[str, Callable[..., str]]:

    def skill_index(*, domain: str | None = None) -> str:
        out: list[dict] = []
        if domain:
            roots = [skills_root / domain]
        else:
            roots = [p for p in skills_root.iterdir() if p.is_dir()] if skills_root.exists() else []
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.md")):
                first = ""
                try:
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.lower().startswith("when to use"):
                                first = line
                                break
                except OSError:
                    pass
                out.append({
                    "name": f"{root.name}/{path.stem}",
                    "when_to_use": first or "(no When-to-use line found)",
                })
        return json.dumps(out)

    def skill_load(*, name: str) -> str:
        parts = name.split("/", 1)
        candidates: list[Path]
        if len(parts) == 2:
            candidates = [skills_root / parts[0] / f"{parts[1]}.md"]
        else:
            candidates = list(skills_root.rglob(f"{parts[0]}.md")) if skills_root.exists() else []
        for p in candidates:
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8")
                except OSError as e:
                    return json.dumps({"ok": False, "error": str(e)})
        return json.dumps({"ok": False, "error": f"skill {name!r} not found"})

    return {"skill_index": skill_index, "skill_load": skill_load}


def _build_file_handlers(workspace_root: Path) -> dict[str, Callable[..., str]]:

    def read_file(*, path: str, max_bytes: int = 200_000) -> str:
        rel = path.lstrip("/")
        if any(rel.startswith(p) for p in _BLOCKED_FILE_PREFIXES):
            return json.dumps({
                "ok": False,
                "error": f"reads under {rel.split('/')[0]}/ are blocked; "
                         f"use mem_* for memory",
            })
        full = (workspace_root / rel).resolve()
        if workspace_root.resolve() not in full.parents and full != workspace_root.resolve():
            return json.dumps({"ok": False, "error": "path escapes workspace root"})
        if not full.exists() or not full.is_file():
            return json.dumps({"ok": False, "error": f"not a file: {rel}"})
        data = full.read_bytes()[:max_bytes]
        try:
            text = data.decode("utf-8")
            return json.dumps({
                "ok": True, "path": rel, "text": text,
                "truncated": len(data) >= max_bytes,
            })
        except UnicodeDecodeError:
            return json.dumps({"ok": False, "error": "binary file (not utf-8)"})

    def list_dir(*, path: str = ".") -> str:
        rel = path.lstrip("/")
        full = (workspace_root / rel).resolve()
        if workspace_root.resolve() not in full.parents and full != workspace_root.resolve():
            return json.dumps({"ok": False, "error": "path escapes workspace root"})
        if not full.exists() or not full.is_dir():
            return json.dumps({"ok": False, "error": f"not a dir: {rel}"})
        entries = []
        for p in sorted(full.iterdir()):
            entries.append({
                "name": p.name, "type": "dir" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else None,
            })
        return json.dumps({"ok": True, "path": rel, "entries": entries})

    return {"read_file": read_file, "list_dir": list_dir}


# ── BackendToolRegistry ──────────────────────────────────────────────


BackendToolRegistry = Mapping[str, Callable[..., Any]]
"""A mapping of backend tool name -> handler. The handler is called with
the tool's input as kwargs and must return a string (typically JSON)."""


# ── Public entry point ──────────────────────────────────────────────


_VALID_ROLES = frozenset(KIND_WHITELIST) | {"orchestrator"}


def build_role_tools(
    role: str,
    *,
    memory: RecipeMemory,
    skills_root: Path,
    workspace_root: Path,
    backend_registry: BackendToolRegistry | None = None,
) -> tuple[list[dict], dict[str, Callable[..., Any]]]:
    """Compose the full (specs, handlers) bundle for one role.

    Reads tool declarations from ``<workspace>/tools/<role>.yaml`` +
    ``_common_model/tools/*.yaml``. Wires platform handlers for known
    names (memory / skill / file / spawn); uses ``backend_registry`` for
    everything else; stub for anything missing.
    """
    if role not in _VALID_ROLES:
        raise ValueError(
            f"unknown role {role!r}; expected one of {sorted(_VALID_ROLES)}"
        )

    role_yaml = load_role_yaml(workspace_root, role)
    role_tools = role_yaml.get("tools", {}) or {}
    common_tools = _load_common_model_tools(workspace_root)

    allowed_kinds = mem_write_kinds_for_role(workspace_root, role)
    mem_handlers = _build_mem_handlers(memory, role, allowed_kinds)
    skill_handlers = _build_skill_handlers(skills_root)
    file_handlers = _build_file_handlers(workspace_root)
    registry = dict(backend_registry or {})

    specs: list[dict] = []
    handlers: dict[str, Callable[..., Any]] = {}

    for name, role_entry in role_tools.items():
        # Orchestrator-only tools should not appear on worker roles.
        if name in _ORCHESTRATOR_TOOL_NAMES and role != "orchestrator":
            raise ValueError(
                f"role {role!r} declares orchestrator-only tool {name!r} — "
                f"move to orchestrator.yaml or remove"
            )
        common_entry = common_tools.get(name)
        spec = _resolve_tool_spec(name, role_entry, common_entry)
        specs.append(spec)

        if name in _MEM_TOOL_NAMES:
            handlers[name] = mem_handlers[name]
        elif name in _SKILL_TOOL_NAMES:
            handlers[name] = skill_handlers[name]
        elif name in _FILE_TOOL_NAMES:
            handlers[name] = file_handlers[name]
        elif name in _ORCHESTRATOR_TOOL_NAMES:
            # Orchestrator tools are wired by the orchestrator path
            # (see orchestrator.py::_build_orchestrator_tools — it passes
            # its SpawnHandler's spawn_* handlers in via backend_registry,
            # so they land here via the registry fall-through). If we get
            # here without a registry entry, surface it as a stub.
            handlers[name] = registry.get(name) or _stub_handler(name)
        else:
            handlers[name] = registry.get(name) or _stub_handler(name)

    return specs, handlers


def build_orchestrator_tools(
    *,
    memory: RecipeMemory,
    skills_root: Path,
    workspace_root: Path,
    spawn_handlers: Mapping[str, Callable[..., Any]],
) -> tuple[list[dict], dict[str, Callable[..., Any]]]:
    """Specialization of :func:`build_role_tools` for the orchestrator.

    The orchestrator has no ``mem_write``; ``_build_mem_handlers`` still
    constructs a full dict, but the role YAML (which doesn't list
    ``mem_write``) filters it out. Spawn handlers are passed through as
    a mini registry keyed by ``spawn_and_run_subagent`` /
    ``call_existing_agent``.
    """
    return build_role_tools(
        "orchestrator",
        memory=memory,
        skills_root=skills_root,
        workspace_root=workspace_root,
        backend_registry=spawn_handlers,
    )
