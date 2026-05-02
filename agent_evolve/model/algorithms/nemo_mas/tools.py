"""Per-role tool builders for nemo_mas workers.

Each role gets a tool set built from three layers:

  1. **Memory tools** (always present): mem_write, mem_search, mem_get,
     mem_recent, mem_link. mem_write is bound to the role identity so
     ``author`` is auto-filled and the kind whitelist is enforced.

  2. **Skill tools** (always present): skill_index, skill_load. Read
     markdown files from ``skills/<role>/`` (or any namespace).

  3. **Backend tools** (caller-supplied): the actual data-/training-side
     handlers like ``run_eval``, ``call_teacher_model``,
     ``launch_training``. The caller passes a ``BackendToolRegistry``
     mapping tool name -> handler; this module wraps them in Bedrock
     toolSpec format and attaches them to the appropriate roles.

Bedrock tool format (per arc bedrock_tools.py):
  {
      "toolSpec": {
          "name": "...",
          "description": "...",
          "inputSchema": {"json": {"type": "object",
                                   "properties": {...},
                                   "required": [...]}},
      }
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .memory import RecipeMemory
from .schema import (
    INTERNAL_KINDS,
    KIND_WHITELIST,
    RecordValidationError,
    kinds_for_role,
)


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


# ── BackendToolRegistry ──────────────────────────────────────────────


# Convention: backend tools per role. The caller wires real handlers in;
# unwired tools default to a ``StubToolHandler`` that returns a clear
# "not implemented" message so the LLM sees a structured failure rather
# than a stack trace.

# Default backend tool catalogue (matches the role yamls). Each entry is
# (name, description, jsonschema_properties, required_fields). Real
# handlers come from the caller via ``BackendToolRegistry``; if missing,
# the stub handler returns a structured error message that the LLM can
# act on (e.g. "this tool isn't wired in for this run; try a different
# approach or write a failed_attempt").

_BACKEND_TOOL_CATALOGUE: dict[str, list[tuple[str, str, dict, list[str]]]] = {
    "analyst": [
        ("sample_jsonl",
         "Sample N rows from a JSONL file. Returns rows + per-field summary.",
         {"path": {"type": "string"}, "n": {"type": "integer", "default": 50},
          "seed": {"type": "integer", "default": 0}}, ["path"]),
        ("count_by_field",
         "Group-by count over a JSONL file. Returns {value: count}.",
         {"path": {"type": "string"}, "field": {"type": "string"}}, ["path", "field"]),
        ("length_distribution",
         "Token-length histogram for a JSONL field.",
         {"path": {"type": "string"}, "field": {"type": "string"},
          "tokenizer": {"type": "string", "default": "nemotron-3-nano"}},
         ["path", "field"]),
        ("run_eval",
         "Run a full eval pass on a checkpoint against a split. Returns per-category accuracy + error buckets + per-row jsonl path.",
         {"ckpt_path": {"type": "string"}, "split": {"type": "string"},
          "limit": {"type": "integer"}}, ["ckpt_path", "split"]),
        ("run_short_training",
         "Launch a short training run for profiling only (default 200 steps).",
         {"recipe_diff": {"type": "string"}, "max_steps": {"type": "integer", "default": 200},
          "log_every": {"type": "integer", "default": 10}}, ["recipe_diff"]),
        ("plot_loss_curve",
         "Generate a PNG of loss vs step for one or more profile runs.",
         {"training_run_ids": {"type": "array", "items": {"type": "string"}}},
         ["training_run_ids"]),
        ("compute_data_gap_table",
         "Cross-tabulate eval errors by (category, length_bucket).",
         {"eval_report_id": {"type": "string"}}, ["eval_report_id"]),
    ],
    "data_engineer": [
        ("call_teacher_model",
         "Call a teacher model on a list of prompts. Refuses if expected cost > 5x budget.",
         {"model": {"type": "string"},
          "prompts": {"type": "array", "items": {"type": "string"}},
          "max_tokens": {"type": "integer", "default": 8000},
          "temperature": {"type": "number", "default": 0.7},
          "system_prompt": {"type": "string"}}, ["model", "prompts"]),
        ("load_checkpoint_for_inference",
         "Load a previously-trained checkpoint for self-distill. Returns inference handle.",
         {"ckpt_path": {"type": "string"}}, ["ckpt_path"]),
        ("batch_generate",
         "Generate completions from a loaded checkpoint with sampling config matching the eval contract.",
         {"handle": {"type": "string"},
          "prompts": {"type": "array", "items": {"type": "string"}},
          "sampling_config": {"type": "object"}}, ["handle", "prompts"]),
        ("filter_by_gold",
         "Rejection sampling: keep only generations whose extracted boxed answer matches gold.",
         {"generations": {"type": "array"}, "golds": {"type": "array"}},
         ["generations", "golds"]),
        ("minhash_dedup",
         "Near-duplicate removal via MinHash LSH.",
         {"input_path": {"type": "string"}, "key_field": {"type": "string"},
          "threshold": {"type": "number", "default": 0.85}},
         ["input_path", "key_field"]),
        ("apply_format_filter",
         "Apply data/recipes/default.yaml filters to a JSONL.",
         {"input_path": {"type": "string"}}, ["input_path"]),
        ("format_validate",
         "Schema check on a JSONL.",
         {"path": {"type": "string"}}, ["path"]),
        ("mix_sources",
         "Build the final train.jsonl per data/mix.yaml weights.",
         {"sources": {"type": "array", "items": {"type": "string"}},
          "weights": {"type": "array", "items": {"type": "number"}},
          "curriculum_yaml": {"type": "string"}}, ["sources", "weights"]),
        ("write_jsonl",
         "Write rows to a JSONL file under data/generated/ or data/final/.",
         {"path": {"type": "string"}, "rows": {"type": "array"}},
         ["path", "rows"]),
    ],
    "theorist": [
        ("diff_yaml",
         "Compute a structural diff between two YAML strings or paths.",
         {"a": {"type": "string"}, "b": {"type": "string"}}, ["a", "b"]),
        ("render_recipe_diff",
         "Take a recipe_proposal body and produce a clean unified-diff.",
         {"proposal_body": {"type": "string"}}, ["proposal_body"]),
    ],
    "engineer": [
        ("scaffold_runner",
         "Generate a training-runner script under runner/.",
         {"stage": {"type": "string", "enum": ["sft", "rl", "distill", "eval"]},
          "template": {"type": "string"}},
         ["stage", "template"]),
        ("read_runner",
         "Read a runner script under runner/.",
         {"path": {"type": "string"}}, ["path"]),
        ("edit_runner",
         "Edit a runner script under runner/.",
         {"path": {"type": "string"}, "old_text": {"type": "string"},
          "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
        ("check_pipeline_coverage",
         "Compare train/pipeline.yaml stages vs runners present.",
         {}, []),
        ("launch_training",
         "Launch a training run. Returns job_id, log_path.",
         {"runner_path": {"type": "string"}, "recipe_path": {"type": "string"},
          "data_path": {"type": "string"}, "ckpt_out": {"type": "string"},
          "max_steps": {"type": "integer"},
          "monitor": {"type": "boolean", "default": True}},
         ["runner_path", "recipe_path", "data_path", "ckpt_out"]),
        ("read_training_log",
         "Read live or final training log.",
         {"job_id": {"type": "string"}}, ["job_id"]),
        ("read_checkpoint_metric",
         "Read metric.json beside a checkpoint.",
         {"ckpt_path": {"type": "string"}}, ["ckpt_path"]),
        ("rerun_recipe_with_seeds",
         "Run the same recipe N times with different seeds.",
         {"recipe_path": {"type": "string"}, "data_path": {"type": "string"},
          "seeds": {"type": "array", "items": {"type": "integer"}},
          "splits": {"type": "array", "items": {"type": "string"}}},
         ["recipe_path", "data_path", "seeds"]),
        ("compute_stability",
         "Compute mean / stddev of the primary metric across training_run ids.",
         {"training_run_ids": {"type": "array", "items": {"type": "string"}}},
         ["training_run_ids"]),
    ],
}


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


# ── Memory tool builders ─────────────────────────────────────────────


def _mem_tools(memory: RecipeMemory, role: str) -> tuple[list[dict], dict]:
    allowed = sorted(KIND_WHITELIST.get(role, frozenset()))
    descr = (
        "Append a typed record to the memory store. Required: kind, title, body. "
        "Optional: tags (list of strings), refs (list of record ids). "
        f"Allowed kinds for this role: {allowed}. "
        "kind=breakthrough requires len(refs)>=1. "
        "Per-kind ref rules are enforced; violation returns an error."
    )

    def mem_write(*, kind: str, title: str, body: str,
                  tags: list[str] | None = None,
                  refs: list[str] | None = None) -> str:
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

    specs = [
        _spec("mem_write", descr, {
            "kind": {"type": "string", "enum": allowed + sorted(INTERNAL_KINDS)},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "refs": {"type": "array", "items": {"type": "string"}},
        }, ["kind", "title", "body"]),
        _spec("mem_search",
              "BM25 search over the typed-record memory.",
              {"query": {"type": "string"},
               "kind": {"type": "string"},
               "author": {"type": "string"},
               "tags": {"type": "array", "items": {"type": "string"}},
               "cycle_range": {"type": "array", "items": {"type": "string"}},
               "top_k": {"type": "integer", "default": 8}},
              ["query"]),
        _spec("mem_recent",
              "Most recent N records, optionally filtered by kind / author / tags.",
              {"kind": {"type": "string"},
               "author": {"type": "string"},
               "tags": {"type": "array", "items": {"type": "string"}},
               "k": {"type": "integer", "default": 10}},
              []),
        _spec("mem_get", "Fetch one full record by id.",
              {"id": {"type": "string"}}, ["id"]),
        _spec("mem_link",
              "Add a refs edge from one record to another (append-only).",
              {"child_id": {"type": "string"},
               "parent_id": {"type": "string"},
               "relation": {"type": "string", "default": "refs"}},
              ["child_id", "parent_id"]),
    ]
    handlers = {
        "mem_write": mem_write,
        "mem_search": mem_search,
        "mem_recent": mem_recent,
        "mem_get": mem_get,
        "mem_link": mem_link,
    }
    return specs, handlers


# ── Skill tool builders ──────────────────────────────────────────────


def _skill_tools(skills_root: Path) -> tuple[list[dict], dict]:
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
        # name is "<domain>/<skill>" or "<skill>" (search all domains).
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

    specs = [
        _spec("skill_index",
              "List skills (optionally under one domain) with their 'When to use' line.",
              {"domain": {"type": "string"}}, []),
        _spec("skill_load",
              "Read full markdown body for one skill (name='<domain>/<skill>').",
              {"name": {"type": "string"}}, ["name"]),
    ]
    handlers = {"skill_index": skill_index, "skill_load": skill_load}
    return specs, handlers


# ── Read-only file access ────────────────────────────────────────────

# Block reads under these subdirs (safety) — checkpoints can be huge,
# memory/ should be accessed via mem_*.
_BLOCKED_PREFIXES = ("memory/", "checkpoints/", ".git/")


def _file_tools(workspace_root: Path) -> tuple[list[dict], dict]:
    def read_file(*, path: str, max_bytes: int = 200_000) -> str:
        rel = path.lstrip("/")
        if any(rel.startswith(p) for p in _BLOCKED_PREFIXES):
            return json.dumps({"ok": False, "error": f"reads under {rel.split('/')[0]}/ are blocked; use mem_* for memory"})
        full = (workspace_root / rel).resolve()
        if workspace_root.resolve() not in full.parents and full != workspace_root.resolve():
            return json.dumps({"ok": False, "error": "path escapes workspace root"})
        if not full.exists() or not full.is_file():
            return json.dumps({"ok": False, "error": f"not a file: {rel}"})
        data = full.read_bytes()[:max_bytes]
        try:
            text = data.decode("utf-8")
            return json.dumps({"ok": True, "path": rel, "text": text,
                               "truncated": len(data) >= max_bytes})
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
            entries.append({"name": p.name, "type": "dir" if p.is_dir() else "file",
                            "size": p.stat().st_size if p.is_file() else None})
        return json.dumps({"ok": True, "path": rel, "entries": entries})

    specs = [
        _spec("read_file", "Read a workspace file (utf-8). Blocked under memory/ and checkpoints/.",
              {"path": {"type": "string"},
               "max_bytes": {"type": "integer", "default": 200000}},
              ["path"]),
        _spec("list_dir", "List a workspace directory.",
              {"path": {"type": "string", "default": "."}}, []),
    ]
    return specs, {"read_file": read_file, "list_dir": list_dir}


# ── Backend tool builders ────────────────────────────────────────────


BackendToolRegistry = Mapping[str, Callable[..., Any]]
"""A mapping of backend tool name -> handler. The handler is called with
the tool's input as kwargs and must return a string (typically JSON)."""


def _backend_tools(role: str, registry: BackendToolRegistry | None) -> tuple[list[dict], dict]:
    catalogue = _BACKEND_TOOL_CATALOGUE.get(role, [])
    specs: list[dict] = []
    handlers: dict[str, Callable[..., Any]] = {}
    registry = registry or {}
    for name, descr, props, required in catalogue:
        specs.append(_spec(name, descr, props, required))
        handlers[name] = registry.get(name) or _stub_handler(name)
    return specs, handlers


# ── Public entry point ──────────────────────────────────────────────


def build_role_tools(
    role: str,
    *,
    memory: RecipeMemory,
    skills_root: Path,
    workspace_root: Path,
    backend_registry: BackendToolRegistry | None = None,
) -> tuple[list[dict], dict[str, Callable[..., Any]]]:
    """Compose all tools for a given role: memory + skill + file + backend."""
    if role not in KIND_WHITELIST:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(KIND_WHITELIST)}")
    specs: list[dict] = []
    handlers: dict[str, Callable[..., Any]] = {}
    for s, h in (
        _mem_tools(memory, role),
        _skill_tools(skills_root),
        _file_tools(workspace_root),
        _backend_tools(role, backend_registry),
    ):
        specs.extend(s)
        handlers.update(h)
    return specs, handlers


def kinds_for_role_text(role: str) -> str:
    """Helper for prompt assembly: comma-separated allowed kinds."""
    return ", ".join(kinds_for_role(role))
