"""Deterministic dedup keys + upsample helpers for the merge stage.

Keeping this as a tiny standalone module because:
  * every worker needs these utilities,
  * the dedup-key strategy is something MCGS might mutate later
    (e.g. "dedup by normalized prompt prefix"), so the key-builder
    surface has to be small and swappable.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from .base import GeneratedRow
from .recipe import DataRecipe, RecipeFilters


def dedup_key(row: GeneratedRow, mode: str) -> str:
    """Canonical bytes to hash for dedup. Deterministic across runs."""
    if mode == "prompt_hash":
        payload = row.prompt
    elif mode == "prompt_and_source_hash":
        payload = row.prompt + "\n__source__\n" + row.source
    else:
        raise ValueError(f"unknown dedup mode: {mode!r}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dedup(rows: Iterable[GeneratedRow], filters: RecipeFilters) -> list[GeneratedRow]:
    """First-seen wins. Stable in iteration order — order only matters
    when callers append multiple stage outputs and want upstream
    priority (e.g. solver output preferred over teacher output for
    the same prompt)."""
    seen: set[str] = set()
    kept: list[GeneratedRow] = []
    for r in rows:
        key = dedup_key(r, filters.dedup_by)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


def upsample(rows: Iterable[GeneratedRow], recipe: DataRecipe) -> list[GeneratedRow]:
    """Repeat each row by ``{source}_upsample`` from its category's recipe.

    Sources without a corresponding ``_upsample`` field default to 1
    (no-op) — unknown sources ride along unchanged so a caller can wire
    a new generator type without touching this module.
    """
    out: list[GeneratedRow] = []
    for r in rows:
        cat = recipe.category(r.category)
        n = _pick_upsample_for_source(cat, r.source)
        out.extend([r] * n)
    return out


def _pick_upsample_for_source(cat, source: str) -> int:
    # Known source → explicit field on the dataclass
    if source == "solver":
        return cat.solver_upsample
    if source == "teacher_llm":
        return cat.teacher_upsample
    # Unknown source → check extras, default 1
    key = f"{source}_upsample"
    if key in cat.extra:
        try:
            n = int(cat.extra[key])
            return n if n > 0 else 1
        except (TypeError, ValueError):
            return 1
    return 1


__all__ = ["dedup", "dedup_key", "upsample"]
