"""Data-recipe YAML loader + shallow validation.

A recipe is an evolvable YAML under ``<workspace>/data/recipes/<name>.yaml``
that tells ``solver_distill`` / ``data_merge`` / ``synth_generate`` which
categories to run, how much to upsample each, and per-stage knobs.

Schema (v1, minimal — room to grow):

    recipe_name: str
    categories:
      <cat>:
        solver: "enabled" | "disabled"
        solver_upsample: int         # default 1
        teacher_upsample: int        # default 1
        cot_template: str            # default "verify_pass_v1"
        # category-specific extras ignored by this layer
    filters:
      require_verify_pass: bool      # default True
      max_cot_tokens: int | null     # default null (no cap)
      dedup_by: "prompt_hash" | "prompt_and_source_hash"  # default "prompt_and_source_hash"

Unknown fields are preserved in ``extra`` so benchmark-specific knobs
(e.g. ``ood_operators``) can ride along without forcing a schema bump
every time we add a mutation axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_COT_TEMPLATE = "verify_pass_v1"
_KNOWN_DEDUP_MODES = {"prompt_hash", "prompt_and_source_hash"}


@dataclass
class CategoryRecipe:
    name: str
    solver: str = "disabled"           # "enabled" | "disabled"
    solver_upsample: int = 1
    teacher_upsample: int = 1
    cot_template: str = _DEFAULT_COT_TEMPLATE
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def solver_enabled(self) -> bool:
        return self.solver == "enabled"


@dataclass
class RecipeFilters:
    require_verify_pass: bool = True
    max_cot_tokens: int | None = None
    dedup_by: str = "prompt_and_source_hash"


@dataclass
class DataRecipe:
    """Parsed form of a recipe YAML.

    Always accessed by category name via ``.category(cat)`` — returns a
    ``CategoryRecipe`` with defaults applied if the YAML omits that
    category. This keeps callers from sprinkling ``.get(cat, {}).get(key, default)``.
    """
    recipe_name: str
    categories: dict[str, CategoryRecipe]
    filters: RecipeFilters
    source_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def category(self, name: str) -> CategoryRecipe:
        """Return the recipe for ``name``, synthesizing a default
        (all-disabled) entry if unspecified."""
        if name in self.categories:
            return self.categories[name]
        return CategoryRecipe(name=name)

    def enabled_solver_categories(self) -> list[str]:
        return sorted(n for n, c in self.categories.items() if c.solver_enabled)


# ── Loader ───────────────────────────────────────────────────────────────

def load_recipe(path: str | Path) -> DataRecipe:
    """Load and minimally validate a recipe YAML. Raises ``ValueError``
    on structural errors (unknown dedup mode, non-int upsample, etc.)."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    return _parse_recipe_dict(raw, source_path=p)


def recipe_from_dict(raw: dict[str, Any], *, name: str = "inline") -> DataRecipe:
    """Parse a recipe from an already-loaded dict. Convenient for tests."""
    raw = dict(raw)
    raw.setdefault("recipe_name", name)
    return _parse_recipe_dict(raw, source_path=None)


def _parse_recipe_dict(raw: dict[str, Any], *, source_path: Path | None) -> DataRecipe:
    if not isinstance(raw, dict):
        raise ValueError("recipe root must be a mapping")

    recipe_name = str(raw.get("recipe_name", "unnamed"))

    # Categories
    cats_raw = raw.get("categories") or {}
    if not isinstance(cats_raw, dict):
        raise ValueError("`categories` must be a mapping of category -> cfg")
    categories: dict[str, CategoryRecipe] = {}
    for cat_name, cfg in cats_raw.items():
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ValueError(f"category `{cat_name}` cfg must be a mapping")

        solver = str(cfg.get("solver", "disabled"))
        if solver not in ("enabled", "disabled"):
            raise ValueError(f"category {cat_name}.solver must be enabled|disabled, got {solver!r}")

        cat = CategoryRecipe(
            name=str(cat_name),
            solver=solver,
            solver_upsample=_as_positive_int(cfg.get("solver_upsample", 1),
                                              f"{cat_name}.solver_upsample"),
            teacher_upsample=_as_positive_int(cfg.get("teacher_upsample", 1),
                                               f"{cat_name}.teacher_upsample"),
            cot_template=str(cfg.get("cot_template", _DEFAULT_COT_TEMPLATE)),
            extra={k: v for k, v in cfg.items()
                   if k not in {"solver", "solver_upsample",
                                "teacher_upsample", "cot_template"}},
        )
        categories[cat.name] = cat

    # Filters
    f_raw = raw.get("filters") or {}
    if not isinstance(f_raw, dict):
        raise ValueError("`filters` must be a mapping")
    dedup_by = str(f_raw.get("dedup_by", "prompt_and_source_hash"))
    if dedup_by not in _KNOWN_DEDUP_MODES:
        raise ValueError(
            f"filters.dedup_by must be one of {sorted(_KNOWN_DEDUP_MODES)}, got {dedup_by!r}"
        )
    max_cot_tokens = f_raw.get("max_cot_tokens")
    if max_cot_tokens is not None:
        max_cot_tokens = _as_positive_int(max_cot_tokens, "filters.max_cot_tokens")
    filters = RecipeFilters(
        require_verify_pass=bool(f_raw.get("require_verify_pass", True)),
        max_cot_tokens=max_cot_tokens,
        dedup_by=dedup_by,
    )

    # Anything else rides along
    extra = {k: v for k, v in raw.items()
             if k not in {"recipe_name", "categories", "filters"}}

    return DataRecipe(
        recipe_name=recipe_name,
        categories=categories,
        filters=filters,
        source_path=source_path,
        extra=extra,
    )


def _as_positive_int(value: Any, field_name: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from None
    if n <= 0:
        raise ValueError(f"{field_name} must be > 0, got {n}")
    return n


__all__ = [
    "CategoryRecipe",
    "DataRecipe",
    "RecipeFilters",
    "load_recipe",
    "recipe_from_dict",
]
