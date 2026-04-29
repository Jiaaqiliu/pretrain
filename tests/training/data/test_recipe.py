"""Recipe YAML loader validation + default-fill semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_evolve.training.data.recipe import (
    CategoryRecipe,
    load_recipe,
    recipe_from_dict,
)


def test_category_default_when_absent_from_yaml() -> None:
    rec = recipe_from_dict({"categories": {"a": {"solver": "enabled"}}})
    # Explicitly declared
    assert rec.category("a").solver_enabled
    # Not declared -> synthesized default, all disabled
    b = rec.category("b")
    assert isinstance(b, CategoryRecipe)
    assert not b.solver_enabled
    assert b.solver_upsample == 1


def test_enabled_solver_categories_sorted() -> None:
    rec = recipe_from_dict({
        "categories": {
            "z": {"solver": "enabled"},
            "a": {"solver": "enabled"},
            "m": {"solver": "disabled"},
        }
    })
    assert rec.enabled_solver_categories() == ["a", "z"]


def test_filters_defaults() -> None:
    rec = recipe_from_dict({})
    assert rec.filters.require_verify_pass is True
    assert rec.filters.max_cot_tokens is None
    assert rec.filters.dedup_by == "prompt_and_source_hash"


def test_bad_solver_value_rejected() -> None:
    with pytest.raises(ValueError, match="enabled|disabled"):
        recipe_from_dict({"categories": {"a": {"solver": "maybe"}}})


def test_negative_upsample_rejected() -> None:
    with pytest.raises(ValueError, match="> 0"):
        recipe_from_dict({"categories": {"a": {"solver_upsample": 0}}})


def test_non_int_upsample_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        recipe_from_dict({"categories": {"a": {"solver_upsample": "many"}}})


def test_unknown_dedup_mode_rejected() -> None:
    with pytest.raises(ValueError, match="dedup_by"):
        recipe_from_dict({"filters": {"dedup_by": "random"}})


def test_extra_fields_preserved_on_category() -> None:
    rec = recipe_from_dict({
        "categories": {"a": {
            "solver": "enabled",
            "ood_ratio": 0.3,
            "ood_operators": ["xor_then_rotate"],
        }}
    })
    assert rec.category("a").extra["ood_ratio"] == 0.3
    assert rec.category("a").extra["ood_operators"] == ["xor_then_rotate"]


def test_extra_top_level_preserved() -> None:
    rec = recipe_from_dict({"categories": {}, "custom_knob": 42})
    assert rec.extra["custom_knob"] == 42


def test_load_recipe_from_file(tmp_path: Path) -> None:
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump({
        "recipe_name": "test_v1",
        "categories": {"bit": {"solver": "enabled", "solver_upsample": 3}},
    }))
    rec = load_recipe(p)
    assert rec.recipe_name == "test_v1"
    assert rec.source_path == p
    assert rec.category("bit").solver_upsample == 3


def test_seed_workspace_recipe_valid() -> None:
    """Regression guard: the default recipe we shipped must parse cleanly."""
    from agent_evolve.training.data.recipe import load_recipe
    p = Path(__file__).resolve().parents[3] / \
        "seed_workspaces" / "nemotron_reasoner" / "data" / "recipes" / "default.yaml"
    assert p.is_file(), f"seed recipe missing: {p}"
    rec = load_recipe(p)
    assert rec.recipe_name == "solver_first_v1"
    assert rec.filters.max_cot_tokens == 7600
    # All categories start disabled (solvers not yet shipped)
    for name, cat in rec.categories.items():
        assert cat.solver == "disabled", \
            f"{name} should ship disabled; flip per-category when solver lands"
