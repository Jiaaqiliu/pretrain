"""Dedup + upsample semantics."""

from __future__ import annotations

from agent_evolve.training.data.base import GeneratedRow
from agent_evolve.training.data.dedup import dedup, dedup_key, upsample
from agent_evolve.training.data.recipe import recipe_from_dict


def _row(prompt: str, source: str = "solver", category: str = "c", answer: str = "a") -> GeneratedRow:
    return GeneratedRow(
        id=f"{source}_{prompt}_{answer}",
        prompt=prompt, answer=answer, category=category,
        cot=f"cot-{prompt}", source=source,
    )


def test_dedup_key_prompt_only_ignores_source() -> None:
    r1 = _row("p", "solver")
    r2 = _row("p", "teacher_llm")
    assert dedup_key(r1, "prompt_hash") == dedup_key(r2, "prompt_hash")


def test_dedup_key_prompt_and_source_distinguishes() -> None:
    r1 = _row("p", "solver")
    r2 = _row("p", "teacher_llm")
    assert dedup_key(r1, "prompt_and_source_hash") != dedup_key(r2, "prompt_and_source_hash")


def test_dedup_first_seen_wins() -> None:
    rec = recipe_from_dict({})
    rows = [_row("a", "solver"), _row("b", "solver"), _row("a", "solver")]
    out = dedup(rows, rec.filters)
    assert [r.id for r in out] == [rows[0].id, rows[1].id]


def test_dedup_preserves_input_order_for_stable_upstream_priority() -> None:
    # With prompt_and_source dedup, same prompt from different sources both
    # survive — important because it means the merge stage can keep
    # solver rows AND teacher rows for the same prompt when desired.
    rec = recipe_from_dict({})
    rows = [_row("p", "solver"), _row("p", "teacher_llm")]
    out = dedup(rows, rec.filters)
    assert len(out) == 2
    assert [r.source for r in out] == ["solver", "teacher_llm"]


def test_upsample_respects_source_specific_ratios() -> None:
    rec = recipe_from_dict({
        "categories": {
            "cat_a": {"solver_upsample": 3, "teacher_upsample": 2},
        }
    })
    rows = [_row("p1", "solver", "cat_a"), _row("p2", "teacher_llm", "cat_a")]
    out = upsample(rows, rec)
    # p1 x 3 + p2 x 2
    assert len(out) == 5
    assert sum(1 for r in out if r.source == "solver") == 3
    assert sum(1 for r in out if r.source == "teacher_llm") == 2


def test_upsample_unknown_source_default_one() -> None:
    rec = recipe_from_dict({"categories": {"c": {"solver": "enabled"}}})
    rows = [_row("p", "ood_augment", "c")]
    out = upsample(rows, rec)
    assert len(out) == 1   # unknown source -> 1x


def test_upsample_unknown_source_extras_ratio() -> None:
    # If the recipe carries an `ood_augment_upsample: 5` in extras,
    # upsample should honor it via the generic lookup.
    rec = recipe_from_dict({
        "categories": {"c": {"solver": "enabled", "ood_augment_upsample": 5}}
    })
    rows = [_row("p", "ood_augment", "c")]
    out = upsample(rows, rec)
    assert len(out) == 5


def test_upsample_category_not_in_recipe_defaults_applied() -> None:
    rec = recipe_from_dict({})
    rows = [_row("p", "solver", "unknown_cat")]
    out = upsample(rows, rec)
    assert len(out) == 1
