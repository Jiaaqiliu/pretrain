"""Verifier gate: drops wrong answers, missing verify marks, over-long CoTs."""

from __future__ import annotations

from agent_evolve.model.data.base import GeneratedRow
from agent_evolve.model.data.cot_template import postprocess_cot
from agent_evolve.model.data.recipe import recipe_from_dict
from agent_evolve.model.data.verifier_gate import apply_verifier_gate

from .fakes import FakeVerifier


def _row(prompt: str, gt: str, boxed: str, cat: str = "fake",
         with_marker: bool = True) -> GeneratedRow:
    body = f"reasoning\n\\boxed{{{boxed}}}"
    if with_marker:
        body = f"reasoning\n[verify]: PASS\n\\boxed{{{boxed}}}"
    return GeneratedRow(
        id=f"{prompt}", prompt=prompt, answer=gt, category=cat,
        cot=body, source="solver",
    )


def test_kept_when_verifier_passes_and_marker_present() -> None:
    rec = recipe_from_dict({})
    verifiers = {"fake": FakeVerifier("fake")}
    rows = [_row("p", "answer", "answer")]
    kept, stats = apply_verifier_gate(rows, verifiers, rec.filters)
    assert len(kept) == 1
    assert stats.kept == 1
    assert stats.per_category_kept["fake"] == 1


def test_dropped_on_wrong_answer() -> None:
    rec = recipe_from_dict({})
    rows = [_row("p", "answer", "wrong")]
    kept, stats = apply_verifier_gate(rows, {"fake": FakeVerifier("fake")}, rec.filters)
    assert kept == []
    assert stats.dropped_wrong_answer == 1


def test_dropped_on_missing_verify_marker_when_required() -> None:
    rec = recipe_from_dict({"filters": {"require_verify_pass": True}})
    rows = [_row("p", "answer", "answer", with_marker=False)]
    kept, stats = apply_verifier_gate(rows, {"fake": FakeVerifier("fake")}, rec.filters)
    assert kept == []
    assert stats.dropped_missing_verify_mark == 1


def test_marker_not_required_when_filter_off() -> None:
    rec = recipe_from_dict({"filters": {"require_verify_pass": False}})
    rows = [_row("p", "answer", "answer", with_marker=False)]
    kept, stats = apply_verifier_gate(rows, {"fake": FakeVerifier("fake")}, rec.filters)
    assert stats.kept == 1


def test_dropped_on_cot_too_long() -> None:
    rec = recipe_from_dict({"filters": {"max_cot_tokens": 3}})
    rows = [_row("p", "answer", "answer")]  # CoT has way more than 3 words
    kept, stats = apply_verifier_gate(rows, {"fake": FakeVerifier("fake")}, rec.filters)
    assert kept == []
    assert stats.dropped_cot_too_long == 1
