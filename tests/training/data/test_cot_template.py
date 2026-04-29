"""CoT post-processing guarantees: \\boxed{answer} + [verify]: PASS."""

from __future__ import annotations

from agent_evolve.training.data.cot_template import (
    CoTPostprocessConfig,
    extract_boxed,
    has_verify_pass,
    postprocess_cot,
)


def test_force_boxed_replaces_existing() -> None:
    raw = "Some reasoning. \\boxed{wrong_answer}"
    out = postprocess_cot(raw, "correct")
    assert "\\boxed{correct}" in out
    assert "wrong_answer" not in out


def test_force_boxed_appends_when_missing() -> None:
    out = postprocess_cot("just words", "42")
    assert out.strip().endswith("\\boxed{42}")


def test_verify_marker_injected_before_final_box() -> None:
    out = postprocess_cot("reasoning\n\\boxed{42}", "42")
    # Marker must appear strictly before the box
    box_idx = out.index("\\boxed{42}")
    mark_idx = out.lower().index("[verify]: pass")
    assert mark_idx < box_idx


def test_verify_marker_not_duplicated() -> None:
    raw = "reasoning\n[verify]: PASS\n\\boxed{42}"
    out = postprocess_cot(raw, "42")
    assert out.lower().count("[verify]: pass") == 1


def test_inject_verify_disabled_leaves_cot_alone() -> None:
    raw = "no marker here \\boxed{x}"
    out = postprocess_cot(raw, "X",
                          config=CoTPostprocessConfig(inject_verify_marker=False))
    assert "[verify]" not in out.lower()


def test_extract_boxed_returns_last() -> None:
    assert extract_boxed("\\boxed{first} middle \\boxed{second}") == "second"
    assert extract_boxed("no box") is None


def test_has_verify_pass_case_insensitive() -> None:
    assert has_verify_pass("[verify]: PASS")
    assert has_verify_pass("[VERIFY] :  pass")
    assert not has_verify_pass("[verify]: FAIL")


def test_postprocess_idempotent() -> None:
    """Running postprocess twice produces the same result — it should
    detect existing markers/boxes and no-op them."""
    once = postprocess_cot("reasoning", "42")
    twice = postprocess_cot(once, "42")
    assert once == twice
