"""Protocol conformance + dataclass round-trips for training.data.base."""

from __future__ import annotations

from agent_evolve.model.data.base import (
    CoTRenderer,
    GeneratedRow,
    Solver,
    SolverResult,
    TrainingExample,
    Verifier,
)

from .fakes import FakeRenderer, FakeSolver, FakeVerifier


def test_solver_result_defaults() -> None:
    r = SolverResult(predicted_answer=None)
    assert r.confidence == "none"
    assert r.method == ""
    assert r.trace_dict == {}


def test_generated_row_roundtrip() -> None:
    row = GeneratedRow(
        id="x", prompt="p", answer="a", category="c",
        cot="cot text", source="solver", metadata={"k": 1},
    )
    d = row.to_dict()
    assert d["id"] == "x"
    assert d["metadata"] == {"k": 1}
    back = GeneratedRow.from_dict(d)
    assert back == row


def test_generated_row_missing_metadata_backfills_empty() -> None:
    d = {"id": "1", "prompt": "p", "answer": "a", "category": "c",
         "cot": "c", "source": "solver"}
    r = GeneratedRow.from_dict(d)
    assert r.metadata == {}


def test_fakes_satisfy_runtime_checkable_protocols() -> None:
    s = FakeSolver("cat", {})
    v = FakeVerifier("cat")
    r = FakeRenderer("cat")
    # runtime_checkable Protocols validate attribute presence only —
    # enough to catch the 80% refactor hazard.
    assert isinstance(s, Solver)
    assert isinstance(v, Verifier)
    assert isinstance(r, CoTRenderer)


def test_training_example_metadata_independent() -> None:
    e1 = TrainingExample(id="1", prompt="p", answer="a", category="c")
    e2 = TrainingExample(id="2", prompt="q", answer="b", category="c")
    # Regression guard: default_factory, not shared dict
    e1.metadata["x"] = 1
    assert "x" not in e2.metadata
