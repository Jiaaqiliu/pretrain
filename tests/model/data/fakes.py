"""In-memory fakes for testing the data pipeline without touching a real
benchmark or any heavy deps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agent_evolve.model.data.base import (
    GeneratedRow,
    SolverResult,
    TrainingExample,
)


@dataclass
class FakeSolver:
    category: str
    answers: dict[str, str]   # prompt -> predicted answer; missing = decline

    def solve(self, prompt: str) -> SolverResult:
        if prompt not in self.answers:
            return SolverResult(predicted_answer=None, confidence="none", method="declined")
        return SolverResult(
            predicted_answer=self.answers[prompt],
            trace_dict={"prompt": prompt, "answer": self.answers[prompt]},
            confidence="high",
            method="fake",
        )


@dataclass
class FakeVerifier:
    category: str

    def check(self, pred: str, gt: str) -> bool:
        return pred.strip().lower() == gt.strip().lower()


@dataclass
class FakeRenderer:
    category: str

    def render(self, prompt: str, answer: str, trace: dict) -> str:
        return (
            f"<think>\n"
            f"Prompt says: {prompt.strip()}\n"
            f"I reason that the answer is {answer}.\n"
            f"</think>\n"
            f"\\boxed{{{answer}}}"
        )


class FakeBenchmark:
    name = "fake_bench"

    def __init__(self, rows: list[TrainingExample]):
        self._rows = rows

    def iter_training_rows(self, workspace) -> Iterable[TrainingExample]:
        yield from self._rows

    def classify_category(self, prompt: str) -> str:
        return "fake"

    def solvers(self):
        return {"fake": FakeSolver("fake", {r.prompt: r.answer for r in self._rows})}

    def verifiers(self):
        return {"fake": FakeVerifier("fake")}

    def cot_renderers(self):
        return {"fake": FakeRenderer("fake")}


class EmptyBenchmark:
    """Benchmark that implements nothing — for testing the 'no solvers'
    graceful-degradation path."""
    name = "empty_bench"


class FakeWorkspace:
    def __init__(self, root):
        from pathlib import Path
        self.root = str(Path(root))
