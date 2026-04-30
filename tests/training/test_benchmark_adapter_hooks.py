"""PR-E targeted tests: benchmark-agnostic hooks for teacher_distill + rl.

Today ``teacher_distill.py`` and ``rl.py`` still ``import ... from
benchmarks.nemo_reasoner``. ``benchmarks.helpers`` is the transitional
shim that prefers ``benchmark.<method>(...)`` when present and falls
back to the legacy imports otherwise. This test fleet locks in both paths.
"""

from __future__ import annotations

from agent_evolve.benchmarks import helpers
from agent_evolve.benchmarks.nemo_reasoner import NemoReasonerBenchmark


class _BenchmarkWithHooks:
    """Benchmark that implements the Protocol hooks directly."""

    def build_eval_prompt(self, row, tokenizer=None):
        return f"CUSTOM::{row}"

    def extract_final_answer(self, text):
        return "CUSTOM_ANSWER"

    def verify(self, pred, gt):
        return True


class _BenchmarkNoHooks:
    """Benchmark lacking the hook methods — helpers fall back to legacy."""


def test_helper_prefers_benchmark_method():
    b = _BenchmarkWithHooks()
    assert helpers.build_eval_prompt(b, "row") == "CUSTOM::row"
    assert helpers.extract_final_answer(b, "text") == "CUSTOM_ANSWER"
    assert helpers.verify(b, "p", "g") is True


def test_helper_falls_back_to_legacy_nemo_reasoner():
    b = _BenchmarkNoHooks()
    # Legacy build_eval_prompt adds the boxed-answer suffix.
    prompt = helpers.build_eval_prompt(b, "what is 2+2?")
    assert "\\boxed" in prompt
    # Legacy extract_final_answer on a boxed text pulls the boxed value.
    assert helpers.extract_final_answer(b, r"the answer is \boxed{42}") == "42"


def test_nemo_reasoner_conforms_to_hooks():
    b = NemoReasonerBenchmark()
    assert callable(b.build_eval_prompt)
    assert callable(b.extract_final_answer)
    assert callable(b.verify)
    # Identity: helpers routed through the benchmark produce the same
    # result as calling the benchmark method directly.
    via_helper = helpers.extract_final_answer(b, r"\boxed{7}")
    via_method = b.extract_final_answer(r"\boxed{7}")
    assert via_helper == via_method == "7"
