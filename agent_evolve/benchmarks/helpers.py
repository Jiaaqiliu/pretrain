"""Cross-benchmark helpers for stage workers.

``teacher_distill`` and ``rl`` used to import ``EVAL_INSTRUCTION_SUFFIX /
build_eval_prompt / extract_final_answer / verify`` directly from
``nemo_reasoner.py``. That coupling meant any new benchmark (code-gen,
multi-choice QA, ...) couldn't use those stages without forking the stage
worker.

The helpers here prefer ``benchmark.<method>(...)`` when the benchmark
implements it, falling back to the legacy ``nemo_reasoner`` imports
otherwise. New benchmarks should implement the Protocol methods; the
legacy fallback keeps Nemotron-Reasoner pipelines green during the
transition.
"""

from __future__ import annotations

from typing import Any


def build_eval_prompt(benchmark: Any, row: Any, tokenizer: Any = None) -> str:
    fn = getattr(benchmark, "build_eval_prompt", None)
    if callable(fn):
        try:
            return fn(row, tokenizer)
        except TypeError:
            # Some benchmarks may define a one-arg form.
            return fn(row)
    from .nemo_reasoner import build_eval_prompt as _legacy

    raw_prompt = row if isinstance(row, str) else getattr(row, "prompt", row)
    return _legacy(raw_prompt, tokenizer)


def extract_final_answer(benchmark: Any, text: str | None) -> str:
    fn = getattr(benchmark, "extract_final_answer", None)
    if callable(fn):
        return fn(text)
    from .nemo_reasoner import extract_final_answer as _legacy

    return _legacy(text)


def verify(benchmark: Any, pred: str, gt: str) -> bool:
    fn = getattr(benchmark, "verify", None)
    if callable(fn):
        return fn(pred, gt)
    from .nemo_reasoner import verify as _legacy

    return _legacy(pred, gt)


__all__ = ["build_eval_prompt", "extract_final_answer", "verify"]
