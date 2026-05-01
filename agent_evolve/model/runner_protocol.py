"""``TrainingJobRunner`` — the one Protocol every training backend implements.

A TrainingJobRunner owns *how* one candidate gets trained and evaluated: which
framework (HF/PEFT, sklearn, JAX, ...), which loss, which artifacts. It does
NOT own:

  * which candidate to try next (that's the search algorithm)
  * when to declare an incumbent or compute reward (algorithm)
  * where to fork the workspace on disk (workspace owns that)

The existing LLM path (`SingleNodeTinkerLiteBackend`, `K8sTinkerLiteBackend`)
is one implementation. A sklearn / tabular / JAX job is another. See
``INTEGRATION.md`` §1 for a minimal non-LLM example.

Why a standalone Protocol rather than the old ``TinkerLiteBackend`` class:
scheduling (k8s fan-out, GPU locks) is orthogonal to "how to train one
candidate". Keeping them distinct means a sklearn runner doesn't grow a k8s
dependency by accident.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import (
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
)


@runtime_checkable
class TrainingJobRunner(Protocol):
    name: str

    def run_trial(
        self,
        workspace: Any,
        node: TrainingSearchNode,
        budget: TrialBudget,
        benchmark: Any,
    ) -> TrainingTrialResult: ...


__all__ = ["TrainingJobRunner"]
