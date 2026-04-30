"""TrialRunner — thin wrapper around backend.run_trial.

Normalizes edge cases (raised exceptions, missing validity reports) into a
uniform :class:`TrainingTrialResult`. MCGS calls this so the algorithm layer
never needs to deal with backend-specific exception types.
"""

from __future__ import annotations

import logging
from typing import Any

from .runner_protocol import TrainingJobRunner
from .types import (
    TrainingSearchNode,
    TrainingTrialResult,
    TrialBudget,
    ValidityReport,
)

logger = logging.getLogger(__name__)


class TrialRunner:
    def __init__(self, backend: TrainingJobRunner, benchmark: Any):
        # ``backend`` is any ``TrainingJobRunner`` — LLM backend, sklearn
        # runner, etc. See ``INTEGRATION.md`` §1.
        self.backend = backend
        self.benchmark = benchmark

    def run(
        self,
        workspace: Any,
        node: TrainingSearchNode,
        budget: TrialBudget,
    ) -> TrainingTrialResult:
        try:
            result = self.backend.run_trial(workspace, node, budget, self.benchmark)
        except Exception as exc:  # unexpected backend crash
            logger.exception("backend crashed for node %s", node.node_id)
            return TrainingTrialResult(
                node_id=node.node_id,
                workspace_path=str(workspace.root),
                status="train_failed",
                train_metrics={"error": repr(exc)},
            )

        # If the benchmark can rule the trial invalid, do so now.
        if result.status == "success" and hasattr(self.benchmark, "check_validity"):
            try:
                validity = self.benchmark.check_validity(workspace, result)
            except Exception as exc:  # validity checker itself broke
                logger.exception("check_validity raised for node %s", node.node_id)
                validity = ValidityReport(
                    is_valid=False, hard_fail_reason=f"check_validity_exception:{exc}"
                )
            result.validity = validity
            if not validity.is_valid:
                # Preserve the success-path checkpoint/metrics so MCGS can
                # still report them; just downgrade the status.
                result.status = "invalid_adapter"
        elif result.validity is None:
            result.validity = ValidityReport(is_valid=result.status == "success")
        return result
