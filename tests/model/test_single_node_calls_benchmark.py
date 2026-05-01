"""PR7 acceptance: backend invokes benchmark.evaluate (or its fallback)."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend
from agent_evolve.benchmarks.nemo_reasoner import NemoReasonerBenchmark
from agent_evolve.model.types import (
    CheckpointRef,
    EvalPlan,
    TrainingSearchNode,
    TrialBudget,
)
from agent_evolve.model.workspace import TrainingWorkspace


class _SpyBenchmark(NemoReasonerBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.eval_calls = 0
        self.parse_calls = 0

    def evaluate(self, workspace, checkpoint, backend, split):
        self.eval_calls += 1
        plan = self.build_eval_plan(workspace, checkpoint, split)
        return backend.run_eval_plan(plan)

    def parse_metrics(self, result_dir):
        self.parse_calls += 1
        return super().parse_metrics(result_dir)


def test_benchmark_evaluate_invoked(minimal_workspace: Path) -> None:
    # Add a tiny holdout so smoke eval produces a non-empty metrics.json.
    holdout = minimal_workspace / "eval" / "local_holdout_small.jsonl"
    holdout.write_text('{"is_correct": true}\n{"is_correct": false, "format_error": true}\n')

    ws = TrainingWorkspace.load(minimal_workspace)
    bench = _SpyBenchmark()
    backend = SingleNodeTinkerLiteBackend(mock=True)
    node = TrainingSearchNode(node_id="n-e", parent_id=None, branch_id=0)
    result = backend.run_trial(ws, node, TrialBudget(seconds=30), bench)

    assert bench.eval_calls == 1
    assert bench.parse_calls == 1
    assert result.eval_metrics is not None
    assert result.eval_metrics.primary_metric_value == 0.5
