"""Cycle outcome classifier (proposal d): what kinds of records map to
which ``CycleOutcome`` enum value.

The classifier is the seam between "orchestrator did stuff" and "driver
decides whether to advance the cycle counter"; getting this wrong means
either burning Bedrock on trivial cycles or losing productive ones.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_evolve.model.algorithms.nemo_mas.orchestrator import (
    _classify_outcome,
    _count_kinds,
)


def _rec(kind: str):
    return SimpleNamespace(kind=kind)


class TestClassifyOutcome:
    def test_budget_exhausted_wins_over_everything(self):
        # Even a promotion becomes budget_exhausted if we ran out of budget.
        assert _classify_outcome(
            [_rec("cv_result"), _rec("training_run")],
            promoted=True,
            budget_exhausted=True,
        ) == "budget_exhausted"

    def test_promoted_when_incumbent_changed(self):
        assert _classify_outcome(
            [_rec("cv_result")], promoted=True, budget_exhausted=False,
        ) == "promoted"

    def test_trained_on_cv_result_without_promotion(self):
        # A cv_result exists but it didn't beat the incumbent — still real
        # training happened, so credit the cycle.
        assert _classify_outcome(
            [_rec("cv_result")], promoted=False, budget_exhausted=False,
        ) == "trained"

    def test_trained_on_training_run(self):
        # Single training_run without cv_result still counts.
        assert _classify_outcome(
            [_rec("training_run")], promoted=False, budget_exhausted=False,
        ) == "trained"

    def test_partial_on_analysis_only(self):
        # The MAS analyzed but didn't train — not nothing, but not enough
        # to advance the evolutionary step.
        for kind in ("eval_report", "recipe_proposal", "data_gap",
                     "dataset_snapshot", "distill_batch", "breakthrough"):
            assert _classify_outcome(
                [_rec(kind)], promoted=False, budget_exhausted=False,
            ) == "partial", f"{kind} should classify as partial"

    def test_null_on_no_load_bearing_records(self):
        # Only a failed_attempt + a data_audit_finding — both recorded
        # but neither moves search forward in a load-bearing way.
        assert _classify_outcome(
            [_rec("failed_attempt"), _rec("data_audit_finding")],
            promoted=False, budget_exhausted=False,
        ) == "null"

    def test_null_on_empty(self):
        assert _classify_outcome(
            [], promoted=False, budget_exhausted=False,
        ) == "null"

    def test_trained_trumps_partial(self):
        # When both load-bearing and partial kinds appear, "trained" wins.
        assert _classify_outcome(
            [_rec("eval_report"), _rec("training_run"), _rec("data_gap")],
            promoted=False, budget_exhausted=False,
        ) == "trained"


class TestCountKinds:
    def test_counts(self):
        got = _count_kinds([_rec("eval_report"), _rec("eval_report"),
                            _rec("training_run")])
        assert got == {"eval_report": 2, "training_run": 1}

    def test_empty(self):
        assert _count_kinds([]) == {}


class TestCycleReportFields:
    """Confirm MCGSCycleReport can be instantiated with the new defaults."""

    def test_defaults(self):
        from agent_evolve.model.types import MCGSCycleReport
        report = MCGSCycleReport(
            cycle=1, selected_parent_id=None, trial_node_ids=[],
            incumbent_node_id=None, incumbent_changed=False,
            best_metric=None, graph_path="", report_path="",
        )
        # Backward-compat: old call sites that don't pass the new fields
        # still get a sensible default and stable reprs.
        assert report.cycle_outcome == "trained"
        assert report.wall_seconds == 0.0
        assert report.orchestrator_turns == 0
        assert report.record_counts == {}
