"""Cycle outcome classifier — what kinds of records map to which outcome.

These helpers used to live in the headless ``orchestrator.run_cycle`` path
that has since been removed. The classification rules are still useful as
a stable reference for downstream consumers (trace viewer, summary
emitters), so we keep them here, co-located with their tests.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


_STRONG_KINDS = frozenset({"cv_result"})
_TRAINED_KINDS = frozenset({"cv_result", "training_run"})
_PARTIAL_KINDS = frozenset({"eval_report", "recipe_proposal", "data_gap",
                            "dataset_snapshot", "distill_batch",
                            "breakthrough"})


def _classify_outcome(
    new_records: list,
    promoted: bool,
    *,
    budget_exhausted: bool,
) -> str:
    """Map the records written this cycle onto the 5-state outcome enum.

    Order of checks matters: budget_exhausted wins over everything else
    (so we don't claim success on a truncated run); promotion then trumps
    plain training; partial only applies when nothing load-bearing ran.
    """
    if budget_exhausted:
        return "budget_exhausted"
    if promoted:
        return "promoted"
    kinds = {r.kind for r in new_records}
    if kinds & _TRAINED_KINDS:
        return "trained"
    if kinds & _PARTIAL_KINDS:
        return "partial"
    return "null"


def _count_kinds(records: list) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r.kind] = out.get(r.kind, 0) + 1
    return out


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
