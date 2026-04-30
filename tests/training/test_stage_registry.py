"""PR-B targeted tests: ``StageRegistry`` decouples stage dispatch."""

from __future__ import annotations

import pytest

from agent_evolve.training.stage_registry import (
    STAGE_TYPES,
    StageContext,
    StageResult,
    register_stage,
    resolve_stage,
)


def test_builtin_stages_registered_on_runners_import():
    # Side-effect of importing runners: all built-in stages are registered.
    import agent_evolve.training.runners  # noqa: F401

    for stype in ("sft", "rl", "synth_generate", "solver_distill", "data_merge"):
        assert stype in STAGE_TYPES, f"{stype} not registered"


def test_resolve_unknown_stage_raises():
    with pytest.raises(KeyError, match="Unknown stage type"):
        resolve_stage("definitely_not_a_real_stage_type")


def test_register_new_stage_plugs_in_via_decorator():
    @register_stage("ut_test_stage_alpha")
    def _run(ctx: StageContext) -> StageResult:
        return StageResult(metrics={"ok": True, "stage": ctx.stage["name"]})

    try:
        fn = resolve_stage("ut_test_stage_alpha")
        ctx = StageContext(
            workspace=None, stage={"name": "demo"}, benchmark=None,
            budget_seconds=None, smoke=True, last_ckpt=None,
        )
        result = fn(ctx)
        assert result.metrics == {"ok": True, "stage": "demo"}
        assert result.checkpoint is None
    finally:
        STAGE_TYPES.pop("ut_test_stage_alpha", None)


def test_reregistering_same_stage_type_raises():
    @register_stage("ut_test_stage_beta")
    def _run1(ctx: StageContext) -> StageResult:
        return StageResult()

    try:
        with pytest.raises(RuntimeError, match="already registered"):
            @register_stage("ut_test_stage_beta")
            def _run2(ctx: StageContext) -> StageResult:  # noqa
                return StageResult()
    finally:
        STAGE_TYPES.pop("ut_test_stage_beta", None)


def test_registering_same_function_twice_is_idempotent():
    @register_stage("ut_test_stage_gamma")
    def _run(ctx: StageContext) -> StageResult:
        return StageResult()

    try:
        # Re-applying the decorator to the same function (e.g. a double
        # import) must not raise — only distinct-function collision does.
        register_stage("ut_test_stage_gamma")(_run)
    finally:
        STAGE_TYPES.pop("ut_test_stage_gamma", None)
