"""PR4 invariant: benchmark never computes reward or chooses an incumbent."""

from __future__ import annotations

from agent_evolve.benchmarks.nemo_reasoner import NemoReasonerBenchmark


def test_no_reward_method() -> None:
    banned = {
        "compute_reward",
        "reward",
        "backprop",
        "score_for_mcgs",
    }
    for name in banned:
        assert not hasattr(NemoReasonerBenchmark, name), (
            f"{name!r} must not live on the benchmark; reward is owned by MCGS"
        )


def test_no_incumbent_method() -> None:
    banned = {"promote_incumbent", "select_incumbent", "is_incumbent"}
    for name in banned:
        assert not hasattr(NemoReasonerBenchmark, name), (
            f"{name!r} must not live on the benchmark; promotion is MCGS-only"
        )


def test_registers_via_registry() -> None:
    from agent_evolve.training.registries import resolve_benchmark

    bench = resolve_benchmark("nemo_reasoner")
    assert bench.__class__.__name__ == "NemoReasonerBenchmark"
