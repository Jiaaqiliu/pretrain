"""PR3 invariant: the loop has no accept/reject gate surface."""

from __future__ import annotations

from agent_evolve.training.loop import TrainingEvolutionLoop


def test_no_gate_methods() -> None:
    banned = {
        "accept_candidate",
        "reject_candidate",
        "gate",
        "check_gate",
        "promote",
        "promote_candidate",
        "compute_reward",
    }
    for name in banned:
        assert not hasattr(TrainingEvolutionLoop, name), (
            f"loop exposes forbidden method {name!r}; promotion logic belongs to MCGS"
        )
