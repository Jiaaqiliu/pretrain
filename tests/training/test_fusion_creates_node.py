"""PR6 acceptance: fusion produces mutation with mutation_type='fusion'."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.training.algorithms.mcgs.fusion import FusionPolicy
from agent_evolve.training.algorithms.mcgs.memory import NodeMemoryStore
from agent_evolve.training.algorithms.mcgs.selection import TopKStore
from agent_evolve.training.types import TrainingSearchNode


def _valid(id_: str, branch: int, metric: float) -> TrainingSearchNode:
    return TrainingSearchNode(
        node_id=id_,
        parent_id="r",
        branch_id=branch,
        metric=metric,
        is_valid=True,
    )


def test_fusion_mutation_type(tmp_path: Path) -> None:
    policy = FusionPolicy(stagnation_cycles=2)
    topk = TopKStore(k=3, per_branch_cap=1).update(
        [
            _valid("a", 0, 0.4),
            _valid("b", 1, 0.4),
            _valid("c", 2, 0.4),
        ]
    )
    memory = NodeMemoryStore(tmp_path)

    # No stagnation yet → no fusion.
    assert policy.maybe_fuse(topk, memory=memory) is None

    # Simulate three cycles without improvement on the same metric plateau.
    for _ in range(3):
        policy.update_streak(best_metric=0.4)

    mutation = policy.maybe_fuse(topk, memory=memory)
    assert mutation is not None
    assert mutation.mutation_type == "fusion"
    # Patch should include the fusion key on data/mix.yaml.
    paths = [op.path for op in mutation.patch.operations]
    assert "data/mix.yaml" in paths


def test_fusion_skipped_when_only_one_branch(tmp_path: Path) -> None:
    policy = FusionPolicy(stagnation_cycles=1)
    topk = [_valid("a", 0, 0.4)]
    for _ in range(3):
        policy.update_streak(best_metric=0.4)
    assert policy.maybe_fuse(topk, memory=NodeMemoryStore(tmp_path)) is None
