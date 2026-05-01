"""PR6 acceptance: top-k preserves branch diversity."""

from __future__ import annotations

from agent_evolve.model.algorithms.mcgs.selection import TopKStore
from agent_evolve.model.types import TrainingSearchNode


def _node(id_: str, branch: int, metric: float) -> TrainingSearchNode:
    return TrainingSearchNode(
        node_id=id_,
        parent_id="r",
        branch_id=branch,
        metric=metric,
        is_valid=True,
    )


def test_topk_caps_per_branch() -> None:
    nodes = [
        _node("a1", 0, 0.9),
        _node("a2", 0, 0.8),
        _node("a3", 0, 0.7),
        _node("b1", 1, 0.6),
        _node("c1", 2, 0.5),
    ]
    store = TopKStore(k=3, per_branch_cap=2)
    entries = store.update(nodes)
    assert [n.node_id for n in entries] == ["a1", "a2", "b1"]
    # branch 0 capped at 2, so a3 is excluded despite being higher than c1.
    assert store.branches_represented() == 2


def test_topk_excludes_invalid() -> None:
    nodes = [_node("a", 0, 0.9)]
    nodes[0].is_valid = False
    store = TopKStore(k=3, per_branch_cap=2)
    entries = store.update(nodes)
    assert entries == []
