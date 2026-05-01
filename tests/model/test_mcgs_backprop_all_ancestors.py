"""PR5 acceptance: backpropagation updates all ancestors."""

from __future__ import annotations

from agent_evolve.model.algorithms.mcgs import MCGSSearch
from agent_evolve.model.types import TrainingSearchNode


def test_backprop_updates_all_ancestors() -> None:
    algo = MCGSSearch()
    # Seed root manually.
    root = TrainingSearchNode(node_id="root", parent_id=None, branch_id=-1, is_valid=True)
    a = TrainingSearchNode(node_id="A", parent_id="root", branch_id=0, is_valid=True)
    b = TrainingSearchNode(node_id="B", parent_id="A", branch_id=0, is_valid=True)
    algo.graph.add_node(root)
    algo.graph.add_node(a)
    algo.graph.add_node(b)

    algo._backpropagate(b, reward=0.5)

    assert b.visits == 1 and b.total_reward == 0.5 and b.mean_reward == 0.5
    assert a.visits == 1 and a.total_reward == 0.5 and a.mean_reward == 0.5
    assert root.visits == 1 and root.total_reward == 0.5 and root.mean_reward == 0.5

    # A second reward on A propagates to root but not B.
    algo._backpropagate(a, reward=0.1)
    assert root.visits == 2
    assert a.visits == 2
    assert b.visits == 1
