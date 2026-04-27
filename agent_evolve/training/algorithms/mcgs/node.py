"""Node helpers for the MCGS graph."""

from __future__ import annotations

from enum import Enum

from ...types import TrainingSearchNode, TrainingSearchNodeSummary


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    INVALID = "invalid"
    TERMINAL = "terminal"


def node_summary(node: TrainingSearchNode) -> TrainingSearchNodeSummary:
    return TrainingSearchNodeSummary(
        node_id=node.node_id,
        parent_id=node.parent_id,
        branch_id=node.branch_id,
        metric=node.metric,
        reward=node.reward,
        is_valid=node.is_valid,
        visits=node.visits,
        mean_reward=node.mean_reward,
    )
