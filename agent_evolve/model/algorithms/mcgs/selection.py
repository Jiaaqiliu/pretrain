"""Selection + top-k helpers for MCGS."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from ...types import TrainingSearchNode


class UCTSelector:
    """UCT with piecewise exploration decay + branch-diversity guard.

    The inspiration is MLEvolve's "piecewise exploration decay" — we shrink
    the exploration constant linearly between cycle 0 and ``decay_until``,
    floored at ``c_floor``.
    """

    def __init__(
        self,
        c_init: float = 1.4,
        c_floor: float = 0.4,
        decay_until: int = 20,
        branch_dominance_cap: float = 0.6,
    ):
        self.c_init = c_init
        self.c_floor = c_floor
        self.decay_until = max(1, decay_until)
        self.branch_dominance_cap = branch_dominance_cap

    def c_for_cycle(self, cycle: int) -> float:
        ratio = min(1.0, cycle / self.decay_until)
        return self.c_init - (self.c_init - self.c_floor) * ratio

    def select(
        self,
        graph,
        *,
        cycle: int,
    ) -> TrainingSearchNode:
        candidates = [n for n in graph.nodes.values() if not n.is_terminal]
        if not candidates:
            root = graph.root()
            assert root is not None
            return root

        branch_counts = defaultdict(int)
        for n in graph.nodes.values():
            if n.branch_id >= 0:
                branch_counts[n.branch_id] += 1
        total = sum(branch_counts.values()) or 1

        c = self.c_for_cycle(cycle)

        def uct(node: TrainingSearchNode) -> float:
            parent = graph.parent(node)
            parent_visits = max(1, parent.visits if parent else 1)
            exploration = c * math.sqrt(math.log(parent_visits + 1) / (node.visits + 1))
            score = node.mean_reward + exploration
            if node.branch_id >= 0:
                branch_share = branch_counts[node.branch_id] / total
                if branch_share > self.branch_dominance_cap:
                    score -= (branch_share - self.branch_dominance_cap)
            return score

        return max(candidates, key=uct)


class TopKStore:
    """Keeps the top-k nodes with a per-branch cap to preserve diversity."""

    def __init__(self, k: int = 5, per_branch_cap: int = 2):
        self.k = k
        self.per_branch_cap = per_branch_cap
        self.entries: list[TrainingSearchNode] = []

    def update(self, nodes: Iterable[TrainingSearchNode]) -> list[TrainingSearchNode]:
        pool = [
            n
            for n in nodes
            if n.is_valid and n.metric is not None and n.node_id != "node-root"
        ]
        pool.sort(key=lambda n: n.metric or float("-inf"), reverse=True)
        kept: list[TrainingSearchNode] = []
        per_branch: dict[int, int] = defaultdict(int)
        for node in pool:
            if per_branch[node.branch_id] >= self.per_branch_cap:
                continue
            kept.append(node)
            per_branch[node.branch_id] += 1
            if len(kept) >= self.k:
                break
        self.entries = kept
        return list(self.entries)

    def branches_represented(self) -> int:
        return len({n.branch_id for n in self.entries})
