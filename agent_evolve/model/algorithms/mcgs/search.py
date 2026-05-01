"""MCGSSearch — the main training search algorithm (PR5 core).

Responsibilities:
  - Pick a parent via basic UCT.
  - Propose a mutation.
  - Fork a candidate workspace.
  - Run the backend.
  - Ask the benchmark for validity + parsed metrics.
  - Compute reward via :class:`DefaultRewardPolicy`.
  - Add the node to the graph and backpropagate.
  - Promote incumbent via :class:`PromotionPolicy`.

PR6 adds memory, top-k diversity, and fusion on top of this.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from ...types import (
    MCGSCycleReport,
    TrainingSearchNode,
    TrainingSearchNodeSummary,
    TrainingTrialResult,
    ValidityReport,
)
from .fusion import FusionPolicy
from .graph import TrainingSearchGraph
from .memory import NodeMemoryStore
from .mutation import BaselineMutationProposer
from .node import node_summary
from .promotion import PromotionPolicy
from .reward import DefaultRewardPolicy
from .selection import TopKStore, UCTSelector

logger = logging.getLogger(__name__)


class MCGSSearch:
    def __init__(
        self,
        *,
        exploration_c: float = 1.4,
        reward_policy: DefaultRewardPolicy | None = None,
        promotion_policy: PromotionPolicy | None = None,
        mutator: Any | None = None,
        selector: UCTSelector | None = None,
        topk: TopKStore | None = None,
        fusion: FusionPolicy | None = None,
    ):
        self.exploration_c = exploration_c
        self.graph = TrainingSearchGraph()
        self.reward_policy = reward_policy or DefaultRewardPolicy()
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.mutator = mutator or BaselineMutationProposer()
        self.selector = selector or UCTSelector(c_init=exploration_c)
        self.topk = topk or TopKStore(k=5, per_branch_cap=2)
        self.fusion = fusion or FusionPolicy()
        self.memory: NodeMemoryStore | None = None
        self._next_branch_id = 0
        self._root_initialized = False
        self.graph_path: str | None = None

    # ── Public API (for the loop) ────────────────────────────────────

    def run_cycle(self, ctx: Any) -> MCGSCycleReport:
        self._ensure_root(ctx)
        self._ensure_memory(ctx)
        parent = self._select_parent(cycle=ctx.cycle)

        mutation = self._propose_mutation(parent)
        node_id = f"node-{uuid.uuid4().hex[:10]}"
        branch_id = self._assign_branch(parent)

        candidate_ws = self._fork_workspace(ctx, parent, node_id, mutation)

        node = TrainingSearchNode(
            node_id=node_id,
            parent_id=parent.node_id,
            branch_id=branch_id,
            mutation_plan=mutation.description,
            workspace_patch=mutation.patch,
        )
        self.graph.add_node(node)

        result = ctx.trial.run(candidate_ws, node, ctx.budget)
        ctx.observer.record_trial(result)
        self._absorb_result(node, parent, result)

        incumbent_id, changed = self.promotion_policy.update_incumbent(self.graph, node)
        self.topk.update(self.graph.nodes.values())
        best_metric = self._best_metric()
        self.fusion.update_streak(best_metric)
        if self.memory is not None:
            self.memory.record(node)
            fusion_mutation = self.fusion.maybe_fuse(self.topk.entries, memory=self.memory)
            if fusion_mutation is not None:
                self.memory.record_fusion_candidate(fusion_mutation)
        self._persist_graph(ctx)

        return MCGSCycleReport(
            cycle=ctx.cycle,
            selected_parent_id=parent.node_id,
            trial_node_ids=[node.node_id],
            incumbent_node_id=incumbent_id,
            incumbent_changed=changed,
            best_metric=best_metric,
            graph_path=self.graph_path or "",
            report_path="",
        )

    def topk_summary(self) -> list[TrainingSearchNodeSummary]:
        ranked = sorted(
            (n for n in self.graph.nodes.values() if n.is_valid),
            key=lambda n: (n.metric if n.metric is not None else float("-inf")),
            reverse=True,
        )
        return [node_summary(n) for n in ranked[:5]]

    # ── Internals ────────────────────────────────────────────────────

    def _ensure_root(self, ctx: Any) -> None:
        if self._root_initialized:
            return
        root = TrainingSearchNode(
            node_id="node-root",
            parent_id=None,
            branch_id=-1,
            mutation_plan="root",
            pipeline_summary="seed workspace",
            is_valid=True,
            is_terminal=False,
        )
        # Seed the root with a sentinel metric drawn from the baseline workspace
        # (zero if nothing is known). This keeps UCT well-defined from cycle 1.
        root.metric = 0.0
        root.reward = 0.0
        root.visits = 1
        self.graph.add_node(root)
        self._root_initialized = True

    def _select_parent(self, cycle: int = 1) -> TrainingSearchNode:
        return self.selector.select(self.graph, cycle=cycle)

    def _propose_mutation(self, parent: TrainingSearchNode) -> Any:
        # Prefer a fusion mutation if fusion is ready, and the memory store
        # has observed enough stagnation.
        if self.fusion.should_fuse() and len(self.topk.entries) >= 2:
            fused = self.fusion.maybe_fuse(self.topk.entries, memory=self.memory)
            if fused is not None:
                # Fusion patches attach to the top-1 parent.
                fused.parent_node_id = parent.node_id
                return fused
        return self.mutator.propose(parent, graph=self.graph)

    def _ensure_memory(self, ctx: Any) -> None:
        if self.memory is not None:
            return
        memory_root = Path(ctx.workspace.root) / "evolution" / "memory"
        self.memory = NodeMemoryStore(memory_root)

    def _assign_branch(self, parent: TrainingSearchNode) -> int:
        if parent.branch_id < 0:
            # Parent is the root — start a new branch.
            bid = self._next_branch_id
            self._next_branch_id += 1
            return bid
        return parent.branch_id

    def _fork_workspace(
        self,
        ctx: Any,
        parent: TrainingSearchNode,  # noqa: ARG002
        node_id: str,
        mutation: Any,
    ) -> Any:
        return ctx.workspace.fork(node_id, mutation, work_dir=ctx.work_dir)

    def _absorb_result(
        self,
        node: TrainingSearchNode,
        parent: TrainingSearchNode,
        result: TrainingTrialResult,
    ) -> None:
        node.trial_status = result.status
        node.checkpoint = result.checkpoint
        node.artifacts = dict(result.artifacts or {})
        node.cost = dict(result.cost or {})
        if result.error_buckets:
            node.error_buckets = dict(result.error_buckets.counts)

        validity = result.validity or ValidityReport(is_valid=result.status == "success")
        node.validity = validity
        node.is_valid = validity.is_valid
        if not validity.is_valid:
            node.is_terminal = True

        if result.eval_metrics is not None:
            node.metric = result.eval_metrics.primary_metric_value

        reward = self.reward_policy.compute(result, validity, parent, self.graph)
        node.reward = reward
        self._backpropagate(node, reward)

    def _backpropagate(self, node: TrainingSearchNode, reward: float) -> None:
        cursor: TrainingSearchNode | None = node
        while cursor is not None:
            cursor.visits += 1
            cursor.total_reward += reward
            cursor.mean_reward = cursor.total_reward / cursor.visits
            cursor = self.graph.parent(cursor)

    def _persist_graph(self, ctx: Any) -> None:
        path = Path(ctx.workspace.root) / "evolution" / "mcgs_graph.json"
        self.graph_path = str(self.graph.save(path))

    def _best_metric(self) -> float | None:
        valid = [
            n.metric for n in self.graph.nodes.values()
            if n.is_valid and n.metric is not None and n.node_id != "node-root"
        ]
        return max(valid) if valid else None
