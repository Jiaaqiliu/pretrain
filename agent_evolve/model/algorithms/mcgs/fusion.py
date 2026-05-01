"""Fusion policy — cross-branch recombination on stagnation."""

from __future__ import annotations

import uuid
from typing import Iterable

from ...types import (
    PatchOperation,
    TrainingSearchNode,
    WorkspaceMutation,
    WorkspacePatch,
)
from .memory import NodeMemoryStore


class FusionPolicy:
    def __init__(
        self,
        *,
        stagnation_cycles: int = 4,
        branch_stagnation: int = 3,
    ):
        self.stagnation_cycles = stagnation_cycles
        self.branch_stagnation = branch_stagnation
        self._no_improvement_streak = 0
        self._last_best: float | None = None

    def update_streak(self, best_metric: float | None) -> None:
        if best_metric is None:
            return
        if self._last_best is None or best_metric > self._last_best + 1e-9:
            self._no_improvement_streak = 0
            self._last_best = best_metric
        else:
            self._no_improvement_streak += 1

    def should_fuse(self) -> bool:
        return self._no_improvement_streak >= self.stagnation_cycles

    def maybe_fuse(
        self,
        topk: list[TrainingSearchNode],
        memory: NodeMemoryStore | None = None,
    ) -> WorkspaceMutation | None:
        if not self.should_fuse() or len(topk) < 2:
            return None
        # Reset streak — we're taking an action.
        self._no_improvement_streak = 0
        return _build_fusion_mutation(topk, memory)


def _build_fusion_mutation(
    topk: Iterable[TrainingSearchNode],
    memory: NodeMemoryStore | None,  # noqa: ARG001 — reserved for richer fusion
) -> WorkspaceMutation:
    entries = list(topk)
    parent = entries[0]
    top_ids = [n.node_id for n in entries]
    mutation_id = f"m-fuse-{uuid.uuid4().hex[:8]}"
    # Baseline fusion: combine each branch's mix signal into a new mix. We
    # encode it as a single patch on ``data/mix.yaml`` whose value is keyed
    # by branch_id — the actual merge lives in the mutation proposer.
    fused_value = {str(n.branch_id): (n.metric or 0.0) for n in entries}
    return WorkspaceMutation(
        mutation_id=mutation_id,
        parent_node_id=parent.node_id,
        description=f"Fuse top-k branches: {', '.join(top_ids)}",
        patch=WorkspacePatch(
            operations=[
                PatchOperation(
                    op="replace",
                    path="data/mix.yaml",
                    key_path=["fusion"],
                    value=fused_value,
                )
            ]
        ),
        mutation_type="fusion",
    )
