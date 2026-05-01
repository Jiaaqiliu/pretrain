"""Mutation proposers for MCGS."""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from ...types import (
    PatchOperation,
    WorkspaceMutation,
    WorkspacePatch,
)


class BaselineMutationProposer:
    """Rotates through a small, deterministic bag of data-mix perturbations."""

    def __init__(self) -> None:
        self._i = 0
        self._bag = [
            ("failure_replay", 0.25),
            ("generic_synthetic", 0.10),
            ("default", 0.75),
            ("long_form", 0.15),
        ]

    def propose(
        self,
        parent: Any,
        graph: Any = None,  # noqa: ARG002 — hook for memory-aware proposer
    ) -> WorkspaceMutation:
        bucket, value = self._bag[self._i % len(self._bag)]
        self._i += 1
        mutation_id = f"m-{uuid.uuid4().hex[:8]}"
        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=f"Set data/mix.yaml:buckets.{bucket} to {value}",
            patch=WorkspacePatch(
                operations=[
                    PatchOperation(
                        op="replace",
                        path="data/mix.yaml",
                        key_path=["buckets", bucket],
                        value=value,
                    )
                ]
            ),
            mutation_type="data_mix",
        )


class LRBagMutationProposer:
    """Rotates through a bag of learning-rate values.

    Each mutation patches ``train/optimizer.yaml::lr``. Used for the Kaggle
    4-cycle sweep that forks a candidate per lr from root.
    """

    def __init__(self, bag: Iterable[float] = (1e-4, 5e-5, 3e-5, 1e-5)) -> None:
        self._i = 0
        self._bag = list(bag)
        if not self._bag:
            raise ValueError("LRBagMutationProposer bag must be non-empty")

    def propose(
        self,
        parent: Any,
        graph: Any = None,  # noqa: ARG002
    ) -> WorkspaceMutation:
        lr = self._bag[self._i % len(self._bag)]
        self._i += 1
        mutation_id = f"m-lr-{uuid.uuid4().hex[:8]}"
        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=f"Set train/optimizer.yaml:lr={lr}",
            patch=WorkspacePatch(
                operations=[
                    PatchOperation(
                        op="replace",
                        path="train/optimizer.yaml",
                        key_path=["lr"],
                        value=lr,
                    )
                ]
            ),
            mutation_type="training_recipe",
        )
