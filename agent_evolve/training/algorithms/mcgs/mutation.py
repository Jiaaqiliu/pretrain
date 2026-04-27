"""Baseline mutation proposer (PR5).

Generates simple, self-contained patches to ``data/mix.yaml``. PR6 replaces
this with a memory-aware proposer.
"""

from __future__ import annotations

import uuid
from typing import Any

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
