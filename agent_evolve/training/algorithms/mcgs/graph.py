"""TrainingSearchGraph — persistent DAG of training search nodes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from ...types import (
    CheckpointRef,
    PatchOperation,
    TrainingSearchNode,
    ValidityReport,
    WorkspacePatch,
)


class TrainingSearchGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, TrainingSearchNode] = {}
        # parent_id -> set of child ids (ordered list preserves insertion)
        self._children: dict[str, list[str]] = {}
        self.root_id: str | None = None

    # ── Mutation ─────────────────────────────────────────────────────

    def add_node(self, node: TrainingSearchNode) -> TrainingSearchNode:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node
        if node.parent_id is None:
            if self.root_id is None:
                self.root_id = node.node_id
        else:
            self._children.setdefault(node.parent_id, []).append(node.node_id)
        return node

    # ── Queries ──────────────────────────────────────────────────────

    def parent(self, node: TrainingSearchNode) -> TrainingSearchNode | None:
        if node.parent_id is None:
            return None
        return self.nodes.get(node.parent_id)

    def children(self, node: TrainingSearchNode) -> list[TrainingSearchNode]:
        return [self.nodes[c] for c in self._children.get(node.node_id, [])]

    def ancestors(self, node: TrainingSearchNode) -> Iterator[TrainingSearchNode]:
        cursor = self.parent(node)
        while cursor is not None:
            yield cursor
            cursor = self.parent(cursor)

    def root(self) -> TrainingSearchNode | None:
        return self.nodes.get(self.root_id) if self.root_id else None

    def __iter__(self) -> Iterator[TrainingSearchNode]:
        return iter(self.nodes.values())

    def __len__(self) -> int:
        return len(self.nodes)

    # ── Serialization ────────────────────────────────────────────────

    def to_json(self) -> dict:
        return {
            "root_id": self.root_id,
            "nodes": [_node_to_dict(n) for n in self.nodes.values()],
            "children": self._children,
        }

    @classmethod
    def from_json(cls, data: dict) -> "TrainingSearchGraph":
        graph = cls()
        graph.root_id = data.get("root_id")
        for raw in data.get("nodes", []):
            node = _node_from_dict(raw)
            graph.nodes[node.node_id] = node
        graph._children = {k: list(v) for k, v in data.get("children", {}).items()}
        return graph

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainingSearchGraph":
        with open(path) as f:
            data = json.load(f)
        return cls.from_json(data)


# ── Private (de)serialization helpers ────────────────────────────────────

def _node_to_dict(node: TrainingSearchNode) -> dict:
    raw = asdict(node)
    return raw


def _node_from_dict(raw: dict) -> TrainingSearchNode:
    patch_raw = raw.get("workspace_patch") or {}
    patch_ops = [
        PatchOperation(**op) if not isinstance(op, PatchOperation) else op
        for op in patch_raw.get("operations", [])
    ]

    ckpt_raw = raw.get("checkpoint")
    checkpoint = CheckpointRef(**ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw

    validity_raw = raw.get("validity")
    validity = (
        ValidityReport(**validity_raw) if isinstance(validity_raw, dict) else validity_raw
    )

    return TrainingSearchNode(
        node_id=raw["node_id"],
        parent_id=raw.get("parent_id"),
        branch_id=int(raw.get("branch_id", 0)),
        mutation_plan=raw.get("mutation_plan", ""),
        workspace_patch=WorkspacePatch(operations=patch_ops),
        pipeline_summary=raw.get("pipeline_summary", ""),
        trial_status=raw.get("trial_status"),
        metric=raw.get("metric"),
        reward=raw.get("reward"),
        validity=validity,
        error_buckets=dict(raw.get("error_buckets") or {}),
        cost=dict(raw.get("cost") or {}),
        checkpoint=checkpoint,
        artifacts=dict(raw.get("artifacts") or {}),
        visits=int(raw.get("visits", 0)),
        total_reward=float(raw.get("total_reward", 0.0)),
        mean_reward=float(raw.get("mean_reward", 0.0)),
        is_valid=raw.get("is_valid"),
        is_terminal=bool(raw.get("is_terminal", False)),
        local_best_node_id=raw.get("local_best_node_id"),
        continue_improve=bool(raw.get("continue_improve", False)),
    )
