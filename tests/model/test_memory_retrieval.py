"""PR6 acceptance: failed and successful mutations are retrievable."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.model.algorithms.mcgs.memory import NodeMemoryStore
from agent_evolve.model.types import TrainingSearchNode


def _valid(id_: str, summary: str) -> TrainingSearchNode:
    return TrainingSearchNode(
        node_id=id_,
        parent_id="root",
        branch_id=0,
        mutation_plan=summary,
        is_valid=True,
        metric=0.5,
        reward=0.1,
    )


def _invalid(id_: str, summary: str) -> TrainingSearchNode:
    return TrainingSearchNode(
        node_id=id_,
        parent_id="root",
        branch_id=0,
        mutation_plan=summary,
        is_valid=False,
        reward=-1.0,
    )


def test_failed_mutation_retrievable(tmp_path: Path) -> None:
    store = NodeMemoryStore(tmp_path)
    store.record(_invalid("bad1", "increase learning rate excessively"))
    store.record(_valid("good1", "reduce batch size"))

    failures = store.retrieve("learning rate", k=3, source=store.failed)
    assert failures and failures[0]["node_id"] == "bad1"


def test_successful_mutation_retrievable(tmp_path: Path) -> None:
    store = NodeMemoryStore(tmp_path)
    store.record(_valid("good1", "increase failure replay in data mix"))
    store.record(_valid("good2", "shorter rollouts"))

    hits = store.retrieve("failure replay", k=3, source=store.successful)
    assert hits and hits[0]["node_id"] == "good1"


def test_empty_store_returns_empty(tmp_path: Path) -> None:
    store = NodeMemoryStore(tmp_path)
    assert store.retrieve("anything") == []
