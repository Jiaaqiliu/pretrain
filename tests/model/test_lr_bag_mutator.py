"""Unit tests for LRBagMutationProposer."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_evolve.model.algorithms.mcgs.mutation import LRBagMutationProposer


@dataclass
class _Parent:
    node_id: str = "node-root"


def test_bag_cycles_over_all_values() -> None:
    bag = (1e-4, 5e-5, 3e-5, 1e-5)
    mut = LRBagMutationProposer(bag=bag)
    values = [mut.propose(_Parent()).patch.operations[0].value for _ in range(len(bag))]
    assert values == list(bag)


def test_bag_wraps_around() -> None:
    bag = (1e-4, 5e-5)
    mut = LRBagMutationProposer(bag=bag)
    seen = [mut.propose(_Parent()).patch.operations[0].value for _ in range(4)]
    assert seen == [1e-4, 5e-5, 1e-4, 5e-5]


def test_patch_targets_optimizer_lr() -> None:
    mut = LRBagMutationProposer(bag=(1e-4,))
    wm = mut.propose(_Parent())
    assert wm.mutation_type == "training_recipe"
    assert len(wm.patch.operations) == 1
    op = wm.patch.operations[0]
    assert op.op == "replace"
    assert op.path == "train/optimizer.yaml"
    assert op.key_path == ["lr"]
    assert op.value == 1e-4


def test_empty_bag_raises() -> None:
    with pytest.raises(ValueError):
        LRBagMutationProposer(bag=())


def test_parent_id_threaded_into_mutation() -> None:
    parent = _Parent(node_id="node-xyz")
    wm = LRBagMutationProposer(bag=(1e-4,)).propose(parent)
    assert wm.parent_node_id == "node-xyz"
    assert wm.mutation_id.startswith("m-lr-")
