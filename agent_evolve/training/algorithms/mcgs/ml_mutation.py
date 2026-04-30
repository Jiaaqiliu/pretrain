"""Mutation strategies for ML hyperparameters (AutoML)

These mutators change traditional ML model configurations
instead of LLM training hyperparameters.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

import yaml

from ...types import WorkspaceMutation, WorkspacePatch, PatchOperation


class MLHyperparameterMutationProposer:
    """Mutates ML model hyperparameters.

    Focuses on key hyperparameters for tree-based models:
    - n_estimators
    - max_depth
    - learning_rate
    - subsample
    - etc.
    """

    def __init__(
        self,
        mutation_rate: float = 0.3,
        random_state: int | None = None,
    ):
        self.mutation_rate = mutation_rate
        self.rng = random.Random(random_state)

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        """Propose a hyperparameter mutation.

        Args:
            parent: Parent node in MCGS graph
            graph: Full MCGS graph (for context)

        Returns:
            WorkspaceMutation with hyperparameter changes
        """
        # Read parent config
        parent_config = self._read_config(parent)

        # Choose what to mutate
        mutation_type = self.rng.choice([
            "learning_rate",
            "n_estimators",
            "max_depth",
            "subsample",
            "model_type",
            "feature_engineering",
        ])

        new_config = parent_config.copy()

        if mutation_type == "learning_rate":
            current_lr = new_config["hyperparameters"].get("learning_rate", 0.1)
            # Multiply by random factor in [0.5, 2.0]
            factor = self.rng.uniform(0.5, 2.0)
            new_lr = max(0.001, min(0.5, current_lr * factor))
            new_config["hyperparameters"]["learning_rate"] = round(new_lr, 4)
            description = f"Mutate learning_rate: {current_lr:.4f} -> {new_lr:.4f}"

        elif mutation_type == "n_estimators":
            current_n = new_config["hyperparameters"].get("n_estimators", 100)
            # Add/subtract random amount
            delta = self.rng.choice([-50, -20, 20, 50, 100])
            new_n = max(10, min(500, current_n + delta))
            new_config["hyperparameters"]["n_estimators"] = new_n
            description = f"Mutate n_estimators: {current_n} -> {new_n}"

        elif mutation_type == "max_depth":
            current_depth = new_config["hyperparameters"].get("max_depth", 6)
            # Change by ±1 or ±2
            delta = self.rng.choice([-2, -1, 1, 2])
            new_depth = max(3, min(15, current_depth + delta))
            new_config["hyperparameters"]["max_depth"] = new_depth
            description = f"Mutate max_depth: {current_depth} -> {new_depth}"

        elif mutation_type == "subsample":
            current_subsample = new_config["hyperparameters"].get("subsample", 0.8)
            # Random value in [0.6, 1.0]
            new_subsample = round(self.rng.uniform(0.6, 1.0), 2)
            new_config["hyperparameters"]["subsample"] = new_subsample
            description = f"Mutate subsample: {current_subsample} -> {new_subsample}"

        elif mutation_type == "model_type":
            current_type = new_config.get("model_type", "xgboost")
            # Switch to different model
            options = ["xgboost", "lightgbm", "random_forest"]
            options = [o for o in options if o != current_type]
            new_type = self.rng.choice(options)
            new_config["model_type"] = new_type
            description = f"Mutate model_type: {current_type} -> {new_type}"

        elif mutation_type == "feature_engineering":
            # This would mutate feature_engineering.yaml
            # For now, just toggle scaling
            mutation_id = f"m-fe-{uuid.uuid4().hex[:8]}"
            return WorkspaceMutation(
                mutation_id=mutation_id,
                parent_node_id=parent.node_id,
                description="Enable feature scaling",
                patch=WorkspacePatch(
                    operations=[
                        PatchOperation(
                            op="replace",
                            path="model/config.yaml",
                            key_path=["feature_engineering", "scale"],
                            value=True,
                        ),
                    ]
                ),
                mutation_type="training_recipe",
            )

        mutation_id = f"m-hyper-{uuid.uuid4().hex[:8]}"

        # Create PatchOperation for the mutation
        if mutation_type == "learning_rate":
            operations = [
                PatchOperation(
                    op="replace",
                    path="model/config.yaml",
                    key_path=["hyperparameters", "learning_rate"],
                    value=new_config["hyperparameters"]["learning_rate"],
                ),
            ]
        elif mutation_type == "n_estimators":
            operations = [
                PatchOperation(
                    op="replace",
                    path="model/config.yaml",
                    key_path=["hyperparameters", "n_estimators"],
                    value=new_config["hyperparameters"]["n_estimators"],
                ),
            ]
        elif mutation_type == "max_depth":
            operations = [
                PatchOperation(
                    op="replace",
                    path="model/config.yaml",
                    key_path=["hyperparameters", "max_depth"],
                    value=new_config["hyperparameters"]["max_depth"],
                ),
            ]
        elif mutation_type == "subsample":
            operations = [
                PatchOperation(
                    op="replace",
                    path="model/config.yaml",
                    key_path=["hyperparameters", "subsample"],
                    value=new_config["hyperparameters"]["subsample"],
                ),
            ]
        elif mutation_type == "model_type":
            operations = [
                PatchOperation(
                    op="replace",
                    path="model/config.yaml",
                    key_path=["model_type"],
                    value=new_config["model_type"],
                ),
            ]

        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=description,
            patch=WorkspacePatch(operations=operations),
            mutation_type="training_recipe",
        )

    def _read_config(self, parent: Any) -> dict:
        """Read config from parent node."""
        # In real implementation, read from parent's workspace
        # For now, return default config
        return {
            "model_type": "xgboost",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
            },
        }


class MLModelTypeMutationProposer:
    """Rotates through different model types.

    Similar to LRBagMutationProposer but for model types.
    """

    def __init__(
        self,
        model_types: tuple[str, ...] = ("xgboost", "lightgbm", "random_forest"),
    ):
        self.model_types = model_types
        self.index = 0

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        """Propose next model type in rotation."""
        model_type = self.model_types[self.index % len(self.model_types)]
        self.index += 1

        mutation_id = f"m-model-{uuid.uuid4().hex[:8]}"

        # Different default hyperparameters for each model
        if model_type == "xgboost":
            hyperparams = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        elif model_type == "lightgbm":
            hyperparams = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        elif model_type == "random_forest":
            hyperparams = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            }

        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=f"Switch to {model_type}",
            patch=WorkspacePatch(
                operations=[
                    PatchOperation(
                        op="replace",
                        path="model/config.yaml",
                        key_path=["model_type"],
                        value=model_type,
                    ),
                    PatchOperation(
                        op="replace",
                        path="model/config.yaml",
                        key_path=["hyperparameters"],
                        value=hyperparams,
                    ),
                ]
            ),
            mutation_type="training_recipe",
        )


class MLLearningRateSweepProposer:
    """Sweeps through learning rates (for tree models).

    Similar to LRBagMutationProposer.
    """

    def __init__(self, learning_rates: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)):
        self.learning_rates = learning_rates
        self.index = 0

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        """Propose next learning rate."""
        lr = self.learning_rates[self.index % len(self.learning_rates)]
        self.index += 1

        mutation_id = f"m-lr-{uuid.uuid4().hex[:8]}"

        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=f"Set learning_rate={lr}",
            patch=WorkspacePatch(
                operations=[
                    PatchOperation(
                        op="replace",
                        path="model/config.yaml",
                        key_path=["hyperparameters", "learning_rate"],
                        value=lr,
                    ),
                ]
            ),
            mutation_type="training_recipe",
        )


class MLDepthSweepProposer:
    """Sweeps through max_depth values for tree models."""

    def __init__(self, depths: tuple[int, ...] = (5, 8, 10, 12, 15, 20)):
        self.depths = depths
        self.index = 0

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        """Propose next max_depth."""
        depth = self.depths[self.index % len(self.depths)]
        self.index += 1

        mutation_id = f"m-depth-{uuid.uuid4().hex[:8]}"

        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=f"Set max_depth={depth}",
            patch=WorkspacePatch(
                operations=[
                    PatchOperation(
                        op="replace",
                        path="model/config.yaml",
                        key_path=["hyperparameters", "max_depth"],
                        value=depth,
                    ),
                ]
            ),
            mutation_type="training_recipe",
        )


class MLNEstimatorsSweepProposer:
    """Sweeps through n_estimators values."""

    def __init__(self, n_estimators: tuple[int, ...] = (50, 100, 150, 200, 300)):
        self.n_estimators = n_estimators
        self.index = 0

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        """Propose next n_estimators."""
        n_est = self.n_estimators[self.index % len(self.n_estimators)]
        self.index += 1

        mutation_id = f"m-nest-{uuid.uuid4().hex[:8]}"

        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=f"Set n_estimators={n_est}",
            patch=WorkspacePatch(
                operations=[
                    PatchOperation(
                        op="replace",
                        path="model/config.yaml",
                        key_path=["hyperparameters", "n_estimators"],
                        value=n_est,
                    ),
                ]
            ),
            mutation_type="training_recipe",
        )


class CombinedMutationProposer:
    """Cycles through multiple mutation strategies."""

    def __init__(self, mutators: list):
        self.mutators = mutators
        self.index = 0

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        """Propose using the next mutator in sequence."""
        mutator = self.mutators[self.index % len(self.mutators)]
        self.index += 1
        return mutator.propose(parent, graph)


class EnsembleMutationProposer:
    """Assembles an ensemble from the top-K validated configs in the graph.

    Rule-based counterpart to LLM-proposed ensembles. Picks the top-K nodes
    by metric, reconstructs their configs, and creates a `train/ensemble.yaml`
    patch with those configs as members.

    Useful as a late-stage bootstrap: once MCGS has found several strong
    single-model configs, this proposer composes them into one ensemble.
    """

    def __init__(
        self,
        top_k: int = 3,
        strategy: str = "voting_soft",
        ensure_diversity: bool = True,
    ):
        self.top_k = top_k
        self.strategy = strategy
        self.ensure_diversity = ensure_diversity

    def propose(self, parent: Any, graph: Any) -> WorkspaceMutation:
        # Gather valid non-root nodes sorted by metric
        valid = [
            n for n in graph.nodes.values()
            if n.node_id != "node-root"
            and getattr(n, "is_valid", False)
            and getattr(n, "metric", None) is not None
        ]
        valid.sort(key=lambda n: n.metric or float("-inf"), reverse=True)

        if not valid:
            # No validated configs yet — fall back to a trivial single-member ensemble
            members = [{"model_type": "lightgbm", "random_state": 42, "hyperparameters": {}}]
        else:
            members = self._build_members_from_topk(valid, graph)

        mutation_id = f"m-ensemble-{uuid.uuid4().hex[:8]}"
        return WorkspaceMutation(
            mutation_id=mutation_id,
            parent_node_id=parent.node_id,
            description=f"Rule ensemble: {self.strategy} over top-{len(members)} configs",
            patch=WorkspacePatch(operations=[
                PatchOperation(
                    op="replace",
                    path="train/ensemble.yaml",
                    key_path=["enabled"],
                    value=True,
                ),
                PatchOperation(
                    op="replace",
                    path="train/ensemble.yaml",
                    key_path=["strategy"],
                    value=self.strategy,
                ),
                PatchOperation(
                    op="replace",
                    path="train/ensemble.yaml",
                    key_path=["members"],
                    value=members,
                ),
            ]),
            mutation_type="training_recipe",
        )

    def _build_members_from_topk(self, valid_nodes: list, graph: Any) -> list[dict]:
        """Reconstruct each top-K node's config and emit as a member spec.

        Each member keeps only keys valid across tree models (keeps hyperparameters
        minimal — learning_rate, n_estimators, max_depth, random_state). Model-specific
        params like gamma, reg_alpha (xgboost) or num_leaves (lightgbm) are dropped
        here; the backend's _train_model pipeline uses model defaults for anything not
        specified.
        """
        from .llm_mutation import _reconstruct_config

        # Safe set of hparams that most tree models accept (or ignore harmlessly)
        SAFE_KEYS = {"n_estimators", "max_depth", "random_state"}

        members = []
        used_types = set()
        for node in valid_nodes:
            cfg = _reconstruct_config(node, graph)
            model_type = cfg.get("model_type", "lightgbm")
            raw_hparams = cfg.get("hyperparameters", {})

            # Keep only safe keys + learning_rate if model supports it
            safe_hparams = {k: v for k, v in raw_hparams.items() if k in SAFE_KEYS}
            if model_type in ("xgboost", "lightgbm"):
                # These accept learning_rate; include if present
                if "learning_rate" in raw_hparams:
                    safe_hparams["learning_rate"] = raw_hparams["learning_rate"]

            seed = safe_hparams.get("random_state", 42)
            if self.ensure_diversity and model_type in used_types:
                seed = seed + 100 * len(members)
            safe_hparams["random_state"] = seed

            spec = {
                "model_type": model_type,
                "random_state": seed,
                "hyperparameters": safe_hparams,
            }
            used_types.add(model_type)
            members.append(spec)
            if len(members) >= self.top_k:
                break
        return members


__all__ = [
    "MLHyperparameterMutationProposer",
    "MLModelTypeMutationProposer",
    "MLLearningRateSweepProposer",
    "MLDepthSweepProposer",
    "MLNEstimatorsSweepProposer",
    "CombinedMutationProposer",
    "EnsembleMutationProposer",
]
