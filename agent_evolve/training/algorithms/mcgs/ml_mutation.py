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
            return WorkspaceMutation(
                path="model/feature_engineering.yaml",
                patch={"scale": True},  # Enable scaling
                description="Enable feature scaling",
            )

        return WorkspaceMutation(
            path="model/config.yaml",
            patch=new_config,
            description=description,
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


__all__ = [
    "MLHyperparameterMutationProposer",
    "MLModelTypeMutationProposer",
    "MLLearningRateSweepProposer",
]
