"""Ensemble strategies for combining multiple ML models.

This module provides ensemble methods to combine predictions from
multiple trained models to achieve better performance.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor


class EnsemblePredictor:
    """Combines predictions from multiple models using various strategies."""

    def __init__(self, strategy: str = "voting", weights: List[float] = None):
        """
        Args:
            strategy: Ensemble strategy ('voting', 'averaging', 'stacking')
            weights: Optional weights for each model (for weighted voting/averaging)
        """
        self.strategy = strategy
        self.weights = weights
        self.models = []
        self.model_names = []

    def add_model(self, model, name: str = None):
        """Add a trained model to the ensemble."""
        self.models.append(model)
        self.model_names.append(name or f"model_{len(self.models)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.models:
            raise ValueError("No models added to ensemble")

        if self.strategy == "voting" or self.strategy == "averaging":
            return self._voting_predict(X)
        elif self.strategy == "stacking":
            return self._stacking_predict(X)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions."""
        if not self.models:
            raise ValueError("No models added to ensemble")

        # Collect predictions from all models
        all_proba = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            else:
                # For models without predict_proba, convert predictions to probabilities
                pred = model.predict(X)
                proba = np.column_stack([1 - pred, pred])
            all_proba.append(proba)

        # Average probabilities (weighted if weights provided)
        if self.weights is None:
            weights = np.ones(len(self.models)) / len(self.models)
        else:
            weights = np.array(self.weights) / np.sum(self.weights)

        ensemble_proba = np.average(all_proba, axis=0, weights=weights)
        return ensemble_proba

    def _voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using voting/averaging strategy."""
        all_preds = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                # Use probability predictions for voting
                proba = model.predict_proba(X)
                pred = np.argmax(proba, axis=1)
            else:
                pred = model.predict(X)
            all_preds.append(pred)

        all_preds = np.array(all_preds)

        if self.weights is None:
            # Simple majority voting
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=all_preds
            )
        else:
            # Weighted voting
            weights = np.array(self.weights)
            weighted_votes = all_preds * weights[:, np.newaxis]
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=weights).argmax(),
                axis=0,
                arr=all_preds
            )

        return ensemble_pred

    def _stacking_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using stacking strategy (not yet implemented)."""
        # TODO: Implement stacking with a meta-learner
        # For now, fallback to voting
        return self._voting_predict(X)


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[Path],
    strategy: str = "voting",
    weights: List[float] = None
) -> EnsemblePredictor:
    """Create an ensemble from multiple model checkpoints.

    Args:
        checkpoint_paths: List of paths to model.pkl files
        strategy: Ensemble strategy
        weights: Optional weights for each model

    Returns:
        EnsemblePredictor instance
    """
    ensemble = EnsemblePredictor(strategy=strategy, weights=weights)

    for i, ckpt_path in enumerate(checkpoint_paths):
        model_path = Path(ckpt_path) / "model.pkl"
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        ensemble.add_model(model, name=f"model_{i}")
        print(f"✓ Loaded model from {ckpt_path}")

    return ensemble


def create_topk_ensemble(
    graph_path: Path,
    k: int = 3,
    strategy: str = "voting"
) -> EnsemblePredictor:
    """Create an ensemble from top-k models in the MCGS graph.

    Args:
        graph_path: Path to mcgs_graph.json
        k: Number of top models to ensemble
        strategy: Ensemble strategy

    Returns:
        EnsemblePredictor instance
    """
    import json

    # Load graph
    with open(graph_path) as f:
        graph = json.load(f)

    # Sort nodes by metric
    nodes = graph.get("nodes", [])
    valid_nodes = [n for n in nodes if n.get("metric") is not None and n.get("checkpoint")]
    valid_nodes.sort(key=lambda n: n["metric"], reverse=True)

    # Get top k
    top_nodes = valid_nodes[:k]

    if not top_nodes:
        raise ValueError("No valid nodes with checkpoints found")

    print(f"\nCreating ensemble from top {len(top_nodes)} models:")
    for i, node in enumerate(top_nodes, 1):
        print(f"  {i}. {node['node_id']}: metric={node['metric']:.5f}")

    # Extract checkpoint paths
    checkpoint_paths = [Path(n["checkpoint"]["path"]) for n in top_nodes]

    # Create weights based on metrics (optional)
    metrics = [n["metric"] for n in top_nodes]
    weights = np.array(metrics) / np.sum(metrics)  # Normalize to sum to 1

    return create_ensemble_from_checkpoints(
        checkpoint_paths,
        strategy=strategy,
        weights=weights.tolist()
    )


__all__ = [
    "EnsemblePredictor",
    "create_ensemble_from_checkpoints",
    "create_topk_ensemble",
]
