"""Data mixture optimization.

Implements automated data mixture ratio optimization based on:
- DoReMi (Xie et al., 2023): Domain reweighting with minimax optimization
- RegMix (2024): Data mixing as regression
- Data Mixing Laws (Ye et al., 2024): Predictive models for mixture performance

The optimizer finds the optimal weights for combining different data domains
(e.g., web, books, code, academic) to minimize downstream loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from autopilot.utils.logging import get_logger

log = get_logger("optimization.data_mixing")


@dataclass
class DataDomain:
    """A single data domain/source."""

    name: str
    path: str
    token_count: int
    quality_score: Optional[float] = None  # 0-1, estimated quality
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MixtureWeights:
    """Weighted data mixture configuration."""

    weights: Dict[str, float]  # domain_name -> weight (sums to 1.0)
    source: str = "manual"  # "manual", "doremi", "regmix", "online"
    confidence: float = 1.0

    def __post_init__(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_weight(self, domain: str) -> float:
        return self.weights.get(domain, 0.0)

    def adjust(self, domain: str, delta: float) -> "MixtureWeights":
        """Adjust weight for a domain and renormalize."""
        new_weights = dict(self.weights)
        new_weights[domain] = max(0.0, new_weights.get(domain, 0.0) + delta)
        return MixtureWeights(weights=new_weights, source="online")

    def blend(self, other: "MixtureWeights", alpha: float = 0.5) -> "MixtureWeights":
        """Blend with another mixture (interpolation)."""
        all_domains = set(self.weights.keys()) | set(other.weights.keys())
        blended = {}
        for d in all_domains:
            w1 = self.weights.get(d, 0.0)
            w2 = other.weights.get(d, 0.0)
            blended[d] = alpha * w1 + (1 - alpha) * w2
        return MixtureWeights(weights=blended, source="blended")


@dataclass
class MixingExperiment:
    """Record of a single data mixing experiment."""

    mixture: MixtureWeights
    model_params: float
    tokens_trained: int
    final_loss: float
    domain_losses: Dict[str, float] = field(default_factory=dict)


class DataMixingOptimizer:
    """Optimizes data mixture ratios for LLM pre-training.

    Strategy:
    1. Run proxy experiments with different mixtures
    2. Fit a predictive model for loss as a function of mixture
    3. Use the model to find the optimal mixture
    4. Optionally adjust online during training based on per-domain losses
    """

    def __init__(self, domains: List[DataDomain]):
        self._domains = {d.name: d for d in domains}
        self._experiments: List[MixingExperiment] = []
        self._current_mixture: Optional[MixtureWeights] = None

    @property
    def domain_names(self) -> List[str]:
        return list(self._domains.keys())

    @property
    def current_mixture(self) -> Optional[MixtureWeights]:
        return self._current_mixture

    def uniform_mixture(self) -> MixtureWeights:
        """Start with uniform weights across all domains."""
        n = len(self._domains)
        weights = {name: 1.0 / n for name in self._domains}
        return MixtureWeights(weights=weights, source="uniform")

    def token_proportional_mixture(self) -> MixtureWeights:
        """Weight proportional to available tokens per domain."""
        total_tokens = sum(d.token_count for d in self._domains.values())
        weights = {name: d.token_count / total_tokens for name, d in self._domains.items()}
        return MixtureWeights(weights=weights, source="token_proportional")

    def quality_weighted_mixture(self) -> MixtureWeights:
        """Weight by estimated quality scores."""
        scored = {
            name: d.quality_score for name, d in self._domains.items() if d.quality_score is not None
        }
        if not scored:
            return self.uniform_mixture()
        total = sum(scored.values())
        weights = {name: score / total for name, score in scored.items()}
        # Include unscored domains with minimum weight
        for name in self._domains:
            if name not in weights:
                weights[name] = 0.01
        return MixtureWeights(weights=weights, source="quality_weighted")

    def suggest_exploration_mixtures(self, n_suggestions: int = 8) -> List[MixtureWeights]:
        """Suggest diverse mixture configurations for exploration.

        Uses a combination of:
        - Corner cases (heavily weighted towards each domain)
        - Random Dirichlet samples
        - Perturbations around the best known mixture
        """
        suggestions = []
        domain_names = self.domain_names
        n_domains = len(domain_names)

        # Include uniform and token-proportional as baselines
        suggestions.append(self.uniform_mixture())
        suggestions.append(self.token_proportional_mixture())

        # Corner cases: emphasize each domain
        for i, name in enumerate(domain_names):
            if len(suggestions) >= n_suggestions:
                break
            weights = {n: 0.05 for n in domain_names}
            weights[name] = 1.0 - 0.05 * (n_domains - 1)
            suggestions.append(MixtureWeights(weights=weights, source="exploration"))

        # Random Dirichlet samples
        rng = np.random.default_rng(42)
        while len(suggestions) < n_suggestions:
            alpha = np.ones(n_domains) * 2.0  # slightly concentrated
            sample = rng.dirichlet(alpha)
            weights = {name: float(w) for name, w in zip(domain_names, sample)}
            suggestions.append(MixtureWeights(weights=weights, source="exploration"))

        return suggestions[:n_suggestions]

    def record_experiment(self, experiment: MixingExperiment) -> None:
        """Record results of a mixing experiment."""
        self._experiments.append(experiment)
        log.info(
            f"Recorded mixing experiment: loss={experiment.final_loss:.4f}, "
            f"mixture={experiment.mixture.weights}"
        )

    def optimize_regmix(self) -> Optional[MixtureWeights]:
        """Optimize mixture using RegMix approach (regression-based prediction).

        Fits a linear model predicting loss from mixture weights,
        then optimizes the weights subject to simplex constraint.
        """
        if len(self._experiments) < 3:
            log.warning("Need at least 3 experiments for RegMix optimization")
            return None

        domain_names = self.domain_names
        n = len(domain_names)

        # Build feature matrix (mixture weights) and target vector (losses)
        X = np.array(
            [[exp.mixture.get_weight(d) for d in domain_names] for exp in self._experiments]
        )
        y = np.array([exp.final_loss for exp in self._experiments])

        # Fit linear regression with interaction terms
        # Simple model: loss = sum(w_i * c_i) + intercept
        # With ridge regularization
        from numpy.linalg import lstsq

        X_with_bias = np.column_stack([X, np.ones(len(X))])
        coeffs, _, _, _ = lstsq(X_with_bias, y, rcond=None)

        domain_coeffs = coeffs[:n]
        # Optimal mixture: lower coefficient = more weight (inversely proportional to loss contribution)
        # Use softmin of coefficients
        inv_coeffs = -domain_coeffs  # negate so lower loss contribution gets higher weight
        inv_coeffs = inv_coeffs - np.max(inv_coeffs)  # numerical stability
        exp_weights = np.exp(inv_coeffs / 0.1)  # temperature=0.1 for sharper distribution
        optimal_weights = exp_weights / exp_weights.sum()

        weights = {name: float(w) for name, w in zip(domain_names, optimal_weights)}
        result = MixtureWeights(weights=weights, source="regmix")
        self._current_mixture = result

        log.info(f"RegMix optimized mixture: {weights}")
        return result

    def optimize_doremi(
        self, domain_losses: Dict[str, float], reference_losses: Dict[str, float]
    ) -> MixtureWeights:
        """Optimize mixture using DoReMi-style group DRO.

        Upweights domains where the model has higher excess loss
        (loss relative to a reference model).

        Args:
            domain_losses: Per-domain loss of the current model
            reference_losses: Per-domain loss of a reference (smaller/baseline) model
        """
        domain_names = self.domain_names

        # Compute excess losses
        excess_losses = {}
        for name in domain_names:
            if name in domain_losses and name in reference_losses:
                excess_losses[name] = max(0, domain_losses[name] - reference_losses[name])
            else:
                excess_losses[name] = 0.0

        total_excess = sum(excess_losses.values())
        if total_excess < 1e-8:
            return self._current_mixture or self.uniform_mixture()

        # DoReMi: weight proportional to excess loss (exponentiated for sharpness)
        eta = 1.0  # step size for multiplicative weight update
        weights = {}
        for name in domain_names:
            w = np.exp(eta * excess_losses.get(name, 0.0) / total_excess)
            weights[name] = w

        result = MixtureWeights(weights=weights, source="doremi")
        self._current_mixture = result

        log.info(f"DoReMi optimized mixture: {result.weights}")
        return result

    def adjust_online(
        self,
        current_mixture: MixtureWeights,
        domain_losses: Dict[str, float],
        step_size: float = 0.05,
    ) -> MixtureWeights:
        """Adjust mixture online based on current per-domain losses.

        Uses exponentiated gradient descent on the simplex.
        Domains with higher loss get upweighted.
        """
        domain_names = self.domain_names
        losses = np.array([domain_losses.get(d, 0.0) for d in domain_names])

        if np.all(losses == 0):
            return current_mixture

        # Normalize losses
        mean_loss = np.mean(losses)
        normalized = (losses - mean_loss) / (np.std(losses) + 1e-8)

        # Exponentiated gradient update
        current_weights = np.array([current_mixture.get_weight(d) for d in domain_names])
        new_weights = current_weights * np.exp(step_size * normalized)
        new_weights = new_weights / new_weights.sum()

        # Smoothing: blend with current to avoid drastic changes
        smoothed = 0.7 * new_weights + 0.3 * current_weights
        smoothed = smoothed / smoothed.sum()

        weights = {name: float(w) for name, w in zip(domain_names, smoothed)}
        result = MixtureWeights(weights=weights, source="online")
        self._current_mixture = result

        return result
