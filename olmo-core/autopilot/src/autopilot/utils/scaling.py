"""Scaling law estimation and performance prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from autopilot.utils.logging import get_logger

log = get_logger("scaling")


@dataclass
class ScalingDataPoint:
    """A single data point for scaling law fitting."""

    model_params: float  # number of parameters
    tokens: float  # number of training tokens
    loss: float  # final training/validation loss
    compute_flops: Optional[float] = None  # total FLOPs used


@dataclass
class ScalingLawFit:
    """Fitted scaling law parameters.

    Models the Chinchilla-style scaling law:
        L(N, D) = E + A / N^alpha + B / D^beta

    Where N = model params, D = dataset tokens, E = irreducible loss.
    """

    E: float  # irreducible loss
    A: float  # parameter scaling coefficient
    alpha: float  # parameter scaling exponent
    B: float  # data scaling coefficient
    beta: float  # data scaling exponent
    r_squared: float = 0.0  # goodness of fit

    def predict_loss(self, params: float, tokens: float) -> float:
        return self.E + self.A / (params**self.alpha) + self.B / (tokens**self.beta)

    def compute_optimal_allocation(self, compute_budget_flops: float) -> Tuple[float, float]:
        """Given a compute budget (in FLOPs), find optimal (N, D) allocation.

        Approximation: C ≈ 6 * N * D (for dense transformers).
        Optimal ratio follows: N* ∝ C^(beta/(alpha+beta)), D* ∝ C^(alpha/(alpha+beta))
        """
        ratio_exp = self.beta / (self.alpha + self.beta)
        scale = (compute_budget_flops / 6.0) ** 0.5

        optimal_params = scale ** (2 * ratio_exp)
        optimal_tokens = compute_budget_flops / (6.0 * optimal_params)
        return optimal_params, optimal_tokens


class ScalingLawEstimator:
    """Fits and applies scaling laws from experimental data.

    Implements Chinchilla-style scaling laws with extensions for:
    - Loss prediction at arbitrary scale
    - Compute-optimal model/data allocation
    - Early training signal extrapolation
    """

    def __init__(self):
        self._data_points: List[ScalingDataPoint] = []
        self._fit: Optional[ScalingLawFit] = None

    def add_data_point(self, point: ScalingDataPoint) -> None:
        self._data_points.append(point)
        self._fit = None  # invalidate cached fit

    def add_data_points(self, points: List[ScalingDataPoint]) -> None:
        self._data_points.extend(points)
        self._fit = None

    @property
    def fit(self) -> Optional[ScalingLawFit]:
        if self._fit is None and len(self._data_points) >= 5:
            self._fit = self._fit_scaling_law()
        return self._fit

    def predict_loss(self, params: float, tokens: float) -> Optional[float]:
        if self.fit is None:
            return None
        return self.fit.predict_loss(params, tokens)

    def compute_optimal_allocation(
        self, compute_budget_flops: float
    ) -> Optional[Tuple[float, float]]:
        if self.fit is None:
            return None
        return self.fit.compute_optimal_allocation(compute_budget_flops)

    def _fit_scaling_law(self) -> ScalingLawFit:
        """Fit the Chinchilla scaling law using least squares in log space."""
        from scipy.optimize import minimize

        points = self._data_points
        log_params = np.array([np.log(p.model_params) for p in points])
        log_tokens = np.array([np.log(p.tokens) for p in points])
        losses = np.array([p.loss for p in points])

        def objective(x):
            E, log_A, alpha, log_B, beta = x
            A, B = np.exp(log_A), np.exp(log_B)
            predicted = E + A * np.exp(-alpha * log_params) + B * np.exp(-beta * log_tokens)
            return np.sum((predicted - losses) ** 2)

        # Initial guesses based on Chinchilla findings
        x0 = [1.5, np.log(400), 0.34, np.log(400), 0.28]
        bounds = [
            (0.1, 5.0),  # E
            (np.log(1), np.log(1e6)),  # log_A
            (0.01, 2.0),  # alpha
            (np.log(1), np.log(1e6)),  # log_B
            (0.01, 2.0),  # beta
        ]

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        E, log_A, alpha, log_B, beta = result.x

        # Compute R-squared
        ss_res = result.fun
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return ScalingLawFit(
            E=E,
            A=np.exp(log_A),
            alpha=alpha,
            B=np.exp(log_B),
            beta=beta,
            r_squared=r_squared,
        )

    def predict_from_early_signal(
        self,
        current_step: int,
        current_loss: float,
        total_steps: int,
        model_params: float,
    ) -> float:
        """Predict final loss from early training signal using power-law extrapolation.

        Uses the empirical observation that training loss follows:
            L(t) = L_final + C * t^(-gamma)
        """
        if current_step < 100:
            return current_loss

        # Rough approximation: assume power-law decay
        # L(t) ≈ L_inf + (L_0 - L_inf) * (t/T)^(-gamma)
        # With gamma ≈ 0.5 for typical LLM training
        gamma = 0.5
        progress = current_step / total_steps
        if progress > 0.9:
            return current_loss

        # Extrapolate assuming constant decay rate in log-log space
        remaining_improvement_factor = (1.0 / progress) ** gamma - 1.0
        estimated_remaining = current_loss * 0.1 * remaining_improvement_factor
        predicted_final = current_loss - estimated_remaining

        return max(predicted_final, current_loss * 0.5)  # sanity bound
