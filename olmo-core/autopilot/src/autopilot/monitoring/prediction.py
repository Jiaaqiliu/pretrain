"""Training outcome prediction.

Predicts final loss and training quality from early signals, using:
- Power-law extrapolation of loss curves
- Scaling law-based predictions
- Comparison against known training trajectories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from autopilot.monitoring.metrics import MetricsWindow
from autopilot.utils.logging import get_logger

log = get_logger("monitoring.prediction")


@dataclass
class TrainingOutlook:
    """Predicted training outcome."""

    predicted_final_loss: float
    confidence: float  # 0-1, how confident in the prediction
    estimated_steps_remaining: int
    estimated_time_remaining_hours: float
    on_track: bool  # is training converging as expected
    notes: str = ""


class LossPredictor:
    """Predicts final training loss from observed trajectory.

    Uses a combination of:
    1. Power-law fitting: L(t) = a * t^(-b) + c
    2. Exponential decay fitting: L(t) = (L0 - L_inf) * exp(-t/tau) + L_inf
    3. Ensemble of both for robust prediction
    """

    def __init__(self, total_steps: int, target_loss: Optional[float] = None):
        self._total_steps = total_steps
        self._target_loss = target_loss

    def predict(self, window: MetricsWindow) -> Optional[TrainingOutlook]:
        loss_values = window.get_values("loss")
        if len(loss_values) < 100:
            return None

        latest = window.latest
        if latest is None:
            return None

        current_step = latest.step
        current_loss = latest.metrics.get("loss")
        if current_loss is None:
            return None

        # Fit power law
        power_law_pred = self._fit_power_law(loss_values, current_step)
        # Fit exponential decay
        exp_decay_pred = self._fit_exponential_decay(loss_values, current_step)

        # Ensemble prediction (weighted by fit quality)
        predictions = [p for p in [power_law_pred, exp_decay_pred] if p is not None]
        if not predictions:
            return None

        predicted_final = np.mean([p[0] for p in predictions])
        confidence = np.mean([p[1] for p in predictions])

        # Estimate time remaining
        steps_remaining = max(0, self._total_steps - current_step)
        throughput = window.mean("throughput")
        if throughput and throughput > 0:
            time_remaining_hours = (steps_remaining / throughput) / 3600
        else:
            time_remaining_hours = 0.0

        # Determine if on track
        on_track = True
        notes = ""
        if self._target_loss and predicted_final > self._target_loss * 1.1:
            on_track = False
            notes = (
                f"Predicted final loss ({predicted_final:.4f}) "
                f"exceeds target ({self._target_loss:.4f}) by "
                f"{((predicted_final - self._target_loss) / self._target_loss * 100):.1f}%"
            )
        elif window.trend("loss") is not None and window.trend("loss") > 0:
            on_track = False
            notes = "Loss is trending upward"

        return TrainingOutlook(
            predicted_final_loss=float(predicted_final),
            confidence=float(confidence),
            estimated_steps_remaining=steps_remaining,
            estimated_time_remaining_hours=time_remaining_hours,
            on_track=on_track,
            notes=notes,
        )

    def _fit_power_law(
        self, loss_values: np.ndarray, current_step: int
    ) -> Optional[Tuple[float, float]]:
        """Fit L(t) = a * t^(-b) + c and extrapolate to total_steps."""
        n = len(loss_values)
        if n < 50:
            return None

        # Use log-spaced subsample for fitting
        indices = np.unique(np.geomspace(1, n - 1, min(100, n)).astype(int))
        steps = indices + (current_step - n + 1)
        values = loss_values[indices]

        # Filter out any NaN/Inf
        mask = np.isfinite(values) & (steps > 0)
        steps = steps[mask]
        values = values[mask]

        if len(steps) < 20:
            return None

        try:
            from scipy.optimize import curve_fit

            def power_law(t, a, b, c):
                return a * np.power(t.astype(float), -b) + c

            # Initial guess
            p0 = [values[0] - values[-1], 0.5, values[-1]]
            bounds = ([0, 0.01, 0], [np.inf, 2.0, values[-1] * 2])

            popt, pcov = curve_fit(power_law, steps, values, p0=p0, bounds=bounds, maxfev=5000)
            a, b, c = popt

            # Predict at total_steps
            predicted = power_law(np.array([self._total_steps]), a, b, c)[0]

            # Estimate confidence from fit quality
            residuals = values - power_law(steps, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            confidence = max(0.0, min(1.0, r_squared))
            return float(predicted), confidence

        except (RuntimeError, ValueError, ImportError):
            return None

    def _fit_exponential_decay(
        self, loss_values: np.ndarray, current_step: int
    ) -> Optional[Tuple[float, float]]:
        """Fit L(t) = (L0 - L_inf) * exp(-t/tau) + L_inf."""
        n = len(loss_values)
        if n < 50:
            return None

        indices = np.unique(np.linspace(0, n - 1, min(100, n)).astype(int))
        steps = indices + (current_step - n + 1)
        values = loss_values[indices]

        mask = np.isfinite(values)
        steps = steps[mask]
        values = values[mask]

        if len(steps) < 20:
            return None

        try:
            from scipy.optimize import curve_fit

            def exp_decay(t, l0, l_inf, tau):
                return (l0 - l_inf) * np.exp(-t.astype(float) / tau) + l_inf

            l0_guess = values[0]
            l_inf_guess = values[-1] * 0.9
            tau_guess = current_step / 2.0

            p0 = [l0_guess, l_inf_guess, tau_guess]
            bounds = (
                [0, 0, 1],
                [values[0] * 2, values[-1] * 1.5, self._total_steps * 10],
            )

            popt, pcov = curve_fit(exp_decay, steps, values, p0=p0, bounds=bounds, maxfev=5000)
            l0, l_inf, tau = popt

            predicted = exp_decay(np.array([self._total_steps]), l0, l_inf, tau)[0]

            residuals = values - exp_decay(steps, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            confidence = max(0.0, min(1.0, r_squared * 0.9))  # slightly lower confidence
            return float(predicted), confidence

        except (RuntimeError, ValueError, ImportError):
            return None
