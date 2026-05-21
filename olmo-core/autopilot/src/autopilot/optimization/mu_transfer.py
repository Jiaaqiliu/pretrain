"""muTransfer (Maximal Update Parametrization) for hyperparameter transfer.

Implements the key insight from Yang et al. (2022): under muP, optimal
hyperparameters remain stable across model scales. This allows tuning on
a small proxy model (~1% compute) and transferring to the target scale.

Reference: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
Hyperparameter Transfer" (Yang et al., 2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from autopilot.utils.logging import get_logger

log = get_logger("optimization.mu_transfer")


@dataclass
class ModelScale:
    """Defines a model's scale parameters."""

    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: Optional[int] = None
    vocab_size: int = 50304

    @property
    def num_params_approx(self) -> float:
        """Approximate parameter count (dense transformer)."""
        d = self.hidden_size
        L = self.num_layers
        ffn = self.intermediate_size or 4 * d
        # Attention: 4*d*d per layer, FFN: 2*d*ffn per layer, embeddings: vocab*d
        params_per_layer = 4 * d * d + 2 * d * ffn
        return L * params_per_layer + self.vocab_size * d

    @property
    def width_multiplier(self) -> float:
        """Width multiplier relative to a base width of 256."""
        return self.hidden_size / 256.0


@dataclass
class MuTransferConfig:
    """Configuration for muTransfer hyperparameter scaling."""

    proxy_scale: ModelScale
    target_scale: ModelScale
    base_width: int = 256  # reference width for muP calculations

    # Which hyperparameters to transfer (vs. scale)
    transfer_params: List[str] = field(
        default_factory=lambda: ["learning_rate", "weight_decay", "beta1", "beta2"]
    )
    # Which hyperparameters scale with width
    width_scaled_params: List[str] = field(
        default_factory=lambda: ["learning_rate"]
    )

    @property
    def width_ratio(self) -> float:
        """Ratio of target width to proxy width."""
        return self.target_scale.hidden_size / self.proxy_scale.hidden_size


class MuTransferEngine:
    """Transfers hyperparameters from proxy to target model using muP scaling rules.

    Under muP:
    - Learning rate for hidden layers scales as 1/width (relative to base)
    - Output layer LR scales as 1/width^2
    - Embedding LR does not scale
    - Initialization scales as 1/sqrt(width) for hidden, 1/width for output
    - Weight decay does not scale with width (direct transfer)
    - Adam betas do not scale (direct transfer)
    """

    def __init__(self, config: MuTransferConfig):
        self._config = config
        self._proxy_results: List[Tuple[Dict[str, Any], float]] = []

    @property
    def proxy_scale(self) -> ModelScale:
        return self._config.proxy_scale

    @property
    def target_scale(self) -> ModelScale:
        return self._config.target_scale

    def transfer_hyperparameters(self, proxy_params: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer optimal hyperparameters from proxy scale to target scale.

        Applies muP scaling rules:
        - LR: proxy_lr * (proxy_width / target_width)
        - Weight decay: direct transfer
        - Adam betas: direct transfer
        - Warmup steps: scale proportionally with total training steps
        """
        target_params = dict(proxy_params)
        width_ratio = self._config.width_ratio

        if "learning_rate" in target_params:
            # Under muP, LR scales as 1/width for hidden layers
            target_params["learning_rate"] = proxy_params["learning_rate"] / width_ratio

        if "init_std" in target_params:
            # Initialization scales as 1/sqrt(width)
            target_params["init_std"] = proxy_params["init_std"] / np.sqrt(width_ratio)

        if "output_lr_multiplier" in target_params:
            # Output layer has additional 1/width factor
            target_params["output_lr_multiplier"] = (
                proxy_params["output_lr_multiplier"] / width_ratio
            )

        log.info(
            f"Transferred HPs from {self._config.proxy_scale.hidden_size}d "
            f"to {self._config.target_scale.hidden_size}d "
            f"(width_ratio={width_ratio:.1f}): {target_params}"
        )
        return target_params

    def add_proxy_result(self, params: Dict[str, Any], loss: float) -> None:
        """Record a proxy training result for analysis."""
        self._proxy_results.append((params, loss))

    def get_best_proxy_params(self) -> Optional[Dict[str, Any]]:
        """Get the best hyperparameters from proxy experiments."""
        if not self._proxy_results:
            return None
        best = min(self._proxy_results, key=lambda x: x[1])
        return best[0]

    def get_transferred_best(self) -> Optional[Dict[str, Any]]:
        """Get best proxy params, transferred to target scale."""
        best = self.get_best_proxy_params()
        if best is None:
            return None
        return self.transfer_hyperparameters(best)

    def suggest_proxy_config(self) -> Dict[str, Any]:
        """Suggest a proxy model configuration for HP search.

        The proxy should be:
        - Same depth as target (num_layers)
        - Reduced width (typically 1/4 to 1/8 of target)
        - Same architecture choices (attention type, activation, etc.)
        """
        proxy = self._config.proxy_scale
        return {
            "hidden_size": proxy.hidden_size,
            "num_layers": proxy.num_layers,
            "num_heads": proxy.num_heads,
            "intermediate_size": proxy.intermediate_size or 4 * proxy.hidden_size,
            "vocab_size": proxy.vocab_size,
            "estimated_params": proxy.num_params_approx,
            "recommended_tokens": int(proxy.num_params_approx * 20),  # Chinchilla-optimal
        }

    @staticmethod
    def design_proxy(target: ModelScale, width_divisor: int = 4) -> ModelScale:
        """Design a proxy model scale for a given target.

        Rules:
        - Same number of layers (depth matters for transfer)
        - Width reduced by width_divisor
        - Heads reduced proportionally
        """
        proxy_hidden = target.hidden_size // width_divisor
        proxy_heads = max(1, target.num_heads // width_divisor)
        # Ensure hidden_size is divisible by num_heads
        proxy_hidden = (proxy_hidden // proxy_heads) * proxy_heads

        return ModelScale(
            hidden_size=proxy_hidden,
            num_layers=target.num_layers,
            num_heads=proxy_heads,
            intermediate_size=(target.intermediate_size or 4 * target.hidden_size) // width_divisor,
            vocab_size=target.vocab_size,
        )
