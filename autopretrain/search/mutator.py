"""Mutation strategies for data mixtures and hyperparameters.

Strategies are selected probabilistically based on the current search phase:
- Early (exploration): more random perturbation, reference interpolation
- Late (exploitation): more targeted domain boost, smaller deltas

Also supports hyperparameter mutations for full-space search.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from autopretrain.core.types import TrialConfig
from autopretrain.search.mixture import DataMixture, REFERENCE_MIXTURES


@dataclass
class MutationConfig:
    """Controls mutation behavior."""

    # Data mix mutations
    perturb_sigma: float = 0.10
    boost_amount: float = 0.15
    interpolation_alpha_range: tuple[float, float] = (0.2, 0.8)

    # Hyperparameter mutations
    lr_log_sigma: float = 0.3
    batch_size_options: list[int] | None = None
    warmup_range: tuple[int, int] = (100, 2000)

    # Annealing: reduce exploration over time
    min_sigma_multiplier: float = 0.3


class MixtureMutator:
    """Mutates data mixtures using multiple strategies."""

    def __init__(self, config: MutationConfig | None = None) -> None:
        self.config = config or MutationConfig()

    def mutate(
        self,
        mixture: DataMixture,
        strategy: str | None = None,
        progress: float = 0.0,
    ) -> DataMixture:
        """Apply a mutation strategy to a mixture.

        Args:
            mixture: Current mixture to mutate
            strategy: Specific strategy (or None for random selection)
            progress: Training progress 0.0 to 1.0 (anneals exploration)
        """
        if strategy is None:
            strategy = self._select_strategy(progress)

        strategies = {
            "random_perturb": self._random_perturb,
            "domain_boost": self._domain_boost,
            "reference_interpolate": self._reference_interpolate,
            "swap_emphasis": self._swap_emphasis,
            "uniform_drift": self._uniform_drift,
        }

        handler = strategies.get(strategy, self._random_perturb)
        sigma_mult = 1.0 - progress * (1.0 - self.config.min_sigma_multiplier)
        return handler(mixture, sigma_mult)

    def _select_strategy(self, progress: float) -> str:
        """Select strategy based on progress (more exploitation later)."""
        if progress < 0.3:
            weights = [0.3, 0.2, 0.3, 0.1, 0.1]
        elif progress < 0.7:
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        else:
            weights = [0.2, 0.4, 0.1, 0.2, 0.1]

        strategies = ["random_perturb", "domain_boost", "reference_interpolate", "swap_emphasis", "uniform_drift"]
        return random.choices(strategies, weights=weights, k=1)[0]

    def _random_perturb(self, mix: DataMixture, sigma_mult: float) -> DataMixture:
        """Perturb a random domain by Gaussian noise."""
        domain = random.choice(mix.domains)
        delta = random.gauss(0, self.config.perturb_sigma * sigma_mult)
        return mix.perturb(domain, delta)

    def _domain_boost(self, mix: DataMixture, sigma_mult: float) -> DataMixture:
        """Boost a specific domain (typically code or math for reasoning)."""
        domain = random.choice(mix.domains)
        amount = self.config.boost_amount * sigma_mult
        if random.random() < 0.5:
            amount = -amount
        return mix.perturb(domain, amount)

    def _reference_interpolate(self, mix: DataMixture, sigma_mult: float) -> DataMixture:
        """Interpolate toward a reference mixture."""
        ref_name = random.choice(list(REFERENCE_MIXTURES.keys()))
        ref = DataMixture.from_reference(ref_name)
        lo, hi = self.config.interpolation_alpha_range
        alpha = random.uniform(lo * sigma_mult, hi * sigma_mult)
        return mix.interpolate(ref, alpha)

    def _swap_emphasis(self, mix: DataMixture, sigma_mult: float) -> DataMixture:
        """Swap emphasis between two domains."""
        if len(mix.domains) < 2:
            return mix
        d1, d2 = random.sample(mix.domains, 2)
        swap_amount = random.uniform(0.05, 0.15) * sigma_mult
        new_mix = mix.perturb(d1, swap_amount)
        return new_mix.perturb(d2, -swap_amount)

    def _uniform_drift(self, mix: DataMixture, sigma_mult: float) -> DataMixture:
        """Drift toward uniform distribution."""
        uniform = DataMixture(weights={d: 1.0 / len(mix.domains) for d in mix.domains})
        alpha = random.uniform(0.1, 0.3) * sigma_mult
        return mix.interpolate(uniform, alpha)


class HyperparamMutator:
    """Mutates training hyperparameters for full-space search."""

    def __init__(self, config: MutationConfig | None = None) -> None:
        self.config = config or MutationConfig()

    def mutate_lr(self, current_lr: float, progress: float = 0.0) -> float:
        """Log-normal perturbation of learning rate."""
        sigma = self.config.lr_log_sigma * (1.0 - progress * 0.5)
        log_lr = math.log(current_lr) + random.gauss(0, sigma)
        new_lr = math.exp(log_lr)
        return max(1e-5, min(1e-2, new_lr))

    def mutate_batch_size(self, current: int) -> int:
        """Mutate batch size to adjacent power of 2."""
        options = self.config.batch_size_options or [64, 128, 256, 512]
        idx = options.index(current) if current in options else len(options) // 2
        delta = random.choice([-1, 0, 1])
        new_idx = max(0, min(len(options) - 1, idx + delta))
        return options[new_idx]

    def mutate_warmup(self, current: int) -> int:
        """Mutate warmup steps."""
        lo, hi = self.config.warmup_range
        delta = random.randint(-200, 200)
        return max(lo, min(hi, current + delta))

    def mutate_trial(self, trial: TrialConfig, progress: float = 0.0) -> TrialConfig:
        """Create a mutated copy of a trial config.

        Randomly selects which parameters to mutate.
        """
        import copy
        new = copy.deepcopy(trial)

        # Always mutate data mix
        if new.data_mix:
            mixer = MixtureMutator(self.config)
            current_mix = DataMixture(weights=dict(new.data_mix.sources))
            mutated_mix = mixer.mutate(current_mix, progress=progress)
            new.data_mix.sources = mutated_mix.weights

        # Optionally mutate hyperparams
        if random.random() < 0.3:
            new.lr = self.mutate_lr(new.lr, progress)

        if random.random() < 0.2:
            new.global_batch_size = self.mutate_batch_size(new.global_batch_size)

        if random.random() < 0.15:
            new.warmup_steps = self.mutate_warmup(new.warmup_steps)

        # Generate new trial_id
        import uuid
        new.trial_id = uuid.uuid4().hex[:8]
        new.name = f"mutated_{new.trial_id}"

        return new
