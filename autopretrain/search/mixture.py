"""Data mixture on the probability simplex.

A DataMixture represents domain weights that sum to 1.0.
Supports perturbation, interpolation, clamping, and distance computation.

Migrated from agent_evolve/model/algorithms/autopretrain/mixture.py
with cleanup and integration into the new type system.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any

from autopretrain.core.types import DataMixSpec

DEFAULT_DOMAINS = ["web", "code", "math", "academic"]

DOMAIN_TOKEN_BUDGET = {
    "web": 38_501_000_000,
    "code": 9_679_000_000,
    "math": 7_016_000_000,
    "academic": 38_501_000_000,
}

DOMAIN_PATHS = {
    "web": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/dclm_web/*.bin",
    "code": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/code/*.bin",
    "math": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/math/*.bin",
    "academic": "/fsx/dev/jiaqi/data/olmo-pretrain-raw/fineweb_edu/*.bin",
}

REFERENCE_MIXTURES = {
    "olmo2_stage1": {"web": 0.95, "code": 0.02, "math": 0.01, "academic": 0.02},
    "olmo2_midtrain": {"web": 0.50, "code": 0.03, "math": 0.25, "academic": 0.22},
    "llama3": {"web": 0.52, "code": 0.18, "math": 0.26, "academic": 0.04},
    "deepseek_v3": {"web": 0.47, "code": 0.26, "math": 0.21, "academic": 0.06},
    "doremi": {"web": 0.88, "code": 0.03, "math": 0.01, "academic": 0.08},
    "uniform": {"web": 0.25, "code": 0.25, "math": 0.25, "academic": 0.25},
    "reasoning_heavy": {"web": 0.25, "code": 0.30, "math": 0.30, "academic": 0.15},
    "late_annealing": {"web": 0.20, "code": 0.30, "math": 0.35, "academic": 0.15},
}


@dataclass
class DataMixture:
    """A point on the probability simplex."""

    weights: dict[str, float]

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.weights = {k: v / total for k, v in self.weights.items()}

    @property
    def domains(self) -> list[str]:
        return list(self.weights.keys())

    def get(self, domain: str) -> float:
        return self.weights.get(domain, 0.0)

    def perturb(self, domain: str, delta: float) -> DataMixture:
        """Perturb one domain's weight, maintaining simplex."""
        new_weights = dict(self.weights)
        old_val = new_weights[domain]
        new_val = max(0.01, min(0.95, old_val + delta))
        actual_delta = new_val - old_val

        new_weights[domain] = new_val
        others = [d for d in new_weights if d != domain]
        other_total = sum(new_weights[d] for d in others)

        if other_total > 0:
            for d in others:
                new_weights[d] -= actual_delta * (new_weights[d] / other_total)
                new_weights[d] = max(0.01, new_weights[d])

        return DataMixture(weights=new_weights)

    def interpolate(self, other: DataMixture, alpha: float) -> DataMixture:
        """Linear interpolation: self * (1-alpha) + other * alpha."""
        new_weights = {}
        for domain in set(self.domains) | set(other.domains):
            new_weights[domain] = self.get(domain) * (1 - alpha) + other.get(domain) * alpha
        return DataMixture(weights=new_weights)

    def distance(self, other: DataMixture) -> float:
        """Jensen-Shannon divergence."""
        domains = set(self.domains) | set(other.domains)
        p = [max(self.get(d), 1e-10) for d in domains]
        q = [max(other.get(d), 1e-10) for d in domains]
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

        def kl(a, b):
            return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b))

        return (kl(p, m) + kl(q, m)) / 2

    def clamp_by_repetition(self, total_tokens: int, max_epochs: float = 4.0) -> DataMixture:
        """Cap weights to respect repetition ceiling per Muennighoff et al."""
        max_weights = {}
        for domain, weight in self.weights.items():
            available = DOMAIN_TOKEN_BUDGET.get(domain, float("inf"))
            max_w = (available * max_epochs) / total_tokens if total_tokens > 0 else 1.0
            max_weights[domain] = min(weight, max_w)
        return DataMixture(weights=max_weights)

    def to_spec(self) -> DataMixSpec:
        """Convert to TrialConfig-compatible DataMixSpec."""
        return DataMixSpec(
            sources=dict(self.weights),
            paths=DOMAIN_PATHS,
            max_repetition_ratio=4.0,
        )

    @classmethod
    def from_reference(cls, name: str) -> DataMixture:
        if name not in REFERENCE_MIXTURES:
            raise ValueError(f"Unknown: {name}. Available: {list(REFERENCE_MIXTURES)}")
        return cls(weights=dict(REFERENCE_MIXTURES[name]))

    @classmethod
    def random(cls, domains: list[str] | None = None) -> DataMixture:
        """Generate a random mixture on the simplex."""
        domains = domains or DEFAULT_DOMAINS
        raw = [random.random() for _ in domains]
        total = sum(raw)
        return cls(weights={d: v / total for d, v in zip(domains, raw)})

    def __repr__(self) -> str:
        parts = [f"{d}={w:.1%}" for d, w in sorted(self.weights.items(), key=lambda x: -x[1])]
        return f"Mix({', '.join(parts)})"
