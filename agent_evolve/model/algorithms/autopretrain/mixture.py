"""Data mixture representation and manipulation.

A DataMixture lives on the probability simplex: domain weights sum to 1.0.
Mutations, interpolations, and curriculum transitions all preserve this invariant.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any


# Default domains — maps to actual data on FSx:
#   web → /fsx/dev/jiaqi/data/olmo-pretrain/dclm_web (38.5B tokens)
#   code → /fsx/dev/jiaqi/data/olmo-pretrain/code (9.7B tokens)
#   math → /fsx/dev/jiaqi/data/olmo-pretrain/math (7.0B tokens)
#   academic → /fsx/dev/jiaqi/data/olmo-pretrain/fineweb_edu (38.5B tokens)
DEFAULT_DOMAINS = ["web", "code", "math", "academic"]

# Actual available tokens on FSx cluster (measured May 2026)
# Used to enforce the 4-epoch repetition ceiling (Muennighoff et al., 2023)
DOMAIN_TOKEN_BUDGET = {
    "web": 38_501_000_000,       # 38.5B (dclm_web on FSx)
    "code": 9_679_000_000,       # 9.7B (code on FSx)
    "math": 7_016_000_000,       # 7.0B (math on FSx)
    "academic": 38_501_000_000,  # 38.5B (fineweb_edu on FSx)
}

# Reference mixtures (4-domain: web, code, math, academic)
# Adapted from published technical reports to our domain mapping
REFERENCE_MIXTURES = {
    # OLMo-2 Stage 1 proportional (Jan 2025): overwhelmingly web
    "olmo2_stage1": {"web": 0.95, "code": 0.02, "math": 0.01, "academic": 0.02},
    # OLMo-2 Mid-training: shifts heavily toward math+academic
    "olmo2_midtrain": {"web": 0.50, "code": 0.03, "math": 0.25, "academic": 0.22},
    # Llama-3 final (general=50%, math+reasoning=25%, code=17%, other=8%)
    "llama3": {"web": 0.52, "code": 0.18, "math": 0.26, "academic": 0.04},
    # DeepSeek-v3 estimated: heavy on code+math
    "deepseek_v3": {"web": 0.47, "code": 0.26, "math": 0.21, "academic": 0.06},
    # DoReMi: DRO pushes toward web dominance
    "doremi": {"web": 0.88, "code": 0.03, "math": 0.01, "academic": 0.08},
    # Uniform (control)
    "uniform": {"web": 0.25, "code": 0.25, "math": 0.25, "academic": 0.25},
    # Our hypothesis: maximize reasoning signal
    "reasoning_heavy": {"web": 0.25, "code": 0.30, "math": 0.30, "academic": 0.15},
    # Blakeney et al.: annealing-phase style (code+math dominant)
    "late_annealing": {"web": 0.20, "code": 0.30, "math": 0.35, "academic": 0.15},
}


@dataclass
class FilterConfig:
    """Per-domain quality filtering configuration.

    retention_rate: fraction of documents kept after filtering (1.0 = no filter).
    Inspired by "A Bitter Lesson for Data Filtering" — the optimal retention
    depends on model size and training compute.
    """

    retention_rates: dict[str, float] = field(default_factory=lambda: {
        "web": 0.20,
        "code": 0.50,
        "math": 0.80,
        "academic": 0.70,
    })

    def effective_tokens(self, raw_tokens: dict[str, int]) -> dict[str, int]:
        """Compute available tokens after filtering."""
        return {
            domain: int(count * self.retention_rates.get(domain, 1.0))
            for domain, count in raw_tokens.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {"retention_rates": dict(self.retention_rates)}

    @classmethod
    def from_dict(cls, d: dict) -> FilterConfig:
        return cls(retention_rates=d.get("retention_rates", {}))


@dataclass
class DataMixture:
    """A point on the probability simplex representing domain weights.

    Invariant: sum(weights.values()) == 1.0 (within floating point tolerance).
    """

    weights: dict[str, float]
    filter_config: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            self.weights = {k: v / total for k, v in self.weights.items()}

    @property
    def domains(self) -> list[str]:
        return list(self.weights.keys())

    @property
    def n_domains(self) -> int:
        return len(self.weights)

    def get(self, domain: str) -> float:
        return self.weights.get(domain, 0.0)

    def perturb(self, domain: str, delta: float) -> DataMixture:
        """Perturb one domain's weight, redistributing to maintain simplex."""
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

        return DataMixture(weights=new_weights, filter_config=copy.deepcopy(self.filter_config))

    def interpolate(self, other: DataMixture, alpha: float) -> DataMixture:
        """Linear interpolation on the simplex: self * (1-alpha) + other * alpha."""
        new_weights = {}
        for domain in set(self.domains) | set(other.domains):
            w1 = self.get(domain)
            w2 = other.get(domain)
            new_weights[domain] = w1 * (1 - alpha) + w2 * alpha
        return DataMixture(weights=new_weights, filter_config=copy.deepcopy(self.filter_config))

    def distance(self, other: DataMixture) -> float:
        """Jensen-Shannon divergence between two mixtures."""
        domains = set(self.domains) | set(other.domains)
        p = [self.get(d) for d in domains]
        q = [other.get(d) for d in domains]
        m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

        def kl(a, b):
            return sum(
                ai * math.log(ai / bi) for ai, bi in zip(a, b)
                if ai > 0 and bi > 0
            )

        return (kl(p, m) + kl(q, m)) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "filter_config": self.filter_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataMixture:
        fc = FilterConfig.from_dict(d.get("filter_config", {}))
        return cls(weights=d["weights"], filter_config=fc)

    @classmethod
    def from_reference(cls, name: str) -> DataMixture:
        if name not in REFERENCE_MIXTURES:
            raise ValueError(f"Unknown reference: {name}. Available: {list(REFERENCE_MIXTURES)}")
        return cls(weights=dict(REFERENCE_MIXTURES[name]))

    def clamp_by_repetition(
        self, total_training_tokens: int, max_epochs: float = 4.0
    ) -> DataMixture:
        """Clamp domain weights to respect the repetition ceiling.

        Per Muennighoff et al. (2023): beyond 4 epochs of a domain's unique data,
        additional repetition yields diminishing returns. This method caps each
        domain's weight such that it won't exceed max_epochs of its available data.
        """
        max_weights = {}
        for domain, weight in self.weights.items():
            available = DOMAIN_TOKEN_BUDGET.get(domain, float("inf"))
            # Max tokens we'd sample from this domain
            max_tokens_for_domain = available * max_epochs
            # Max weight this domain can have
            max_w = max_tokens_for_domain / total_training_tokens if total_training_tokens > 0 else 1.0
            max_weights[domain] = min(weight, max_w)

        return DataMixture(weights=max_weights, filter_config=copy.deepcopy(self.filter_config))

    def __repr__(self) -> str:
        parts = [f"{d}={w:.1%}" for d, w in sorted(self.weights.items(), key=lambda x: -x[1])]
        return f"DataMixture({', '.join(parts)})"


@dataclass
class CurriculumPhase:
    """A single phase in a data curriculum."""

    mixture: DataMixture
    start_fraction: float  # fraction of total training steps where this phase begins
    end_fraction: float    # fraction where this phase ends

    def active_at(self, progress: float) -> bool:
        return self.start_fraction <= progress < self.end_fraction


@dataclass
class CurriculumSchedule:
    """Multi-phase data curriculum: different mixes at different training stages.

    The simplest curriculum has 1 phase (= static mix).
    A 2-phase curriculum might start web-heavy then shift to reasoning-heavy.
    """

    phases: list[CurriculumPhase]

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    @property
    def is_static(self) -> bool:
        return len(self.phases) == 1

    def mixture_at(self, progress: float) -> DataMixture:
        """Get the data mixture at a given training progress (0.0 to 1.0).

        Linearly interpolates between phases at transition boundaries.
        """
        progress = max(0.0, min(1.0, progress))

        for phase in self.phases:
            if phase.active_at(progress):
                return phase.mixture

        return self.phases[-1].mixture

    @classmethod
    def static(cls, mixture: DataMixture) -> CurriculumSchedule:
        """Create a single-phase (static) curriculum."""
        return cls(phases=[CurriculumPhase(mixture=mixture, start_fraction=0.0, end_fraction=1.0)])

    @classmethod
    def two_phase(
        cls,
        phase1: DataMixture,
        phase2: DataMixture,
        transition_point: float = 0.5,
    ) -> CurriculumSchedule:
        """Create a 2-phase curriculum with transition at given point."""
        return cls(phases=[
            CurriculumPhase(mixture=phase1, start_fraction=0.0, end_fraction=transition_point),
            CurriculumPhase(mixture=phase2, start_fraction=transition_point, end_fraction=1.0),
        ])

    @classmethod
    def three_phase(
        cls,
        phase1: DataMixture,
        phase2: DataMixture,
        phase3: DataMixture,
        t1: float = 0.33,
        t2: float = 0.67,
    ) -> CurriculumSchedule:
        """Create a 3-phase curriculum."""
        return cls(phases=[
            CurriculumPhase(mixture=phase1, start_fraction=0.0, end_fraction=t1),
            CurriculumPhase(mixture=phase2, start_fraction=t1, end_fraction=t2),
            CurriculumPhase(mixture=phase3, start_fraction=t2, end_fraction=1.0),
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": [
                {
                    "mixture": p.mixture.to_dict(),
                    "start_fraction": p.start_fraction,
                    "end_fraction": p.end_fraction,
                }
                for p in self.phases
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> CurriculumSchedule:
        phases = []
        for p in d["phases"]:
            phases.append(CurriculumPhase(
                mixture=DataMixture.from_dict(p["mixture"]),
                start_fraction=p["start_fraction"],
                end_fraction=p["end_fraction"],
            ))
        return cls(phases=phases)

    def __repr__(self) -> str:
        if self.is_static:
            return f"CurriculumSchedule(static={self.phases[0].mixture})"
        return f"CurriculumSchedule({self.n_phases} phases)"
