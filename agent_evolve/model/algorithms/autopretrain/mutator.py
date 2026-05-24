"""LLM-guided and structured mutations for data mixture search.

The MixtureMutator produces candidate mixtures from parent mixtures.
It combines structured operations (simplex perturbation, interpolation)
with LLM-guided proposals that encode "research intuition" about what
data compositions might help specific downstream capabilities.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from .mixture import (
    CurriculumSchedule,
    CurriculumPhase,
    DataMixture,
    FilterConfig,
    REFERENCE_MIXTURES,
)

logger = logging.getLogger(__name__)


@dataclass
class MutationRecord:
    """Records what mutation was applied and why."""

    mutation_type: str
    description: str
    parent_mixture: dict[str, Any]
    child_mixture: dict[str, Any]
    reasoning: str = ""


@dataclass
class MixtureMutator:
    """Proposes new data mixtures from parent mixtures.

    Mutation strategies (in order of exploration vs exploitation):
    1. Random simplex perturbation (high exploration)
    2. Domain-targeted boost/reduce (medium exploration)
    3. Reference interpolation (medium - borrows from known-good mixes)
    4. Curriculum introduction (high novelty - adds phases)
    5. Filter adjustment (Bitter Lesson axis)
    6. LLM-guided proposal (highest quality, uses eval feedback)
    """

    # Controls exploration vs exploitation
    perturbation_scale: float = 0.10
    # Probability of each mutation type
    strategy_weights: dict[str, float] = field(default_factory=lambda: {
        "random_perturb": 0.20,
        "domain_boost": 0.25,
        "reference_interpolate": 0.15,
        "curriculum_mutate": 0.15,
        "filter_adjust": 0.10,
        "llm_guided": 0.15,
    })
    # LLM backend for guided mutations (None = skip LLM mutations)
    llm_backend: Any = None

    def mutate(
        self,
        parent: CurriculumSchedule,
        eval_feedback: dict[str, Any] | None = None,
        cycle: int = 0,
    ) -> tuple[CurriculumSchedule, MutationRecord]:
        """Produce a child curriculum from a parent.

        Args:
            parent: The parent curriculum to mutate.
            eval_feedback: Optional eval results to guide mutation direction.
            cycle: Current search cycle (used for annealing).

        Returns:
            (child_curriculum, mutation_record)
        """
        strategy = self._select_strategy(cycle)
        mutate_fn = {
            "random_perturb": self._random_perturb,
            "domain_boost": self._domain_boost,
            "reference_interpolate": self._reference_interpolate,
            "curriculum_mutate": self._curriculum_mutate,
            "filter_adjust": self._filter_adjust,
            "llm_guided": self._llm_guided,
        }[strategy]

        child, description, reasoning = mutate_fn(parent, eval_feedback, cycle)

        record = MutationRecord(
            mutation_type=strategy,
            description=description,
            parent_mixture=parent.to_dict(),
            child_mixture=child.to_dict(),
            reasoning=reasoning,
        )
        return child, record

    def _select_strategy(self, cycle: int) -> str:
        """Select mutation strategy with annealing: more exploitation later."""
        weights = dict(self.strategy_weights)
        # Anneal: reduce random exploration, increase targeted mutations
        anneal = min(1.0, cycle / 30.0)
        weights["random_perturb"] *= (1.0 - 0.5 * anneal)
        weights["domain_boost"] *= (1.0 + 0.3 * anneal)
        weights["llm_guided"] *= (1.0 + 0.5 * anneal)

        strategies = list(weights.keys())
        probs = list(weights.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        return random.choices(strategies, weights=probs, k=1)[0]

    def _random_perturb(
        self, parent: CurriculumSchedule, feedback: Any, cycle: int
    ) -> tuple[CurriculumSchedule, str, str]:
        """Randomly perturb one domain in a random phase."""
        phase_idx = random.randint(0, parent.n_phases - 1)
        phase = parent.phases[phase_idx]
        domain = random.choice(phase.mixture.domains)
        delta = random.gauss(0, self.perturbation_scale)

        new_mixture = phase.mixture.perturb(domain, delta)
        new_phases = list(parent.phases)
        new_phases[phase_idx] = CurriculumPhase(
            mixture=new_mixture,
            start_fraction=phase.start_fraction,
            end_fraction=phase.end_fraction,
        )

        desc = f"Perturb {domain} by {delta:+.3f} in phase {phase_idx}"
        return CurriculumSchedule(phases=new_phases), desc, "Random exploration"

    def _domain_boost(
        self, parent: CurriculumSchedule, feedback: Any, cycle: int
    ) -> tuple[CurriculumSchedule, str, str]:
        """Boost or reduce a specific domain based on eval feedback."""
        phase_idx = 0 if parent.is_static else random.randint(0, parent.n_phases - 1)
        phase = parent.phases[phase_idx]

        # If we have feedback, boost domains correlated with weak performance
        if feedback and "domain_losses" in feedback:
            domain_losses = feedback["domain_losses"]
            # Boost domain with highest loss (it needs more data)
            domain = max(domain_losses, key=domain_losses.get)
            delta = random.uniform(0.03, 0.12)
            reasoning = f"Domain '{domain}' has highest loss ({domain_losses[domain]:.4f}), boosting"
        else:
            # Random domain with larger delta
            domain = random.choice(phase.mixture.domains)
            delta = random.choice([-1, 1]) * random.uniform(0.05, 0.15)
            reasoning = "No feedback available, random domain boost"

        new_mixture = phase.mixture.perturb(domain, delta)
        new_phases = list(parent.phases)
        new_phases[phase_idx] = CurriculumPhase(
            mixture=new_mixture,
            start_fraction=phase.start_fraction,
            end_fraction=phase.end_fraction,
        )

        desc = f"Boost {domain} by {delta:+.3f} in phase {phase_idx}"
        return CurriculumSchedule(phases=new_phases), desc, reasoning

    def _reference_interpolate(
        self, parent: CurriculumSchedule, feedback: Any, cycle: int
    ) -> tuple[CurriculumSchedule, str, str]:
        """Interpolate current mixture toward a reference mixture."""
        ref_name = random.choice(list(REFERENCE_MIXTURES.keys()))
        ref_mixture = DataMixture.from_reference(ref_name)
        alpha = random.uniform(0.1, 0.4)

        phase_idx = 0 if parent.is_static else random.randint(0, parent.n_phases - 1)
        phase = parent.phases[phase_idx]
        new_mixture = phase.mixture.interpolate(ref_mixture, alpha)

        new_phases = list(parent.phases)
        new_phases[phase_idx] = CurriculumPhase(
            mixture=new_mixture,
            start_fraction=phase.start_fraction,
            end_fraction=phase.end_fraction,
        )

        desc = f"Interpolate {alpha:.0%} toward '{ref_name}' in phase {phase_idx}"
        reasoning = f"Exploring direction of {ref_name} mixture (known successful recipe)"
        return CurriculumSchedule(phases=new_phases), desc, reasoning

    def _curriculum_mutate(
        self, parent: CurriculumSchedule, feedback: Any, cycle: int
    ) -> tuple[CurriculumSchedule, str, str]:
        """Modify curriculum structure: add phase, shift transition, or merge phases."""
        if parent.is_static:
            # Promote to 2-phase: create a shifted version for phase 2
            base = parent.phases[0].mixture
            # Phase 2 boosts reasoning domains
            phase2_weights = dict(base.weights)
            for domain in ["code", "math"]:
                if domain in phase2_weights:
                    phase2_weights[domain] = min(0.9, phase2_weights[domain] * 1.5)
            phase2 = DataMixture(weights=phase2_weights, filter_config=base.filter_config)
            transition = random.uniform(0.3, 0.7)
            child = CurriculumSchedule.two_phase(base, phase2, transition)
            desc = f"Promote to 2-phase (transition at {transition:.0%}): boost code+math in phase 2"
            reasoning = "Testing hypothesis: reasoning-heavy data helps more in later training"
        elif parent.n_phases == 2:
            # Shift transition point
            shift = random.uniform(-0.15, 0.15)
            old_t = parent.phases[0].end_fraction
            new_t = max(0.2, min(0.8, old_t + shift))
            child = CurriculumSchedule(phases=[
                CurriculumPhase(
                    mixture=parent.phases[0].mixture,
                    start_fraction=0.0,
                    end_fraction=new_t,
                ),
                CurriculumPhase(
                    mixture=parent.phases[1].mixture,
                    start_fraction=new_t,
                    end_fraction=1.0,
                ),
            ])
            desc = f"Shift transition: {old_t:.0%} → {new_t:.0%}"
            reasoning = "Adjusting when curriculum phase transition occurs"
        else:
            # Simplify back to static (use weighted average)
            avg_weights: dict[str, float] = {}
            for phase in parent.phases:
                duration = phase.end_fraction - phase.start_fraction
                for d, w in phase.mixture.weights.items():
                    avg_weights[d] = avg_weights.get(d, 0) + w * duration
            child = CurriculumSchedule.static(DataMixture(weights=avg_weights))
            desc = "Collapse multi-phase to static (weighted average)"
            reasoning = "Testing whether curriculum complexity helps vs simpler static mix"

        return child, desc, reasoning

    def _filter_adjust(
        self, parent: CurriculumSchedule, feedback: Any, cycle: int
    ) -> tuple[CurriculumSchedule, str, str]:
        """Adjust filter aggressiveness (Bitter Lesson dimension)."""
        import copy as _copy
        phase_idx = 0
        phase = parent.phases[phase_idx]
        new_filter = _copy.deepcopy(phase.mixture.filter_config)

        domain = random.choice(list(new_filter.retention_rates.keys()))
        old_rate = new_filter.retention_rates[domain]
        # Bias toward LESS filtering (Bitter Lesson)
        delta = random.gauss(0.05, 0.10)
        new_rate = max(0.05, min(1.0, old_rate + delta))
        new_filter.retention_rates[domain] = new_rate

        new_mixture = DataMixture(
            weights=dict(phase.mixture.weights),
            filter_config=new_filter,
        )
        new_phases = list(parent.phases)
        new_phases[phase_idx] = CurriculumPhase(
            mixture=new_mixture,
            start_fraction=phase.start_fraction,
            end_fraction=phase.end_fraction,
        )

        direction = "less" if delta > 0 else "more"
        desc = f"Filter {domain}: {old_rate:.0%} → {new_rate:.0%} ({direction} filtering)"
        reasoning = (
            f"Bitter Lesson hypothesis: {direction} filtering for {domain} "
            f"(model may be large enough to handle noise)"
        )
        return CurriculumSchedule(phases=new_phases), desc, reasoning

    def _llm_guided(
        self, parent: CurriculumSchedule, feedback: Any, cycle: int
    ) -> tuple[CurriculumSchedule, str, str]:
        """Use LLM to propose a mutation based on eval feedback.

        Falls back to domain_boost if no LLM backend configured.
        """
        if self.llm_backend is None:
            return self._domain_boost(parent, feedback, cycle)

        # TODO: Implement LLM-guided mutation via self.llm_backend
        # The LLM receives: current mixture, eval feedback, search history
        # and proposes a specific mutation with reasoning.
        # For now, fall back to structured mutation.
        return self._domain_boost(parent, feedback, cycle)
