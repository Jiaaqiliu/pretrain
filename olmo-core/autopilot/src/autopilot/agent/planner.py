"""Training strategy planner.

Plans multi-phase training campaigns based on:
- User-specified targets (model size, performance goals, compute budget)
- Scaling laws (Chinchilla optimal allocation)
- muTransfer (proxy search strategy)
- Best practices from OLMo-2, Llama-3, DeepSeek reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autopilot.experiment.config_builder import (
    ComputeBudget,
    ConfigBuilder,
    ModelSize,
    TrainingPhase,
    TrainingTarget,
)
from autopilot.optimization.mu_transfer import ModelScale, MuTransferEngine
from autopilot.utils.logging import get_logger

log = get_logger("agent.planner")


@dataclass
class PhaseSpec:
    """Specification for a single phase in the training plan."""

    phase_id: str
    name: str
    phase_type: str  # "proxy_search", "validation", "full_training", "midtrain", "sft"
    target: TrainingTarget
    depends_on: List[str] = field(default_factory=list)
    estimated_gpu_hours: float = 0.0
    n_experiments: int = 1
    description: str = ""


@dataclass
class TrainingPlan:
    """A complete multi-phase training campaign plan."""

    name: str
    phases: List[PhaseSpec]
    total_estimated_gpu_hours: float
    target_model_size: ModelSize
    target_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def phase_names(self) -> List[str]:
        return [p.name for p in self.phases]

    def get_phase(self, phase_id: str) -> Optional[PhaseSpec]:
        for p in self.phases:
            if p.phase_id == phase_id:
                return p
        return None


class PlannerAgent:
    """Plans training campaigns from high-level goals.

    The planner decomposes a training target into phases:
    1. Proxy HP search (muTransfer on small model)
    2. Data mixture optimization (DoReMi/RegMix)
    3. Validation run (medium scale)
    4. Full-scale training
    5. Optional: mid-training / post-training

    Each phase produces outputs consumed by the next phase.
    """

    def __init__(self, config_builder: Optional[ConfigBuilder] = None):
        self._builder = config_builder or ConfigBuilder()

    def plan(
        self,
        model_size: ModelSize,
        compute_budget: Optional[ComputeBudget] = None,
        target_loss: Optional[float] = None,
        data_domains: Optional[List[str]] = None,
        include_sft: bool = False,
        skip_proxy_search: bool = False,
    ) -> TrainingPlan:
        """Generate a complete training plan."""
        target_scale = model_size.to_scale()
        phases: List[PhaseSpec] = []
        total_hours = 0.0

        # Phase 1: Proxy HP search
        if not skip_proxy_search:
            proxy_phase = self._plan_proxy_search(target_scale, compute_budget, data_domains)
            phases.append(proxy_phase)
            total_hours += proxy_phase.estimated_gpu_hours

        # Phase 2: Data mixture optimization
        if data_domains and len(data_domains) > 1:
            mix_phase = self._plan_mixture_optimization(target_scale, data_domains, compute_budget)
            mix_phase.depends_on = [phases[-1].phase_id] if phases else []
            phases.append(mix_phase)
            total_hours += mix_phase.estimated_gpu_hours

        # Phase 3: Validation run
        val_phase = self._plan_validation(target_scale, compute_budget, data_domains)
        val_phase.depends_on = [phases[-1].phase_id] if phases else []
        phases.append(val_phase)
        total_hours += val_phase.estimated_gpu_hours

        # Phase 4: Full-scale training
        full_phase = self._plan_full_training(
            model_size, target_scale, compute_budget, target_loss, data_domains
        )
        full_phase.depends_on = [val_phase.phase_id]
        phases.append(full_phase)
        total_hours += full_phase.estimated_gpu_hours

        # Phase 5: SFT (optional)
        if include_sft:
            sft_phase = self._plan_sft(model_size, compute_budget)
            sft_phase.depends_on = [full_phase.phase_id]
            phases.append(sft_phase)
            total_hours += sft_phase.estimated_gpu_hours

        plan = TrainingPlan(
            name=f"autopilot-{model_size.value}-campaign",
            phases=phases,
            total_estimated_gpu_hours=total_hours,
            target_model_size=model_size,
            target_loss=target_loss,
        )

        log.info(
            f"Created training plan '{plan.name}' with {len(phases)} phases, "
            f"estimated {total_hours:.0f} GPU-hours"
        )
        return plan

    def _plan_proxy_search(
        self,
        target_scale: ModelScale,
        budget: Optional[ComputeBudget],
        data_domains: Optional[List[str]],
    ) -> PhaseSpec:
        """Plan the proxy model hyperparameter search phase."""
        proxy_scale = MuTransferEngine.design_proxy(target_scale, width_divisor=4)
        proxy_tokens = int(proxy_scale.num_params_approx * 20)  # Chinchilla optimal

        # Estimate cost: N_trials * proxy_cost
        n_trials = 20
        flops_per_trial = 6 * proxy_scale.num_params_approx * proxy_tokens
        gpu_tflops = 312  # A100
        hours_per_trial = flops_per_trial / (gpu_tflops * 1e12 * 3600 * 8)  # 8 GPUs
        total_hours = hours_per_trial * n_trials

        proxy_target = TrainingTarget(
            model_size=ModelSize.SMALL,
            phase=TrainingPhase.PRETRAIN,
            target_tokens=proxy_tokens,
            data_domains=data_domains or [],
            compute_budget=ComputeBudget(num_nodes=1, gpus_per_node=8),
        )

        return PhaseSpec(
            phase_id="proxy_search",
            name="Proxy HP Search (muTransfer)",
            phase_type="proxy_search",
            target=proxy_target,
            estimated_gpu_hours=total_hours,
            n_experiments=n_trials,
            description=(
                f"Search hyperparameters on {proxy_scale.hidden_size}d proxy model "
                f"({proxy_scale.num_params_approx/1e6:.0f}M params, "
                f"{proxy_tokens/1e9:.1f}B tokens) using muTransfer"
            ),
        )

    def _plan_mixture_optimization(
        self,
        target_scale: ModelScale,
        data_domains: List[str],
        budget: Optional[ComputeBudget],
    ) -> PhaseSpec:
        """Plan data mixture optimization phase."""
        MuTransferEngine.design_proxy(target_scale, width_divisor=8)
        n_experiments = min(len(data_domains) * 3, 12)

        hours_estimate = n_experiments * 2.0  # ~2 GPU-hours per proxy experiment

        mix_target = TrainingTarget(
            model_size=ModelSize.SMALL,
            phase=TrainingPhase.PRETRAIN,
            data_domains=data_domains,
            compute_budget=ComputeBudget(num_nodes=1, gpus_per_node=8),
        )

        return PhaseSpec(
            phase_id="mixture_opt",
            name="Data Mixture Optimization",
            phase_type="mixture_search",
            target=mix_target,
            estimated_gpu_hours=hours_estimate,
            n_experiments=n_experiments,
            description=(
                f"Optimize data mixture across {len(data_domains)} domains "
                f"using DoReMi/RegMix approach"
            ),
        )

    def _plan_validation(
        self,
        target_scale: ModelScale,
        budget: Optional[ComputeBudget],
        data_domains: Optional[List[str]],
    ) -> PhaseSpec:
        """Plan medium-scale validation run."""
        # Use 1/4 width model for ~10% compute verification
        val_scale = MuTransferEngine.design_proxy(target_scale, width_divisor=2)
        val_tokens = int(val_scale.num_params_approx * 10)  # shorter run for validation

        flops = 6 * val_scale.num_params_approx * val_tokens
        hours = flops / (312 * 1e12 * 3600 * 8)

        val_target = TrainingTarget(
            model_size=ModelSize.MEDIUM,
            phase=TrainingPhase.PRETRAIN,
            target_tokens=val_tokens,
            data_domains=data_domains or [],
            compute_budget=budget or ComputeBudget(num_nodes=1, gpus_per_node=8),
        )

        return PhaseSpec(
            phase_id="validation",
            name="Validation Run",
            phase_type="validation",
            target=val_target,
            estimated_gpu_hours=hours,
            n_experiments=1,
            description=(
                f"Validate transferred HPs on {val_scale.hidden_size}d model "
                f"({val_scale.num_params_approx/1e6:.0f}M params)"
            ),
        )

    def _plan_full_training(
        self,
        model_size: ModelSize,
        target_scale: ModelScale,
        budget: Optional[ComputeBudget],
        target_loss: Optional[float],
        data_domains: Optional[List[str]],
    ) -> PhaseSpec:
        """Plan the full-scale training run."""
        tokens = int(target_scale.num_params_approx * 20)  # Chinchilla optimal
        flops = 6 * target_scale.num_params_approx * tokens
        n_gpus = (budget.total_gpus if budget else 64)
        hours = flops / (312 * 1e12 * 3600 * n_gpus)

        full_target = TrainingTarget(
            model_size=model_size,
            phase=TrainingPhase.PRETRAIN,
            target_loss=target_loss,
            target_tokens=tokens,
            data_domains=data_domains or [],
            compute_budget=budget or ComputeBudget(num_nodes=8, gpus_per_node=8),
        )

        return PhaseSpec(
            phase_id="full_training",
            name="Full-Scale Training",
            phase_type="full_training",
            target=full_target,
            estimated_gpu_hours=hours,
            n_experiments=1,
            description=(
                f"Full training at {target_scale.hidden_size}d "
                f"({target_scale.num_params_approx/1e9:.1f}B params, "
                f"{tokens/1e12:.1f}T tokens)"
            ),
        )

    def _plan_sft(self, model_size: ModelSize, budget: Optional[ComputeBudget]) -> PhaseSpec:
        """Plan supervised fine-tuning phase."""
        sft_target = TrainingTarget(
            model_size=model_size,
            phase=TrainingPhase.SFT,
            compute_budget=budget or ComputeBudget(num_nodes=1, gpus_per_node=8),
        )

        return PhaseSpec(
            phase_id="sft",
            name="Supervised Fine-Tuning",
            phase_type="sft",
            target=sft_target,
            estimated_gpu_hours=50.0,
            n_experiments=1,
            description="SFT on instruction-following data",
        )
