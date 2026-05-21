"""AutoPilot Orchestrator — the central brain of the autonomous training system.

Coordinates all components:
- Planner: generates multi-phase training strategies
- Launcher: submits and manages jobs
- Monitor: observes running experiments
- Analyzer: extracts insights from results
- Decision Engine: makes autonomous decisions
- HPO Engine: optimizes hyperparameters

Supports long-running operation with state persistence and recovery.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from autopilot.agent.analyzer import AnalyzerAgent, AnalysisReport
from autopilot.agent.decision import Action, ActionType, DecisionContext, DecisionEngine
from autopilot.agent.planner import PlannerAgent, TrainingPlan
from autopilot.backends.base import ComputeBackend
from autopilot.experiment.config_builder import (
    ComputeBudget,
    ConfigBuilder,
    ModelSize,
    TrainingTarget,
)
from autopilot.experiment.launcher import ExperimentLauncher
from autopilot.experiment.monitor import ExperimentMonitor
from autopilot.monitoring.metrics import MetricsCollector
from autopilot.optimization.data_mixing import DataDomain, DataMixingOptimizer
from autopilot.optimization.hpo import HPOEngine, SearchSpace
from autopilot.optimization.mu_transfer import MuTransferConfig, MuTransferEngine
from autopilot.utils.logging import get_logger
from autopilot.utils.persistence import StateStore

log = get_logger("agent.orchestrator")


class AutonomyLevel(enum.Enum):
    FULL = "full"  # act on all decisions automatically
    SEMI = "semi"  # low-risk auto, high-risk confirm
    ADVISORY = "advisory"  # suggest only, never act


class CampaignStatus(enum.Enum):
    PLANNING = "planning"
    PROXY_SEARCH = "proxy_search"
    MIXTURE_OPTIMIZATION = "mixture_optimization"
    VALIDATION = "validation"
    FULL_TRAINING = "full_training"
    POST_TRAINING = "post_training"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class AgentConfig:
    """Configuration for the AutoPilot agent."""

    store_dir: str = "./autopilot_state"
    autonomy_level: AutonomyLevel = AutonomyLevel.SEMI
    poll_interval_seconds: int = 60
    max_parallel_experiments: int = 8
    model_size: ModelSize = ModelSize.MEDIUM
    target_loss: Optional[float] = None
    compute_budget: Optional[ComputeBudget] = None
    data_domains: List[str] = field(default_factory=list)
    search_space: Optional[SearchSpace] = None
    include_sft: bool = False
    notification_callback: Optional[Callable[[str, str], None]] = None


class AutoPilotAgent:
    """The main autonomous training agent.

    Usage:
        agent = AutoPilotAgent(config=AgentConfig(...), backend=my_backend)
        agent.start(target=TrainingTarget(...))
        # Agent runs autonomously, making decisions and managing experiments
        agent.run_loop()  # blocking, or call agent.step() periodically
    """

    def __init__(self, config: AgentConfig, backend: ComputeBackend):
        self._config = config
        self._backend = backend

        # Initialize persistence
        self._store = StateStore(config.store_dir)

        # Initialize sub-components
        self._metrics = MetricsCollector()
        self._planner = PlannerAgent()
        self._builder = ConfigBuilder()
        self._launcher = ExperimentLauncher(backend, self._store)
        self._monitor = ExperimentMonitor(
            backend,
            total_steps=100000,
            target_loss=config.target_loss,
            on_anomaly=self._on_anomaly,
            on_stop_decision=self._on_stop_decision,
        )
        self._analyzer = AnalyzerAgent(self._metrics)
        self._decision_engine = DecisionEngine(
            store=self._store,
            autonomy_level=config.autonomy_level.value,
            on_action=self._execute_action,
        )

        # HPO and optimization
        search_space = config.search_space or SearchSpace.for_llm_pretraining()
        self._hpo = HPOEngine(search_space=search_space)
        self._mu_transfer: Optional[MuTransferEngine] = None
        self._data_optimizer: Optional[DataMixingOptimizer] = None

        # State
        self._plan: Optional[TrainingPlan] = None
        self._status = CampaignStatus.PLANNING
        self._current_phase_idx = 0
        self._experiment_configs: Dict[str, Dict[str, Any]] = {}
        self._running = False

    @property
    def status(self) -> CampaignStatus:
        return self._status

    @property
    def plan(self) -> Optional[TrainingPlan]:
        return self._plan

    @property
    def active_experiments(self) -> List[str]:
        return self._launcher.active_experiments

    def start(self, target: Optional[TrainingTarget] = None) -> TrainingPlan:
        """Start a new training campaign.

        Generates a plan and begins execution from Phase 1.
        """
        log.info("Starting AutoPilot training campaign")

        # Generate training plan
        self._plan = self._planner.plan(
            model_size=self._config.model_size,
            compute_budget=self._config.compute_budget,
            target_loss=self._config.target_loss,
            data_domains=self._config.data_domains or None,
            include_sft=self._config.include_sft,
        )

        self._store.set_state("plan", {"name": self._plan.name, "num_phases": len(self._plan.phases)})
        self._store.set_state("status", CampaignStatus.PROXY_SEARCH.value)
        self._status = CampaignStatus.PROXY_SEARCH
        self._current_phase_idx = 0

        # Initialize muTransfer engine
        target_scale = self._config.model_size.to_scale()
        proxy_scale = MuTransferEngine.design_proxy(target_scale)
        self._mu_transfer = MuTransferEngine(
            MuTransferConfig(proxy_scale=proxy_scale, target_scale=target_scale)
        )

        # Initialize data optimizer if domains specified
        if self._config.data_domains:
            domains = [
                DataDomain(name=d, path=f"data/{d}", token_count=int(1e10))
                for d in self._config.data_domains
            ]
            self._data_optimizer = DataMixingOptimizer(domains)

        log.info(f"Plan created: {self._plan.name} ({len(self._plan.phases)} phases)")
        self._notify("Campaign Started", f"Plan: {self._plan.name}")

        # Launch first phase
        self._launch_current_phase()

        return self._plan

    def step(self) -> Dict[str, Any]:
        """Execute one monitoring/decision cycle.

        Call this periodically (e.g., every 60 seconds) to:
        1. Poll experiment metrics
        2. Detect anomalies
        3. Make decisions
        4. Check phase completion
        5. Advance to next phase if ready

        Returns a status dict summarizing what happened.
        """
        step_result = {
            "timestamp": time.time(),
            "status": self._status.value,
            "actions_taken": [],
            "anomalies": [],
            "phase": self._current_phase_idx,
        }

        if self._status in (CampaignStatus.COMPLETED, CampaignStatus.FAILED, CampaignStatus.PAUSED):
            return step_result

        # 1. Poll all experiments
        states = self._monitor.poll()

        # 2. Process each experiment's state
        for eid, state in states.items():
            # Build decision context
            context = DecisionContext(
                experiment_id=eid,
                anomalies=state.anomalies,
                stopping_decision=state.stopping_decision,
                current_step=state.last_metrics.step if state.last_metrics else 0,
                current_loss=(
                    state.last_metrics.metrics.get("loss") if state.last_metrics else None
                ),
                config=self._experiment_configs.get(eid, {}),
            )

            # 3. Get decisions
            actions = self._decision_engine.decide(context)
            for action in actions:
                if not action.requires_confirmation:
                    self._execute_action(action)
                    step_result["actions_taken"].append(action.action_type.value)

            step_result["anomalies"].extend(
                [a.anomaly_type.value for a in state.anomalies]
            )

        # 4. Check if current phase is complete
        if self._is_phase_complete():
            self._advance_phase()
            step_result["phase_advanced"] = True

        # 5. Refresh statuses
        self._launcher.refresh_all_statuses()

        return step_result

    def run_loop(self, max_iterations: Optional[int] = None) -> None:
        """Run the agent in a blocking loop until campaign completes."""
        self._running = True
        iteration = 0

        log.info("Entering main agent loop")
        while self._running:
            if max_iterations and iteration >= max_iterations:
                break

            try:
                self.step()
                iteration += 1

                if self._status == CampaignStatus.COMPLETED:
                    log.info("Campaign completed successfully!")
                    break
                elif self._status == CampaignStatus.FAILED:
                    log.error("Campaign failed.")
                    break

                time.sleep(self._config.poll_interval_seconds)

            except KeyboardInterrupt:
                log.info("Agent interrupted by user")
                self._status = CampaignStatus.PAUSED
                break
            except Exception as e:
                log.error(f"Error in agent loop: {e}")
                time.sleep(self._config.poll_interval_seconds * 2)

    def stop(self) -> None:
        """Gracefully stop the agent."""
        self._running = False
        self._status = CampaignStatus.PAUSED
        self._store.set_state("status", self._status.value)
        log.info("Agent stopped")

    def get_report(self) -> AnalysisReport:
        """Generate a comprehensive analysis report of the campaign."""
        return self._analyzer.analyze_sweep(self._experiment_configs, self._hpo)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for the dashboard display."""
        return {
            "status": self._status.value,
            "plan": self._plan.__dict__ if self._plan else None,
            "current_phase": self._current_phase_idx,
            "active_experiments": self._launcher.active_experiments,
            "rankings": self._monitor.get_rankings(),
            "pending_actions": [
                {"type": a.action_type.value, "experiment": a.experiment_id, "reasoning": a.reasoning}
                for a in self._decision_engine.pending_actions
            ],
            "hpo_best": self._hpo.best_params,
            "hpo_completed": self._hpo.n_completed,
        }

    # --- Internal methods ---

    def _launch_current_phase(self) -> None:
        """Launch experiments for the current phase."""
        if self._plan is None:
            return

        phase = self._plan.phases[self._current_phase_idx]
        log.info(f"Launching phase: {phase.name}")

        if phase.phase_type == "proxy_search":
            self._launch_proxy_search(phase)
        elif phase.phase_type == "mixture_search":
            self._launch_mixture_search(phase)
        elif phase.phase_type == "validation":
            self._launch_validation(phase)
        elif phase.phase_type == "full_training":
            self._launch_full_training(phase)
        elif phase.phase_type == "sft":
            self._launch_sft(phase)

    def _launch_proxy_search(self, phase) -> None:
        """Launch proxy model HP search experiments."""
        self._status = CampaignStatus.PROXY_SEARCH
        n_trials = min(phase.n_experiments, self._config.max_parallel_experiments)

        for _ in range(n_trials):
            trial = self._hpo.suggest_next()
            config = self._builder.build(phase.target, params=trial.params)
            eid = self._launcher.launch(config)
            self._experiment_configs[eid] = trial.params
            self._monitor.add_experiment(eid, self._launcher.get_handle(eid))

    def _launch_mixture_search(self, phase) -> None:
        """Launch data mixture optimization experiments."""
        self._status = CampaignStatus.MIXTURE_OPTIMIZATION
        if self._data_optimizer is None:
            return

        mixtures = self._data_optimizer.suggest_exploration_mixtures(
            n_suggestions=phase.n_experiments
        )
        for mixture in mixtures:
            params = {"data_mixture": mixture.weights}
            config = self._builder.build(phase.target, params=params)
            eid = self._launcher.launch(config)
            self._experiment_configs[eid] = params
            self._monitor.add_experiment(eid, self._launcher.get_handle(eid))

    def _launch_validation(self, phase) -> None:
        """Launch validation run with best found configuration."""
        self._status = CampaignStatus.VALIDATION

        # Get best params from proxy search
        best_params = self._hpo.best_params
        if self._mu_transfer and best_params:
            transferred = self._mu_transfer.transfer_hyperparameters(best_params)
        else:
            transferred = best_params or {}

        config = self._builder.build(phase.target, params=transferred)
        eid = self._launcher.launch(config)
        self._experiment_configs[eid] = transferred
        self._monitor.add_experiment(eid, self._launcher.get_handle(eid))

    def _launch_full_training(self, phase) -> None:
        """Launch the full-scale training run."""
        self._status = CampaignStatus.FULL_TRAINING

        best_params = self._hpo.best_params or {}
        if self._mu_transfer:
            best_params = self._mu_transfer.transfer_hyperparameters(best_params)

        config = self._builder.build(phase.target, params=best_params)
        eid = self._launcher.launch(config)
        self._experiment_configs[eid] = best_params
        self._monitor.add_experiment(eid, self._launcher.get_handle(eid))

    def _launch_sft(self, phase) -> None:
        """Launch supervised fine-tuning."""
        self._status = CampaignStatus.POST_TRAINING
        config = self._builder.build(phase.target)
        eid = self._launcher.launch(config)
        self._monitor.add_experiment(eid, self._launcher.get_handle(eid))

    def _is_phase_complete(self) -> bool:
        """Check if all experiments in the current phase are done."""
        active = self._launcher.active_experiments
        if not active:
            return True

        statuses = self._launcher.refresh_all_statuses()
        all_done = all(
            s in ("completed", "failed", "stopped") for s in statuses.values()
        )
        return all_done

    def _advance_phase(self) -> None:
        """Move to the next phase in the plan."""
        if self._plan is None:
            return

        self._current_phase_idx += 1
        if self._current_phase_idx >= len(self._plan.phases):
            self._status = CampaignStatus.COMPLETED
            self._store.set_state("status", self._status.value)
            self._notify("Campaign Complete", "All phases finished successfully")
            log.info("All phases complete!")
            return

        phase = self._plan.phases[self._current_phase_idx]
        log.info(f"Advancing to phase {self._current_phase_idx}: {phase.name}")
        self._store.set_state("current_phase", self._current_phase_idx)
        self._launch_current_phase()

    def _execute_action(self, action: Action) -> bool:
        """Execute a decided action."""
        log.info(f"Executing action: {action.action_type.value} on {action.experiment_id}")

        if action.action_type == ActionType.EARLY_STOP:
            self._launcher.cancel(action.experiment_id)
            self._monitor.remove_experiment(action.experiment_id)
            return True

        elif action.action_type == ActionType.ROLLBACK_CHECKPOINT:
            # Cancel current, resume from checkpoint
            self._launcher.cancel(action.experiment_id)
            # In production, this would find the last good checkpoint
            return True

        elif action.action_type == ActionType.REDUCE_LR:
            # Would modify the running experiment's LR
            factor = action.parameters.get("lr_factor", 0.5)
            log.info(f"Reducing LR by factor {factor} for {action.experiment_id}")
            return True

        elif action.action_type == ActionType.NOTIFY_USER:
            self._notify(
                f"Action needed: {action.experiment_id}",
                action.reasoning,
            )
            return True

        elif action.action_type == ActionType.SKIP_STEP:
            log.info(f"Skipping step for {action.experiment_id}")
            return True

        return False

    def _on_anomaly(self, experiment_id: str, anomaly) -> None:
        """Callback when an anomaly is detected."""
        log.warning(f"Anomaly in {experiment_id}: {anomaly.message}")

    def _on_stop_decision(self, experiment_id: str, decision) -> None:
        """Callback when early stopping is recommended."""
        log.info(f"Stop recommended for {experiment_id}: {decision.message}")

    def _notify(self, title: str, message: str) -> None:
        """Send notification via configured callback."""
        if self._config.notification_callback:
            self._config.notification_callback(title, message)
        log.info(f"[NOTIFICATION] {title}: {message}")
