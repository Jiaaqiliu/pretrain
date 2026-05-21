"""Tests for the agent orchestrator and decision engine."""

import tempfile


from autopilot.agent.decision import (
    ActionType,
    DecisionContext,
    DecisionEngine,
)
from autopilot.agent.orchestrator import AgentConfig, AutoPilotAgent, AutonomyLevel, CampaignStatus
from autopilot.agent.planner import PlannerAgent, TrainingPlan
from autopilot.backends.base import JobConfig, JobHandle, JobMetrics, JobStatus
from autopilot.experiment.config_builder import ComputeBudget, ModelSize
from autopilot.monitoring.anomaly import AnomalyType, Severity, TrainingAnomaly
from autopilot.optimization.early_stopping import StopReason, StoppingDecision
from autopilot.utils.persistence import StateStore


class MockBackend:
    """Mock compute backend for testing."""

    def __init__(self):
        self._jobs = {}
        self._job_counter = 0

    @property
    def name(self) -> str:
        return "mock"

    def submit_job(self, config: JobConfig) -> JobHandle:
        self._job_counter += 1
        handle = JobHandle(
            job_id=f"mock_{self._job_counter}",
            backend="mock",
            name=config.name,
            status=JobStatus.RUNNING,
        )
        self._jobs[handle.job_id] = handle
        return handle

    def cancel_job(self, handle: JobHandle) -> None:
        handle.status = JobStatus.CANCELLED

    def get_status(self, handle: JobHandle) -> JobStatus:
        return handle.status

    def get_logs(self, handle: JobHandle, tail: int = 100) -> str:
        return "step=100, loss=2.5, lr=3e-4"

    def get_metrics(self, handle: JobHandle) -> JobMetrics:
        return JobMetrics(step=100, loss=2.5, learning_rate=3e-4, grad_norm=1.2)

    def list_jobs(self, status=None, tags=None):
        return list(self._jobs.values())

    def get_available_resources(self):
        return {"gpus": 64, "nodes": 8}

    def stream_logs(self, handle: JobHandle):
        yield "step=100, loss=2.5"


class TestDecisionEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = StateStore(self.tmpdir)
        self.engine = DecisionEngine(store=self.store, autonomy_level="semi")

    def test_handles_critical_anomaly(self):
        context = DecisionContext(
            experiment_id="exp_001",
            anomalies=[
                TrainingAnomaly(
                    anomaly_type=AnomalyType.LOSS_SPIKE,
                    severity=Severity.CRITICAL,
                    step=1000,
                    value=15.0,
                    threshold=5.0,
                    message="Critical loss spike",
                    suggested_action="rollback_checkpoint",
                )
            ],
        )

        actions = self.engine.decide(context)
        assert len(actions) > 0
        assert actions[0].action_type == ActionType.ROLLBACK_CHECKPOINT

    def test_handles_stopping_decision(self):
        context = DecisionContext(
            experiment_id="exp_002",
            stopping_decision=StoppingDecision(
                should_stop=True,
                reason=StopReason.DIVERGING,
                confidence=0.9,
                message="Loss is diverging",
            ),
        )

        actions = self.engine.decide(context)
        assert len(actions) > 0
        assert actions[0].action_type == ActionType.EARLY_STOP

    def test_full_autonomy_executes_immediately(self):
        engine = DecisionEngine(store=self.store, autonomy_level="full")
        context = DecisionContext(
            experiment_id="exp_003",
            anomalies=[
                TrainingAnomaly(
                    anomaly_type=AnomalyType.LOSS_SPIKE,
                    severity=Severity.CRITICAL,
                    step=500,
                    value=10.0,
                    threshold=5.0,
                    message="Critical spike",
                    suggested_action="rollback",
                )
            ],
        )

        actions = engine.decide(context)
        # In full autonomy, all actions should be executable
        for action in actions:
            assert not action.requires_confirmation

    def test_advisory_never_executes(self):
        engine = DecisionEngine(store=self.store, autonomy_level="advisory")
        context = DecisionContext(
            experiment_id="exp_004",
            anomalies=[
                TrainingAnomaly(
                    anomaly_type=AnomalyType.GRADIENT_EXPLOSION,
                    severity=Severity.HIGH,
                    step=200,
                    value=100.0,
                    threshold=10.0,
                    message="Gradient explosion",
                    suggested_action="reduce_lr",
                )
            ],
        )

        actions = engine.decide(context)
        # In advisory mode, all actions require confirmation
        for action in actions:
            assert action.requires_confirmation


class TestPlannerAgent:
    def test_generates_valid_plan(self):
        planner = PlannerAgent()
        plan = planner.plan(
            model_size=ModelSize.LARGE,
            data_domains=["web", "code", "math"],
            compute_budget=ComputeBudget(num_nodes=8, gpus_per_node=8),
        )

        assert isinstance(plan, TrainingPlan)
        assert len(plan.phases) >= 3  # at least proxy_search, validation, full_training
        assert plan.total_estimated_gpu_hours > 0
        assert plan.target_model_size == ModelSize.LARGE

    def test_plan_phases_have_dependencies(self):
        planner = PlannerAgent()
        plan = planner.plan(model_size=ModelSize.MEDIUM, data_domains=["web", "code"])

        # Later phases should depend on earlier ones
        for i in range(1, len(plan.phases)):
            phase = plan.phases[i]
            assert len(phase.depends_on) > 0 or i == 0

    def test_skip_proxy_search(self):
        planner = PlannerAgent()
        plan = planner.plan(model_size=ModelSize.MEDIUM, skip_proxy_search=True)

        phase_types = [p.phase_type for p in plan.phases]
        assert "proxy_search" not in phase_types


class TestAutoPilotAgent:
    def test_agent_creation(self):
        backend = MockBackend()
        config = AgentConfig(
            store_dir=tempfile.mkdtemp(),
            model_size=ModelSize.SMALL,
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        agent = AutoPilotAgent(config=config, backend=backend)
        assert agent.status == CampaignStatus.PLANNING

    def test_agent_start(self):
        backend = MockBackend()
        config = AgentConfig(
            store_dir=tempfile.mkdtemp(),
            model_size=ModelSize.SMALL,
            data_domains=["web", "code"],
            autonomy_level=AutonomyLevel.FULL,
            max_parallel_experiments=4,
        )
        agent = AutoPilotAgent(config=config, backend=backend)
        plan = agent.start()

        assert plan is not None
        assert agent.status in (CampaignStatus.PROXY_SEARCH, CampaignStatus.MIXTURE_OPTIMIZATION)
        assert len(agent.active_experiments) > 0

    def test_agent_step(self):
        backend = MockBackend()
        config = AgentConfig(
            store_dir=tempfile.mkdtemp(),
            model_size=ModelSize.TINY,
            autonomy_level=AutonomyLevel.FULL,
        )
        agent = AutoPilotAgent(config=config, backend=backend)
        agent.start()

        result = agent.step()
        assert "status" in result
        assert "timestamp" in result

    def test_agent_dashboard_data(self):
        backend = MockBackend()
        config = AgentConfig(
            store_dir=tempfile.mkdtemp(),
            model_size=ModelSize.TINY,
            autonomy_level=AutonomyLevel.SEMI,
        )
        agent = AutoPilotAgent(config=config, backend=backend)
        agent.start()

        dashboard = agent.get_dashboard_data()
        assert "status" in dashboard
        assert "active_experiments" in dashboard
        assert "rankings" in dashboard
