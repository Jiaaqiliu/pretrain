"""Integration tests for HoloPretrain core flow.

Tests the full pipeline WITHOUT actual K8s:
- State machine transitions
- Failure diagnosis
- Recovery strategy selection
- Self-diagnosis learning
- Anomaly detection
- Mixture mutation
- Script generation
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

from autopretrain.core.types import (
    ActionType,
    Budget,
    DataMixSpec,
    FailureType,
    Heartbeat,
    JobState,
    MetricSnapshot,
    TrialConfig,
)
from autopretrain.engine.olmo_adapter import OLMoAdapter
from autopretrain.monitor.anomaly_detector import AnomalyDetector
from autopretrain.orchestrator.state_machine import (
    InvalidTransitionError,
    JobStateMachine,
)
from autopretrain.resilience.diagnoser import MultiLayerDiagnoser
from autopretrain.resilience.heartbeat import FSxHeartbeatReader
from autopretrain.resilience.recovery import AdaptiveRecoveryStrategy
from autopretrain.resilience.self_diagnoser import SelfDiagnoser
from autopretrain.search.mixture import DataMixture, REFERENCE_MIXTURES
from autopretrain.search.mutator import HyperparamMutator, MixtureMutator
from autopretrain.store.event_log import EventLog


def test_state_machine_full_lifecycle():
    """Test complete happy path through state machine."""
    sm = JobStateMachine()
    config = TrialConfig(trial_id="t-001", name="test")
    sm.register_trial(config)

    sm.transition("t-001", JobState.PENDING, "submitted")
    sm.transition("t-001", JobState.RUNNING, "pod running")
    sm.transition("t-001", JobState.COMPLETED, "done")

    assert sm.get_trial("t-001").state == JobState.COMPLETED
    assert sm.all_done()
    assert len(sm.get_trial("t-001").events) == 4  # register + 3 transitions


def test_state_machine_failure_recovery():
    """Test failure → diagnose → recover → resubmit flow."""
    sm = JobStateMachine()
    config = TrialConfig(trial_id="t-002", name="test")
    sm.register_trial(config)

    sm.transition("t-002", JobState.PENDING)
    sm.transition("t-002", JobState.RUNNING)
    sm.transition("t-002", JobState.DIAGNOSING, "heartbeat dead")
    sm.transition("t-002", JobState.RECOVERING, "OOM diagnosed")
    sm.transition("t-002", JobState.PENDING, "retry 1")
    sm.transition("t-002", JobState.RUNNING, "pod running")
    sm.transition("t-002", JobState.COMPLETED)

    trial = sm.get_trial("t-002")
    assert trial.state == JobState.COMPLETED


def test_state_machine_invalid_transition():
    """Test that invalid transitions are rejected."""
    sm = JobStateMachine()
    config = TrialConfig(trial_id="t-003", name="test")
    sm.register_trial(config)

    try:
        sm.transition("t-003", JobState.RUNNING)  # Can't go CREATED → RUNNING
        assert False, "Should have raised"
    except InvalidTransitionError:
        pass

    # Valid path
    sm.transition("t-003", JobState.PENDING)
    try:
        sm.transition("t-003", JobState.COMPLETED)  # Can't go PENDING → COMPLETED
        assert False, "Should have raised"
    except InvalidTransitionError:
        pass


def test_diagnoser_all_types():
    """Test that diagnoser classifies all known failure types."""
    diag = MultiLayerDiagnoser()

    cases = [
        ("CUDA out of memory. Tried to allocate 2.00 GiB", FailureType.OOM),
        ("NCCL WARN Timeout after 300000 ms", FailureType.NCCL_TIMEOUT),
        ("Connection closed by peer", FailureType.NETWORK_DISCONNECT),
        ("FileNotFoundError: /fsx/data/missing.bin", FailureType.DATA_ERROR),
        ("ImportError: No module named 'nonexistent'", FailureType.CODE_BUG),
        ("node ip-10-0-1-5 is NotReady", FailureType.PREEMPTION),
        ("Insufficient nvidia.com/gpu", FailureType.RESOURCE_UNAVAILABLE),
    ]

    for log_text, expected_type in cases:
        result = diag.diagnose(log_text)
        assert result.failure_type == expected_type, \
            f"Expected {expected_type.value} for '{log_text[:50]}', got {result.failure_type.value}"

    print(f"  All {len(cases)} diagnosis cases passed")


def test_diagnoser_silent_hang():
    """Test silent hang detection via dead heartbeat."""
    diag = MultiLayerDiagnoser()
    dead_hb = Heartbeat(
        trial_id="t-004",
        timestamp=time.time() - 600,  # 10 min ago
        step=2341,
    )

    result = diag.diagnose("", heartbeat=dead_hb)
    assert result.failure_type == FailureType.SILENT_HANG


def test_diagnoser_correlation():
    """Test cluster-wide failure detection."""
    diag = MultiLayerDiagnoser()
    result = diag.diagnose("some generic error", concurrent_failures=3)
    assert result.failure_type == FailureType.NETWORK_DISCONNECT


def test_recovery_escalation():
    """Test OOM recovery escalates across retries."""
    recovery = AdaptiveRecoveryStrategy()
    config = TrialConfig(rank_microbatch_size=4)

    # First OOM → reduce batch
    r0 = recovery.select_action(FailureType.OOM, config, retry_count=0)
    assert r0.action_type == ActionType.REDUCE_BATCH
    assert r0.config_changes["rank_microbatch_size"] == 2

    # Second OOM → enable grad ckpt
    r1 = recovery.select_action(FailureType.OOM, config, retry_count=1)
    assert r1.action_type == ActionType.ENABLE_GRAD_CKPT

    # Third OOM → aggressive reduction
    r2 = recovery.select_action(FailureType.OOM, config, retry_count=2)
    assert r2.action_type == ActionType.REDUCE_BATCH

    # Fourth OOM → alert human
    r3 = recovery.select_action(FailureType.OOM, config, retry_count=3)
    assert r3.action_type == ActionType.ALERT_HUMAN

    print("  OOM escalation: reduce→grad_ckpt→aggressive→alert")


def test_recovery_non_retryable():
    """Test non-retryable failures go straight to alert."""
    recovery = AdaptiveRecoveryStrategy()
    config = TrialConfig()

    r = recovery.select_action(FailureType.CODE_BUG, config, retry_count=0)
    assert r.action_type == ActionType.ALERT_HUMAN


def test_recovery_node_exclusion():
    """Test that repeated failures on same node trigger exclusion."""
    recovery = AdaptiveRecoveryStrategy()
    config = TrialConfig()

    recovery.select_action(FailureType.NETWORK_DISCONNECT, config, retry_count=0, node_name="node-a")
    recovery.select_action(FailureType.NETWORK_DISCONNECT, config, retry_count=1, node_name="node-a")

    assert "node-a" in recovery.excluded_nodes


def test_self_diagnoser_learning():
    """Test that self-diagnoser learns from outcomes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_path = Path(tmpdir) / "learned.json"
        sd = SelfDiagnoser(knowledge_path=kb_path)

        # Simulate a successful fix
        from autopretrain.core.types import RecoveryAction
        action = RecoveryAction(
            action_type=ActionType.REDUCE_BATCH,
            description="test fix",
            config_changes={"rank_microbatch_size": 2},
        )

        signature = "RuntimeError: memory allocation <MEM> failed"
        sd.learn_from_outcome(signature, action, success=True)

        # Verify it was learned
        assert signature in sd._learned_fixes
        assert sd._learned_fixes[signature].success_count == 1

        # Verify persistence
        assert kb_path.exists()
        data = json.loads(kb_path.read_text())
        assert len(data) == 1

    print("  Self-diagnosis learning + persistence passed")


def test_anomaly_detector_spike():
    """Test loss spike detection."""
    import random
    detector = AnomalyDetector()

    # Need non-zero variance in history for spike detection
    history = [
        MetricSnapshot(step=i, timestamp=time.time(), metrics={"train/ce_loss": 3.0 + random.gauss(0, 0.05)})
        for i in range(50)
    ]
    spike = MetricSnapshot(step=50, timestamp=time.time(), metrics={"train/ce_loss": 8.0})

    anomalies = detector.check(spike, history)
    assert len(anomalies) >= 1
    assert any(a.type.value == "loss_spike" for a in anomalies)


def test_anomaly_detector_nan():
    """Test NaN detection."""
    detector = AnomalyDetector()
    nan_snap = MetricSnapshot(step=100, timestamp=time.time(), metrics={"train/ce_loss": float("nan")})
    anomalies = detector.check(nan_snap, [])
    assert len(anomalies) == 1
    assert anomalies[0].type.value == "nan_detected"


def test_anomaly_detector_throughput():
    """Test throughput drop detection."""
    detector = AnomalyDetector()
    history = [
        MetricSnapshot(step=i, timestamp=time.time(), metrics={"throughput/tokens_per_second": 100000.0})
        for i in range(20)
    ]
    drop = MetricSnapshot(step=20, timestamp=time.time(), metrics={"throughput/tokens_per_second": 30000.0})
    anomalies = detector.check(drop, history)
    assert any(a.type.value == "throughput_drop" for a in anomalies)


def test_mixture_operations():
    """Test data mixture math."""
    mix = DataMixture.from_reference("llama3")
    assert abs(sum(mix.weights.values()) - 1.0) < 1e-6

    # Perturbation maintains simplex
    perturbed = mix.perturb("math", 0.1)
    assert abs(sum(perturbed.weights.values()) - 1.0) < 1e-6
    assert perturbed.get("math") > mix.get("math")

    # Interpolation
    uniform = DataMixture.from_reference("uniform")
    mid = mix.interpolate(uniform, 0.5)
    assert abs(sum(mid.weights.values()) - 1.0) < 1e-6

    # Distance
    d = mix.distance(uniform)
    assert d > 0
    assert mix.distance(mix) < 1e-10

    print(f"  Mixture ops: perturb, interpolate, distance all correct")


def test_mutator():
    """Test mixture and hyperparam mutation."""
    mix = DataMixture.from_reference("llama3")
    mutator = MixtureMutator()

    # All strategies should maintain simplex
    for strategy in ["random_perturb", "domain_boost", "reference_interpolate", "swap_emphasis", "uniform_drift"]:
        result = mutator.mutate(mix, strategy=strategy)
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-6, f"{strategy} broke simplex: {total}"

    # Hyperparam mutation
    hp_mutator = HyperparamMutator()
    config = TrialConfig(
        data_mix=DataMixSpec(sources=dict(mix.weights), paths={}),
        lr=3e-4,
        global_batch_size=128,
    )
    mutated = hp_mutator.mutate_trial(config)
    assert mutated.trial_id != config.trial_id
    assert abs(sum(mutated.data_mix.sources.values()) - 1.0) < 1e-6

    print("  Mutation: all strategies maintain simplex, hyperparam mutation works")


def test_olmo_adapter_script_generation():
    """Test that OLMo adapter generates valid scripts."""
    adapter = OLMoAdapter(olmo_core_path="/tmp/olmo-core")

    config = TrialConfig(
        trial_id="test-gen",
        name="gen_test",
        model_factory="olmo2_1B",
        max_steps=100,
        data_mix=DataMixSpec(
            sources={"web": 0.5, "code": 0.3, "math": 0.2},
            paths={
                "web": "/data/web/*.bin",
                "code": "/data/code/*.bin",
                "math": "/data/math/*.bin",
            },
        ),
        save_folder="/tmp/checkpoints/test",
    )

    script = adapter.generate_script(config)
    assert "TransformerConfig.olmo2_1B" in script
    assert "SkipStepAdamWConfig" in script
    assert "StabilityMonitorCallback" in script
    assert "glob.glob" in script
    assert "domain_ratios" in script
    assert "heartbeat" in script.lower()

    manifest = adapter.generate_manifest(config, script)
    assert "nvidia.com/gpu: 8" in manifest
    assert "autopretrain-agent" in manifest or "test-gen" in manifest
    assert "activeDeadlineSeconds" in manifest

    print("  Script generation: all callbacks, SkipStep, SourceMixture, heartbeat present")


def test_heartbeat_reader():
    """Test heartbeat file reading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a heartbeat file
        hb_data = {
            "trial_id": "t-005",
            "timestamp": time.time(),
            "step": 1000,
            "loss": 2.5,
            "status": "training",
        }
        Path(tmpdir, "t-005.json").write_text(json.dumps(hb_data))

        reader = FSxHeartbeatReader(tmpdir)
        hb = asyncio.run(reader.read_heartbeat("t-005"))

        assert hb is not None
        assert hb.step == 1000
        assert hb.loss == 2.5
        assert not hb.is_stale
        assert not hb.is_dead

        # Non-existent trial
        none_hb = asyncio.run(reader.read_heartbeat("nonexistent"))
        assert none_hb is None

    print("  Heartbeat reader: read, stale check, dead check all work")


def test_event_log():
    """Test event log persistence and query."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "events.jsonl"
        log = EventLog(log_path)

        from autopretrain.core.types import JobEvent
        log.append(JobEvent(trial_id="t-006", event_type="submitted", from_state="created", to_state="pending"))
        log.append(JobEvent(trial_id="t-006", event_type="started", from_state="pending", to_state="running"))
        log.append(JobEvent(trial_id="t-007", event_type="submitted", from_state="created", to_state="pending"))

        # Query all
        all_events = log.query(limit=100)
        assert len(all_events) == 3

        # Query by trial
        t6_events = log.query(trial_id="t-006")
        assert len(t6_events) == 2

        # Tail
        recent = log.tail(n=2)
        assert len(recent) == 2

    print("  Event log: append, query, filter all work")


def test_budget_tracking():
    """Test budget consumption and limits."""
    budget = Budget(max_gpu_hours=10.0, max_trials=5, max_retries_total=10)

    assert budget.has_budget
    budget.consume_gpu_hours(5.0)
    assert budget.has_budget
    assert budget.utilization_pct == 50.0

    budget.consume_gpu_hours(5.1)
    assert not budget.has_budget  # Exceeded


def run_all_tests():
    """Run all integration tests."""
    tests = [
        test_state_machine_full_lifecycle,
        test_state_machine_failure_recovery,
        test_state_machine_invalid_transition,
        test_diagnoser_all_types,
        test_diagnoser_silent_hang,
        test_diagnoser_correlation,
        test_recovery_escalation,
        test_recovery_non_retryable,
        test_recovery_node_exclusion,
        test_self_diagnoser_learning,
        test_anomaly_detector_spike,
        test_anomaly_detector_nan,
        test_anomaly_detector_throughput,
        test_mixture_operations,
        test_mutator,
        test_olmo_adapter_script_generation,
        test_heartbeat_reader,
        test_event_log,
        test_budget_tracking,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    success = run_all_tests()
    sys.exit(0 if success else 1)
