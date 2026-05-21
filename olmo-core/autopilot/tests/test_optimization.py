"""Tests for optimization modules."""

import pytest

from autopilot.optimization.data_mixing import DataDomain, DataMixingOptimizer, MixtureWeights
from autopilot.optimization.early_stopping import EarlyStoppingStrategy, StopReason
from autopilot.optimization.hpo import HPOEngine, SearchSpace
from autopilot.optimization.mu_transfer import ModelScale, MuTransferConfig, MuTransferEngine


class TestHPOEngine:
    def test_suggest_and_report(self):
        space = SearchSpace.for_llm_pretraining()
        engine = HPOEngine(search_space=space, n_trials=10, seed=42)

        trial = engine.suggest_next()
        assert trial.trial_id == 0
        assert "learning_rate" in trial.params
        assert "weight_decay" in trial.params

        engine.report_complete(trial.trial_id, 2.5)
        assert engine.n_completed == 1
        assert engine.best_value == 2.5

    def test_multiple_trials(self):
        space = SearchSpace()
        space.add_float("lr", 1e-5, 1e-2, log=True)
        space.add_float("wd", 0.0, 0.3)

        engine = HPOEngine(search_space=space, n_trials=20, seed=123)

        for i in range(5):
            trial = engine.suggest_next()
            # Simulate: lower lr is better
            value = trial.params["lr"] * 1000
            engine.report_complete(trial.trial_id, value)

        assert engine.n_completed == 5
        best = engine.best_params
        assert best is not None

    def test_pruning(self):
        space = SearchSpace()
        space.add_float("x", 0.0, 10.0)

        engine = HPOEngine(search_space=space, n_trials=10)
        trial = engine.suggest_next()

        # Report bad intermediate values
        for step in range(10):
            should_prune = engine.report_intermediate(trial.trial_id, step, 100.0)
            if should_prune:
                break

    def test_importance(self):
        space = SearchSpace()
        space.add_float("important_param", 0.0, 1.0)
        space.add_float("noise_param", 0.0, 1.0)

        engine = HPOEngine(search_space=space, n_trials=50)

        for _ in range(20):
            trial = engine.suggest_next()
            # Only important_param affects the objective
            value = trial.params["important_param"] ** 2
            engine.report_complete(trial.trial_id, value)

        importance = engine.get_importance()
        if importance:
            assert "important_param" in importance


class TestMuTransfer:
    def test_proxy_design(self):
        target = ModelScale(hidden_size=4096, num_layers=32, num_heads=32)
        proxy = MuTransferEngine.design_proxy(target, width_divisor=4)

        assert proxy.hidden_size == 1024
        assert proxy.num_layers == 32  # same depth
        assert proxy.num_heads == 8

    def test_hp_transfer(self):
        proxy = ModelScale(hidden_size=1024, num_layers=32, num_heads=8)
        target = ModelScale(hidden_size=4096, num_layers=32, num_heads=32)

        config = MuTransferConfig(proxy_scale=proxy, target_scale=target)
        engine = MuTransferEngine(config)

        proxy_params = {"learning_rate": 1e-3, "weight_decay": 0.1, "beta1": 0.9}
        transferred = engine.transfer_hyperparameters(proxy_params)

        # LR should be scaled down by width ratio (4096/1024 = 4)
        assert transferred["learning_rate"] == pytest.approx(1e-3 / 4.0)
        # Weight decay should be preserved
        assert transferred["weight_decay"] == 0.1
        # Betas should be preserved
        assert transferred["beta1"] == 0.9

    def test_proxy_result_tracking(self):
        proxy = ModelScale(hidden_size=512, num_layers=12, num_heads=8)
        target = ModelScale(hidden_size=2048, num_layers=12, num_heads=16)

        engine = MuTransferEngine(MuTransferConfig(proxy_scale=proxy, target_scale=target))
        engine.add_proxy_result({"learning_rate": 3e-4}, 3.0)
        engine.add_proxy_result({"learning_rate": 1e-3}, 2.8)
        engine.add_proxy_result({"learning_rate": 5e-4}, 2.7)

        best = engine.get_best_proxy_params()
        assert best["learning_rate"] == 5e-4

        transferred = engine.get_transferred_best()
        assert transferred["learning_rate"] < 5e-4  # should be scaled down


class TestDataMixing:
    def test_uniform_mixture(self):
        domains = [
            DataDomain(name="web", path="/data/web", token_count=int(1e12)),
            DataDomain(name="code", path="/data/code", token_count=int(2e11)),
            DataDomain(name="math", path="/data/math", token_count=int(5e10)),
        ]
        optimizer = DataMixingOptimizer(domains)
        mixture = optimizer.uniform_mixture()

        assert abs(sum(mixture.weights.values()) - 1.0) < 1e-6
        assert abs(mixture.weights["web"] - 1 / 3) < 1e-6

    def test_token_proportional(self):
        domains = [
            DataDomain(name="web", path="/data/web", token_count=int(800e9)),
            DataDomain(name="code", path="/data/code", token_count=int(200e9)),
        ]
        optimizer = DataMixingOptimizer(domains)
        mixture = optimizer.token_proportional_mixture()

        assert mixture.weights["web"] == pytest.approx(0.8, abs=1e-6)
        assert mixture.weights["code"] == pytest.approx(0.2, abs=1e-6)

    def test_online_adjustment(self):
        domains = [
            DataDomain(name="web", path="/data/web", token_count=int(1e12)),
            DataDomain(name="code", path="/data/code", token_count=int(1e12)),
        ]
        optimizer = DataMixingOptimizer(domains)
        initial = optimizer.uniform_mixture()

        # Code domain has higher loss -> should be upweighted
        adjusted = optimizer.adjust_online(
            initial, domain_losses={"web": 2.0, "code": 3.0}, step_size=0.1
        )

        assert adjusted.weights["code"] > adjusted.weights["web"]

    def test_mixture_normalization(self):
        mixture = MixtureWeights(weights={"a": 2.0, "b": 3.0, "c": 5.0})
        assert abs(sum(mixture.weights.values()) - 1.0) < 1e-6
        assert mixture.weights["c"] == pytest.approx(0.5)

    def test_blend(self):
        m1 = MixtureWeights(weights={"web": 0.8, "code": 0.2})
        m2 = MixtureWeights(weights={"web": 0.4, "code": 0.6})

        blended = m1.blend(m2, alpha=0.5)
        assert blended.weights["web"] == pytest.approx(0.6, abs=0.01)
        assert blended.weights["code"] == pytest.approx(0.4, abs=0.01)


class TestEarlyStopping:
    def test_diverging_loss(self):
        from autopilot.monitoring.metrics import MetricsSnapshot, MetricsWindow

        strategy = EarlyStoppingStrategy(total_steps=10000, patience=500)
        window = MetricsWindow(window_size=256)

        # Simulate diverging training
        for i in range(200):
            loss = 2.5 + i * 0.05  # steadily increasing
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": loss}))

        decision = strategy.should_stop("exp_1", window)
        assert decision.should_stop
        assert decision.reason == StopReason.DIVERGING

    def test_plateau_detection(self):
        from autopilot.monitoring.metrics import MetricsSnapshot, MetricsWindow

        strategy = EarlyStoppingStrategy(
            total_steps=10000, patience=100, min_improvement=0.01
        )
        window = MetricsWindow(window_size=256)

        # Initial improvement — call should_stop each step to track best
        for i in range(50):
            loss = 3.0 - i * 0.02
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": loss}))
            strategy.should_stop("exp_1", window)

        # Then plateau — keep calling should_stop
        decision = None
        for i in range(50, 300):
            loss = 2.0 + 0.001 * (i % 5 - 2)  # essentially flat
            window.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": loss}))
            decision = strategy.should_stop("exp_1", window)
            if decision.should_stop:
                break

        assert decision is not None
        assert decision.should_stop
        assert decision.reason == StopReason.PLATEAU

    def test_asha_evaluation(self):
        from autopilot.monitoring.metrics import MetricsSnapshot, MetricsWindow

        strategy = EarlyStoppingStrategy(total_steps=10000)

        windows = {}
        # Create 6 experiments with different quality
        for idx, final_loss in enumerate([2.0, 2.5, 3.0, 3.5, 4.0, 4.5]):
            w = MetricsWindow(window_size=256)
            for i in range(600):
                loss = 4.0 - (4.0 - final_loss) * (i / 600)
                w.add(MetricsSnapshot(timestamp=float(i), step=i, metrics={"loss": loss}))
            windows[f"exp_{idx}"] = w

        decisions = strategy.asha_evaluate(windows, rung_step=500)
        # Should prune bottom 2/3 (keep top 2)
        pruned_ids = [eid for eid, _ in decisions]
        assert len(pruned_ids) >= 2  # at least some are pruned
