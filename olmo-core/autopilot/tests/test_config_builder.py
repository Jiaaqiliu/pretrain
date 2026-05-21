"""Tests for the configuration builder."""


from autopilot.experiment.config_builder import (
    ComputeBudget,
    ConfigBuilder,
    GeneratedConfig,
    ModelSize,
    TrainingTarget,
)


class TestConfigBuilder:
    def setup_method(self):
        self.builder = ConfigBuilder()

    def test_build_basic_config(self):
        target = TrainingTarget(model_size=ModelSize.MEDIUM)
        config = self.builder.build(target)

        assert isinstance(config, GeneratedConfig)
        assert config.model_config["hidden_size"] == 2048
        assert config.model_config["num_layers"] == 24
        assert config.optimizer_config["type"] == "adamw"
        assert config.trainer_config["max_steps"] > 0

    def test_builds_correct_model_sizes(self):
        for size in ModelSize:
            target = TrainingTarget(model_size=size)
            config = self.builder.build(target)
            assert config.model_config["hidden_size"] > 0
            assert config.model_config["num_layers"] > 0

    def test_respects_compute_budget(self):
        budget = ComputeBudget(num_nodes=4, gpus_per_node=8, gpu_type="H100")
        target = TrainingTarget(model_size=ModelSize.LARGE, compute_budget=budget)
        config = self.builder.build(target)

        assert config.launch_config["num_nodes"] == 4
        assert config.launch_config["gpus_per_node"] == 8

    def test_custom_hyperparameters(self):
        target = TrainingTarget(model_size=ModelSize.SMALL)
        params = {"learning_rate": 5e-4, "weight_decay": 0.05}
        config = self.builder.build(target, params=params)

        assert config.optimizer_config["lr"] == 5e-4
        assert config.optimizer_config["weight_decay"] == 0.05

    def test_target_tokens_respected(self):
        target = TrainingTarget(model_size=ModelSize.MEDIUM, target_tokens=int(1e11))
        config = self.builder.build(target)
        assert config.trainer_config["total_tokens"] == int(1e11)

    def test_auto_parallelism_selection(self):
        # Small model -> DDP
        small_target = TrainingTarget(model_size=ModelSize.SMALL)
        small_config = self.builder.build(small_target)
        assert small_config.launch_config["dp_strategy"] == "ddp"

        # Large model -> FSDP
        large_target = TrainingTarget(model_size=ModelSize.LARGE)
        large_config = self.builder.build(large_target)
        assert large_config.launch_config["dp_strategy"] == "fsdp"

    def test_olmo_overrides_generation(self):
        target = TrainingTarget(model_size=ModelSize.MEDIUM)
        params = {"learning_rate": 1e-3}
        config = self.builder.build(target, params=params)

        overrides = config.to_olmo_overrides()
        assert isinstance(overrides, list)
        # Should contain optimizer-related overrides
        assert any("lr" in o for o in overrides)
