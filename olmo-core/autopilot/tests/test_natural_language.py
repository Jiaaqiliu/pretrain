"""Tests for the natural language interface."""

import pytest

from autopilot.agent.natural_language import NaturalLanguageInterface
from autopilot.experiment.config_builder import ModelSize, TrainingPhase


class TestNaturalLanguageInterface:
    def setup_method(self):
        self.nl = NaturalLanguageInterface()

    def test_parse_train_command(self):
        intent = self.nl.parse("Train a 7B model on web and code data")
        assert intent.action == "train"
        assert intent.model_size == ModelSize.LARGE  # 7B
        assert "web" in intent.data_config["domains"]
        assert "code" in intent.data_config["domains"]

    def test_parse_model_sizes(self):
        assert self.nl.parse("Train a 190M model").model_size == ModelSize.SMALL
        assert self.nl.parse("Train a 1B model").model_size == ModelSize.MEDIUM
        assert self.nl.parse("Train a 7B model").model_size == ModelSize.LARGE
        assert self.nl.parse("Train a 13B model").model_size == ModelSize.XL
        assert self.nl.parse("Train a 70B model").model_size == ModelSize.XXL

    def test_parse_compute_config(self):
        intent = self.nl.parse("Use 4 nodes with 8 H100 GPUs")
        assert intent.compute_config is not None
        assert intent.compute_config["num_nodes"] == 4
        assert intent.compute_config["gpu_type"] == "H100"

    def test_parse_target_loss(self):
        intent = self.nl.parse("Train to target loss of 2.5")
        assert intent.target_loss == pytest.approx(2.5)

    def test_parse_token_count(self):
        intent = self.nl.parse("Train for 2T tokens")
        assert intent.target_tokens == int(2e12)

        intent = self.nl.parse("Train for 500B tokens")
        assert intent.target_tokens == int(500e9)

    def test_parse_phase(self):
        intent = self.nl.parse("Fine-tune the model on instructions")
        assert intent.phase == TrainingPhase.SFT

        intent = self.nl.parse("Do RLHF training")
        assert intent.phase == TrainingPhase.RLHF

        intent = self.nl.parse("Pretrain from scratch")
        assert intent.phase == TrainingPhase.PRETRAIN

    def test_parse_stop_command(self):
        intent = self.nl.parse("Stop the failing experiment")
        assert intent.action == "stop"

    def test_parse_status_query(self):
        intent = self.nl.parse("Show me the current loss curves")
        assert intent.action == "status"

    def test_parse_data_spec(self):
        spec = self.nl.parse_data_spec(
            "I have 500B tokens of web data and 100B tokens of code"
        )
        assert len(spec.available_domains) == 2
        web = next(d for d in spec.available_domains if d.name == "web")
        assert web.token_count == int(500e9)

    def test_parse_mixture_percentages(self):
        spec = self.nl.parse_data_spec("Mix: 60% web, 25% code, 10% math, 5% academic")
        assert spec.mixture_preferences["web"] == pytest.approx(0.6)
        assert spec.mixture_preferences["code"] == pytest.approx(0.25)
        assert spec.mixture_preferences["math"] == pytest.approx(0.1)

    def test_parse_missing_data(self):
        spec = self.nl.parse_data_spec("We need more math data and want to add science data")
        assert "math" in spec.missing_domains
        assert "science" in spec.missing_domains

    def test_to_training_target(self):
        intent = self.nl.parse("Train a 7B model on web and code data with 4 nodes of H100 GPUs")
        target = self.nl.to_training_target(intent)

        assert target.model_size == ModelSize.LARGE
        assert "web" in target.data_domains
        assert target.compute_budget is not None
        assert target.compute_budget.num_nodes == 4

    def test_parse_deploy_command(self):
        intent = self.nl.parse("Deploy the training on our SLURM cluster")
        assert intent.action == "deploy"

    def test_parse_learning_rate(self):
        intent = self.nl.parse("Set learning rate to 3e-4")
        assert "learning_rate" in intent.parameters
        assert intent.parameters["learning_rate"] == pytest.approx(3e-4)
