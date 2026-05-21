"""Integration tests for the OLMo-core backend.

Tests that the MCGS evolution loop can orchestrate training via the
OLMo-core backend in mock mode.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest


class TestOLMoCoreConfigTranslator:
    """Test workspace YAML → OLMoCoreTrainingConfig translation."""

    def test_translate_from_workspace(self, tmp_path):
        from agent_evolve.backends.olmo_core.config_translator import OLMoCoreConfigTranslator

        # Create a minimal workspace
        (tmp_path / "model").mkdir()
        (tmp_path / "train").mkdir()
        (tmp_path / "data").mkdir()

        (tmp_path / "model" / "base.yaml").write_text(
            "hidden_size: 768\nnum_layers: 12\nnum_heads: 12\nvocab_size: 50304\n"
        )
        (tmp_path / "train" / "optimizer.yaml").write_text(
            "lr: 6.0e-4\nweight_decay: 0.1\nbetas: [0.9, 0.95]\nwarmup_steps: 2000\n"
        )
        (tmp_path / "train" / "batching.yaml").write_text(
            "per_device_train_batch_size: 4\nmax_seq_len: 4096\nlog_every: 10\n"
        )
        (tmp_path / "train" / "pipeline.yaml").write_text(
            "stages:\n  - name: pretrain\n    type: pretrain\n    enabled: true\n    max_steps: 5000\n"
        )
        (tmp_path / "data" / "sources.yaml").write_text(
            "sources:\n  - path: /data/web\n    split: train\n    format: numpy\n"
        )
        (tmp_path / "data" / "mix.yaml").write_text(
            "ratios:\n  web: 0.7\n  code: 0.3\n"
        )

        translator = OLMoCoreConfigTranslator()
        config = translator.translate(tmp_path)

        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12
        assert config.lr == 6.0e-4
        assert config.weight_decay == 0.1
        assert config.warmup_steps == 2000
        assert config.max_steps == 5000
        assert config.phase == "pretrain"
        assert "/data/web" in config.data_paths
        assert config.data_mix_ratios == {"web": 0.7, "code": 0.3}

    def test_translate_missing_files(self, tmp_path):
        """Should handle missing YAML files gracefully."""
        from agent_evolve.backends.olmo_core.config_translator import OLMoCoreConfigTranslator

        translator = OLMoCoreConfigTranslator()
        config = translator.translate(tmp_path)

        # Should return defaults
        assert config.d_model == 768
        assert config.lr == 3e-4
        assert config.max_steps == 10000


class TestOLMoCoreScriptGenerator:
    """Test training script generation."""

    def test_generate_script(self, tmp_path):
        from agent_evolve.backends.olmo_core.config_translator import OLMoCoreTrainingConfig
        from agent_evolve.backends.olmo_core.script_generator import OLMoCoreScriptGenerator

        config = OLMoCoreTrainingConfig(
            d_model=768,
            n_layers=12,
            n_heads=12,
            lr=6e-4,
            max_steps=5000,
            save_folder=str(tmp_path / "checkpoints"),
        )

        generator = OLMoCoreScriptGenerator()
        script_path = generator.generate(config, tmp_path / "scripts")

        assert script_path.exists()
        content = script_path.read_text()
        assert "olmo_core" in content
        assert "d_model=768" in content
        assert "n_layers=12" in content
        assert "lr=0.0006" in content
        assert "Duration.steps(5000)" in content

    def test_generated_script_is_valid_python(self, tmp_path):
        """Generated script should be syntactically valid Python."""
        from agent_evolve.backends.olmo_core.config_translator import OLMoCoreTrainingConfig
        from agent_evolve.backends.olmo_core.script_generator import OLMoCoreScriptGenerator

        config = OLMoCoreTrainingConfig()
        generator = OLMoCoreScriptGenerator()
        script_path = generator.generate(config, tmp_path)

        content = script_path.read_text()
        compile(content, str(script_path), "exec")


class TestOLMoCoreBackend:
    """Test the OLMoCoreBackend (mock mode)."""

    def _make_workspace(self, tmp_path):
        """Create a minimal workspace compatible with TrainingWorkspace."""
        ws_root = tmp_path / "workspace"
        for d in ["model", "train", "data", "eval", "memory", "checkpoints", "evolution"]:
            (ws_root / d).mkdir(parents=True, exist_ok=True)

        (ws_root / "manifest.yaml").write_text(
            "name: test_olmo\ncontract_version: train-1.0\n"
            "defaults:\n  backend: olmo_core\n  algorithm: mcgs\n  benchmark: test\n"
            "evolvable_layers:\n  - train/optimizer.yaml\n"
            "protected_layers:\n  - model/base.yaml\n"
            "artifact_layers:\n  - checkpoints\n  - evolution\n"
        )
        (ws_root / "model" / "base.yaml").write_text(
            "hidden_size: 768\nnum_layers: 12\nnum_heads: 12\n"
        )
        (ws_root / "train" / "optimizer.yaml").write_text(
            "lr: 6.0e-4\nweight_decay: 0.1\nbetas: [0.9, 0.95]\nwarmup_steps: 100\n"
        )
        (ws_root / "train" / "pipeline.yaml").write_text(
            "stages:\n  - name: pretrain\n    type: pretrain\n    enabled: true\n    max_steps: 100\n"
        )
        (ws_root / "train" / "batching.yaml").write_text(
            "per_device_train_batch_size: 2\nmax_seq_len: 512\n"
        )
        (ws_root / "data" / "sources.yaml").write_text("sources: []\n")
        (ws_root / "data" / "mix.yaml").write_text("ratios: {}\n")
        (ws_root / "eval" / "local_splits.yaml").write_text("dev:\n  path: /dev/null\n")
        (ws_root / "eval" / "error_taxonomy.yaml").write_text("buckets: {}\n")

        class MockWorkspace:
            def __init__(self, root):
                self.root = root

        return MockWorkspace(ws_root)

    def test_mock_run_trial(self, tmp_path):
        from agent_evolve.backends.olmo_core.backend import OLMoCoreBackend
        from agent_evolve.model.types import TrainingSearchNode, TrialBudget

        workspace = self._make_workspace(tmp_path)
        backend = OLMoCoreBackend(mock=True)

        node = TrainingSearchNode(
            node_id="test_node_001",
            parent_id=None,
            branch_id=0,
        )
        budget = TrialBudget(seconds=60, steps=100)

        result = backend.run_trial(workspace, node, budget, benchmark=None)

        assert result.status == "success"
        assert result.checkpoint is not None
        assert result.eval_metrics is not None
        assert result.eval_metrics.primary_metric_name == "loss"
        assert result.eval_metrics.primary_metric_value > 0
        assert result.node_id == "test_node_001"
        assert result.cost["seconds"] > 0

    def test_backend_registered_in_registry(self):
        from agent_evolve.model.registries import TRAINING_JOB_RUNNERS

        assert "olmo_core" in TRAINING_JOB_RUNNERS
        assert "OLMoCoreBackend" in TRAINING_JOB_RUNNERS["olmo_core"]

    def test_backend_satisfies_protocol(self):
        from agent_evolve.backends.olmo_core.backend import OLMoCoreBackend
        from agent_evolve.model.runner_protocol import TrainingJobRunner

        backend = OLMoCoreBackend(mock=True)
        assert isinstance(backend, TrainingJobRunner)
        assert backend.name == "olmo_core"


class TestEndToEndIntegration:
    """Test the full MCGS → OLMo-core pipeline (mock mode)."""

    def test_training_evolver_with_olmo_core(self, tmp_path):
        """TrainingEvolver should run successfully with OLMo-core backend."""
        from agent_evolve.model.api import TrainingEvolver
        from agent_evolve.model.types import TrainingEvolveConfig

        ws_src = Path(__file__).parent.parent / "seed_workspaces" / "olmo_core_pretrain"
        if not ws_src.exists():
            pytest.skip("olmo_core_pretrain workspace not found")

        config = TrainingEvolveConfig(
            max_cycles=1,
            smoke=True,
            trial_budget_seconds=30,
            trial_budget_steps=10,
        )

        evolver = TrainingEvolver(
            workspace=ws_src,
            benchmark="nemo_reasoner",
            algorithm="mcgs",
            backend="olmo_core",
            config=config,
            work_dir=tmp_path / "workdir",
        )

        # The backend should be in mock mode
        assert evolver.backend.mock is True
        assert evolver.backend.name == "olmo_core"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
