"""OLMo-core training backend for A-EVOLVE.

This backend implements the TrainingJobRunner protocol using OLMo-core's
training engine. It enables the MCGS evolution loop to orchestrate real
pre-training and fine-tuning runs via OLMo-core's Trainer/TrainModule system.

Architecture:
    MCGS loop → OLMoCoreBackend.run_trial(workspace, node, budget, benchmark)
        → generates olmo-core training script from workspace config
        → launches via torchrun (local) or SLURM/Beaker
        → monitors via metric callbacks
        → evaluates checkpoint via benchmark
        → returns TrainingTrialResult
"""

from agent_evolve.backends.olmo_core.backend import OLMoCoreBackend
from agent_evolve.backends.olmo_core.config_translator import OLMoCoreConfigTranslator
from agent_evolve.backends.olmo_core.script_generator import OLMoCoreScriptGenerator

__all__ = [
    "OLMoCoreBackend",
    "OLMoCoreConfigTranslator",
    "OLMoCoreScriptGenerator",
]
