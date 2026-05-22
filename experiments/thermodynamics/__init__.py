"""Thermodynamics of Pretraining — measurement, analysis, and visualization.

Core modules:
    measures        — compute thermodynamic state variables from checkpoints
    schedules       — LR schedule implementations (Gaussian, WSD, Cosine)
    checkpoint_loader — load OLMo checkpoints for measurement
    analysis        — fitting (state equation, KWW, correlation)
    viz             — figure generation for the paper
"""

from experiments.thermodynamics.measures import (
    spectral_entropy_layer,
    order_parameter_layer,
    global_spectral_entropy,
    global_order_parameter,
    weight_volume,
    effective_temperature,
    free_energy,
    entropy_production_rate,
    cumulative_entropy_production,
    thermodynamic_efficiency,
)
from experiments.thermodynamics.schedules import (
    gaussian_lr,
    wsd_linear_lr,
    wsd_exponential_lr,
    cosine_lr,
)
