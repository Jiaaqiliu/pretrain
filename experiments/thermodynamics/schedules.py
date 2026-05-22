"""Learning rate schedule implementations for proxy experiments.

Four schedules compared at matched total compute (Section 4.5):
  1. Cosine: standard cosine annealing from peak to η_min = η_max/100
  2. WSD-Linear: warmup (2%), stable (78%), linear decay (20%)
  3. WSD-Exponential: warmup (2%), stable (78%), exponential decay (20%)
  4. Gaussian (ours): warmup (2%), stable (78%), Gaussian decay (20%)
"""

import math


def cosine_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    min_ratio: float = 0.01,
    warmup_steps: int = 0,
) -> float:
    """Standard cosine annealing with optional warmup.

    Args:
        step: current training step
        total_steps: total number of training steps
        peak_lr: maximum learning rate
        min_ratio: η_min / η_max (default 0.01 → η_min = peak_lr/100)
        warmup_steps: linear warmup steps (default: 2% of total)
    """
    if warmup_steps == 0:
        warmup_steps = int(0.02 * total_steps)

    if step < warmup_steps:
        return peak_lr * step / warmup_steps

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_ratio + (1.0 - min_ratio) * cosine_value)


def wsd_linear_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_frac: float = 0.02,
    stable_frac: float = 0.78,
    min_ratio: float = 0.01,
) -> float:
    """WSD schedule with linear decay phase.

    Phases: warmup (2%) → stable (78%) → linear decay (20%)
    """
    warmup_steps = int(warmup_frac * total_steps)
    stable_end = int((warmup_frac + stable_frac) * total_steps)

    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)

    if step < stable_end:
        return peak_lr

    # Linear decay from peak_lr to peak_lr * min_ratio
    decay_steps = total_steps - stable_end
    progress = (step - stable_end) / max(decay_steps, 1)
    return peak_lr * (1.0 - progress * (1.0 - min_ratio))


def wsd_exponential_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_frac: float = 0.02,
    stable_frac: float = 0.78,
    min_ratio: float = 0.01,
) -> float:
    """WSD schedule with exponential decay phase.

    Phases: warmup (2%) → stable (78%) → exponential decay (20%)
    """
    warmup_steps = int(warmup_frac * total_steps)
    stable_end = int((warmup_frac + stable_frac) * total_steps)

    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)

    if step < stable_end:
        return peak_lr

    # Exponential decay: peak_lr * exp(-λt) where λ chosen so final = min_ratio * peak_lr
    decay_steps = total_steps - stable_end
    progress = (step - stable_end) / max(decay_steps, 1)
    decay_rate = -math.log(min_ratio)
    return peak_lr * math.exp(-decay_rate * progress)


def gaussian_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_frac: float = 0.02,
    stable_frac: float = 0.78,
    min_ratio: float = 0.01,
) -> float:
    """Gaussian decay schedule derived from minimum entropy production principle.

    From the variational optimization (Appendix A.2):
        T*(t) = T_max · exp(-(t-t_s)² / (2τ²_opt))
    where τ_opt = (T-t_s) / √(2·ln(T_max/T_min))

    Phases: warmup (2%) → stable (78%) → Gaussian decay (20%)
    """
    warmup_steps = int(warmup_frac * total_steps)
    stable_end = int((warmup_frac + stable_frac) * total_steps)

    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)

    if step < stable_end:
        return peak_lr

    # Gaussian decay
    decay_steps = total_steps - stable_end
    t = (step - stable_end) / max(decay_steps, 1)
    tau = 1.0 / math.sqrt(2.0 * math.log(1.0 / min_ratio))
    return peak_lr * math.exp(-0.5 * (t / tau) ** 2)


SCHEDULE_REGISTRY = {
    "cosine": cosine_lr,
    "wsd_linear": wsd_linear_lr,
    "wsd_exponential": wsd_exponential_lr,
    "gaussian": gaussian_lr,
}


def get_schedule(name: str):
    """Get a schedule function by name."""
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule: {name}. Available: {list(SCHEDULE_REGISTRY.keys())}")
    return SCHEDULE_REGISTRY[name]
