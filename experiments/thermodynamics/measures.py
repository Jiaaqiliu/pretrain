"""Thermodynamic state variable computation from model checkpoints.

Implements all measurements defined in the framework (Section 3):
  - Spectral entropy S (Eq. 1)
  - Order parameter ψ (Eq. 2)
  - Volume V = ||θ||²_F
  - Effective temperature T = η·σ̂²_∇ / (2B)
  - Free energy F = U - T·S
  - Entropy production rate σ(t) = -(1/T)·(dF/dt)
  - Cumulative entropy production ΔS_tot
  - Thermodynamic efficiency η_thermo
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class LayerThermo:
    """Thermodynamic measurements for a single layer."""

    name: str
    shape: tuple[int, int]
    num_params: int
    spectral_entropy: float
    order_parameter: float
    frobenius_norm_sq: float
    top_singular_values: list[float] = field(default_factory=list)


@dataclass
class CheckpointThermo:
    """Full thermodynamic measurement for one checkpoint."""

    step: int
    model_name: str
    num_params: int
    num_layers: int

    # State variables
    volume: float  # V = ||θ||²_F
    spectral_entropy: float  # S (global, parameter-weighted)
    order_parameter: float  # ψ (global, layer-averaged)
    internal_energy: float  # U = training loss
    temperature: float  # T = η·σ̂²_∇ / (2B)
    pressure: float  # P = λ (weight decay)
    free_energy: float  # F = U - T·S

    # Derived
    pv_over_nt: float  # P·V / (N·T)
    specific_heat: Optional[float] = None  # C_v = ∂U/∂T|_V

    # Per-layer details
    layer_measurements: list[LayerThermo] = field(default_factory=list)

    # Metadata
    lr: float = 0.0
    batch_size: int = 0
    grad_variance: float = 0.0
    checkpoint_path: str = ""


def spectral_entropy_layer(
    weight: Tensor, k: int = 256, power_iter: int = 1
) -> tuple[float, list[float]]:
    """Compute spectral entropy for a single weight matrix.

    Uses randomized SVD for large matrices (min(m,n) > 2048).

    Args:
        weight: 2D tensor (m × n)
        k: number of top singular values to retain
        power_iter: power iteration steps for randomized SVD

    Returns:
        (spectral_entropy, top_k_singular_values)
    """
    assert weight.ndim == 2, f"Expected 2D tensor, got {weight.ndim}D"
    m, n = weight.shape
    min_dim = min(m, n)

    if min_dim <= 2048:
        # Full SVD for small matrices
        sv = torch.linalg.svdvals(weight.float())
    else:
        # Randomized SVD (Halko et al. 2011)
        actual_k = min(k, min_dim)
        oversampling = 16
        total = actual_k + oversampling

        omega = torch.randn(n, total, device=weight.device, dtype=torch.float32)
        y = weight.float() @ omega

        q, _ = torch.linalg.qr(y)

        # Power iteration for improved accuracy
        for _ in range(power_iter):
            z = weight.float().T @ q
            q, _ = torch.linalg.qr(weight.float() @ z)

        b = q.T @ weight.float()
        sv = torch.linalg.svdvals(b)
        sv = sv[:actual_k]

    # Normalize to probability distribution
    sv_pos = sv[sv > 0]
    if len(sv_pos) == 0:
        return 0.0, []

    total_sv = sv_pos.sum()
    p = sv_pos / total_sv

    # Spectral entropy: S = -Σ p_i log(p_i)
    log_p = torch.log(p)
    entropy = -(p * log_p).sum().item()

    return entropy, sv_pos[:10].tolist()


def order_parameter_layer(weight: Tensor) -> float:
    """Compute order parameter ψ for a single weight matrix.

    ψ = (σ₁ - σ₂) / (σ₁ + σ₂)

    Uses power iteration for efficiency (only needs top-2 singular values).
    """
    assert weight.ndim == 2
    m, n = weight.shape

    if min(m, n) < 4:
        return 0.0

    # Power iteration to get top-2 singular values
    w = weight.float()
    v1 = torch.randn(n, device=w.device, dtype=torch.float32)
    v1 = v1 / v1.norm()

    # ~20 iterations is sufficient for convergence
    for _ in range(30):
        u1 = w @ v1
        u1 = u1 / (u1.norm() + 1e-10)
        v1 = w.T @ u1
        v1 = v1 / (v1.norm() + 1e-10)

    sigma1 = (w @ v1).norm().item()

    # Deflate and get second singular value
    w_deflated = w - sigma1 * (u1.unsqueeze(1) @ v1.unsqueeze(0))
    v2 = torch.randn(n, device=w.device, dtype=torch.float32)
    v2 = v2 / v2.norm()

    for _ in range(30):
        u2 = w_deflated @ v2
        u2 = u2 / (u2.norm() + 1e-10)
        v2 = w_deflated.T @ u2
        v2 = v2 / (v2.norm() + 1e-10)

    sigma2 = (w_deflated @ v2).norm().item()

    denom = sigma1 + sigma2
    if denom < 1e-10:
        return 0.0

    return (sigma1 - sigma2) / denom


def global_spectral_entropy(
    model: torch.nn.Module, k: int = 256
) -> tuple[float, list[LayerThermo]]:
    """Compute global spectral entropy (parameter-weighted average over layers).

    S = Σ_l (N_l / N) · S_l

    Returns:
        (global_entropy, per_layer_measurements)
    """
    total_params = 0
    weighted_entropy = 0.0
    layer_measurements = []

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue

        m, n = param.shape
        n_params = m * n
        total_params += n_params

        with torch.no_grad():
            entropy, top_svs = spectral_entropy_layer(param.data, k=k)
            psi = order_parameter_layer(param.data)
            fnorm_sq = param.data.float().pow(2).sum().item()

        lm = LayerThermo(
            name=name,
            shape=(m, n),
            num_params=n_params,
            spectral_entropy=entropy,
            order_parameter=psi,
            frobenius_norm_sq=fnorm_sq,
            top_singular_values=top_svs,
        )
        layer_measurements.append(lm)
        weighted_entropy += n_params * entropy

    if total_params == 0:
        return 0.0, []

    return weighted_entropy / total_params, layer_measurements


def global_order_parameter(model: torch.nn.Module) -> float:
    """Compute global order parameter (layer-averaged).

    ψ = (1/L) Σ_l ψ_l
    """
    psi_values = []
    for _name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        with torch.no_grad():
            psi = order_parameter_layer(param.data)
        psi_values.append(psi)

    if not psi_values:
        return 0.0
    return sum(psi_values) / len(psi_values)


def weight_volume(model: torch.nn.Module) -> float:
    """Compute volume V = ||θ||²_F (squared Frobenius norm of all parameters)."""
    v = 0.0
    for param in model.parameters():
        with torch.no_grad():
            v += param.data.float().pow(2).sum().item()
    return v


def effective_temperature(
    lr: float, batch_size: int, grad_variance: float
) -> float:
    """Compute effective temperature T = η·σ̂²_∇ / (2B).

    Args:
        lr: learning rate η
        batch_size: batch size B
        grad_variance: per-parameter gradient variance σ̂²_∇
    """
    if batch_size == 0:
        return 0.0
    return (lr * grad_variance) / (2 * batch_size)


def estimate_gradient_variance(
    model: torch.nn.Module,
    data_loader,
    loss_fn,
    num_samples: int = 2,
) -> float:
    """Estimate per-parameter gradient variance using double-batch method.

    σ̂²_∇ ≈ (1/2N) · ||g_A - g_B||²₂

    Args:
        model: the model
        data_loader: provides mini-batches
        loss_fn: loss function
        num_samples: number of batch pairs to average over
    """
    model.eval()
    total_variance = 0.0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    data_iter = iter(data_loader)

    for _ in range(num_samples):
        # Batch A
        batch_a = next(data_iter)
        model.zero_grad()
        loss_a = loss_fn(model, batch_a)
        loss_a.backward()
        grad_a = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

        # Batch B
        batch_b = next(data_iter)
        model.zero_grad()
        loss_b = loss_fn(model, batch_b)
        loss_b.backward()
        grad_b = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

        # Variance estimate
        diff = grad_a - grad_b
        variance = diff.pow(2).sum().item() / (2 * num_params)
        total_variance += variance

    model.zero_grad()
    return total_variance / num_samples


def free_energy(loss: float, temperature: float, spectral_entropy: float) -> float:
    """Compute free energy F = U - T·S."""
    return loss - temperature * spectral_entropy


def entropy_production_rate(
    free_energy_series: np.ndarray,
    temperature_series: np.ndarray,
    step_series: np.ndarray,
    window: int = 5,
    polyorder: int = 3,
) -> np.ndarray:
    """Compute entropy production rate σ(t) = -(1/T)·(dF/dt).

    Uses Savitzky-Golay smoothing for numerical differentiation.

    Args:
        free_energy_series: F values at each measurement point
        temperature_series: T values at each measurement point
        step_series: step numbers for computing dt
        window: Savitzky-Golay window size
        polyorder: Savitzky-Golay polynomial order

    Returns:
        σ(t) array (same length as input, edges may be noisy)
    """
    from scipy.signal import savgol_filter

    n = len(free_energy_series)
    if n < window:
        window = max(3, n if n % 2 == 1 else n - 1)

    # Smooth free energy
    f_smooth = savgol_filter(free_energy_series, window, min(polyorder, window - 1))

    # Numerical derivative dF/dt
    dt = np.diff(step_series).astype(float)
    df = np.diff(f_smooth)
    dfdt = np.zeros(n)
    dfdt[1:] = df / dt
    dfdt[0] = dfdt[1]

    # σ(t) = -(1/T)·(dF/dt)
    sigma = np.zeros(n)
    nonzero_t = temperature_series > 1e-15
    sigma[nonzero_t] = -dfdt[nonzero_t] / temperature_series[nonzero_t]

    # Entropy production should be non-negative (second law)
    sigma = np.maximum(sigma, 0.0)

    return sigma


def cumulative_entropy_production(
    sigma_series: np.ndarray, step_series: np.ndarray
) -> np.ndarray:
    """Compute cumulative entropy production ΔS_tot = ∫₀ᵗ σ(t') dt'.

    Uses trapezoidal integration.
    """
    dt = np.diff(step_series).astype(float)
    sigma_avg = (sigma_series[:-1] + sigma_series[1:]) / 2
    cumulative = np.zeros(len(sigma_series))
    cumulative[1:] = np.cumsum(sigma_avg * dt)
    return cumulative


def thermodynamic_efficiency(
    delta_f: float, delta_s_irr: float, mean_temperature: float
) -> float:
    """Compute thermodynamic efficiency η = |ΔF| / (|ΔF| + T̄·ΔS_irr).

    Args:
        delta_f: total free energy change |F(T) - F(0)|
        delta_s_irr: irreversible entropy production
        mean_temperature: time-averaged temperature T̄

    Returns:
        efficiency in [0, 1]
    """
    abs_df = abs(delta_f)
    waste = mean_temperature * delta_s_irr
    denom = abs_df + waste
    if denom < 1e-15:
        return 0.0
    return abs_df / denom


def measure_checkpoint(
    model: torch.nn.Module,
    step: int,
    loss: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    grad_variance: float,
    model_name: str = "",
    k: int = 256,
    checkpoint_path: str = "",
) -> CheckpointThermo:
    """Compute all thermodynamic state variables for a single checkpoint.

    This is the main entry point for measurement jobs.
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_2d_layers = sum(1 for p in model.parameters() if p.ndim == 2)

    # Compute state variables
    vol = weight_volume(model)
    s_global, layer_measurements = global_spectral_entropy(model, k=k)
    psi = sum(lm.order_parameter for lm in layer_measurements) / max(len(layer_measurements), 1)
    temp = effective_temperature(lr, batch_size, grad_variance)
    f = free_energy(loss, temp, s_global)

    # P·V / (N·T)
    pressure = weight_decay
    if temp > 1e-15 and num_params > 0:
        pv_over_nt = (pressure * vol) / (num_params * temp)
    else:
        pv_over_nt = 0.0

    return CheckpointThermo(
        step=step,
        model_name=model_name,
        num_params=num_params,
        num_layers=num_2d_layers,
        volume=vol,
        spectral_entropy=s_global,
        order_parameter=psi,
        internal_energy=loss,
        temperature=temp,
        pressure=pressure,
        free_energy=f,
        pv_over_nt=pv_over_nt,
        layer_measurements=layer_measurements,
        lr=lr,
        batch_size=batch_size,
        grad_variance=grad_variance,
        checkpoint_path=checkpoint_path,
    )
