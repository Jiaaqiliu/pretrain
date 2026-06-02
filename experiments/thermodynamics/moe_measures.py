"""Spectral measurements for Mixture-of-Experts (MoE) models.

Extends the dense-model measurement pipeline to handle:
  - Per-expert SR/d, α, spectral entropy
  - Cross-expert spectral alignment (subspace cosine similarity)
  - Shared vs routed expert separation
  - Router matrix spectral analysis
  - Expert-level aggregation statistics

Supported architectures:
  - OLMoE (allenai/OLMoE-1B-7B): 64 experts, top-8, no shared expert
  - Mixtral (mistralai/Mixtral-8x7B): 8 experts, top-2
  - DeepSeek-V2/V3: fine-grained MoE with shared experts
  - Phi-3.5-MoE: 16 experts, top-2
  - DBRX: 16 fine-grained experts, top-4
  - Qwen2-MoE / Qwen3: shared + routed experts
  - Generic HuggingFace MoE models
"""

import json
import math
import re
import gc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpertSpectral:
    """Spectral measurements for a single expert's weight matrix."""
    expert_id: int
    layer_idx: int
    layer_type: str  # "gate_proj", "up_proj", "down_proj", etc.
    shape: tuple[int, int]
    alpha: float
    stable_rank: float
    srd: float  # SR / d (d = min(m,n) or hidden_dim)
    spectral_entropy: float
    frobenius_norm: float
    order_parameter: float = 0.0  # ψ = (σ₁ - σ₂)/(σ₁ + σ₂) [U5]
    volume: float = 0.0  # ||W||²_F [N3, U1]
    top10_sv: list[float] = field(default_factory=list)


@dataclass
class LayerMoESpectral:
    """Spectral measurements for one MoE layer (all experts + router)."""
    layer_idx: int
    num_experts: int
    experts: list[ExpertSpectral] = field(default_factory=list)

    # Aggregated
    alpha_mean: float = 0.0
    alpha_std: float = 0.0
    alpha_min: float = 0.0
    alpha_max: float = 0.0
    srd_mean: float = 0.0
    srd_std: float = 0.0

    # Cross-expert alignment (avg cosine similarity of top-k singular vectors)
    cross_expert_alignment: float = 0.0

    # Energy equipartition [N3]
    expert_volumes: list[float] = field(default_factory=list)  # ||W_i||²_F per expert
    epr: float = 0.0  # Var(V_i) / <V_i>² — 0=equipartition, high=specialized

    # Order parameter [U5]
    psi_mean: float = 0.0
    psi_std: float = 0.0

    # Router
    router_stable_rank: Optional[float] = None
    router_spectral_norm: Optional[float] = None
    router_srd: Optional[float] = None

    # Shared expert (if present)
    shared_expert_alpha: Optional[float] = None
    shared_expert_srd: Optional[float] = None


@dataclass
class MoECheckpointSpectral:
    """Full spectral measurement for one MoE model checkpoint."""
    model_name: str
    step: int
    revision: str
    total_params: int
    active_params: int
    num_experts: int
    top_k: int
    has_shared_expert: bool
    hidden_dim: int

    # Global (all experts aggregated)
    alpha_mean: float = 0.0
    alpha_std_across_experts: float = 0.0
    srd_mean: float = 0.0
    srd_std_across_experts: float = 0.0

    # Attention layers (non-MoE)
    alpha_attn: float = 0.0
    srd_attn: float = 0.0

    # MoE FFN layers (all routed experts)
    alpha_moe: float = 0.0
    srd_moe: float = 0.0

    # Shared expert (if present)
    alpha_shared: Optional[float] = None
    srd_shared: Optional[float] = None

    # Cross-expert alignment (mean across layers)
    cross_expert_alignment_mean: float = 0.0

    # Router health
    router_srd_mean: Optional[float] = None

    # Energy equipartition ratio (EPR) [N3]
    epr_mean: float = 0.0  # mean EPR across layers

    # Order parameter [U5]
    psi_mean: float = 0.0
    psi_attn: float = 0.0
    psi_moe: float = 0.0

    # PV/NkT state equation [U1]
    pv_over_nt_total: Optional[float] = None  # using N_total
    pv_over_nt_active: Optional[float] = None  # using N_active

    # Per-layer details
    layers: list[LayerMoESpectral] = field(default_factory=list)

    # Per-expert alpha distribution (all experts across all layers)
    per_expert_alphas: list[float] = field(default_factory=list)
    per_expert_volumes: list[float] = field(default_factory=list)  # [N3]
    per_expert_psis: list[float] = field(default_factory=list)  # [U5]

    n_layers_measured: int = 0


# ---------------------------------------------------------------------------
# Core spectral computations (reused from measures.py with extensions)
# ---------------------------------------------------------------------------

def compute_svd(weight: Tensor, k: int = 256) -> Tensor:
    """Compute singular values of a 2D weight matrix."""
    assert weight.ndim == 2
    m, n = weight.shape
    min_dim = min(m, n)

    if min_dim <= 2048:
        return torch.linalg.svdvals(weight.float())
    else:
        actual_k = min(k, min_dim)
        omega = torch.randn(n, actual_k + 16, device=weight.device, dtype=torch.float32)
        y = weight.float() @ omega
        q, _ = torch.linalg.qr(y)
        for _ in range(2):
            z = weight.float().T @ q
            q, _ = torch.linalg.qr(weight.float() @ z)
        b = q.T @ weight.float()
        return torch.linalg.svdvals(b)[:actual_k]


def compute_alpha(sv: Tensor, xmin_percentile: float = 0.1) -> float:
    """Estimate power-law exponent α from singular values.

    Uses the ESD (Empirical Spectral Density) of W^T W, i.e., λ = σ².
    Fits P(λ) ~ λ^(-α) via Hill estimator on the tail.
    """
    eigenvalues = (sv ** 2).cpu().numpy()
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) < 10:
        return float('nan')

    eigenvalues = np.sort(eigenvalues)[::-1]

    n = len(eigenvalues)
    k_start = max(1, int(n * xmin_percentile))
    tail = eigenvalues[:n - k_start]

    if len(tail) < 5:
        return float('nan')

    xmin = tail[-1]
    if xmin <= 0:
        return float('nan')

    log_ratios = np.log(tail / xmin)
    log_ratios = log_ratios[log_ratios > 0]

    if len(log_ratios) < 5:
        return float('nan')

    alpha = 1.0 + len(log_ratios) / np.sum(log_ratios)
    return float(alpha)


def compute_stable_rank(sv: Tensor) -> float:
    """SR = ||W||_F^2 / σ_1^2 = Σσ_i^2 / σ_1^2."""
    sv_sq = sv ** 2
    if sv_sq[0] < 1e-10:
        return 0.0
    return (sv_sq.sum() / sv_sq[0]).item()


def compute_spectral_entropy(sv: Tensor) -> float:
    """S = -Σ p_i log(p_i) where p_i = σ_i / Σσ_j."""
    sv_pos = sv[sv > 0]
    if len(sv_pos) == 0:
        return 0.0
    p = sv_pos / sv_pos.sum()
    return -(p * torch.log(p)).sum().item()


def compute_order_parameter(sv: Tensor) -> float:
    """ψ = (σ₁ - σ₂)/(σ₁ + σ₂) — measures spectral gap dominance."""
    if len(sv) < 2:
        return 0.0
    s1, s2 = sv[0].item(), sv[1].item()
    denom = s1 + s2
    if denom < 1e-10:
        return 0.0
    return (s1 - s2) / denom


def compute_epr(volumes: list[float]) -> float:
    """Energy Equipartition Ratio: Var(V_i) / <V_i>².
    0 = perfect equipartition, high = strong specialization."""
    if len(volumes) < 2:
        return 0.0
    arr = np.array(volumes)
    mean_v = arr.mean()
    if mean_v < 1e-10:
        return 0.0
    return float(arr.var() / (mean_v ** 2))


def compute_cross_expert_alignment(
    sv_matrices: list[Tensor], top_k: int = 10
) -> float:
    """Average cosine similarity of top-k right singular vectors across expert pairs.

    sv_matrices: list of (U, S, Vh) tuples or just weight matrices.
    Returns mean pairwise alignment in [0, 1].
    """
    if len(sv_matrices) < 2:
        return 1.0

    # Extract top-k right singular vectors for each expert
    top_vectors = []
    for W in sv_matrices:
        if W.ndim != 2 or min(W.shape) < top_k:
            continue
        try:
            _, _, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            top_vectors.append(Vh[:top_k])  # (k, n)
        except Exception:
            continue

    if len(top_vectors) < 2:
        return 1.0

    # Pairwise subspace alignment via principal angles
    alignments = []
    for i in range(len(top_vectors)):
        for j in range(i + 1, len(top_vectors)):
            # Cosine of principal angles = singular values of V_i @ V_j^T
            try:
                cos_angles = torch.linalg.svdvals(top_vectors[i] @ top_vectors[j].T)
                alignments.append(cos_angles.mean().item())
            except Exception:
                continue

    return float(np.mean(alignments)) if alignments else 1.0


# ---------------------------------------------------------------------------
# Architecture-specific weight extraction
# ---------------------------------------------------------------------------

# Pattern registry: maps architecture name to regex patterns for weight names
MOE_ARCHITECTURES = {
    "olmoe": {
        "expert_pattern": r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
        "fused_expert_pattern": r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)",
        "router_pattern": r"model\.layers\.(\d+)\.mlp\.gate\.weight",
        "attn_pattern": r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
        "shared_expert_pattern": None,
    },
    "mixtral": {
        "expert_pattern": r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight",
        "router_pattern": r"model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight",
        "attn_pattern": r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
        "shared_expert_pattern": None,
    },
    "deepseek_v2": {
        "expert_pattern": r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
        "router_pattern": r"model\.layers\.(\d+)\.mlp\.gate\.weight",
        "attn_pattern": r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
        "shared_expert_pattern": r"model\.layers\.(\d+)\.mlp\.shared_experts?\.(gate_proj|up_proj|down_proj)\.weight",
    },
    "phi3_moe": {
        "expert_pattern": r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight",
        "router_pattern": r"model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight",
        "attn_pattern": r"model\.layers\.(\d+)\.self_attn\.(qkv_proj|o_proj)\.weight",
        "shared_expert_pattern": None,
    },
    "dbrx": {
        "expert_pattern": r"transformer\.blocks\.(\d+)\.ffn\.experts\.mlp\.(\d+)\.(w1|w2|v1)\.weight",
        "router_pattern": r"transformer\.blocks\.(\d+)\.ffn\.router\.layer\.weight",
        "attn_pattern": r"transformer\.blocks\.(\d+)\.norm_attn_norm\.attn\.(Wqkv|out_proj)\.weight",
        "shared_expert_pattern": None,
    },
    "qwen2_moe": {
        "expert_pattern": r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
        "router_pattern": r"model\.layers\.(\d+)\.mlp\.gate\.weight",
        "attn_pattern": r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
        "shared_expert_pattern": r"model\.layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight",
    },
    "llama4_moe": {
        "expert_pattern": r"model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(w1|w2|w3)\.weight",
        "router_pattern": r"model\.layers\.(\d+)\.feed_forward\.router\.weight",
        "attn_pattern": r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
        "shared_expert_pattern": None,
    },
}


def detect_architecture(state_dict_keys: list[str]) -> str:
    """Auto-detect MoE architecture from weight names."""
    keys_str = "\n".join(state_dict_keys[:200])

    if "block_sparse_moe.experts" in keys_str:
        if any("Phi" in k or "phi" in k for k in state_dict_keys[:5]):
            return "phi3_moe"
        return "mixtral"
    if "mlp.experts" in keys_str:
        if "shared_expert" in keys_str:
            if "deepseek" in keys_str.lower() or any(
                re.search(r"layers\.\d+\.mlp\.experts\.\d{2,}", k) for k in state_dict_keys[:200]
            ):
                return "deepseek_v2"
            return "qwen2_moe"
        return "olmoe"
    if "ffn.experts" in keys_str:
        return "dbrx"
    if "feed_forward.experts" in keys_str:
        return "llama4_moe"

    return "olmoe"  # fallback


# ---------------------------------------------------------------------------
# Main measurement functions
# ---------------------------------------------------------------------------

def measure_expert_weight(
    weight: Tensor, expert_id: int, layer_idx: int, layer_type: str,
    hidden_dim: int, k: int = 256
) -> ExpertSpectral:
    """Measure spectral properties of a single expert weight matrix."""
    m, n = weight.shape
    d = hidden_dim

    with torch.no_grad():
        sv = compute_svd(weight, k=k)
        alpha = compute_alpha(sv)
        sr = compute_stable_rank(sv)
        srd = sr / d if d > 0 else 0.0
        entropy = compute_spectral_entropy(sv)
        fnorm = weight.float().norm().item()
        vol = fnorm ** 2
        psi = compute_order_parameter(sv)
        top10 = sv[:10].tolist()

    return ExpertSpectral(
        expert_id=expert_id,
        layer_idx=layer_idx,
        layer_type=layer_type,
        shape=(m, n),
        alpha=alpha,
        stable_rank=sr,
        srd=srd,
        spectral_entropy=entropy,
        frobenius_norm=fnorm,
        order_parameter=psi,
        volume=vol,
        top10_sv=top10,
    )


def measure_moe_checkpoint(
    model_name_or_path: str,
    revision: str = "main",
    step: int = 0,
    hidden_dim: int = 0,
    num_experts: int = 0,
    top_k_routing: int = 0,
    device: str = "cpu",
    max_experts_per_layer: int = 0,
    measure_alignment: bool = True,
    alignment_top_k: int = 10,
    output_path: Optional[str] = None,
) -> MoECheckpointSpectral:
    """Measure spectral properties of a MoE model checkpoint.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        revision: git revision / checkpoint tag
        step: training step number
        hidden_dim: model hidden dimension (auto-detected if 0)
        num_experts: number of routed experts (auto-detected if 0)
        top_k_routing: experts activated per token (for metadata)
        device: "cpu" or "cuda"
        max_experts_per_layer: if >0, only measure this many experts per layer (for speed)
        measure_alignment: whether to compute cross-expert spectral alignment
        alignment_top_k: number of top singular vectors for alignment computation
        output_path: if set, write results to this JSONL file
    """
    from transformers import AutoConfig

    print(f"Loading config for {model_name_or_path} (revision={revision})...")
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, revision=revision, trust_remote_code=True)
    except Exception:
        config = None

    if config and hidden_dim == 0:
        hidden_dim = getattr(config, 'hidden_size', 0) or getattr(config, 'd_model', 0)
    if config and num_experts == 0:
        num_experts = (
            getattr(config, 'num_experts', 0)
            or getattr(config, 'num_local_experts', 0)
            or getattr(config, 'n_routed_experts', 0)
            or 0
        )
    if config and top_k_routing == 0:
        top_k_routing = (
            getattr(config, 'num_experts_per_tok', 0)
            or getattr(config, 'num_experts_per_token', 0)
            or getattr(config, 'top_k', 0)
            or 0
        )
    has_shared = bool(
        config and (
            getattr(config, 'num_shared_experts', 0) > 0
            or getattr(config, 'n_shared_experts', 0) > 0
        )
    )

    print(f"Config: hidden_dim={hidden_dim}, num_experts={num_experts}, top_k={top_k_routing}, shared={has_shared}")

    # Load state dict (memory-efficient: don't instantiate full model)
    print(f"Loading weights...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        torch_dtype=torch.float16,
        device_map=device if device != "cpu" else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    state_dict_keys = list(model.state_dict().keys())
    arch = detect_architecture(state_dict_keys)
    patterns = MOE_ARCHITECTURES[arch]
    print(f"Detected architecture: {arch}")

    total_params = sum(p.numel() for p in model.parameters())

    # Compile patterns
    expert_re = re.compile(patterns["expert_pattern"]) if patterns["expert_pattern"] else None
    fused_re = re.compile(patterns["fused_expert_pattern"]) if patterns.get("fused_expert_pattern") else None
    router_re = re.compile(patterns["router_pattern"]) if patterns["router_pattern"] else None
    attn_re = re.compile(patterns["attn_pattern"]) if patterns["attn_pattern"] else None
    shared_re = re.compile(patterns["shared_expert_pattern"]) if patterns.get("shared_expert_pattern") else None

    # Organize weights by layer
    expert_weights = {}   # {layer_idx: {expert_id: {proj_type: tensor}}}
    router_weights = {}   # {layer_idx: tensor}
    attn_weights = {}     # {layer_idx: {proj_type: tensor}}
    shared_weights = {}   # {layer_idx: {proj_type: tensor}}

    for name, param in model.named_parameters():
        # Handle fused 3D expert weights [num_experts, out_dim, in_dim]
        if param.ndim == 3 and fused_re:
            m = fused_re.match(name)
            if m:
                li = int(m.group(1))
                pt = m.group(2)
                n_exp = param.shape[0]
                for ei in range(n_exp):
                    expert_weights.setdefault(li, {}).setdefault(ei, {})[pt] = param.data[ei]
                continue

        if param.ndim != 2:
            continue

        if expert_re:
            m = expert_re.match(name)
            if m:
                li, ei = int(m.group(1)), int(m.group(2))
                pt = m.group(3)
                expert_weights.setdefault(li, {}).setdefault(ei, {})[pt] = param.data
                continue

        if router_re:
            m = router_re.match(name)
            if m:
                li = int(m.group(1))
                router_weights[li] = param.data
                continue

        if attn_re:
            m = attn_re.match(name)
            if m:
                li = int(m.group(1))
                pt = m.group(2)
                attn_weights.setdefault(li, {})[pt] = param.data
                continue

        if shared_re:
            m = shared_re.match(name)
            if m:
                li = int(m.group(1))
                pt = m.group(2)
                shared_weights.setdefault(li, {})[pt] = param.data
                continue

    print(f"Found {len(expert_weights)} MoE layers, {len(attn_weights)} attention layers")

    # Measure per-expert spectral properties
    all_expert_alphas = []
    all_expert_srds = []
    all_expert_volumes = []
    all_expert_psis = []
    all_attn_alphas = []
    all_attn_srds = []
    all_attn_psis = []
    all_shared_alphas = []
    all_shared_srds = []
    all_router_srds = []
    all_layer_eprs = []
    moe_layers = []
    total_volume = 0.0

    for li in sorted(expert_weights.keys()):
        experts_in_layer = expert_weights[li]
        n_exp = len(experts_in_layer)
        layer_result = LayerMoESpectral(layer_idx=li, num_experts=n_exp)

        expert_ids = sorted(experts_in_layer.keys())
        if max_experts_per_layer > 0 and len(expert_ids) > max_experts_per_layer:
            indices = np.linspace(0, len(expert_ids) - 1, max_experts_per_layer, dtype=int)
            expert_ids = [expert_ids[i] for i in indices]

        alignment_matrices = []
        layer_expert_volumes = []

        for ei in expert_ids:
            proj_weights = experts_in_layer[ei]
            expert_vol = 0.0
            for pt, w in proj_weights.items():
                es = measure_expert_weight(w, ei, li, pt, hidden_dim)
                layer_result.experts.append(es)
                all_expert_alphas.append(es.alpha)
                all_expert_srds.append(es.srd)
                all_expert_psis.append(es.order_parameter)
                expert_vol += es.volume
                total_volume += es.volume

                if measure_alignment and pt in ("gate_proj", "w1") and w.shape[0] <= 16384:
                    alignment_matrices.append(w)

            layer_expert_volumes.append(expert_vol)
            all_expert_volumes.append(expert_vol)

        # Cross-expert alignment [N1, N11]
        if measure_alignment and len(alignment_matrices) >= 2:
            layer_result.cross_expert_alignment = compute_cross_expert_alignment(
                alignment_matrices, top_k=alignment_top_k
            )

        # Energy equipartition [N3]
        layer_result.expert_volumes = layer_expert_volumes
        layer_result.epr = compute_epr(layer_expert_volumes)
        all_layer_eprs.append(layer_result.epr)

        # Order parameter [U5]
        layer_psis = [e.order_parameter for e in layer_result.experts]
        if layer_psis:
            layer_result.psi_mean = float(np.mean(layer_psis))
            layer_result.psi_std = float(np.std(layer_psis))

        # Router spectral analysis [N2]
        if li in router_weights:
            rw = router_weights[li]
            with torch.no_grad():
                r_sv = compute_svd(rw, k=min(64, min(rw.shape)))
                layer_result.router_stable_rank = compute_stable_rank(r_sv)
                layer_result.router_spectral_norm = r_sv[0].item()
                layer_result.router_srd = layer_result.router_stable_rank / rw.shape[1] if rw.shape[1] > 0 else 0
                all_router_srds.append(layer_result.router_srd)

        # Shared expert [N6]
        if li in shared_weights:
            for pt, w in shared_weights[li].items():
                es = measure_expert_weight(w, -1, li, f"shared_{pt}", hidden_dim)
                all_shared_alphas.append(es.alpha)
                all_shared_srds.append(es.srd)
                total_volume += es.volume
            layer_result.shared_expert_alpha = float(np.nanmean([a for a in all_shared_alphas if not np.isnan(a)])) if all_shared_alphas else None
            layer_result.shared_expert_srd = float(np.nanmean(all_shared_srds)) if all_shared_srds else None

        # Aggregate per-layer
        layer_alphas = [e.alpha for e in layer_result.experts if not np.isnan(e.alpha)]
        layer_srds = [e.srd for e in layer_result.experts]
        if layer_alphas:
            layer_result.alpha_mean = float(np.mean(layer_alphas))
            layer_result.alpha_std = float(np.std(layer_alphas))
            layer_result.alpha_min = float(np.min(layer_alphas))
            layer_result.alpha_max = float(np.max(layer_alphas))
        if layer_srds:
            layer_result.srd_mean = float(np.mean(layer_srds))
            layer_result.srd_std = float(np.std(layer_srds))

        moe_layers.append(layer_result)
        print(f"  Layer {li}: {n_exp} experts, α={layer_result.alpha_mean:.2f}±{layer_result.alpha_std:.2f}, "
              f"SR/d={layer_result.srd_mean:.4f}, align={layer_result.cross_expert_alignment:.3f}, "
              f"EPR={layer_result.epr:.4f}")

    # Measure attention layers
    for li in sorted(attn_weights.keys()):
        for pt, w in attn_weights[li].items():
            with torch.no_grad():
                sv = compute_svd(w)
                a = compute_alpha(sv)
                sr = compute_stable_rank(sv)
                srd = sr / hidden_dim if hidden_dim > 0 else 0
                psi = compute_order_parameter(sv)
                total_volume += w.float().pow(2).sum().item()
            if not np.isnan(a):
                all_attn_alphas.append(a)
            all_attn_srds.append(srd)
            all_attn_psis.append(psi)

    # Clean valid values
    valid_expert_alphas = [a for a in all_expert_alphas if not np.isnan(a)]
    valid_shared_alphas = [a for a in all_shared_alphas if not np.isnan(a)]

    # Estimate active params
    if num_experts > 0 and top_k_routing > 0:
        moe_param_fraction = top_k_routing / num_experts
        active_params = getattr(config, 'num_active_params', 0) if config else 0
        if active_params == 0:
            active_params = int(total_params * (0.3 + 0.7 * moe_param_fraction))
    else:
        active_params = total_params

    # PV/NkT state equation [U1]
    # P = weight_decay (from config), V = total_volume, T ~ lr/batch_size
    weight_decay = getattr(config, 'weight_decay', None) if config else None
    if weight_decay is None:
        weight_decay = 0.1  # common default
    lr_peak = getattr(config, 'learning_rate', None) if config else None
    batch_size_cfg = getattr(config, 'per_device_train_batch_size', None) if config else None

    pv_nt_total = None
    pv_nt_active = None
    if total_volume > 0 and total_params > 0:
        t_proxy = 1e-8  # placeholder; proper T requires gradient variance
        pv_nt_total = (weight_decay * total_volume) / (total_params * t_proxy) if t_proxy > 0 else None
        pv_nt_active = (weight_decay * total_volume) / (active_params * t_proxy) if t_proxy > 0 else None

    result = MoECheckpointSpectral(
        model_name=model_name_or_path,
        step=step,
        revision=revision,
        total_params=total_params,
        active_params=active_params,
        num_experts=num_experts,
        top_k=top_k_routing,
        has_shared_expert=has_shared,
        hidden_dim=hidden_dim,
        alpha_mean=float(np.mean(valid_expert_alphas)) if valid_expert_alphas else 0,
        alpha_std_across_experts=float(np.std(valid_expert_alphas)) if valid_expert_alphas else 0,
        srd_mean=float(np.mean(all_expert_srds)) if all_expert_srds else 0,
        srd_std_across_experts=float(np.std(all_expert_srds)) if all_expert_srds else 0,
        alpha_attn=float(np.mean(all_attn_alphas)) if all_attn_alphas else 0,
        srd_attn=float(np.mean(all_attn_srds)) if all_attn_srds else 0,
        alpha_moe=float(np.mean(valid_expert_alphas)) if valid_expert_alphas else 0,
        srd_moe=float(np.mean(all_expert_srds)) if all_expert_srds else 0,
        alpha_shared=float(np.mean(valid_shared_alphas)) if valid_shared_alphas else None,
        srd_shared=float(np.mean(all_shared_srds)) if all_shared_srds else None,
        cross_expert_alignment_mean=float(np.mean([l.cross_expert_alignment for l in moe_layers if l.cross_expert_alignment > 0])) if moe_layers else 0,
        router_srd_mean=float(np.mean(all_router_srds)) if all_router_srds else None,
        epr_mean=float(np.mean(all_layer_eprs)) if all_layer_eprs else 0,
        psi_mean=float(np.mean(all_expert_psis)) if all_expert_psis else 0,
        psi_attn=float(np.mean(all_attn_psis)) if all_attn_psis else 0,
        psi_moe=float(np.mean(all_expert_psis)) if all_expert_psis else 0,
        pv_over_nt_total=pv_nt_total,
        pv_over_nt_active=pv_nt_active,
        layers=moe_layers,
        per_expert_alphas=valid_expert_alphas,
        per_expert_volumes=all_expert_volumes,
        per_expert_psis=all_expert_psis,
        n_layers_measured=len(moe_layers),
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"MoE Spectral Summary: {model_name_or_path}")
    print(f"{'='*60}")
    print(f"Total params: {total_params/1e9:.1f}B, Active: {active_params/1e9:.1f}B")
    print(f"Experts: {num_experts}, top-{top_k_routing}, shared={has_shared}")
    print(f"MoE layers measured: {len(moe_layers)}")
    print(f"Expert α: {result.alpha_moe:.3f} ± {result.alpha_std_across_experts:.3f}")
    print(f"Expert SR/d: {result.srd_moe:.4f} ± {result.srd_std_across_experts:.4f}")
    print(f"Attention α: {result.alpha_attn:.3f}, SR/d: {result.srd_attn:.4f}")
    if result.alpha_shared is not None:
        print(f"Shared expert α: {result.alpha_shared:.3f}, SR/d: {result.srd_shared:.4f}")
    print(f"Cross-expert alignment: {result.cross_expert_alignment_mean:.3f}")
    print(f"EPR (energy equipartition): {result.epr_mean:.4f}")
    print(f"ψ (order param): MoE={result.psi_moe:.4f}, Attn={result.psi_attn:.4f}")
    if result.router_srd_mean is not None:
        print(f"Router SR/d: {result.router_srd_mean:.4f}")

    # Save
    if output_path:
        output = _serialize(result)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(output) + "\n")
        print(f"Saved to {output_path}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _serialize(result: MoECheckpointSpectral) -> dict:
    """Serialize to JSON-compatible dict."""
    return {
        "model_name": result.model_name,
        "step": result.step,
        "revision": result.revision,
        "total_params": result.total_params,
        "active_params": result.active_params,
        "num_experts": result.num_experts,
        "top_k": result.top_k,
        "has_shared_expert": result.has_shared_expert,
        "hidden_dim": result.hidden_dim,
        # Core spectral metrics
        "alpha_mean": result.alpha_mean,
        "alpha_std_across_experts": result.alpha_std_across_experts,
        "srd_mean": result.srd_mean,
        "srd_std_across_experts": result.srd_std_across_experts,
        "alpha_attn": result.alpha_attn,
        "srd_attn": result.srd_attn,
        "alpha_moe": result.alpha_moe,
        "srd_moe": result.srd_moe,
        "alpha_shared": result.alpha_shared,
        "srd_shared": result.srd_shared,
        # Cross-expert [N1, N11]
        "cross_expert_alignment_mean": result.cross_expert_alignment_mean,
        # Router health [N2]
        "router_srd_mean": result.router_srd_mean,
        # Energy equipartition [N3]
        "epr_mean": result.epr_mean,
        # Order parameter [U5]
        "psi_mean": result.psi_mean,
        "psi_attn": result.psi_attn,
        "psi_moe": result.psi_moe,
        # State equation [U1]
        "pv_over_nt_total": result.pv_over_nt_total,
        "pv_over_nt_active": result.pv_over_nt_active,
        # Counts
        "n_layers_measured": result.n_layers_measured,
        "n_expert_alphas": len(result.per_expert_alphas),
        "alpha_expert_min": float(np.min(result.per_expert_alphas)) if result.per_expert_alphas else None,
        "alpha_expert_max": float(np.max(result.per_expert_alphas)) if result.per_expert_alphas else None,
        "alpha_expert_median": float(np.median(result.per_expert_alphas)) if result.per_expert_alphas else None,
        # Volume stats [N3]
        "volume_expert_cv": float(np.std(result.per_expert_volumes) / np.mean(result.per_expert_volumes)) if result.per_expert_volumes and np.mean(result.per_expert_volumes) > 0 else None,
        # Per-layer detail
        "per_layer_summary": [
            {
                "layer": l.layer_idx,
                "alpha_mean": l.alpha_mean,
                "alpha_std": l.alpha_std,
                "srd_mean": l.srd_mean,
                "alignment": l.cross_expert_alignment,
                "epr": l.epr,
                "psi_mean": l.psi_mean,
                "router_srd": l.router_srd,
            }
            for l in result.layers
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure spectral properties of MoE models")
    parser.add_argument("model", help="HuggingFace model ID or local path")
    parser.add_argument("--revision", default="main", help="Git revision / checkpoint tag")
    parser.add_argument("--step", type=int, default=0, help="Training step number")
    parser.add_argument("--hidden-dim", type=int, default=0, help="Hidden dimension (auto-detect if 0)")
    parser.add_argument("--num-experts", type=int, default=0, help="Number of experts (auto-detect if 0)")
    parser.add_argument("--top-k", type=int, default=0, help="Experts per token (auto-detect if 0)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max-experts", type=int, default=0, help="Max experts to measure per layer (0=all)")
    parser.add_argument("--no-alignment", action="store_true", help="Skip cross-expert alignment computation")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL file path")

    args = parser.parse_args()

    measure_moe_checkpoint(
        model_name_or_path=args.model,
        revision=args.revision,
        step=args.step,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k_routing=args.top_k,
        device=args.device,
        max_experts_per_layer=args.max_experts,
        measure_alignment=not args.no_alignment,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
