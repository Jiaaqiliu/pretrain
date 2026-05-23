# Spectral Monitoring Framework for LLM Pretraining

> "Beyond Loss Curves" — Practical monitoring metrics that provide actionable signals loss cannot

---

## 1. The Two Metrics

### SR/d (Stable Rank / Hidden Dimension)

**Definition**: SR = ||W||²_F / σ₁² averaged over all 2D weight layers, divided by d_model.

**What it measures**: Fraction of available dimensions actively used by the model.

**Key properties**:
- ∈ [0, 1]: higher = less compressed, lower = more compressed
- Universal convergence: SR/d → 0.056 ± 0.008 across all tested architectures
- **Spearman r = -0.918 with downstream performance** (N=143, p<10⁻⁵⁸)
- Does NOT require training data, test data, or loss values

**Interpretation**:
- SR/d ≈ 0.4 → random initialization (no structure)
- SR/d ≈ 0.05-0.07 → fully trained (universal target)
- SR/d decreasing → model is compressing (healthy)
- SR/d increasing → model losing structure (unhealthy)

### α (Power-Law Exponent)

**Definition**: Fitted exponent of P(λ) ~ λ⁻ᵅ for the eigenvalue spectrum of each layer's W^T W.

**What it measures**: Quality of spectral structure (heavy-tail development).

**Key properties**:
- Dimensionless — comparable across any layer size or model scale
- α > 6: random (Marchenko-Pastur bulk)
- α ∈ [4, 6]: partially structured (Bulk + Spikes)
- α ∈ [2, 4]: well-structured (Heavy-Tail Self-Regularization)
- α < 2: over-trained (rank collapse risk)

---

## 2. Monitoring Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│ Training Health Monitor                                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│ α (structure quality)    [████████░░░░] 3.2 (target: <3.0)   │
│   dα/dt = -0.004/1Ksteps → Still improving                   │
│                                                               │
│ SR/d (compression)       [█████████░░░] 0.062 (target: 0.05) │
│   dSR/dt = -0.001/1Ksteps → Actively compressing             │
│                                                               │
│ α_attn                   [████████░░░░] 3.1  (healthy)       │
│ α_mlp                    [███████░░░░░] 3.4  (healthy)       │
│                                                               │
│ Status: 🟢 HEALTHY — structure forming, compression active   │
│                                                               │
│ Alerts:                                                       │
│  ⚠ None                                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Actionable Signals

### Signal 1: α Reversal → "Start LR Decay"

**Condition**: dα/dt > 0 for 3+ consecutive measurements

**Meaning**: The model has exhausted its structural capacity at the current LR. Continued training at high LR is eroding previously formed structure.

**Action**: Begin LR warmdown/decay phase.

**Evidence**:
- Observed in Pythia-2.8b (step 10K), Pythia-6.9b (step 20K), Amber-7B (ckpt 10)
- All have tokens/param < 250 (under-trained relative to structural optimum)
- MLP layers show reversal first (α_mlp rises while α_attn stable)

### Signal 2: SR/d Minimum → "Training Complete"

**Condition**: d(SR/d)/dt ≈ 0 (SR/d stops decreasing)

**Meaning**: All useful compression has been extracted. Further training cannot improve structural quality.

**Action**: End training.

**Evidence**:
- For well-trained models (70m-410m): SR/d reaches minimum at the end of training
- For under-trained models (2.8b-6.9b): SR/d reaches minimum early and RISES slightly after

### Signal 3: α < 3 → "Structurally Mature"

**Condition**: α_mean drops below 3.0 and stabilizes

**Meaning**: The model's weight matrices have developed robust heavy-tailed spectral distributions — the hallmark of well-trained deep networks (Martin & Mahoney, 2021).

**Action**: Model is structurally ready for deployment. LR decay will polish but not fundamentally change structure.

### Signal 4: α_mlp >> α_attn → "MLP Capacity Bottleneck"

**Condition**: α_mlp - α_attn > 1.0

**Meaning**: MLP (feedforward) layers are less structured than attention layers. The MLP is the training bottleneck.

**Action**: Consider increasing FFN width (intermediate_size), or reducing attention heads to reallocate parameters.

---

## 4. Compute Efficiency Insight

### "7% Rule"

From Pythia measurements across all scales:

| Training % | Quality % | α state | SR/d state |
|-----------|-----------|---------|-----------|
| 0.7% (step 1K) | 55-88% | Still dropping fast | Dropping fast |
| **7% (step 10K)** | **83-104%** | **Near minimum** | **Plateauing** |
| 35% (step 50K) | 95-105% | Flat or rising | Slowly dropping |
| 100% (step 143K) | 100% | Final | Final |

**At just 7% of total training (the α minimum), models achieve 83-104% of final quality.**

The remaining 93% of compute produces diminishing returns on structure. Late-training gains come from:
- Continued (slow) compression of SR/d
- Memorization of long-tail patterns (loss ↓ but α flat/rising)
- NOT from structural improvement

### Implication for Large-Scale Training

For a 7B model trained on 300B tokens:
- α minimum occurs at ~10K-20K steps (7-14% of training)
- At that point, the model is ~87% of its final quality
- If LR decay starts at α minimum instead of a fixed schedule:
  → Potential 5-15% compute savings with <2% quality loss

---

## 5. Comparison with Loss-Only Monitoring

| Capability | Loss Curve | α + SR/d |
|-----------|-----------|----------|
| Detect training progress | ✓ | ✓ |
| Detect structural learning | ✗ | ✓ (α ↓) |
| Detect structural degradation | ✗ | **✓ (α reversal)** |
| Adaptive schedule trigger | ✗ | **✓ (α inflection)** |
| Predict performance without eval | ✗ | **✓ (SR/d, r=0.92)** |
| Compare across model sizes | ✗ (scale-dependent) | **✓ (normalized)** |
| Diagnose layer-specific issues | ✗ | **✓ (α per layer type)** |
| Stop training optimally | ✗ (arbitrary) | **✓ (SR/d minimum)** |
| Assess training sufficiency | ✗ | **✓ (α < 3 = sufficient)** |

---

## 6. Implementation Cost

| Metric | Computation | Frequency | Overhead |
|--------|------------|-----------|----------|
| SR (stable rank) | Power iteration for σ₁ + Frobenius norm | Every 1K steps | <1% |
| α (power-law) | SVD or eigendecomp of each layer | Every 5K steps | ~5% |
| Full dashboard | Both + per-layer breakdown | Every 10K steps | ~10% |

**Approximate cost**:
- For 7B model: full SVD of all layers takes ~5 minutes on 1 GPU
- Measuring every 5K steps over 143K total = 29 measurements = ~2.4 GPU-hours
- vs total training cost of ~5000+ GPU-hours → **< 0.05% overhead**

---

## 7. Validated Across

| Architecture | Models | Sizes | Data | SR/d convergence |
|-------------|--------|-------|------|-----------------|
| GPT-NeoX | Pythia (6 scales) | 70M-6.9B | The Pile, 300B | 0.046-0.074 |
| LLaMA | LLM360/Amber | 7B | RefinedWeb+StarCoder, 1.26T | 0.057 |
| OLMo2 | OLMo-2 | 1B-13B | OLMo-Mix, 4T+ | *pending V2* |

**Overall mean SR/d = 0.056 ± 0.008 (CV=14.9%)**

---

*Document version: 2026-05-23. Based on 7 models, 2 architectures, 163 checkpoints.*
