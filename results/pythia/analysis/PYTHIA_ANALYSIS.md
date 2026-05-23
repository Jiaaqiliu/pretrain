# Pythia 全规模热力学测量分析报告

> 6 个规模 (70M-6.9B), 132 个 checkpoint, 全量 SVD 测量
> Generated: 2026-05-23

---

## 1. 实验概要

| 模型 | N (params) | Checkpoints | S_init → S_final | ψ_init → ψ_final | V growth |
|------|-----------|-------------|------------------|-------------------|----------|
| pythia-70m | 70.4M | 25 | 6.2232 → 5.9082 (ΔS=-0.315) | 0.003 → 0.196 | ×56.2 |
| pythia-160m | 162.3M | 23 | 6.6152 → 6.4408 (ΔS=-0.174) | 0.003 → 0.209 | ×31.9 |
| pythia-410m | 405.3M | 25 | 6.8914 → 6.8199 (ΔS=-0.071) | 0.002 → 0.184 | ×9.2 |
| pythia-1b | 1,011.8M | 15 | 7.5815 → 7.5378 (ΔS=-0.044) | 0.002 → 0.214 | ×5.4 |
| pythia-2.8b | 2,775.2M | 21 | 5.5433 → 5.4948 (ΔS=-0.049) | 0.002 → 0.188 | ×5.0 |
| pythia-6.9b | 6,857.3M | 23 | 5.5440 → 5.4917 (ΔS=-0.052) | 0.001 → 0.178 | ×1.6 |

所有模型: cosine schedule, 143K steps, 300B tokens (The Pile deduped), WD=0.1, batch=2M tokens

---

## 2. 核心发现

### 发现 1: ψ_final 在 ~0.2 饱和 — 不存在 ψ(N) scaling law

| Model | N | ψ_final |
|-------|---|---------|
| 70m | 70.4M | 0.196 |
| 160m | 162.3M | 0.209 |
| 410m | 405.3M | 0.184 |
| 1b | 1,011.8M | 0.214 |
| 2.8b | 2,775.2M | 0.188 |
| 6.9b | 6,857.3M | 0.178 |

Power law fit: ψ(N) = 0.315 × N^{-0.023}, R² = 0.31 (极差)

**结论**: ψ_final ∈ [0.178, 0.214], 仅 18% 变化范围跨越 100× 参数量。ψ 是训练"充分度"的指标，不是规模的指标。

### 发现 2: ΔS 随规模急剧减小 — "大模型谱冻结"

| Model | ΔS | ΔS/S_init (相对变化) |
|-------|-----|---------------------|
| 70m | -0.315 | -5.1% |
| 160m | -0.174 | -2.6% |
| 410m | -0.071 | -1.0% |
| 1b | -0.044 | -0.6% |
| 2.8b | -0.049 | -0.9% |
| 6.9b | -0.052 | -0.9% |

大模型 (≥410m) 的谱熵变化 < 1%。训练不显著改变大模型的奇异值分布全局形态。

### 发现 3: V_ratio(N) 是最强的 scaling signal

| Model | V_final/V_init | log₁₀(N) |
|-------|----------------|-----------|
| 70m | 56.2 | 7.85 |
| 160m | 31.9 | 8.21 |
| 410m | 9.2 | 8.61 |
| 1b | 5.4 | 9.01 |
| 2.8b | 5.0 | 9.44 |
| 6.9b | 1.6 | 9.84 |

近似关系: V_ratio ∝ N^{-0.8} (清晰的 power law)

### 发现 4: ψ 增长在 warmup (前 1000 步) 就完成 83-100%

| Model | Δψ_warmup | Δψ_stable | Δψ_late | warmup占总Δψ比例 |
|-------|-----------|-----------|---------|-----------------|
| 70m | +0.139 | +0.009 | +0.006 | 90% |
| 160m | +0.136 | +0.015 | +0.009 | 83% |
| 410m | +0.116 | +0.004 | +0.001 | 96% |
| 2.8b | +0.166 | -0.008 | +0.000 | 100% |
| 6.9b | +0.118 | +0.004 | +0.001 | 94% |

前 0.7% 的训练步数 (1000/143000) 就建立了绝大部分低秩结构。

### 发现 5: PV/(NT) state equation — 原始形式不成立

| Model | k_eff 均值 | CV (变异系数) |
|-------|-----------|-------------|
| 70m | 3.84 | 100% |
| 160m | 2.17 | 97% |
| 410m | 1.23 | 79% |
| 1b | 0.42 | 41% |
| 2.8b | 0.71 | 77% |
| 6.9b | 0.62 | 57% |

CV > 40% 在所有规模上，PV/(NT) 不收敛。原因: LR 作为温度代理不够准确 (cosine 下 LR 持续变化，V 同时增长)。

---

## 3. 与其他实验数据的综合对比

| Model | Family | N | ψ_final | Tokens | Schedule |
|-------|--------|---|---------|--------|----------|
| 190M (自训) | OLMo2 | 267M | 0.079 | 25B | WSD |
| Pythia-70m | GPT-NeoX | 70M | 0.196 | 300B | Cosine |
| Pythia-6.9b | GPT-NeoX | 6.9B | 0.178 | 300B | Cosine |
| OLMo-2-1B | OLMo2 | 1B | 0.182 | 4T+ | Cosine |
| OLMo-2-7B | OLMo2 | 7B | 0.213 | 4T+ | Cosine |
| OLMo-2-13B | OLMo2 | 13B | ~0.23 | 5T+ | Cosine |

**关键结论**:
1. 190M 的 ψ=0.079 远低于其他 → 原因是训练 token 太少 (25B vs 300B+)
2. Pythia 全家族 ψ ≈ 0.18-0.21 → 架构+训练达到同一平衡点
3. OLMo-2 也在同一范围 (0.18-0.23) → 跨架构的通用性
4. 唯一例外: OLMo-2-13B (~0.23) 略高 → 可能是 Stage 2 训练的效果

---

## 4. 对论文假设的最终判定

| # | 原始预测 | 实验结果 | 判定 |
|---|---------|---------|------|
| P1 | S 单调下降 | S 确实下降，但大模型变化极小 (<1%) | ⚠️ 技术上成立，但无实际意义 |
| P2 | ψ 随训练上升 | ✓ 确实上升 (0.001 → 0.2) | ✓ 成立 |
| P3 | ψ(N) 有 scaling law | ✗ ψ 饱和在 ~0.2，不随 N 变化 | ✗ 不成立 |
| P4 | PV/(NT) 收敛 | ✗ CV > 40%，不收敛 | ✗ 不成立 |
| P5 | Gaussian < WSD 熵产 | 未在此实验测试 | — 待 3B 结果 |

---

## 5. 论文策略建议

### 可发表的新发现 (positive results):

1. **ψ universality**: 训练充分的模型 ψ ≈ 0.2 (跨 Pythia + OLMo-2, 70M-13B)
2. **V_ratio scaling law**: V_growth ∝ N^{-0.8} — 明确的 power law
3. **Instant ordering**: 低秩结构在前 1000 步 (0.7% 训练) 形成 90%+
4. **Large-model spectral freeze**: ≥410M 模型的谱熵变化 < 1%
5. **Training sufficiency indicator**: ψ < 0.15 → 训练不足; ψ ≈ 0.2 → 充分训练

### 需要修正的主张:

1. ~~ψ(N) scaling law~~ → ψ universality constant
2. ~~PV/(NT) = k_eff(N) state equation~~ → 需要更好的温度定义
3. ~~Decay phase 是结构形成关键~~ → warmup/early phase 才是关键

---

## 6. 下一步实验方向

1. **Dense sampling (step 0-1000)**: 用 Pythia 的 log-spaced early checkpoints 精确刻画 ψ 从 0 到 0.2 的 phase transition
2. **真实温度计算**: 用 gradient variance 替代 LR 作为温度，重新验证 state equation
3. **下游 benchmark 相关性 (E5)**: 用 Pythia evals 数据计算 ψ 与 accuracy 的 Spearman r
4. **3B 自训**: 等待完成，看 schedule 差异在更大规模上是否可测

---

## 7. 数据文件索引

```
results/pythia/
├── pythia_70m.jsonl        (25 records, 12KB)
├── pythia_160m.jsonl       (23 records, 11KB)
├── pythia_410m.jsonl       (25 records, 12KB)
├── pythia_1b.jsonl         (15 records, 7KB)
├── pythia_2.8b.jsonl       (21 records, 11KB)
├── pythia_6.9b.jsonl       (23 records, 12KB)
└── analysis/
    ├── E1_state_equation.json
    ├── trajectory_summary.json
    ├── training_phases.json
    └── PYTHIA_ANALYSIS.md (本文档)
```
