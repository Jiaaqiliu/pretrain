# MoE Spectral Thermodynamics: Research Overview

> 将 "Beyond Loss Curves" 的谱热力学框架从 Dense 模型扩展到 Mixture-of-Experts 架构。
> 这是论文的核心扩展方向，填补一个完全空白的研究领域。

## 核心动机

截至 2026 年中，**没有任何人将 WeightWatcher/HTSR 的 α 分析或 SR/d 测量应用于 MoE 专家权重矩阵**。这是一个完全开放的研究空白。同时，前沿模型（GPT-4/5、Gemini、Grok-5、DeepSeek-V3/V4、Kimi K2 等）几乎全部采用 MoE 架构，使得 Dense-only 的结论在应用覆盖面上存在根本性局限。

## 文件索引

| 文件 | 内容 |
|------|------|
| [01_frontier_moe_models.md](01_frontier_moe_models.md) | 2024-2026 前沿 MoE 模型全景调研 |
| [02_spectral_moe_literature.md](02_spectral_moe_literature.md) | MoE 谱分析相关文献综述 |
| [03_hypotheses_dense_verified.md](03_hypotheses_dense_verified.md) | 在 Dense 上已验证的发现，在 MoE 上的重新验证方案 |
| [04_hypotheses_dense_unverified.md](04_hypotheses_dense_unverified.md) | 在 Dense 上未验证/被否定的假设，在 MoE 上的新机会 |
| [05_hypotheses_moe_novel.md](05_hypotheses_moe_novel.md) | MoE 特有的全新假设 |
| [06_experiment_plan.md](06_experiment_plan.md) | 分阶段实验方案与优先级 |
| [07_measurement_code.md](07_measurement_code.md) | 测量代码使用指南 |

## 现有结果总结（Dense 模型）

| 发现 | 状态 | Dense 结果 |
|------|------|-----------|
| SR/d 通用收敛 | ✅ 已验证 | 13 模型, 4 架构, SR/d → 0.040 + 0.61/√d |
| α reversal 早期预警 | ✅ 已验证 | OLMo-2-13B Δα = +2.71 |
| 下游性能预测 | ✅ 已验证 | ρ = -0.90, R² = 0.84 |
| 结构相变 N ≈ 1.7B | ✅ 已验证 | Sigmoid fit R² = 0.97 |
| α-guided schedule | ✅ 已验证 | 410M +1.95%, 1B +2.56% vs cosine |
| MLP 结构瓶颈 | ✅ 已验证 | α_mlp >> α_attn, gap 最高 5.43 |
| PV = NkT 状态方程 | ⚠️ 部分 | Liu & Tegmark 在小模型验证, 大模型未确认 |
| KWW 玻璃弛豫 | ❌ 未验证 | 计划中但未执行 |
| Gaussian schedule | ❌ 被放弃 | 原计划从最小熵产生原理推导, 改为 α-guided |
| 热力学效率对比 | ❌ 未验证 | WSD vs Cosine 的 ΔS_tot 未完成 |
