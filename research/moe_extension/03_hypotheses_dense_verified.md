# 在 Dense 上已验证的发现 → MoE 重新验证方案

对我们在 Dense 模型上已经确认的每个发现，分析其在 MoE 架构下的预期行为、可能的偏差、以及具体验证方案。

---

## F1: SR/d 通用压缩定律

### Dense 结论
SR/d → 0.040 + 0.61/√d，所有 Transformer 训练精确压缩 ~2 nats 谱熵（ΔH₂ ≈ -2.04 ± 0.17）。

### MoE 预期
**假设 M1a**: 逐专家 SR/d 仍然收敛，但收敛值可能系统性更低。

理由：
- Jacobian-PCA 研究表明专家局部结构更低秩（SD-MoE, 2026）
- 每个专家只看到 token 子集 → 更集中的表示 → 更低的有效维度
- 共享专家（如果存在）看到所有 token → 应接近 dense 的 SR/d 值

**假设 M1b**: SR/d 的收敛可能取决于每专家的隐藏维度 d_expert，而非全局 d_model。

例如 OLMoE: d_model = 2048, 但每个专家 FFN 的中间维度不同。需要区分使用哪个 d 来归一化。

### 验证方案
```
模型: OLMoE-1B-7B (244 checkpoints, 64 experts, d=2048)
对照: OLMo-2-1B (dense, d=2048, 已有数据)
测量: 每个专家的 gate_proj, up_proj, down_proj 的 SR/d
分析:
  1. 逐专家 SR/d 轨迹是否收敛？
  2. 收敛值是否在 dense 的 0.04-0.07 范围内？
  3. 共享专家 vs 路由专家的 SR/d 是否有系统差异？
  4. ΔH₂ 是否仍然 ≈ -2 nats？
```

---

## F2: α Reversal 早期预警

### Dense 结论
当 dα/dt > 0 连续 3+ 次时，表示结构退化。OLMo-2-13B 展示了从 4.25 到 6.95 的 α reversal。

### MoE 预期
**假设 M2a**: α reversal 在 MoE 中是 expert-specific 的，不会在所有专家同时发生。

理由：
- 高负载专家可能因过拟合高频模式而先反转
- 低负载专家可能因梯度信号不足而先反转
- 中等负载专家最稳定

**假设 M2b**: MoE 的全局 α 可能掩盖专家级别的 reversal — 当 50% 的专家 reversal 而另外 50% 仍在改善时，全局平均可能不变。

这意味着逐专家监控比全局监控更重要。

### 验证方案
```
模型: OLMoE-1B-7B (244 checkpoints)
测量: 每个专家每层的 α 轨迹
分析:
  1. 是否存在 expert-specific reversal？在哪些 checkpoint 出现？
  2. Reversal 是否与 expert load 相关？（需要路由统计数据）
  3. 全局 α 是否掩盖了局部 reversal？
  4. 能否用逐专家 reversal 构建更灵敏的预警系统？
```

---

## F3: 下游性能预测 (ρ = -0.90)

### Dense 结论
SR/d 与下游 benchmark 的 Spearman ρ = -0.90, R² = 0.84。

### MoE 预期
**假设 M3**: SR/d 的预测力在 MoE 中可能更弱或需要修正。

理由：
- MoE 的有效参数不等于总参数 → SR/d 需要用 active params 归一化
- 专家间的谱冗余（SD-MoE 发现）意味着直接平均可能不准确
- 更好的指标可能是"有效专家 SR/d" = 考虑专家多样性后的加权 SR/d

### 验证方案
```
模型: Mixtral-8x7B, Phi-3.5-MoE, OLMoE (有benchmark数据)
对照: Mistral-7B, Phi-3-mini (dense 等价)
测量: 多种 SR/d 聚合方式：
  - 简单均值
  - 参数加权均值  
  - 考虑路由频率的加权均值
  - 仅共享专家的 SR/d
分析: 哪种聚合方式与下游性能相关性最高？
```

---

## F4: 结构相变 N ≈ 1.7B

### Dense 结论
N ≈ 1.7B 处存在 sharp sigmoid phase transition，小模型容易达到 α < 3，大模型即使 D/N > 300 也停留在 α > 5。

### MoE 预期
**假设 M4a**: 相变阈值取决于 per-expert 参数量而非总参数量。

OLMoE: 6.9B total / 64 experts ≈ 108M per expert → 远低于 1.7B → 每个专家应该容易结构化。

**假设 M4b**: 相变阈值取决于激活参数量。

OLMoE: 1.3B active → 接近但低于 1.7B → 处于过渡区。

**假设 M4c**: 相变阈值取决于"有效参数" = f(total, active, routing_quality)。

### 验证方案
```
模型: 跨多个 MoE 模型的 final α:
  - OLMoE (6.9B total, 1.3B active)
  - Phi-3.5-MoE (42B total, 6.6B active)
  - Mixtral 8x7B (47B total, 13B active)
  - Mixtral 8x22B (141B total, 39B active)
  - DBRX (132B total, 36B active)
  - DeepSeek-V2 (236B total, 21B active)
分析:
  1. 绘制 α vs total_params: 是否在 N_total ≈ 1.7B 处有相变？
  2. 绘制 α vs active_params: 阈值是否不同？
  3. 绘制 α vs per_expert_params: 阈值是否更好对齐？
  4. Sigmoid 拟合: 哪种参数量定义给出最高 R²？
```

---

## F5: MLP 结构瓶颈 (α_mlp >> α_attn)

### Dense 结论
Attention 层已结构化 (α < 4)，MLP 层仍随机 (α > 7)，gap 最高达 5.43。

### MoE 预期
**假设 M5a**: α_attention 在 dense 和 MoE 中相似（attention 层通常不做 MoE 稀疏化）。

**假设 M5b**: α_moe_expert < α_mlp_dense（在匹配激活参数时），因为每个专家处理更聚焦的 token 子集，有利于更深的自正则化。

**假设 M5c**: 共享专家的 α 接近 α_mlp_dense（因为它看到所有 token）。

### 验证方案
```
模型: DeepSeek-V2 (有 shared + routed experts)
对照: Mistral-7B (dense, 同 d=4096, 已有数据)
测量:
  - attention 层 α (两个模型)
  - MoE routed expert α (DeepSeek)
  - shared expert α (DeepSeek)
  - dense MLP α (Mistral)
分析: MLP 瓶颈在 MoE 中是否被缓解？
```

---

## F6: α-Guided Adaptive Schedule

### Dense 结论
基于 dα/dt > 0 触发 LR decay，410M +1.95%，1B +2.56%。

### MoE 预期
**假设 M6**: α-guided 在 MoE 中可能需要修改：用全局 α 还是逐专家 α 的某种聚合？

可能的改进方向：
- 用"最差专家的 α"作为触发信号（最脆弱的专家决定全局）
- 用"中位专家的 α"（更鲁棒）
- 用"共享专家的 α"（如果存在，它是最 dense-like 的）

### 验证方案
这需要实际训练 MoE 模型，成本较高。可以先在 OLMoE 训练 checkpoint 上做 post-hoc 分析：
```
数据: OLMoE 244 checkpoints 的 α 轨迹
分析: 
  1. 不同 α 聚合方式在哪个 checkpoint 触发 decay？
  2. 回溯分析: 如果在这些点开始 decay，预期的下游性能改善是多少？
```
