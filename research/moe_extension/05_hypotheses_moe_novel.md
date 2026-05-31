# MoE 特有的全新假设

这些假设只在 MoE 架构中有意义，无法在 dense 模型上测试。
综合了两轮调研的全部发现，包括最新的统计物理/信息几何文献。

---

## N1: 专家坍缩的谱信号 (Spectral Precursors of Expert Collapse)

### 假设
专家坍缩（多个专家收敛到近乎相同的函数）在谱空间中产生可检测的前兆信号，先于 loss 异常和路由熵下降。

1. 坍缩专家的 α 值趋同（跨专家标准差 ↓）
2. 坍缩专家的主导奇异向量高度对齐（余弦相似度 > 0.95）
3. 坍缩专家的 SR/d 值趋同
4. 这些谱信号 **先于** 路由熵下降

### 物理类比
类似于铁磁相变的**逆过程**——从有序的多晶体退化为无差别的无定形态。统计物理中这种相变有明确的序参量前兆。

### 热力学解释
专家坍缩是**不可逆的** (Okanohara, 2026 Part II)：恢复专家多样性需要外部功（重新初始化、调整 load-balancing）。坍缩过程中，Hessian 有效秩下降，恢复所需的熵与丢失的秩成正比。

### 验证方案
```
数据: OpenMoE-8B (5 checkpoints: 200B, 400B, 600B, 800B, 1T tokens)
测量:
  1. 每层的 cross-expert α 方差: Var(α) across experts
  2. 每层的 cross-expert 主导子空间对齐: 逐对余弦相似度
  3. 路由熵: H(routing)
分析: Var(α) 下降 + alignment 上升 = 坍缩前兆信号
```

### 文献支持
- SD-MoE (2026.02): 专家间主导谱方向高度对齐 (>0.9) — arxiv.org/abs/2602.12556
- Stochastic Collapse (Chen et al., J. Stat. Mech. 2024): SGD 噪声驱动网络走向更简单的子网络 — 在 MoE 中表现为专家坍缩
- Thermodynamic Irreversibility (Ziyin et al., 2026.05): 不可逆性框架 — arxiv.org/abs/2605.21933

---

## N2: 路由矩阵的谱健康度 (Router Spectral Health)

### 假设
路由矩阵 G ∈ ℝ^{d×E} 的谱性质是 MoE 健康的**独立诊断信号**：

- Router SR/d → 0: 路由退化到只用少数方向 → 大量专家被浪费
- Router SR/d ≈ E/d: 路由充分利用所有方向 → 健康专家选择
- Router 谱范数过大: 路由对输入过于敏感 → 训练不稳定

### 物理类比
路由矩阵类似于分子筛/催化剂——它决定了输入"分子"被引导到哪个"反应器"。催化效率取决于其内部结构多样性。

### 文献支持
- SR-MoE (2026.01): 约束路由谱范数+stable rank → 已证实有效 — arxiv.org/abs/2601.03889
- Routing Absorption (2026.02): 路由信号可能被 Q/K/V 吸收 — arxiv.org/abs/2603.02227

### 验证方案
```
模型: OLMoE (64 experts, d=2048 → 路由矩阵 2048×64)
测量: 每层路由矩阵的 SR, 谱范数, 条件数
分析: Router SR 随训练变化 + 与 expert utilization 的关系
```

---

## N3: 专家能量均分定理 (Expert Energy Equipartition)

### 假设
如果 SGD 类似于热平衡的 Langevin 动力学，那么在训练稳态阶段，**能量应该在专家间近似均分**。

定义 Expert Energy Equipartition Ratio (EPR):
```
EPR = Var(||W_i||²_F) / <||W_i||²_F>²
```
- EPR ≈ 0: 均分（未特化/热平衡/"热死寂"）
- EPR >> 0: 强特化（偏离平衡 = 有序化）

### 物理意义
EPR 量化了 MoE 系统离最大熵态（所有专家等价）有多远。特化的专家具有不同"能量"，这种不均匀性是有序度的度量。

### 关键预测
模型质量与均分的关系不是线性的——**适度偏离均分**（EPR 在某个最优范围内）对应最好的泛化，过高或过低都不好。

### 文献支持
- PV=NkT for neural networks (Sadrtdinov et al., 2025): arxiv.org/abs/2511.07308
- Liquid and Solid Layers (2025): arxiv.org/abs/2506.06789 — 过参数化 MLP 发展出固/液层结构

### 验证方案
```
模型: OLMoE 244 checkpoints
测量: 每个专家每层的 ||W||²_F (moe_measures.py 已实现 frobenius_norm)
分析: EPR 随训练演化 + 与 routing entropy、下游性能的关系
成本: 几乎为零（测量中已包含）
```

---

## N4: 逐专家涨落-耗散定理 (Per-Expert Fluctuation-Dissipation)

### 假设
每个专家的"有效温度"取决于其实际处理的有效 batch size：
```
T_eff^(i) = η · σ²_grad^(i) / (2 · B_eff^(i))
```
其中 B_eff^(i) = B_total × routing_frequency_i。

### 关键推论
- 高频使用的专家: B_eff 大 → T_eff 低 → 更低噪声 → **更好结构形成** (低 α)
- 低频使用的专家: B_eff 小 → T_eff 高 → 更多噪声 → **结构形成困难** (高 α)

这可以解释为什么低频专家容易保持"随机"谱状态。

### 文献支持
- FDT for SGD: arxiv.org/abs/1810.00004
- Liu & Tegmark (2025): 通过 FDR 确认 learning rate = effective temperature

### 验证方案
```
模型: OLMoE (需要路由频率统计 + 梯度信息)
方法:
  1. 从路由统计计算每个专家的 B_eff
  2. T_eff^(i) 估计
  3. 验证: T_eff^(i) 与 α^(i) 的 Spearman 相关性
  预测: ρ(T_eff, α) > 0.7 (温度越高, α 越高)
```

---

## N5: MoE 训练的三个动力学阶段 (Three Dynamic Phases)

### 假设
MoE 训练可划分为三个谱可区分的动力学阶段：

**Phase I: 路由形成期 (Gas → Liquid)**
- cross-expert alignment 从高（所有专家类似）开始下降
- α 快速下降，SR/d 快速压缩
- 物理类比: 气体冷凝为液体（对称性破缺开始）

**Phase II: 专家特化期 (Liquid → Multi-Crystal)**
- 跨专家 α 方差增大，routing entropy 稳定
- α 继续下降但速率放缓
- 物理类比: 液体结晶为多晶体

**Phase III: 结构冻结期 (Crystal → Glass)**
- α 变化极小或 reversal，routing 模式固化
- SR/d 已收敛
- 物理类比: 玻璃化转变（结构冻结在亚稳态）

### 文献支持
- Liquid and Solid Layers (2025): 确认过参数化网络发展出固/液层结构
- Phase Diagram of SGD (2025): SGD 稳态分布展现相变和遍历性破缺 — Phys. Rev. E 111, 065303
- Phase Transitions Reveal Hierarchical Structure in DNNs (2025): arxiv.org/abs/2512.11866

### 验证方案
```
数据: OLMoE 244 checkpoints (5T tokens)
测量: 全局 α, SR/d, cross-expert α方差, cross-expert alignment, routing entropy
分析: 是否存在明确相变点？位置在哪？各阶段持续时间比例？
```

---

## N6: 共享专家 vs 路由专家的热力学不对称性

### 假设
在有共享专家的架构中，共享专家和路由专家展示根本不同的热力学行为。

**N6a**: 共享专家 = "固态"（constrained by all data, 低自由度）→ α 类似 dense MLP
**N6b**: 路由专家 = "液态"（only constrained by subset, 更多自由度）→ α 可能更低
**N6c**: 共享专家的 SR/d 在 dense 范围内，路由专家的 SR/d 系统性更低

### 文献支持
- Liquid and Solid Layers (2025): 过参数化层 = liquid, under-parameterized = solid
- SD-MoE (2026): 即使路由专家也共享主导谱方向 → "液态"中的长程有序

### 验证方案
```
模型: DeepSeek-V2 (2 shared + 160 routed)
      Qwen2-MoE-57B (shared + routed)
对照: Mistral-7B (dense, 同 d=4096)
测量: 分别 shared expert, routed expert, dense MLP 的 α, SR/d
```

---

## N7: MoE 的谱可塑性窗口 (Spectral Plasticity Window)

### 假设
每个专家存在"谱可塑性窗口"——一旦谱结构固化（α 稳定 + SR/d 收敛），该专家就难以通过进一步训练改变行为。

### 推论
- 如果在可塑性窗口关闭后才切换数据分布，已固化的专家无法适应
- 这可以解释 MoE 的 mid-training 困难

### 文献支持
- SPHERE (2025.05): 形式化了 MoE 谱可塑性损失 — arxiv.org/abs/2605.04712
- Spectral Collapse Drives Loss of Plasticity (NeurIPS 2025) — arxiv.org/abs/2509.22335

---

## N8: Supercollapse 在 MoE 中的验证 (Scaling Collapse Universality)

### 假设
Qiu et al. (ICML 2025 Oral) 发现 dense Transformer 的归一化 loss 曲线在不同模型规模上"坍缩"到同一条曲线（supercollapse）。

**N8a**: 如果 MoE 的 loss 曲线在按 **active compute** 归一化时也 supercollapse，则 MoE 和 dense 属于同一普适类。

**N8b**: 如果按 **total compute** 归一化才 supercollapse，则总参数量（包括非激活的）也在学习中发挥作用。

**N8c**: 如果 MoE 无法 supercollapse → MoE 定义了一个新的普适类。

### 文献支持
- Scaling Collapse (Qiu et al., ICML 2025 Oral): arxiv.org/abs/2507.02119
- Sparsity and Superposition in MoE (2025): MoE 不展现与 dense 相同的 sharp phase changes — arxiv.org/abs/2510.23671

### 验证方案
```
数据: 不同规模 MoE 的训练曲线（公开训练日志）
  - OLMoE (6.9B)
  - Phi-3.5-MoE (42B)
  - Mixtral (47B, 141B)
分析: 按三种归一化分别检查 loss curve collapse
```

---

## N9: 专家剪枝作为重整化群粗粒化 (Expert Pruning as RG Flow)

### 假设
MoE 专家剪枝（移除"不重要"的专家）在物理上等价于重整化群(RG)的粗粒化：积分掉短波长（专家特异性）涨落，保留长波长（共享）行为。

### 关键预测
剪枝后剩余专家的 α 应该 **下降**（更 heavy-tailed），因为剩余专家吸收了被剪枝专家的本质自由度。

具体形式: α(E) = α_∞ + c · E^{-1/ν}，其中 E 是专家数，ν 是相关长度指数。

### 文献支持
- RG for DNNs (2025): arxiv.org/abs/2510.25553
- AlphaPruning (NeurIPS 2024): 用 HTSR α 指导剪枝 — stat.berkeley.edu/~mmahoney/pubs/neurips-2024-alphapruning.pdf
- MoE Comprehensive Scaling (2025): 三维 MoE 缩放框架 — arxiv.org/abs/2509.23678

### 验证方案
```
方法: 对 Mixtral-8x7B 做逐步剪枝（8→7→6→...→2 experts）
每步测量剩余专家的 α
检验: α(E) 是否遵循 RG 幂律？
```

---

## N10: Clausius 不等式与训练不可逆性 (Clausius Inequality)

### 假设
MoE 训练中，Clausius 不等式 dS_total ≥ 0 约束了学习速度：
- loss 每步的改善量 ≤ 熵产生率 × 有效温度
- 高 T_eff 的专家有更松的约束 → 可以学得更快但浪费更多"热量"

### MoE 特有推论
- 专家坍缩事件产生 **熵产生尖峰**（不可逆熵突增）
- load-balancing loss 相当于"外部做功"以维持非平衡稳态

### 文献支持
- Thermodynamic Irreversibility (Ziyin et al., 2026): arxiv.org/abs/2605.21933
- Speed Limits for Deep Learning (Seroussi et al., 2023): arxiv.org/abs/2307.14653

### 验证方案
```
数据: OLMoE 244 checkpoints
计算: 累积熵产生沿训练轨迹 → 是否单调递增？
检验: "collapse events" 处是否有熵产生尖峰？
```

---

## N11: 信息几何与 Coherence Barrier

### 假设
Su & Liu (2026) 证明的 "Coherence Barrier"：当专家表示的互相关性高时，贪婪路由失败，专家正交性是最优的。

### 谱翻译
"Coherence" = 专家权重矩阵主导奇异向量的对齐度（我们的 cross-expert alignment）。因此：
- cross-expert alignment 越低 → 越接近正交性 → 路由越有效
- cross-expert alignment 高 → 路由退化为几乎随机选择

### 与 α 的联系
正交的专家更可能发展出不同的谱尾结构 → α 方差更大 → 更好特化

### 文献支持
- Coherence Barrier (Su & Liu, 2026): arxiv.org/abs/2601.03577
- SD-MoE (2026): 证实专家间主导子空间高对齐

### 验证方案
```
模型: 多个 MoE 模型
测量: cross-expert alignment vs routing entropy vs downstream performance
预测: alignment↓ → routing entropy↑ → performance↑
```

---

## 优先级排序（综合可行性、新颖性、影响力）

| # | 假设 | 优先级 | 可行性 | 新颖性 | 数据/成本 |
|---|------|--------|--------|--------|-----------|
| N5 | 三阶段动力学 | ⭐⭐⭐⭐⭐ | 高 | 极高 | OLMoE 244 ckpts |
| N3 | 能量均分 | ⭐⭐⭐⭐⭐ | 极高 | 高 | 几乎零成本(已有数据) |
| N1 | 专家坍缩谱信号 | ⭐⭐⭐⭐ | 高 | 高 | OpenMoE ckpts |
| N2 | 路由矩阵谱健康 | ⭐⭐⭐⭐ | 高 | 中 | 任何 MoE |
| N8 | Supercollapse 普适类 | ⭐⭐⭐⭐ | 高 | 极高 | 训练日志 |
| N4 | 逐专家 FDT | ⭐⭐⭐ | 中 | 高 | 需路由统计 |
| N6 | 共享 vs 路由 | ⭐⭐⭐ | 高 | 中 | DeepSeek-V2 |
| N11 | Coherence Barrier 验证 | ⭐⭐⭐ | 高 | 中 | 多模型对比 |
| N10 | Clausius 不等式 | ⭐⭐ | 中 | 高 | OLMoE ckpts |
| N7 | 谱可塑性窗口 | ⭐⭐ | 中 | 高 | OLMoE ckpts |
| N9 | RG 剪枝 | ⭐⭐ | 低 | 极高 | 剪枝实验 |
