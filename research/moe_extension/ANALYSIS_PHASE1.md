# Phase 1 深度分析：假设验证、修正与新解释

> 基于 OLMoE-1B-7B 10 个 checkpoint (step 5K–1.22M) 的谱测量数据  
> 2026-06-02

---

## 一、假设验证状态

### ✅ 已验证

| 假设 | 证据 | 置信度 |
|------|------|--------|
| **M1a**: SR/d 逐专家收敛 | SR/d 从 0.047 收敛到 ~0.053，与 Dense 公式预测 (0.0535) 偏差仅 -2.3% | 高 |
| **N2**: 路由矩阵低秩且稳定 | Router SR/d ≈ 0.0033 全程不变 | 高 |
| **N3**: 能量均分 → 训练促进 | EPR 从 0.0076 → 0.0018 最低点，U-型曲线 | 高 |
| **U5**: 序参数 ψ 追踪特化 | ψ 单调上升 +15.5%，是 specialization 的独立度量 | 高 |

### ❌ 被否定或需大幅修正

| 假设 | 预期 | 实际 | 修正方向 |
|------|------|------|---------|
| **M2a**: α reversal 是 expert-specific | 预期存在 reversal | **α 全程极稳定** (Δ < 0.3%)，无任何 reversal | MoE 不存在 α reversal 这一概念 |
| **M2b**: 全局 α 掩盖局部 reversal | 预期 σ(α) 变化揭示局部 reversal | σ(α) 也稳定 (0.161-0.169) | 彻底否定 |
| **U2-MoE-c**: 三阶段动力学 (gas→liquid→glass) | α 大幅下降 + 特化过程 | α 几乎不动，没有 "gas→liquid" 相变 | MoE 从初始化起就是"固态"，不经历相变 |
| **N5**: 明确的三阶段 | routing formation → specialization → saturation | 只看到 2 个阶段，且都非常平缓 | 需重新定义 MoE 的动力学阶段 |

### ⚠️ 部分验证 / 需要更多数据

| 假设 | 状态 | 下一步 |
|------|------|--------|
| **M1b**: SR/d 依赖 d_expert 而非 d_model | 无法在单一模型上验证 | 需 Phase 2 跨模型 |
| **M4**: 相变阈值 | OLMoE 的 α < 2 与 Dense 的 α > 3 完全不同的 regime | 需跨模型确认 MoE 是否独立于参数量 |
| **M5**: MLP 瓶颈 | α_attn = 1.29 vs α_moe = 1.45, 差异方向反转! | Dense: α_mlp >> α_attn; MoE: α_attn < α_moe |
| **U2**: KWW 弛豫 | SR/d 拟合 beta=1.85 (>1)，不是典型玻璃弛豫 | β>1 说明"超指数"收敛，与玻璃态不同 |
| **N7**: 谱可塑性窗口 | Router 极早固化 (step 5K 已定型) | 可塑性窗口可能只在最初几千步 |

---

## 二、需要大幅修正的理论框架

### 2.1 MoE 不存在 Dense 意义上的"结构相变"

**原始假设**: MoE 训练类似 Dense——先有高 α (随机) → 训练后 α 下降 (结构化) → 可能 reversal (退化)

**实际观察**: α ≈ 1.45 从头到尾。**MoE 没有经历相变**。

**修正解释**: MoE 专家的结构从**初始化阶段就被确定**了。每个专家只有 108M 参数 (6.9B/64)，远低于 Dense 的相变阈值 1.7B。小参数量 + top-k routing 的信息瓶颈使得：
- 专家从一开始就只能在低秩子空间中表达
- α < 2 (Lévy stable) 是小规模矩阵 + 有限信息的自然结果
- **训练改变的不是"什么结构"而是"结构的方向"**

### 2.2 "两阶段"替代"三阶段"

**实际观察的两个阶段**:

1. **Compression Phase** (step 0-400K, ~1.7T tokens):
   - SR/d 快速上升 (0.047 → 0.053)
   - EPR 快速下降 (0.0076 → 0.0029)
   - α 微降 (1.463 → 1.451)
   - 物理：能量重新分配 + 共同压缩子空间形成

2. **Specialization Phase** (step 400K+):
   - SR/d 稳定并微降 (0.053 → 0.052)
   - EPR 开始回升 (0.0018 → 0.0034)
   - ψ 持续上升
   - σ(SR/d) 持续增大 (+37%)
   - 物理：专家开始分化功能，同时维持整体结构稳定

### 2.3 α reversal 在 MoE 中无意义

**为什么 MoE 没有 reversal**: α reversal 在 Dense 模型中反映的是"过拟合导致的结构退化"。但在 MoE 中：
- 每个专家只看 ~12.5% 的 token (top-8/64)
- 这天然地提供了正则化效果
- 专家的"过拟合"不表现为 α 上升，而表现为 EPR 偏离和 ψ 分化

**实际观察**: α 确实有极微小的上升趋势 (step 140K 后 +0.001/100K steps)，但幅度小到可以忽略。如果需要早期预警，**应该监控 EPR 和 σ(SR/d) 而非 α**。

### 2.4 KWW 弛豫 β > 1：不是玻璃，是"弹道收敛"

SR/d 的 KWW 拟合给出 β = 1.85，这**不是**玻璃态弛豫 (β < 1) 的特征。

- β < 1: stretched exponential (多尺度、慢弛豫) — 典型玻璃行为
- β = 1: 普通指数衰减
- **β > 1: compressed exponential (超指数收敛) — 弹道动力学**

这意味着 MoE 的 SR/d 收敛是"弹道式"的——快速到达目标然后停止。这与 Dense 模型的缓慢、多尺度收敛完全不同。

**物理解释**: 专家之间的弱耦合允许每个专家独立、快速地收敛到自己的局部最优，而不是像 Dense 那样受全局耦合拖累。这验证了 U2-MoE-b 的精神（但方向相反：不是 β 更接近 1，而是 β > 1）。

---

## 三、新发现与新解释

### 3.1 "MoE 是从初始化就确定的固态系统"

核心洞察：MoE 不是一个经历"液→固"相变的系统，而是一个**从出生就是固态**的系统，训练只是让晶粒（专家）旋转到正确方向。

证据链：
- α < 2 从 step 5K 起就不变 → 结构从初始化就确定
- Router SR/d = 0.003 全程不变 → 路由策略极早固化
- 变化的只是 ψ (方向) 和 σ(SR/d) (分化程度)

### 3.2 "信息瓶颈决定 α 而非训练决定 α"

在 Dense 模型中，α 由训练动力学决定：初始随机 (α高) → 训练压缩 (α低)。  
在 MoE 中，**α 由架构的信息瓶颈决定**：

- 每个专家维度 d=2048 但 intermediate_size=1024
- 每个专家只看 1/8 的 token
- 这两个约束使得有效自由度极少 → 权重自然形成 heavy-tail (α≈1.45)

**推论**: 如果这个解释正确，那么改变 MoE 的 top-k 或 intermediate_size 应该系统性地改变 α 值。具体预测：
- top-k 增大 → 更多 token → α 上升（更接近 Dense）
- intermediate_size 增大 → 更多自由度 → α 上升

### 3.3 "EPR 是 MoE 健康的最灵敏指标"

EPR (能量均分比) 比 α 和 SR/d 变化幅度大得多：
- α: 变化 0.3%
- SR/d: 变化 12%
- **EPR: 变化 76%** (从 0.0076 降到 0.0018)

而且 EPR 的 U-型曲线精确标记了训练从"均衡化"到"分化"的转折点 (step ~680K)。

**实用价值**: 在 MoE 训练监控中，EPR 可能比 α 或 loss curve 更早发现问题（如专家坍缩、load imbalance）。

### 3.4 "SR/d 通用公式跨架构成立"但有微妙偏差

SR/d 收敛到 0.053 vs Dense 预测 0.0535（偏差 -2.3%）。

但有趣的是 SR/d 在 step 680K 达到峰值 0.0532 后**微微下降**到 0.0523。这可能意味着：
- Dense 的公式是渐近下界 (asymptotic lower bound)
- MoE 的 specialization phase 造成额外的压缩 → SR/d 略低于 Dense 预测
- 或者：Dense 公式中的 d 对于 MoE 不应该用 hidden_dim，而是某个"effective d" = f(hidden_dim, intermediate_size, top_k)

---

## 四、文献对照与关键问题

### Q1: α < 2 在文献中的含义

**Martin & Mahoney HT-SR 理论 (JMLR 2021, Nature Comm 2021)**:
- α < 2 对应 "Very Heavy Tailed" (VHT) 普适类
- 在 Dense 模型中，α < 2 通常标志**过拟合/过训练** — 少数极端方向主导
- 物理意义：方差发散 (Lévy regime)，极端事件支配分布，缺乏良好定义的尺度

**HT-SR 理论中 α < 2 的产生条件** (Hodgkinson & Mahoney, 2020):
- SGD 的乘性噪声 (multiplicative noise) 产生 heavy-tailed 平稳分布
- 当插值阈值 P=N 被穿越时 (double descent peak)，α 可降至 ~1.5
- Simsekli et al. (2019): SGD 梯度噪声本身就是 α-stable (非高斯)

**关键争议**: 在 Dense 模型中 α < 2 = "过拟合"。但在 MoE 中：
- OLMoE 是一个 well-trained 模型 (5T tokens, 良好的 benchmark 性能)
- 它的 α < 2 不是过拟合的结果，而是**架构约束的结果**
- 每个专家只有 108M 参数 + 只看 12.5% token → 天然的"过参数化"效应被消除
- **这挑战了 HT-SR 理论的"α < 2 = 过拟合"解释，提出了新的可能：α < 2 = 信息瓶颈的必然结果**

**文献空白**: 没有人在 MoE 专家矩阵上单独测量过 α。我们的测量是**首次在 MoE 专家级别报告 α < 2 的稳定存在**。

### Q2: Compressed Exponential (β > 1) — 全新发现

**文献搜索结果: 没有任何 ML/DNN 文献报告过 β > 1 的训练动力学弛豫**。

β > 1 在凝聚态物理中出现于：
- 非晶固体的应力驱动弛豫 (strain-driven relaxation)
- 胶体凝胶、堵塞软物质中的内部应力释放
- 共同特征：弛豫由内部应力驱动 (非热涨落驱动)

**对应到 MoE 训练的解释**: β > 1 意味着 SR/d 的收敛不是由随机梯度噪声驱动 (β < 1, 扩散式)，而是由内部结构约束的"弹道式"确定性收敛。这与 MoE 专家间弱耦合、各自独立收敛的图像一致。

**这可能是一个可发表的独立观察**。

### Q3: MoE Routing 早期固化 — 已有文献支持

**OpenMoE (Xue et al., 2024)**: "Token-to-expert assignments are determined early in pre-training and remain largely unchanged."

**FLAME-MoE (Kang et al., 2025)**: "Routing behavior stabilizes early in training."

**Three Phases of Expert Routing (Mouzouni, 2026)**: 确认三阶段: surge → stabilization → relaxation。

我们的 Router SR/d ≈ 0.003 全程不变的观察**为这些文献提供了谱证据**：路由矩阵的内部结构从 step 5K 起就不变化，不仅路由决策固化，路由矩阵的几何结构也固化。

**Lottery Ticket 联系**: 文献中尚未有人明确将 MoE 路由早期固化与 Lottery Ticket Hypothesis 联系。这是一个潜在的理论方向。

### Q4: Per-expert 谱分析 — 文献空白

Martin & Mahoney 的 WeightWatcher 已被广泛应用于 Dense 模型 (CNN, Transformer, BERT, GPT)，但**没有人对 MoE 的单个专家矩阵做过 α 测量**。

**我们的工作填补了一个明确的文献空白**。

---

## 五、Phase 2 决策

### 值得做 Phase 2 的理由

1. **验证 Q1**: α < 2 是否普遍。如果 Mixtral (大专家) 的 α > 2 而 OLMoE (小专家) 的 α < 2，那就有一个清晰的 story：**per-expert size 决定 α regime**

2. **验证 SR/d 公式**: 不同 hidden_dim 的模型 (2048, 3584, 4096, 6144) 可以验证 SR/d = 0.040 + 0.61/√d 是否在 MoE 上成立

3. **M4 相变图**: Dense 的 sigmoid 相变发生在 N ≈ 1.7B 参数处。MoE 该用什么 N 来做类比？Phase 2 可以给出答案

### 优先级建议

**Tier 1 (必做, 最高价值):**
- Mixtral-8x7B: 大专家 (intermediate=14336), 8 experts → 验证 α vs expert size
- Phi-3.5-MoE: 16 experts, intermediate=8192 → 中间地带

**Tier 2 (加分项):**  
- Qwen2-57B-A14B: 有共享专家 → 验证 N6 (shared vs routed)

**可以跳过:**
- DBRX/Mixtral-8x22B: 太大，且不提供独特信息
- DeepSeek-V2: 需要 4 卡，成本太高

### 建议: 启动 Phase 2 Tier 1

只测 **Mixtral-8x7B + Phi-3.5-MoE** 的 final checkpoint。  
预计资源: ~20 A100-hours (或 CPU ~8 hours per model)。  
关键验证: α 是否依赖 per-expert 维度。
