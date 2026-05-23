# 测量方法修正记录

> 从 V1 (谱熵/ψ) 到 V2 (power-law α / stable rank / concentration)
> 时间: 2026-05-23

---

## 1. 为什么需要修正

### V1 方法的核心缺陷

| 指标 | 问题 | 根本原因 |
|------|------|---------|
| ψ = (σ₁-σ₂)/(σ₁+σ₂) | 饱和在 ~0.2，不随 N 变化 | 只看 top-2 SV，忽略 spectrum 整体形态 |
| S (谱熵) | 大模型 ΔS < 1%，无区分力 | 全局加权平均 + 高维空间中大部分 SV 不变 |
| PV/(NT) 状态方程 | CV > 40%，不收敛 | LR 不是真温度，训练不在平衡态 |

### 关键发现触发修正

1. **ψ 不 scale**: 70M-6.9B 都在 [0.178, 0.214]，100× 参数变化仅 18% 变动
2. **ψ 过早饱和**: 90%+ 在前 1000 步完成 → 不是结构涌现，是初始化效应
3. **S 对大模型无感**: 6.9B 训练全程 ΔS = 0.9% → 几乎测不到

---

## 2. 文献依据

### Martin & Mahoney (2019-2023) — Heavy-Tail Self-Regularization (HTSR)

**核心思想**: 训练好的 NN 层的奇异值分布遵循 power law (重尾分布)。分布的尾部指数 α 是模型质量的**尺度无关**预测器。

**关键公式**:
- 特征值 (eigenvalue) ESD: P(λ) ~ λ^{-α}
- 在 log-log 坐标下 rank-ordered 特征值呈直线
- α ∈ [2, 6]: 良好训练的层 (heavy-tailed)
- α > 6: 训练不足 (接近随机矩阵/Marchenko-Pastur)
- α < 2: 过拟合/rank collapse

**关键发现**:
- α 与 test accuracy 的 Spearman 相关 r > 0.9
- α 跨模型尺寸可比 (dimensionless)
- 不需要训练数据或测试数据就能预测模型质量

### WeightWatcher 工具

- `pip install weightwatcher`
- 核心指标: α (power-law exponent), α̂ = α × log₁₀(λ_max/λ_min)
- α̂ 是 "weighted alpha"，同时考虑尾部重度和谱范围

### 有效秩与稳定秩

- **Stable rank**: sr(W) = ||W||²_F / ||W||²_op = Σσ_i² / σ₁²
  - 直觉: "如果把所有方差集中到一个方向，能填满几个 σ₁ 大小的方向？"
  - 值域: [1, min(m,n)]
  - 训练使 stable rank 下降 → 信息集中在少数方向
  
- **Effective rank** (Roy & Vetterli 2007): exp(H(σ)) 其中 H 是归一化 SV 的 Shannon 熵
  - 我们的 S 就是 H(σ)，exp(S) = effective rank

### 与热力学的重新连接

**修正后的类比**:

| 热力学量 | V1 定义 (有问题) | V2 定义 (修正) |
|---------|----------------|---------------|
| 温度 T | LR (太粗糙) | 梯度噪声 σ²_∇ / batch_size (待实现) |
| 熵 S | 全局谱熵 (scale-dependent) | **归一化谱熵 S/log(d)** ∈ [0,1] |
| 有序度 | ψ (只看 top-2) | **1/α** (整个谱的结构性) |
| 自由度 | exp(S) | **stable rank** |
| 相变指标 | ψ 跳变 | **α 从 >6 跌到 <4** |

**为什么 α 是更好的序参数**:
1. **Dimensionless**: 与矩阵大小无关
2. **全谱信息**: 不只看 top-2，看整个 ESD tail
3. **理论支撑**: Random Matrix Theory 预测未训练矩阵 α→∞ (Gaussian)
4. **实证验证**: Martin & Mahoney 在 >1000 个模型上验证了 α 与质量的相关性

---

## 3. V2 测量方法

### 新增指标

| 指标 | 公式 | 物理意义 | 预期行为 |
|------|------|---------|---------|
| α (power-law) | fit log(λ) ~ -2/α × log(rank) | 谱分布重尾程度 | 训练中下降 (2→6回到2-4) |
| α̂ (weighted alpha) | α × log₁₀(λ_max/λ_min) | 加权的结构指标 | 训练中下降 |
| Stable rank | ||W||²_F / σ₁² | 有效维度 | 训练中下降 |
| S_norm | S / log(min(m,n)) | 归一化熵 [0,1] | 训练中下降，可跨规模比 |
| C_k (concentration) | Σσ²_top_k / Σσ²_all | 信息集中度 | 训练中上升 |

### 保留的指标 (向后兼容)

- V (volume = ||θ||²_F)
- S (raw spectral entropy)
- ψ (order parameter)

### 代码实现

`scripts/thermo/measure_pythia_v2.py`:
- `fit_power_law_alpha()`: 在 log-log 空间拟合 ESD，返回 α 和 R²
- `measure_model_v2()`: 计算全部新旧指标
- 每层分类: attention / MLP 分开统计 α

---

## 4. 预期结果

如果修正后的指标支撑理论框架，我们应该看到:

### α 的行为
1. **α_init ≈ ∞** (step 0): 随机初始化 = Gaussian = MP distribution
2. **α 在训练中下降**: 从 >6 逐渐降到 2-4
3. **α_final 随 N 递减**: 更大模型最终 α 更低 (更 heavy-tailed = 更结构化)
4. **α 区分 training phases**: warmup 快速下降, stable 缓慢下降, late 趋平

### Stable rank 的行为
1. **sr_init ≈ d**: 随机矩阵的 stable rank ≈ min(m,n) × (1 - small correction)
2. **sr 在训练中下降**: 信息集中到少数方向
3. **sr/d 可跨规模比**: 归一化后应该有 scaling law

### Normalized entropy 的行为
1. **S_norm_init ≈ 1**: 随机初始化 → 最大熵
2. **S_norm_final < 1**: 训练后结构化 → 熵减少
3. **ΔS_norm 应该跨规模可比**: 因为已经归一化

### 状态方程的修正
用 stable rank 代替原来的 V 和 T:
- **新状态方程**: P × V / (N × sr) = ？ (weight_decay × total_norm² / (params × stable_rank))
- 或者: α 本身就是 state variable (不需要 PV=NkT 形式)

---

## 5. 实验状态

| 任务 | 状态 | 预计时间 |
|------|------|---------|
| V2 代码编写 | ✅ 完成 | - |
| V2 小模型测量 (70m+160m+410m) | 🔄 已提交 | ~15 min |
| V2 大模型测量 (1b+2.8b+6.9b) | 🔄 已提交 | ~30 min |
| 文献调研 (Martin & Mahoney) | 🔄 进行中 | - |
| 文献调研 (WeightWatcher) | 🔄 进行中 | - |
| 结果分析 & 理论修正 | ⏳ 待 V2 数据 | - |

---

## 6. 经验教训

### 原始方法为什么失败

1. **ψ = (σ₁-σ₂)/(σ₁+σ₂) 是局部指标**: 只看谱的"顶端"，对全局结构视而不见
2. **全局 S 的加权方式掩盖信号**: 少数大层的变化被多数小层稀释
3. **LR ≠ T**: 学习率是外部控制参数，不是系统内禀量
4. **平衡态假设不适用**: 但如果我们换到非平衡统计力学框架（用 α 描述远离平衡的 heavy-tail），问题可能解决

### 为什么 α 可能有效

Martin & Mahoney 的理论基础:
- SGD + weight decay 的隐式正则化效果是让权重矩阵的 ESD 趋向 heavy-tail
- α 衡量了这种隐式正则化的"成熟度"
- 这不要求系统处于平衡态 — α 描述的是**非平衡稳态** (NESS) 的特征
- 非平衡热力学中，NESS 也有类似状态方程的约束 (fluctuation theorems)

### 关键风险

- 如果 α 也在所有规模上饱和 (类似 ψ) → 需要进一步修正
- Power-law fitting 对小矩阵 (70m 的 512×512) 精度有限
- 不同层类型 (attention vs MLP) 可能有不同的 α 分布

---

*文档将根据 V2 实验结果持续更新*
