# 在 Dense 上未验证/被否定的假设 → 在 MoE 上的新机会

这些假设在原始实验计划中提出但未完成验证，或者被修改/放弃。MoE 架构可能为它们提供新的验证路径。

---

## U1: PV = NkT 状态方程

### 原始假设 (P1)
SGD 训练的稳定阶段满足类理想气体状态方程：P·V/(N·T) 收敛到 k_eff(N) = k₀ + α·N^{-1/3}。

### Dense 上的状态
- Liu & Tegmark (2025) 在小规模网络上验证了 PV = NkT
- 我们在 Experiment Plan 中计划验证但**未完成**
- 主要困难：需要从 optimizer state 提取有效温度 T，大模型的 optimizer state 很难获取

### MoE 上的新机会

**假设 U1-MoE**: 在 MoE 中，状态方程可能需要修正。

关键问题：N 是什么？
- 如果 N = 总参数量 → 每个 token 只"激活"部分参数，P·V/(N·T) 会系统性偏低
- 如果 N = 激活参数量 → 更接近 dense 的行为
- 可能需要 "effective N" = N_active + β·N_inactive (某种衰减的非激活参数贡献)

**新假设**: MoE 的状态方程可能是：
```
P·V_active = N_active · k_eff · T + coupling_term(N_total - N_active)
```
其中 coupling_term 描述非激活专家对激活专家的耦合效应。

### MoE 特有优势
OLMoE 有 244 个 checkpoint + 完整训练日志 → 可以直接计算 T(从 LR, batch size, 梯度方差)，不需要 optimizer state。

### 验证方案
```
模型: OLMoE-1B-7B (244 checkpoints)
测量: V, T (from training config), P = weight_decay
计算: 
  P·V/(N_total·T)  vs  P·V/(N_active·T)  vs  P·V/(N_expert·T)
分析: 哪个归一化使 P·V/(N·T) 收敛最好？
```

---

## U2: KWW 玻璃弛豫

### 原始假设 (P3)
Mid-training 的谱熵弛豫遵循 KWW 拉伸指数：φ(t) = exp[-(t/τ)^β]，β ∈ (0.5, 0.8)。

### Dense 上的状态
- **完全未验证** — 计划用 OLMo-2 的 Stage 2 checkpoint 验证，但未执行
- 玻璃态弛豫是统计物理的经典现象，β < 1 表示多重弛豫时间尺度

### MoE 上的新机会

**假设 U2-MoE-a**: MoE 的专家特化过程类似于"结晶"，不同专家在不同时间尺度上"固化"。

如果专家特化是一个有序化过程（类似于从液态到固态的相变），那么：
- 初始阶段：所有专家类似（"液态"，高对称性）
- 中间阶段：部分专家开始特化（"部分结晶"）
- 最终阶段：专家完全分化（"多晶体"，每个晶粒 = 一个专家的专业领域）

**假设 U2-MoE-b**: MoE 的弛豫应该比 dense 更快（β 更接近 1），因为不同专家可以独立弛豫。

理由：在 dense 模型中，所有参数耦合 → 集体弛豫更慢。在 MoE 中，专家间弱耦合 → 每个专家可以独立达到局部平衡。

**假设 U2-MoE-c**: OLMoE 的 5T token 训练可以分为三个动力学阶段：
1. **Onset phase** (0-100B tokens): 路由形成，专家开始分化
2. **Specialization phase** (100B-2T tokens): 专家深化特化，α 持续下降
3. **Saturation phase** (2T-5T tokens): 结构冻结，类似"玻璃化转变"

### 验证方案
```
数据: OLMoE 244 checkpoints 的逐专家 α 轨迹
分析:
  1. 对全局 S(t) 拟合 KWW → 提取 τ, β
  2. 对逐专家 α(t) 分别拟合 KWW → 不同专家是否有不同的 τ, β？
  3. 计算 cross-expert α 方差随时间的演化 → 特化进程的量化
  4. BIC 模型选择: KWW vs 简单指数 vs 幂律
```

---

## U3: Gaussian Schedule (最小熵产生原理)

### 原始假设 (P4)
从最小熵产生原理推导的 Gaussian decay schedule 优于 WSD。

### Dense 上的状态
- **被放弃** — 原论文改为更实用的 α-guided approach
- 理论动机：如果 SGD 是 Langevin 动力学，最小化不可逆熵产生的最优路径应该是 Gaussian 型 LR 衰减

### MoE 上的新机会

**假设 U3-MoE**: 在 MoE 中，最优 LR schedule 可能是 expert-specific 的。

理由：
- 不同专家的 α 轨迹不同 → 它们的"最优衰减时机"也不同
- 一个全局 LR 对所有专家是次优的
- 可以设计 "per-expert-aware" schedule：全局 LR 的衰减时机由"最多专家开始 reversal"决定

这不需要真正实现 per-expert LR（路由器使得这几乎不可能），但可以用来**诊断**当前全局 schedule 对各专家的适配程度。

---

## U4: 热力学效率 η_thermo

### 原始假设 (Q2)
WSD 的累积熵产生 ΔS_tot < Cosine → WSD 热力学效率更高。

### Dense 上的状态
- **未完成** — 计划对比但缺乏实际数据
- 定义：η = |ΔF| / (|ΔF| + T̄·ΔS_irr)，其中 ΔS_irr 是不可逆熵产生

### MoE 上的新机会

**假设 U4-MoE**: MoE 训练比 dense 更热力学高效。

理由：
- MoE 的有效参数比总参数少 → 相同 compute 下做的"有用功"更多
- 专家稀疏激活减少了"无效"梯度更新（只更新被路由到的专家）
- 但 load balancing loss 增加了额外的"无效功" → 可能降低效率

**假设 U4-MoE-b**: DeepSeek-V3 的 auxiliary-loss-free 路由比 Mixtral 的有损路由更高效。

### 验证方案
```
需要: 完整训练曲线 (loss + LR + α + SR/d at each step)
数据源:
  - OLMoE (5T tokens, 有详细训练日志)
  - 对比 OLMo-2-1B (dense, 4T tokens)
计算:
  1. 有效温度 T(t) = η(t) · σ²_grad / (2B)
  2. 谱熵 S(t) from SR/d measurements
  3. 自由能 F(t) = Loss(t) - T(t)·S(t)
  4. 累积熵产生 ΔS_tot = ∫ σ(t) dt
  5. 热力学效率 η_thermo = |ΔF| / (|ΔF| + T̄·ΔS_irr)
对比: MoE η_thermo vs Dense η_thermo
```

---

## U5: 序参数 ψ 作为训练信号

### 原始假设 (Q4)
ψ = (σ₁ - σ₂)/(σ₁ + σ₂) 作为序参数，与下游性能高度相关。

### Dense 上的状态
- ψ 被计算了但**未作为独立指标深入分析**
- 在论文中被 SR/d 和 α 取代

### MoE 上的新机会

**假设 U5-MoE**: 逐专家 ψ 可以量化专家特化程度。

理由：
- ψ 接近 1 表示权重矩阵被一个主导方向支配 → 过度特化
- ψ 接近 0 表示多个方向均匀分布 → 未特化
- 不同专家的 ψ 分布反映整个 MoE 系统的特化状态

**假设 U5-MoE-b**: 专家间 ψ 的方差是特化质量的指标。
- 高方差 → 专家高度分化（好）
- 低方差 → 专家同质化/坍缩（坏）

### 验证方案
```
模型: OLMoE 244 checkpoints
测量: 每个专家的 ψ
分析: 
  1. ψ 的跨专家分布随训练如何变化？
  2. ψ 的方差是否在特化阶段增加？
  3. ψ 与 routing entropy 的关系？
```
