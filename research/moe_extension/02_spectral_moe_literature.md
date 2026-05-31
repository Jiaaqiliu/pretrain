# MoE 谱分析相关文献综述

## 直接相关论文

### 1. SD-MoE: Spectral Decomposition for Expert Specialization (2026.02)
- **论文**: arxiv.org/abs/2602.12556
- **核心发现**: 专家间共享高度对齐的主导谱成分（主导子空间余弦相似度 >0.9）
- **关键洞察**: 即使模型有显式共享专家（DeepSeek/Qwen），路由专家的权重矩阵仍然展现强烈重叠的主导谱方向
- **方法**: 将每个专家矩阵分解为共享的低秩谱子空间 + 正交的专家特异性补集
- **结果**: 约3% 下游增益，30% 训练效率提升
- **对我们的意义**: α 测量可能在专家间显示相似值（因为主导谱结构共享），专家特异性信息集中在谱尾部

### 2. SR-MoE: Spectral Manifold Regularization for Stable and Modular Routing (2026.01)
- **论文**: arxiv.org/abs/2601.03889
- **核心发现**: 约束路由权重的谱范数以限制 Lipschitz 常数；正则化路由权重的 stable rank 以维持高维特征多样性
- **关键洞察**: 传统线性门控随深度增加导致精度下降最高 4.72%（专家纠缠），谱正则化可维持结构完整性
- **对我们的意义**: 路由矩阵的 stable rank 本身就是一个诊断指标

### 3. MoE as Soft Clustering: Dual Jacobian-PCA Spectral Geometry (2025.12)
- **论文**: arxiv.org/abs/2601.11616
- **核心发现**: 专家局部 Jacobian 的主奇异值更小，谱衰减更快（单个专家有效秩低于 dense）
- **关键洞察**: Top-k 路由产生更集中、更低秩的专家局部结构
- **对我们的意义**: MoE 的 SR/d 逐专家应该系统性低于等价 dense 层

### 4. SPHERE: Spectral Plasticity in MoE for RL (2025.05)
- **论文**: arxiv.org/abs/2605.04712
- **核心发现**: 将 MoE 的可塑性损失形式化为谱可塑性损失（NTK 理论）
- **方法**: 可按单个专家特征矩阵追踪谱可塑性
- **对我们的意义**: 可以逐专家追踪谱可塑性作为训练健康指标

### 5. MoE-SVD: Compression via Singular Value Analysis (ICML 2025)
- **论文**: proceedings.mlr.press/v267/li25az.html
- **模型**: Mixtral, Phi-3.5, DeepSeek, Qwen2 MoE
- **核心发现**: 专家间共享大量谱结构，可共享单个 V 矩阵，实现 60% 压缩
- **对我们的意义**: 直接证实专家谱结构的冗余性

### 6. Geometric Metrics for MoE Specialization (2026.04)
- **论文**: arxiv.org/abs/2604.14500
- **核心发现**: 标准启发式指标（余弦相似度等）违反参数化不变性；Fisher Specialization Index 与下游性能 r=0.91
- **关键洞察**: 专家路由分布在配备 Fisher 信息度量的概率单纯形上演化
- **对我们的意义**: 可与我们的 α 指标互补，建立更完整的诊断体系

## 间接相关论文

### 7. Spectral Collapse Drives Loss of Plasticity (NeurIPS 2025)
- **论文**: arxiv.org/abs/2509.22335
- 可塑性损失由 Hessian 谱坍缩先导
- 维持高有效特征秩（close to stable rank）+ L2 惩罚可保持可塑性

### 8. From SGD to Spectra: Theory of Weight Dynamics (2025.07)
- **论文**: arxiv.org/pdf/2507.12709
- 将权重更新映射到 Dyson 布朗运动
- 极限分布为 gamma 型密度+幂律尾
- 为幂律尾指数的普遍性提供理论基础

### 9. Routing Absorption in Sparse Attention (2026.02)
- **论文**: arxiv.org/pdf/2603.02227
- Q/K/V 投影与稀疏掩码共同适应，吸收路由信号
- 意味着路由矩阵的谱结构可能不携带独立于专家权重的信息

### 10. Load Balancing Loss 的危害 (ACL 2025)
- **论文**: aclanthology.org/2025.acl-long.249/
- 辅助负载均衡损失可能损害性能
- DeepSeek-V3 的无辅助损失策略产生更干净的梯度

## 研究空白（我们可以填补的）

1. **无人对 MoE 专家权重矩阵做过 HTSR α 分析** — 这是最大的空白
2. **无人测量过 MoE 的 SR/d 逐专家收敛行为** — 我们的通用压缩定律完全未测试
3. **无人研究过 MoE 训练中的 α reversal** — 是否在专家级别发生？
4. **N ≈ 1.7B 相变在 MoE 中从未研究** — 阈值取决于总参数还是每专家参数？
5. **OLMoE 的 244 个训练 checkpoint 是未开发的金矿** — 无人用它做谱动力学分析
