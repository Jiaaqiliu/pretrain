# MoE 谱热力学实验方案

## 分阶段执行计划

### Phase 0: 概念验证 (1-2 天, CPU)

**目标**: 验证测量代码可用 + 获取第一批 MoE 谱数据

```bash
# 测量 OLMoE 最终 checkpoint (main branch)
python -m experiments.thermodynamics.moe_measures \
    allenai/OLMoE-1B-7B-0924 \
    --device cpu \
    --max-experts 16 \
    --no-alignment \
    -o results/olmoe_moe/olmoe_quick_test.jsonl

# 测量 Mixtral-8x7B
python -m experiments.thermodynamics.moe_measures \
    mistralai/Mixtral-8x7B-v0.1 \
    --device cpu \
    --max-experts 0 \
    -o results/moe_cross_model/mixtral_8x7b.jsonl
```

**验证清单**:
- [ ] 代码成功运行，无报错
- [ ] 输出 JSONL 格式正确
- [ ] α, SR/d 值在合理范围内
- [ ] cross-expert alignment 计算正确

---

### Phase 1: OLMoE 训练动力学 (3-7 天, CPU/GPU)

**目标**: 获取 OLMoE 完整训练轨迹，验证 M1a, M1b, M2a, M2b, U2, N5

```bash
# 10 个均匀分布的 checkpoint
python scripts/measure_moe_olmoe.py \
    --max-ckpts 10 \
    --device cpu \
    --output-dir results/olmoe_moe

# 如果 Phase 0 OK，扩展到 25 个 checkpoint
python scripts/measure_moe_olmoe.py \
    --max-ckpts 25 \
    --device cuda \
    --output-dir results/olmoe_moe \
    --resume
```

**关键分析**:
1. **SR/d 收敛 (M1a/M1b)**: 逐专家 SR/d 轨迹 → 是否收敛？收敛值？
2. **α reversal (M2a/M2b)**: 逐专家 α 轨迹 → 全局 vs 逐专家差异
3. **三阶段动力学 (N5)**: 识别 route formation → specialization → saturation
4. **KWW 弛豫 (U2)**: 全局/逐专家 S(t) 拟合 KWW
5. **能量均分 (N3)**: EPR = Var(||W_i||²) / <||W_i||²>² 随训练变化
6. **路由矩阵健康 (N2)**: Router SR/d 随训练变化

**对比基准**: 已有的 OLMo-2-1B dense 测量数据 (`results/olmo2_v2/olmo2_1b.jsonl`)

---

### Phase 2: 跨模型相变分析 (5-10 天, GPU)

**目标**: 测量多个 MoE 模型的 final checkpoint，验证 M4 (相变阈值)

```bash
# Tier 1 (小模型)
python scripts/measure_moe_cross_model.py --tier 1 --device cuda

# Tier 2 (中型模型)
python scripts/measure_moe_cross_model.py --tier 2 --device cuda
```

**模型列表与关键参数**:

| 模型 | 总参数 | 激活参数 | per-expert 参数 | hidden_dim |
|------|--------|---------|----------------|-----------|
| OLMoE | 6.9B | 1.3B | ~108M | 2048 |
| Phi-3.5-MoE | 42B | 6.6B | ~2.6B | 4096 |
| Mixtral-8x7B | 47B | 12.9B | ~5.8B | 4096 |
| Qwen2-57B-A14B | 57B | 14B | ~890M | 3584 |
| DBRX | 132B | 36B | ~8.3B | 6144 |
| Mixtral-8x22B | 141B | 39B | ~17.6B | 6144 |

**关键分析**:
1. **相变阈值 (M4)**: α vs N_total, α vs N_active, α vs N_per_expert 三种图
2. **MLP 瓶颈 (M5)**: α_moe vs α_attn for each model
3. **SR/d 跨架构 (M1)**: 是否仍然由 d 决定？
4. **下游性能预测 (M3)**: SR/d → benchmark (如果有数据)

---

### Phase 3: 架构对比 (5-10 天, multi-GPU)

**目标**: 共享专家 vs 纯 MoE，验证 N6

```bash
# DeepSeek-V2 (需要多 GPU)
python -m experiments.thermodynamics.moe_measures \
    deepseek-ai/DeepSeek-V2 \
    --device auto \
    --max-experts 32 \
    -o results/moe_cross_model/deepseek_v2.jsonl

# 对比 Mistral-7B dense (已有数据)
```

**关键分析**:
1. **共享 vs 路由 (N6)**: α_shared vs α_routed vs α_dense_mlp
2. **SR/d 三方对比**: shared expert SR/d ≈ dense? routed < dense?
3. **cross-expert alignment**: shared 与 routed 的主导子空间重叠度

---

### Phase 4: 深度分析 + 论文写作

**目标**: 综合所有数据，提取物理规律

**分析任务**:

1. **通用压缩定律 MoE 扩展**:
   - 修改公式 SR/d_MoE = f(d, E, k) → 加入专家数和 top-k 的影响
   - 计算 ΔH₂ per expert → 是否仍然 ≈ -2 nats？

2. **MoE 相变图**:
   - 绘制 (N_active, D/N_active) → α 的相图
   - 与 dense 相图对比
   - 确定 MoE 相变阈值

3. **三阶段动力学特征化**:
   - 确定相变点
   - 各阶段的持续时间
   - 层级差异（浅层 vs 深层）

4. **实用监控协议**:
   - MoE 的 α-guided schedule 如何修改？
   - 需要监控哪些额外指标（cross-expert alignment, router SR/d）？
   - 专家坍缩的早期预警协议

---

## 资源需求估算

| 阶段 | GPU 时间 | 存储 | 网络下载 |
|------|---------|------|---------|
| Phase 0 | ~4 CPU-hours | 20GB | ~15GB (OLMoE + Mixtral) |
| Phase 1 | ~50 GPU-hours (A100) | 50GB | ~150GB (10-25 checkpoints) |
| Phase 2 | ~100 GPU-hours (A100) | 200GB | ~400GB (6 models) |
| Phase 3 | ~200 GPU-hours (A100×4) | 500GB | ~500GB (DeepSeek-V2) |

**总计**: ~350 A100 GPU-hours (对比 dense 实验的 ~20,000 H200-hours，成本极低)

---

## 测量代码

所有代码位于:
- `experiments/thermodynamics/moe_measures.py` — 核心测量模块
- `scripts/measure_moe_olmoe.py` — OLMoE 批量测量
- `scripts/measure_moe_cross_model.py` — 跨模型对比

使用方法见 [07_measurement_code.md](07_measurement_code.md)

---

## 假设-实验映射

| 假设 | Phase | 所需模型 | 关键指标 |
|------|-------|---------|---------|
| M1a SR/d 逐专家收敛 | 1 | OLMoE | SR/d per expert |
| M1b SR/d 依赖 d_expert | 2 | 跨模型 | SR/d vs d |
| M2a α reversal expert-specific | 1 | OLMoE | α per expert trajectory |
| M2b 全局 α 掩盖局部 reversal | 1 | OLMoE | global vs expert α |
| M3 下游性能预测 | 2 | 多模型 | SR/d vs benchmark |
| M4 相变阈值 | 2 | 多模型 | α vs N variants |
| M5 MLP 瓶颈 | 2+3 | 多模型 | α_moe vs α_attn |
| M6 α-guided for MoE | 1 | OLMoE | decay trigger analysis |
| U1 PV=NkT | 1 | OLMoE | PV/(NT) convergence |
| U2 KWW 弛豫 | 1 | OLMoE | S(t) fitting |
| U4 热力学效率 | 1 | OLMoE vs OLMo-2-1B | η_thermo |
| U5 序参数 ψ | 1 | OLMoE | ψ per expert |
| N1 专家坍缩谱信号 | 1+2 | OpenMoE / OLMoE | Var(α), alignment |
| N2 路由矩阵健康 | 1 | OLMoE | Router SR/d |
| N3 能量均分 | 1 | OLMoE | EPR trajectory |
| N4 逐专家涨落-耗散 | 1 | OLMoE | T_eff vs α per expert |
| N5 三阶段动力学 | 1 | OLMoE | phase identification |
| N6 共享 vs 路由 | 3 | DeepSeek-V2 | α/SR/d comparison |
| N7 谱可塑性窗口 | 1 | OLMoE | solidification time |
| N8 RG 缩放 | - | 需要控制实验 | α vs E scaling |
