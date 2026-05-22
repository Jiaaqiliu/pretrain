# Thermodynamics of Pretraining — 完整实验执行计划

> **本文档目标**: 另一个 Agent 阅读后，可以独立完成本研究的全部剩余工作——从下载检查点、运行实验、到分析数据、生成图表、填充论文中的 [tbd] 数值。

---

## 1. 项目概述

### 1.1 研究目标

论文标题: **"Beyond Loss Curves: Thermodynamics of Pretraining"**

核心论点: 预训练不是单纯的优化过程，而是一个热力学过程。SGD 最小化的是**自由能 F = U - T·S**，而非 loss 本身。

论文提出4个可验证预测:
- **P1**: 稳定阶段 P·V/(N·T) 收敛到 k_eff(N) = k₀ + α·N^(-1/3)
- **P2**: WSD 的累积熵产生 < Cosine（效率优势 23-37%）
- **P3**: Mid-training 谱熵遵循 KWW 拉伸指数 φ(t) = exp[-(t/τ)^β]，β ∈ (0.5, 0.8)
- **P4**: 从最小熵产生原理推导的 Gaussian decay schedule 优于 WSD

### 1.2 论文位置

论文初稿和参考文献在: `/Users/jiaqi/Projects/PreTrain/ThermodynamicsOfPretraining/`
- `paper/main.tex` — 主文件
- `paper/sections/` — 各章节 (framework, experiments, discussion, appendix 等)
- `research/` — 调研笔记、竞争分析、综合思考

### 1.3 代码位置

实验代码在当前仓库 (`A-EVOLVE-V2`), 分支 `nemo-reasoning-single-cc-olmo-core`:
- `experiments/thermodynamics/` — 核心测量和分析 Python 库
- `scripts/thermo/` — 训练和测量可执行脚本
- `scripts/k8s/thermo/` — K8s PyTorchJob YAML 清单

---

## 2. 热力学状态变量定义

所有公式均来自论文 Section 3 (framework.tex)。实现在 `experiments/thermodynamics/measures.py`。

| 变量 | 符号 | 定义 | 代码函数 |
|------|------|------|---------|
| 温度 | T | η·σ̂²_∇ / (2B) | `effective_temperature()` |
| 压力 | P | λ (weight decay) | 直接读取配置 |
| 体积 | V | ‖θ‖²_F | `weight_volume()` |
| 内能 | U | Loss(θ) | 从训练日志读取 |
| 谱熵 | S | -Σ p_i log p_i (SVD归一化) | `spectral_entropy_layer()` / `global_spectral_entropy()` |
| 自由能 | F | U - T·S | `free_energy()` |
| 序参数 | ψ | (σ₁ - σ₂)/(σ₁ + σ₂) | `order_parameter_layer()` / `global_order_parameter()` |
| 比热 | C_v | ∂U/∂T\|_V | 数值微分 |

### 谱熵计算细节

对每个 2D 权重矩阵 W_l:
1. 若 min(m,n) ≤ 2048: 全 SVD
2. 若 min(m,n) > 2048: randomized SVD, 保留 top-k=256 奇异值, 1步 power iteration
3. 归一化: p_i = σ_i / Σ_j σ_j
4. 层熵: S_l = -Σ p_i log p_i
5. 全局熵: S = Σ_l (N_l/N) · S_l （参数加权平均）

### 有效温度计算

两种方式（优先级从高到低）:
1. **从 optimizer state 计算** (最佳): Adam 的 `exp_avg_sq` (v_t) 就是梯度二阶矩的 EMA，`σ̂²_∇ ≈ mean(v_t)`
2. **从 double-batch 估计**: 两个独立 mini-batch 的梯度差 `σ̂²_∇ ≈ (1/2N)·‖g_A - g_B‖²`

---

## 3. 公开检查点复用方案

### 3.1 可用的 OLMo-2 中间检查点

| 模型 | HuggingFace repo | 检查点数 | 间隔 | Token数 | Schedule | Optimizer State |
|------|------------------|---------|------|---------|----------|----------------|
| OLMo-2-1B | `allenai/OLMo-2-0425-1B` | 267 | 10K步(S1) / 1K步(S2) | 4T | Cosine | ✓ |
| OLMo-2-1B-early | `allenai/OLMo-2-0425-1B-early-training` | 38 | 1K步 | 78B | Cosine | ✓ |
| OLMo-2-7B | `allenai/OLMo-2-1124-7B` | 970 | 1K步 | 4T | Cosine | ✓ |
| OLMo-2-13B | `allenai/OLMo-2-1124-13B` | 717 | 1K步 | 5T | Cosine | ✓ |
| OLMo-2-32B | `allenai/OLMo-2-0325-32B` | 752 | 1K步 | ~1.8T | Cosine | ✓ |

**关键**: 所有 OLMo-2 模型使用 **Cosine schedule**，不是 WSD。因此:
- Cosine 基线数据: **免费获得** (直接测量公开检查点)
- WSD/Gaussian 数据: **必须自己训练**

### 3.2 检查点下载方式

**方法 1: HuggingFace (推荐)**
```python
from huggingface_hub import list_repo_refs
from transformers import AutoModelForCausalLM

# 列出所有可用的 revision (每个 revision = 一个检查点)
refs = list_repo_refs("allenai/OLMo-2-1124-7B")
for branch in refs.branches:
    print(branch.name)  # e.g. "stage1-step1000-tokens5B"

# 加载特定检查点
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-1124-7B",
    revision="stage1-step1000-tokens5B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

**方法 2: 直接 URL (7B confirmed)**
```
https://olmo-checkpoints.org/ai2-llm/peteish7/step{N}-unsharded/model.safetensors
https://olmo-checkpoints.org/ai2-llm/peteish7/step{N}-unsharded/optim.safetensors   # Adam state!
https://olmo-checkpoints.org/ai2-llm/peteish7/step{N}-unsharded/train.pt            # step/token metadata
https://olmo-checkpoints.org/ai2-llm/peteish7/step{N}-unsharded/config.yaml         # 完整训练配置
```

### 3.3 OLMo-2 训练配置 (从 config.yaml)

| 参数 | 1B | 7B | 13B |
|------|-----|-----|------|
| LR (peak) | 4e-4 | 3e-4 | 3e-4 |
| Weight Decay (P) | 0.1 | 0.1 | 0.1 |
| Warmup Steps | 2000 | 2000 | 2000 |
| Schedule | CosWithWarmup | CosWithWarmup | CosWithWarmup |
| alpha_f (min_ratio) | 0.1 | 0.1 | 0.1 |
| Batch Size (seqs) | 2048 | 2048 | 2048 |
| Seq Length | 4096 | 4096 | 4096 |
| Optimizer | AdamW | AdamW | AdamW |
| Betas | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) |

### 3.4 不存在 190M 的 OLMo 模型

OLMo 最小尺度是 1B。如果论文需要 sub-1B 数据点:
- **方案 A**: 自己用 OLMo-core 的 `TransformerConfig.olmo2_190M()` 训练（已有代码）
- **方案 B**: 使用 Pythia-160M / Pythia-410M（EleutherAI，154个检查点），但架构不同（LayerNorm 非 RMSNorm）
- **推荐**: 方案 A，成本低（~183 H200-hours/run），且与 OLMo-2 架构一致

---

## 4. 五个研究问题的详细执行方案

### 4.1 Q1: 状态方程 P·V = N·k_eff·T + α·N^(2/3)·T^(3/2) - γ/V

**数据源**: OLMo-2 公开检查点 (1B/7B/13B) + 自训练 190M

**执行步骤**:
1. 下载/加载每个 scale 的所有检查点
2. 对每个检查点计算: V, S, ψ, T (从 optimizer state), F
3. 计算 P·V/(N·T) 时间序列
4. 识别 stable phase（WSD 的 constant-LR 阶段；Cosine 的前 60% 训练）
5. 在 stable phase 内取 P·V/(N·T) 中位数作为 k_eff
6. 拟合 k_eff(N) = k₀ + α·N^(-1/3) 跨 4 个尺度
7. 拟合完整状态方程（包含非平衡校正项）
8. 识别三个热力学 regime: ideal gas / liquid / glass

**输出**:
- Table 3 (论文): k_eff values per scale, R², finite-size correction α
- Figure 2 (论文): P·V/(N·T) convergence plot per scale
- Figure (论文): Phase diagram per scale

**代码入口**: `scripts/thermo/run_analysis.py → analyze_state_equation()`

**注意**: OLMo-2 用的是 Cosine schedule（没有明确的 "stable phase"），但前 60% 训练时 LR 变化缓慢（cosine 的前半部分），可以近似为 quasi-stable。论文需要说明这一点。

### 4.2 Q2: WSD 为什么优于 Cosine

**数据源**:
- Cosine 基线: OLMo-2 公开检查点 (免费)
- WSD 基线: 自训练 190M/1B (必须训练)

**执行步骤**:
1. 从 OLMo-2-7B 检查点计算 Cosine trajectory 的 σ(t) 和 ΔS_tot
2. 从自训练的 190M/1B WSD 检查点计算 WSD trajectory 的 σ(t) 和 ΔS_tot
3. 绘制 T-S 相空间轨迹对比
4. 计算热力学效率 η_thermo
5. 统计对比（但注意: 190M 和 1B 的 Cosine 数据与 WSD 数据的模型参数量可能不完全一致，需要控制变量）

**重要**: OLMo-2-1B (Cosine) 和我们训练的 1B (WSD) 使用相同的 OLMo-2 架构和 olmo2_1B factory，所以直接可比。但训练数据可能不同——需要用相同数据源训练 WSD 1B。

**输出**:
- Table 4 (论文): ΔS comparison per scale
- Figure 3 (论文): Cumulative entropy production curves
- Figure 4 (论文): T-S phase-space trajectories

**代码入口**: `scripts/thermo/run_analysis.py → analyze_wsd_vs_cosine()`

### 4.3 Q3: Mid-Training 的 KWW 玻璃弛豫

**数据源**: OLMo-2 Stage 2 检查点 (完美匹配!)

OLMo-2 的 mid-training 过程:
- Stage 1: 大规模 web 数据预训练 (OLMo-Mix-1124)
- Stage 2: 切换到高质量数据 (Dolmino-Mix-1124) + LR 线性退火到 0

可用的 Stage 2 检查点:
- 7B: 3 ingredients × ~12K steps @ 1K步 = ~36 个检查点
- 13B: 4 ingredients × ~35K steps @ 1K步 = ~140 个检查点
- 1B: 3 ingredients × ~24K steps @ 1K步 = ~72 个检查点

**执行步骤**:
1. 加载每个 scale 的 Stage 2 检查点序列
2. 计算 S(t) 时间序列
3. 检测 onset（数据切换点，OLMo-2 有明确标记）
4. 归一化: φ(t) = (S(t) - S_∞) / (S₀ - S_∞)
5. 拟合 KWW: φ(t) = exp[-(t/τ)^β]
6. Bootstrap 95% CI (1000 次重采样)
7. BIC 模型选择: KWW vs 简单指数 vs 幂律
8. 计算最优 mid-training 时长: 3τ

**输出**:
- Table 5 (论文): τ, β, R² per scale
- Figure 5 (论文): KWW fit curves
- 实践建议: 最优 mid-training 时长 ≈ 3τ tokens

**代码入口**: `scripts/thermo/run_analysis.py → analyze_kww()`

### 4.4 Q4: 序参数 ψ 作为在线监控信号

**数据源**: OLMo-2 检查点 + OLMo-2 论文中的 downstream benchmark 分数

OLMo-2 论文 (arXiv 2501.00656) 报告了每个检查点的 downstream 性能:
- MMLU, GSM8K, HumanEval, HellaSwag, ARC-Challenge
- 这些分数可能在 WandB 上有 (project: `ai2-llm/olmo-medium`, `ai2-llm/olmo-small`)

**执行步骤**:
1. 对每个检查点计算 ψ（只需 top-2 奇异值，用 power iteration，极快）
2. 同时计算 S, F
3. 获取对应的 downstream benchmark 分数
4. 计算 Spearman rank correlation: ψ vs avg_downstream_score
5. 对比 loss vs S vs F vs ψ 的预测力

**注意**: 如果无法获取逐检查点的 benchmark 分数，需要自己对部分检查点跑评估。可以用 `lm-evaluation-harness` 工具。

**输出**:
- Table 6 (论文): Spearman r per metric per scale
- 监控协议: ψ (primary) + σ(t) (secondary) + F (tertiary)

**代码入口**: `scripts/thermo/run_analysis.py`（需要扩展以支持 benchmark 数据加载）

### 4.5 Q5: Gaussian Schedule 对比 (核心贡献, 必须训练)

**数据源**: 全部自训练

**配置矩阵**:
| 模型 | Schedule | Seeds | GPU配置 | 每run H200-hours | 总 H200-hours |
|------|----------|-------|---------|-----------------|---------------|
| 190M | cosine | 42, 123, 456 | 1节点×8 GPU | 183 | 549 |
| 190M | wsd_linear | 42, 123, 456 | 1节点×8 GPU | 183 | 549 |
| 190M | wsd_exponential | 42, 123, 456 | 1节点×8 GPU | 183 | 549 |
| 190M | gaussian | 42, 123, 456 | 1节点×8 GPU | 183 | 549 |
| 1B | wsd_linear | 42, 123, 456 | 2节点×8 GPU | 1463 | 4389 |
| 1B | wsd_exponential | 42, 123, 456 | 2节点×8 GPU | 1463 | 4389 |
| 1B | gaussian | 42, 123, 456 | 2节点×8 GPU | 1463 | 4389 |

**注意**: 1B cosine 基线直接复用 OLMo-2-0425-1B, 省去 4,389 H200-hours!

**每个 run 的额外配置**:
- 检查点间隔: 200 步 (密集保存, 用于热力学测量)
- 热力学测量间隔: 200 步 (ThermoMeasurementCallback)
- 梯度噪声估计间隔: 1000 步 (GradNoiseCallback)
- WandB project: `thermo-pretraining`

**训练数据**:
- 190M: DCLM-web + Dolma-web + Code
- 1B: DCLM-web + Dolma-web + Code + Math + Books
- 数据路径在 FSx: `/fsx/dev/jiaqi/data/olmo-pretrain/`

**执行步骤**:
1. 确保训练数据已就位 (tokenized numpy format)
2. 提交所有 K8s jobs (see Section 6)
3. 等待训练完成
4. 收集 `thermo_measurements.jsonl` 和最终 loss
5. 统计对比: paired t-test, 3 seeds, p < 0.05

**输出**:
- Table 7 (论文): Final loss per schedule per scale, Δ% vs WSD-Linear
- Figure (论文): Schedule comparison bar chart
- 热力学效率对比: η_thermo per schedule

**代码入口**: `scripts/thermo/train_schedule_comparison.py` + `scripts/thermo/run_analysis.py → analyze_schedule_comparison()`

---

## 5. 附录实验

### 5.1 WSD Stable-Phase Ablation (Appendix E)

验证 WSD 优势在 stable phase 太短时消失。

**配置**: 190M, stable_frac ∈ {0.2, 0.4, 0.6, 0.8, 0.95}, seed=42

**代码**: `scripts/thermo/train_wsd_ablation.py`
**K8s**: `scripts/k8s/thermo/thermo_ablation_stable_frac.yaml`

### 5.2 Mid-Training vs Direct Decay (Q3 验证)

隔离 data switching 的退火效应。

**配置**: 190M, mode ∈ {midtrain, direct}, seed=42
- midtrain: Stage 1 → decay → data switch + LR warmup → decay
- direct: Stage 1 → extended decay (no data switch)

**代码**: `scripts/thermo/train_midtraining_comparison.py`
**K8s**: `scripts/k8s/thermo/thermo_midtrain_comparison.yaml`

---

## 6. 执行指南

### 6.1 环境准备

```bash
# 在 K8s 节点或本地
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e olmo-core/.[all]
pip install scipy matplotlib huggingface_hub transformers safetensors
```

### 6.2 分阶段执行

#### Phase 0: 快速验证 (~500 H200 GPU-hours, ~3天)

目标: 验证代码正确性 + 初步确认热力学信号

```bash
# 1. 测量 OLMo-2-7B 前 50 个检查点
python scripts/thermo/measure_checkpoints.py \
    --model-size 7B \
    --use-hf \
    --output /fsx/dev/jiaqi/thermo_results/measurements_7b_pilot.jsonl \
    --step-range 1000,50000 \
    --step-interval 1000

# 2. 训练 1 个 190M Gaussian vs WSD-Linear (seed=42)
kubectl apply -f scripts/k8s/thermo/thermo_train_190m_gaussian_s42.yaml
kubectl apply -f scripts/k8s/thermo/thermo_train_190m_wsd_linear_s42.yaml

# 3. 快速分析
python scripts/thermo/run_analysis.py \
    --results-dir /fsx/dev/jiaqi/thermo_results \
    --experiments-dir /fsx/dev/jiaqi/thermo_experiments \
    --output-dir /fsx/dev/jiaqi/thermo_pilot_results
```

验证清单:
- [ ] S 随训练单调下降
- [ ] ψ 随训练单调上升
- [ ] P·V/(N·T) 在 stable phase 大致收敛
- [ ] Gaussian vs WSD-Linear: Gaussian 的最终 loss 更低

#### Phase 1: 190M 全量 (~3,100 H200 GPU-hours, ~7天)

```bash
# 测量 190M (需要先训练 190M 基线)
# 提交所有 190M 训练 jobs
for yaml in scripts/k8s/thermo/thermo_train_190m_*.yaml; do
    kubectl apply -f $yaml -n default --context $K8S_CONTEXT
done

# WSD ablation
kubectl apply -f scripts/k8s/thermo/thermo_ablation_stable_frac.yaml

# Mid-training comparison
kubectl apply -f scripts/k8s/thermo/thermo_midtrain_comparison.yaml
```

#### Phase 2: 多尺度测量 (~3,600 H200 GPU-hours, ~12天)

```bash
# 测量公开 OLMo-2 检查点 (无需训练!)
kubectl apply -f scripts/k8s/thermo/thermo_measure_1b.yaml
kubectl apply -f scripts/k8s/thermo/thermo_measure_7b.yaml
kubectl apply -f scripts/k8s/thermo/thermo_measure_13b.yaml
```

#### Phase 3: 1B 训练 (~13,200 H200 GPU-hours, ~17天)

```bash
# 提交所有 1B 非 cosine 训练 jobs (cosine 复用 OLMo-2)
for yaml in scripts/k8s/thermo/thermo_train_1b_wsd_linear_*.yaml \
            scripts/k8s/thermo/thermo_train_1b_wsd_exponential_*.yaml \
            scripts/k8s/thermo/thermo_train_1b_gaussian_*.yaml; do
    kubectl apply -f $yaml -n default --context $K8S_CONTEXT
done
```

#### Phase 4: 分析 + 图表生成

```bash
python scripts/thermo/run_analysis.py \
    --results-dir /fsx/dev/jiaqi/thermo_results \
    --experiments-dir /fsx/dev/jiaqi/thermo_experiments \
    --output-dir /fsx/dev/jiaqi/thermo_paper_figures
```

### 6.3 一键提交

```bash
# 提交所有 jobs
./scripts/thermo/submit_all.sh all

# 只提交测量
./scripts/thermo/submit_all.sh measure

# 检查状态
./scripts/thermo/submit_all.sh status
```

---

## 7. 代码结构详解

### 7.1 核心库 (`experiments/thermodynamics/`)

```
experiments/thermodynamics/
├── __init__.py              # 导出所有公共 API
├── measures.py              # 热力学状态变量计算
│   ├── spectral_entropy_layer()     # 单层谱熵 (randomized SVD)
│   ├── order_parameter_layer()      # 单层序参数 (power iteration)
│   ├── global_spectral_entropy()    # 全局谱熵 (参数加权)
│   ├── global_order_parameter()     # 全局序参数 (层平均)
│   ├── weight_volume()              # 体积 V = ||θ||²_F
│   ├── effective_temperature()      # 有效温度 T
│   ├── free_energy()                # 自由能 F = U - T·S
│   ├── entropy_production_rate()    # 熵产生率 σ(t)
│   ├── cumulative_entropy_production()  # 累积熵产生 ΔS_tot
│   ├── thermodynamic_efficiency()   # 热力学效率 η
│   ├── estimate_gradient_variance() # 梯度方差估计 (double-batch)
│   └── measure_checkpoint()         # 一站式: 计算一个检查点的所有状态变量
│
├── schedules.py             # LR schedule 实现
│   ├── cosine_lr()                  # 标准 cosine annealing
│   ├── wsd_linear_lr()              # WSD + 线性 decay
│   ├── wsd_exponential_lr()         # WSD + 指数 decay
│   ├── gaussian_lr()                # Gaussian decay (我们的贡献)
│   └── SCHEDULE_REGISTRY            # name → function 映射
│
├── checkpoint_loader.py     # OLMo 检查点加载
│   ├── discover_checkpoints()       # 发现本地检查点
│   ├── discover_hf_checkpoints()    # 发现 HuggingFace 检查点
│   ├── load_olmo_checkpoint()       # 加载模型权重
│   ├── load_training_logs()         # 加载训练日志
│   └── OLMO_CONFIGS                 # 模型配置字典
│
├── analysis.py              # 拟合和统计分析
│   ├── fit_state_equation()         # Q1: 拟合 P·V = f(N, T, V)
│   ├── fit_k_eff()                  # Q1: 拟合 k_eff(N) = k₀ + α·N^(-1/3)
│   ├── fit_kww()                    # Q3: KWW 拉伸指数拟合 + BIC
│   ├── compare_entropy_production() # Q2: 两个 schedule 的熵产生对比
│   ├── compute_monitoring_correlations() # Q4: Spearman 相关性
│   └── compare_schedules()          # Q5: paired t-test 统计对比
│
└── viz.py                   # 论文图表生成
    ├── plot_phase_diagram()         # T-S 相图 (Figure 1)
    ├── plot_state_equation_convergence()  # P·V/(N·T) 收敛 (Figure 2)
    ├── plot_entropy_comparison()     # 累积熵产生对比 (Figure 3)
    ├── plot_ts_trajectories()        # T-S 轨迹对比 (Figure 4)
    ├── plot_kww_fit()                # KWW 拟合图 (Figure 5)
    ├── plot_monitoring_correlation() # 相关性热图 (Figure 6)
    ├── plot_schedule_comparison()    # Schedule 对比柱状图 (Figure 7)
    └── plot_lr_schedules()           # LR schedule 曲线对比 (Figure 0)
```

### 7.2 脚本 (`scripts/thermo/`)

| 脚本 | 用途 | 典型用法 |
|------|------|---------|
| `measure_checkpoints.py` | 批量测量检查点 | `python measure_checkpoints.py --model-size 7B --use-hf --output ...` |
| `train_schedule_comparison.py` | Q5 代理训练 | `torchrun --nproc_per_node=8 train_schedule_comparison.py --model-size 190M --schedule gaussian --seed 42` |
| `train_midtraining_comparison.py` | Q3 对照实验 | `torchrun ... --mode midtrain` |
| `train_wsd_ablation.py` | Appendix E | `torchrun ... --stable-frac 0.6` |
| `run_analysis.py` | 后处理分析 | `python run_analysis.py --results-dir ... --output-dir ...` |
| `submit_all.sh` | K8s 提交 | `./submit_all.sh all` |

### 7.3 K8s 清单 (`scripts/k8s/thermo/`)

- 4 个测量 job: `thermo_measure_{190m,1b,7b,13b}.yaml`
- 24 个训练 job: `thermo_train_{190m,1b}_{cosine,wsd_linear,wsd_exponential,gaussian}_s{42,123,456}.yaml`
- 1 个 ablation job: `thermo_ablation_stable_frac.yaml`
- 1 个 mid-training job: `thermo_midtrain_comparison.yaml`
- 1 个生成器: `gen_training_jobs.py`

集群配置:
- 集群: ap-south-1 p5-llm (H200)
- 节点组: trainer5
- 镜像: `verl-multiturn:1.0.2`
- 存储: FSx 挂载在 `/fsx`
- 代码路径: `/fsx/dev/jiaqi/A-EVOLVE-V2/`

---

## 8. 输出格式

### 8.1 测量结果 (JSONL)

每行一个 JSON 对象:
```json
{
    "step": 1000,
    "model_name": "OLMo-7B",
    "num_params": 7000000000,
    "volume": 12345.6789,
    "spectral_entropy": 4.2345,
    "order_parameter": 0.1234,
    "internal_energy": 2.4567,
    "temperature": 1.23e-8,
    "pressure": 0.1,
    "free_energy": 2.4566,
    "pv_over_nt": 0.4912,
    "lr": 3e-4,
    "batch_size": 2048,
    "grad_variance": 1.5e-4
}
```

### 8.2 分析结果 (JSON)

- `q1_state_equation.json`: k_eff, α, γ, R² per scale + scaling law
- `q2_entropy_comparison.json`: ΔS_tot, η_thermo per schedule per scale
- `q3_kww_fitting.json`: τ, β, R², BIC per scale
- `q5_schedule_comparison.json`: final loss per schedule, Δ%, p-value

---

## 9. 需要填充的论文 [tbd] 值

所有 [tbd] 标记在 `paper/sections/experiments.tex` 中:

| 表格 | 字段 | 数据来源 |
|------|------|---------|
| Table 3 (Eq 9) | k₀, α | `q1_state_equation.json → scaling_law` |
| Table 4 (WSD vs Cos) | ΔS^WSD, ΔS^Cos, η^WSD, η^Cos per scale | `q2_entropy_comparison.json` |
| Table 5 (KWW) | τ, 3τ per scale | `q3_kww_fitting.json` |
| Table 7 (Schedules) | 190M Loss, 1B Loss per schedule | `q5_schedule_comparison.json` |
| Table 7 (Schedules) | η_thermo per schedule | 从训练 thermo_measurements.jsonl 计算 |

---

## 10. 资源估算总结

### 优化后总计: ~20,265 H200 GPU-hours

| 实验 | H200 GPU-hours | 来源 |
|------|-----------|------|
| 测量 OLMo-2 检查点 (1B/7B/13B) | 3,625 | 公开检查点, 仅需计算 |
| 训练 190M (4 schedules × 3 seeds) | 2,194 | 必须自训练 |
| 训练 1B (3 schedules × 3 seeds, 省去 cosine) | 13,166 | 必须自训练 |
| WSD Ablation | 914 | 必须自训练 |
| Mid-training 对比 | 366 | 必须自训练 |

### 比原方案节省 ~8,154 H200 GPU-hours (29%)

节省来源:
1. 1B Cosine 基线复用 OLMo-2-0425-1B: 省 ~4,389 GPU-hours
2. 不需要训练 7B/13B（仅测量公开检查点）: 省 ~3,500 GPU-hours
3. 1B mid-training 对照可从 OLMo-2 Stage 2 获得: 省 ~265 GPU-hours

### 墙钟时间 (4 节点 = 32 H200 GPU)

| 阶段 | 时间 | 可与上一阶段并行? |
|------|------|------------------|
| Phase 0: 快速验证 | ~3 天 | — |
| Phase 1: 190M 全量 | ~7 天 | 与 Phase 2 并行 |
| Phase 2: 多尺度测量 | ~12 天 | 与 Phase 1 并行 |
| Phase 3: 1B 训练 | ~17 天 | 在 Phase 1+2 后 |
| Phase 4: 分析 | <1 天 | 在全部完成后 |
| **总计 (含并行)** | **~30 天** | |

---

## 11. 需要补充完成的代码工作

当前代码已基本完整，但以下部分可能需要调整:

### 11.1 checkpoint_loader.py 需要增强

- [ ] 支持从 HuggingFace revision 自动发现并加载中间检查点
- [ ] 支持加载 optimizer state (safetensors 格式) 用于计算有效温度
- [ ] 支持从 OLMo-2 的 `config.yaml` 自动提取 LR/WD/batch_size
- [ ] 支持 OLMo-2 的 revision 命名格式 (`stage1-step1000-tokens5B`)

### 11.2 measure_checkpoints.py 需要增强

- [ ] 添加 `--use-hf` 模式的完整实现 (逐 revision 下载 + 测量)
- [ ] 添加从 optimizer state 计算 gradient variance 的逻辑
- [ ] 支持批量下载 (避免每个检查点重复下载模型配置)

### 11.3 run_analysis.py 需要增强

- [ ] 集成 Q4 分析 (需要加载 downstream benchmark 分数)
- [ ] 添加论文表格自动生成 (LaTeX 格式)
- [ ] 增加跨 scale 的 scaling law 拟合和可视化

### 11.4 训练 Callback 可能需要调整

- [ ] `LRScheduleCallback` 可能与 OLMo-core 内置 scheduler 冲突，需要确保只有一个生效
- [ ] `ThermoMeasurementCallback` 需要验证在分布式训练中只在 rank 0 执行
- [ ] 训练数据路径需要根据实际 FSx 目录调整

### 11.5 K8s 清单可能需要调整

- [ ] 验证 Docker 镜像 (`verl-multiturn:1.0.2`) 包含所有依赖
- [ ] 测量 job 的内存可能需要根据实际模型大小调整
- [ ] 确认 WandB secret 名称和 key

---

## 12. 关键风险和应对

| 风险 | 影响 | 应对 |
|------|------|------|
| OLMo-2 检查点下载慢/不可用 | 测量延迟 | 先测量 7B (直接 URL 可用), 1B/13B 用 HF |
| 190M 训练数据不在 FSx | 训练无法启动 | 准备 tokenized data 或使用 OLMo-core 的 data prep 工具 |
| Gaussian schedule 未见显著优势 | 论文核心结论受挑战 | 报告 null result, 分析原因, 可能是 190M scale 太小 |
| P·V/(N·T) 不收敛 | P1 被 falsify | 如实报告, 讨论非平衡效应 |
| KWW β ≈ 1 | P3 被 falsify | 报告为简单指数弛豫, 讨论与物理 glass 的差异 |

---

## 13. 论文表格模板

### Table 3: State Equation (Q1)
```
Scale    k_eff    α (correction)    γ (attraction)    R²
190M     [tbd]    [tbd]             [tbd]             [tbd]
1B       [tbd]    [tbd]             [tbd]             [tbd]
7B       [tbd]    [tbd]             [tbd]             [tbd]
13B      [tbd]    [tbd]             [tbd]             [tbd]

Scaling law: k_eff(N) = [k0] + [alpha] · N^(-1/3)
```

### Table 4: WSD vs Cosine (Q2)
```
Scale    ΔS_WSD    ΔS_Cos    Reduction    η_WSD    η_Cos
190M     [tbd]     [tbd]     [tbd]%       [tbd]    [tbd]
1B       [tbd]     [tbd]     [tbd]%       [tbd]    [tbd]
7B       [tbd]     [tbd]     [tbd]%       [tbd]    [tbd]
```

### Table 5: KWW Fits (Q3)
```
Scale    τ (B tokens)    β       R²      3τ (B tokens)
190M     [tbd]           [tbd]   [tbd]   [tbd]
1B       [tbd]           [tbd]   [tbd]   [tbd]
7B       [tbd]           [tbd]   [tbd]   [tbd]
```

### Table 7: Schedule Comparison (Q5)
```
Schedule          190M Loss    Δ          1B Loss    Δ
Cosine            [tbd]        [tbd]%     [tbd]      [tbd]%
WSD-Linear        [tbd]        baseline   [tbd]      baseline
WSD-Exponential   [tbd]        [tbd]%     [tbd]      [tbd]%
Gaussian (ours)   [tbd]        [tbd]%     [tbd]      [tbd]%
```
