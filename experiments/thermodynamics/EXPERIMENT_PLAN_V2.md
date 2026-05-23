# Beyond Loss Curves: Thermodynamics of Pretraining — 完整实验计划

> **项目**: Beyond Loss Curves: Thermodynamics of Pretraining
> **目标**: 首次在大规模预训练上测量热力学状态变量，验证四个核心预测
> **日期**: 2026-05-23
> **状态**: 待执行

---

## 一、实验总览

### 核心思路

我们不需要从头训练大模型。EleutherAI 的 Pythia 和 AI2 的 OLMo 提供了完整的中间 checkpoint（含全部权重），覆盖 14M 到 13B 共 10+ 个规模。我们在这些已有 checkpoint 上做热力学测量（SVD → 谱熵、序参量、自由能），只在必须验证新 schedule 时才做小规模训练。

### 五个实验及数据来源

| 编号 | 实验 | 核心问题 | 数据来源 | 需要训练？ |
|------|------|---------|---------|----------|
| **E1** | 状态方程验证 | PV/(NkT) 是否收敛到常数？ | Pythia 6 个规模 × ~25 ckpts | 否 |
| **E2** | WSD vs Cosine 效率 | WSD 熵产是否更低？ | 自训 70M × 2 runs | 是（便宜） |
| **E3** | Mid-training KWW 弛豫 | 谱熵是否遵循拉伸指数？ | OLMo-2 Stage 转换 + 自训 70M | 部分 |
| **E4** | Gaussian schedule 验证 | 新 schedule 是否优于 WSD？ | 自训 70M × 3 runs | 是（便宜） |
| **E5** | ψ 监控指标验证 | ψ 与下游性能的相关性？ | Pythia 全规模 + benchmark 结果 | 否 |

### 预计总成本

| 阶段 | 内容 | Compute | 存储 |
|------|------|---------|------|
| Phase 1（零 compute） | Pythia/OLMo checkpoint 分析 | 0 GPU-hours（仅 SVD） | ~200 GB |
| Phase 2（低 compute） | 70M 补训 4-5 runs | ~1,024 H100-hours | ~50 GB |
| Phase 3（可选验证） | 410M 复现关键实验 | ~4,096 H100-hours | ~100 GB |

---

## 二、开源资源详细介绍

### 2.1 Pythia（EleutherAI）

**论文**: "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling" (Biderman et al., ICML 2023)

**GitHub**: https://github.com/EleutherAI/pythia

**HuggingFace**: https://huggingface.co/EleutherAI

**核心价值**: 10 个规模的模型，全部在相同数据上以相同顺序训练，每个保存 154 个中间 checkpoint（含完整权重）。这是目前全球最完整的预训练动态分析资源。

#### 模型列表

| 模型 | 参数量 | 层数 | Hidden | Heads | 学习率 | 单 ckpt 大小 |
|------|-------|------|--------|-------|--------|------------|
| pythia-14m | 14M | — | — | — | — | 0.03 GB |
| pythia-31m | 31M | — | — | — | — | 0.06 GB |
| pythia-70m | 70.4M | 6 | 512 | 8 | 1.0e-3 | 0.13 GB |
| pythia-160m | 162.3M | 12 | 768 | 12 | 6.0e-4 | 0.30 GB |
| pythia-410m | 405.3M | 24 | 1024 | 16 | 3.0e-4 | 0.75 GB |
| pythia-1b | 1,011.8M | 16 | 2048 | 8 | 2.5e-4 | 1.88 GB |
| pythia-1.4b | 1,414.6M | 24 | 2048 | 16 | 2.0e-4 | 2.64 GB |
| pythia-2.8b | 2,775.2M | 32 | 2560 | 32 | 1.6e-4 | 5.17 GB |
| pythia-6.9b | 6,857.3M | 32 | 4096 | 32 | 1.2e-4 | 12.77 GB |
| pythia-12b | 11,846.1M | 36 | 5120 | 40 | 1.2e-4 | 22.06 GB |

每个模型有两个变体：standard（The Pile）和 deduped（去重后的 The Pile）。

#### 训练配置（所有规模统一）

```yaml
optimizer: Adam (NOT AdamW; weight decay via DeepSpeed)
betas: [0.9, 0.95]
epsilon: 1.0e-8
weight_decay: 0.1                  # ← 我们的 P（压力）
lr_schedule: cosine                # ← 从 max_lr 衰减到 0.1 × max_lr
warmup: 1,430 steps (1% of 143K)
gradient_clipping: 1.0
sequence_length: 2048
batch_size: 2,097,152 tokens (2M)  # ← 所有规模统一
total_steps: 143,000
total_tokens: ~300B
training_data: The Pile (825 GiB)
precision: FP16
architecture: GPT-NeoX (RoPE, parallel attention, GELU)
```

#### Checkpoint 保存策略

154 个 checkpoint per model：
- Log-spaced early: steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512（11 个）
- 线性后段: steps 1000, 2000, 3000, ..., 143000（143 个）

**关键**: 每个 checkpoint 保存的是**完整模型权重**（HuggingFace 格式），不只是 loss 日志。

#### Benchmark 评估结果

GitHub 仓库的 `results/json/` 目录包含各 checkpoint 的下游 benchmark 分数，可直接用于 E5 实验。

### 2.2 OLMo-2（AI2）

**论文**: "OLMo 2: Furious" (AI2, 2025)

**GitHub**: https://github.com/allenai/OLMo-core

**HuggingFace**: https://huggingface.co/allenai

#### 模型列表

| 模型 | 参数量 | Stage 1 Tokens | Stage 2 Tokens | Checkpoints |
|------|-------|---------------|---------------|-------------|
| OLMo-2-1B | 1B | 4T | 50B | ~267 |
| OLMo-2-7B | 7B | 4T | 50B | ~965 |
| OLMo-2-13B | 13B | 5T | 100-300B | 可用 |
| OLMo-2-32B | 32B | — | — | 可用 |

#### 训练配置

```yaml
optimizer: SkipStepAdamW
betas: [0.9, 0.95]
weight_decay: 0.1
lr_schedule_stage1: Cosine with warmup
lr_schedule_stage2: Linear with warmup    # ← 注意：不是 WSD！
gradient_clipping: 1.0
sequence_length: 4096
```

#### 对我们的特殊价值

OLMo-2 有**多阶段训练**（Stage 1 → Stage 2 数据切换），这是唯一可用于 E3（KWW 弛豫）的开源资源。Stage 2 切换了数据分布（OLMo-mix → Dolmino-mix），正是我们需要观察谱熵弛豫的时刻。

---

## 三、环境搭建

### 3.1 硬件需求

- **Phase 1（checkpoint 分析）**: 单 GPU（A100/H100），主要做 SVD 计算
- **Phase 2（70M 训练）**: 4-8 × H100（单节点即可）
- **存储**: 至少 500 GB 可用空间

### 3.2 软件环境

```bash
# 创建虚拟环境
conda create -n thermo python=3.11 -y
conda activate thermo

# 核心依赖
pip install torch torchvision  # PyTorch >= 2.0
pip install transformers       # HuggingFace（加载 Pythia checkpoints）
pip install datasets           # HuggingFace Datasets
pip install scipy              # SVD 计算、KWW 拟合
pip install scikit-learn       # 随机化 SVD
pip install matplotlib seaborn # 绘图
pip install pandas             # 数据处理
pip install wandb              # 训练日志（Phase 2）
pip install tqdm               # 进度条

# OLMo-core（Phase 2 训练用）
pip install ai2-olmo-core

# 可选：加速 SVD
pip install cupy-cuda12x       # GPU 加速 SVD（如果需要）
```

---

## 四、数据下载

### 4.1 下载 Pythia Checkpoints

**策略**: 不需要下载全部 154 个 checkpoint。采样关键步骤即可。

#### 采样方案

对每个规模，下载以下 ~25 个 checkpoint：
- Early phase（11 个）: steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
- Main training（14 个）: steps 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 143000

#### 下载脚本

```python
"""download_pythia_checkpoints.py
下载 Pythia 指定规模和步数的 checkpoints。
用法: python download_pythia_checkpoints.py --scales 70m 410m 2.8b --output_dir ./checkpoints
"""
import os
import argparse
from huggingface_hub import snapshot_download

SAMPLE_STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 10000, 20000, 30000, 40000, 50000, 60000,
    70000, 80000, 90000, 100000, 110000, 120000, 143000
]

SCALES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]

def download_checkpoint(scale: str, step: int, output_dir: str, deduped: bool = True):
    suffix = "-deduped" if deduped else ""
    repo_id = f"EleutherAI/pythia-{scale}{suffix}"
    revision = f"step{step}"
    local_dir = os.path.join(output_dir, f"pythia-{scale}{suffix}", revision)

    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        print(f"  [skip] {repo_id} @ {revision} already exists")
        return

    print(f"  [download] {repo_id} @ {revision}")
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5"],  # 只下载 PyTorch/SafeTensors
        )
    except Exception as e:
        print(f"  [error] {repo_id} @ {revision}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", default=["70m", "410m", "2.8b"],
                        help="Model scales to download")
    parser.add_argument("--output_dir", default="./checkpoints",
                        help="Output directory")
    parser.add_argument("--deduped", action="store_true", default=True,
                        help="Use deduped variant")
    parser.add_argument("--all_steps", action="store_true",
                        help="Download all 154 steps instead of sampled 25")
    args = parser.parse_args()

    steps = list(range(0, 143001, 1000)) if args.all_steps else SAMPLE_STEPS
    # 补上 log-spaced early steps
    if args.all_steps:
        steps = sorted(set(steps + [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))

    for scale in args.scales:
        print(f"\n{'='*60}")
        print(f"Downloading pythia-{scale} ({len(steps)} checkpoints)")
        print(f"{'='*60}")
        for step in steps:
            download_checkpoint(scale, step, args.output_dir, args.deduped)

if __name__ == "__main__":
    main()
```

#### 执行下载

```bash
# Phase 1 最小集（~150 GB）: 3 个规模 × 25 个 checkpoint
python download_pythia_checkpoints.py \
    --scales 70m 410m 2.8b \
    --output_dir ./checkpoints

# Phase 1 完整集（~600 GB）: 6 个规模 × 25 个 checkpoint
python download_pythia_checkpoints.py \
    --scales 70m 160m 410m 1b 2.8b 6.9b \
    --output_dir ./checkpoints
```

#### 预估下载量

| 规模组合 | ~25 ckpts 每个 | 总量 |
|---------|---------------|------|
| 70m + 410m + 2.8b | 3.25 + 18.75 + 129.25 | ~151 GB |
| + 160m + 1b + 6.9b | + 7.5 + 47 + 319 | ~524 GB |

### 4.2 下载 OLMo-2 Checkpoints（E3 用）

```bash
# OLMo-2-7B 的 Stage 1 → Stage 2 过渡区 checkpoints
# 需要找到 stage 切换点附近的 checkpoints
# 具体路径待从 AI2 的 HuggingFace 确认

# 先查看可用的 checkpoints:
python -c "
from huggingface_hub import list_repo_refs
refs = list_repo_refs('allenai/OLMo-2-1124-7B')
for branch in refs.branches[:20]:
    print(branch.name)
"
```

### 4.3 下载 Pythia Benchmark 结果（E5 用）

```bash
# 克隆 Pythia 仓库获取 benchmark 结果
git clone --depth 1 https://github.com/EleutherAI/pythia.git ./pythia-repo
ls ./pythia-repo/results/json/
```

---

## 五、实验详细设计

### E1: 热力学状态方程验证

#### 目标
验证预测 P1：在 WSD/cosine 的稳态阶段，PV/(NkT) 是否收敛到依赖于 N 的常数 k_eff(N) = k₀ + α·N^{-1/3}。

#### 输入
Pythia 6 个规模（70M, 160M, 410M, 1B, 2.8B, 6.9B）× ~25 checkpoints = ~150 个测量点

#### 测量脚本

```python
"""measure_thermodynamic_vars.py
对单个 checkpoint 测量所有热力学状态变量。
"""
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM
from sklearn.utils.extmath import randomized_svd

def compute_spectral_entropy(weight_matrix: np.ndarray, k: int = 256) -> float:
    """计算权重矩阵的谱熵（使用 randomized SVD）。"""
    m, n = weight_matrix.shape
    actual_k = min(k, min(m, n))

    if min(m, n) <= 2 * k:
        # 小矩阵用 full SVD
        _, sigmas, _ = np.linalg.svd(weight_matrix, full_matrices=False)
    else:
        # 大矩阵用 randomized SVD
        _, sigmas, _ = randomized_svd(weight_matrix, n_components=actual_k, random_state=42)

    # 归一化为概率分布
    sigmas = sigmas[sigmas > 1e-10]  # 过滤掉数值零
    p = sigmas / sigmas.sum()

    # Shannon 熵
    entropy = -np.sum(p * np.log(p + 1e-12))
    return entropy

def compute_order_parameter(weight_matrix: np.ndarray) -> float:
    """计算谱间隙序参量 ψ = (σ₁ - σ₂)/(σ₁ + σ₂)。"""
    m, n = weight_matrix.shape
    if min(m, n) < 2:
        return 0.0
    # 只需要 top-2 奇异值，用 power iteration 更高效
    _, sigmas, _ = randomized_svd(weight_matrix, n_components=2, random_state=42)
    if len(sigmas) < 2 or sigmas[0] + sigmas[1] < 1e-10:
        return 0.0
    return (sigmas[0] - sigmas[1]) / (sigmas[0] + sigmas[1])

def measure_checkpoint(model_name: str, step: int, cache_dir: str = "./checkpoints"):
    """测量单个 checkpoint 的所有热力学状态变量。"""
    # 加载模型
    revision = f"step{step}"
    local_path = f"{cache_dir}/{model_name}/{revision}"
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=revision,
        cache_dir=local_path,
        torch_dtype=torch.float32,
    )

    results = {
        "model": model_name,
        "step": step,
        "layer_entropies": [],
        "layer_psis": [],
        "layer_sizes": [],
    }

    total_params = 0
    volume = 0.0  # V = ||θ||²_F

    for name, param in model.named_parameters():
        p = param.detach().cpu().numpy()
        total_params += p.size
        volume += np.sum(p ** 2)

        # 只对 2D 权重矩阵（非 bias、非 layernorm）做 SVD
        if p.ndim == 2 and min(p.shape) >= 2:
            entropy = compute_spectral_entropy(p)
            psi = compute_order_parameter(p)
            results["layer_entropies"].append(entropy)
            results["layer_psis"].append(psi)
            results["layer_sizes"].append(p.size)

    # 全局聚合
    sizes = np.array(results["layer_sizes"])
    weights = sizes / sizes.sum()

    results["N"] = total_params
    results["V"] = float(volume)                                        # 体积
    results["S"] = float(np.average(results["layer_entropies"], weights=weights))  # 谱熵
    results["psi"] = float(np.average(results["layer_psis"], weights=weights))     # 序参量

    # 从已知训练配置读取
    # Pythia 配置：所有规模 WD=0.1，LR 从 config 获取
    LR_MAP = {
        "pythia-70m-deduped": 1.0e-3,
        "pythia-160m-deduped": 6.0e-4,
        "pythia-410m-deduped": 3.0e-4,
        "pythia-1b-deduped": 2.5e-4,
        "pythia-1.4b-deduped": 2.0e-4,
        "pythia-2.8b-deduped": 1.6e-4,
        "pythia-6.9b-deduped": 1.2e-4,
        "pythia-12b-deduped": 1.2e-4,
    }
    WD = 0.1                                                           # 压力 P
    max_lr = LR_MAP.get(model_name, 3e-4)
    total_steps = 143000
    warmup_steps = 1430

    # Cosine LR at this step
    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        min_lr = 0.1 * max_lr
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

    results["P"] = WD                                                   # 压力
    results["T_proxy"] = lr  # 简化温度（精确温度需要梯度方差）
    results["PV_over_NT"] = (WD * volume) / (total_params * lr + 1e-20)  # PV/(NT)

    # 自由能 (简化版，用 lr 近似 T)
    # 注意：精确版本需要梯度方差 σ²_∇，此处先用 lr 作为 proxy
    results["F_proxy"] = 0.0  # 需要 training loss，从日志获取

    del model
    torch.cuda.empty_cache()

    return results
```

#### 分析流程

```python
"""analyze_state_equation.py
分析 PV/(NkT) 在不同规模和训练步数上的行为。
"""
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_measurements(results_dir: str) -> list:
    """加载所有测量结果。"""
    all_results = []
    for f in sorted(os.listdir(results_dir)):
        if f.endswith(".json"):
            with open(os.path.join(results_dir, f)) as fp:
                all_results.append(json.load(fp))
    return all_results

def state_equation_fit(N_values, k0, alpha):
    """k_eff(N) = k0 + alpha * N^(-1/3)"""
    return k0 + alpha * np.array(N_values) ** (-1/3)

def analyze():
    results = load_measurements("./results/E1/")

    # 按规模分组
    by_scale = {}
    for r in results:
        scale = r["model"]
        if scale not in by_scale:
            by_scale[scale] = []
        by_scale[scale].append(r)

    # 1. 对每个规模，取 cosine 稳态阶段（step 10K-100K）的 PV/(NT) 平均值
    scale_keff = {}
    for scale, runs in by_scale.items():
        stable_runs = [r for r in runs if 10000 <= r["step"] <= 100000]
        if stable_runs:
            keff = np.mean([r["PV_over_NT"] for r in stable_runs])
            keff_std = np.std([r["PV_over_NT"] for r in stable_runs])
            N = stable_runs[0]["N"]
            scale_keff[scale] = {"N": N, "keff": keff, "keff_std": keff_std}

    # 2. 拟合 k_eff(N) = k0 + alpha * N^(-1/3)
    N_vals = [v["N"] for v in scale_keff.values()]
    keff_vals = [v["keff"] for v in scale_keff.values()]
    keff_stds = [v["keff_std"] for v in scale_keff.values()]

    popt, pcov = curve_fit(state_equation_fit, N_vals, keff_vals,
                           sigma=keff_stds, p0=[0.5, 1.0])
    k0, alpha = popt
    print(f"State equation fit: k_eff(N) = {k0:.4f} + {alpha:.2f} * N^(-1/3)")
    print(f"R² = {1 - np.sum((np.array(keff_vals) - state_equation_fit(N_vals, *popt))**2) / np.sum((np.array(keff_vals) - np.mean(keff_vals))**2):.4f}")

    # 3. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图 1: PV/(NT) vs training step for each scale
    ax = axes[0]
    for scale, runs in sorted(by_scale.items()):
        steps = [r["step"] for r in runs if r["step"] > 100]
        pvnt = [r["PV_over_NT"] for r in runs if r["step"] > 100]
        ax.plot(steps, pvnt, label=scale, alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("PV / (NT)")
    ax.set_title("State Equation: PV/(NT) During Training")
    ax.legend(fontsize=8)
    ax.set_xscale("log")

    # 图 2: k_eff vs N with fit
    ax = axes[1]
    ax.errorbar(N_vals, keff_vals, yerr=keff_stds, fmt='o', capsize=5, label="Measured")
    N_fit = np.logspace(np.log10(min(N_vals)*0.5), np.log10(max(N_vals)*2), 100)
    ax.plot(N_fit, state_equation_fit(N_fit, *popt), 'r--',
            label=f"Fit: k₀={k0:.3f}, α={alpha:.1f}")
    ax.set_xlabel("N (parameters)")
    ax.set_ylabel("k_eff")
    ax.set_title("Finite-Size Scaling: k_eff(N) = k₀ + α·N^(-1/3)")
    ax.set_xscale("log")
    ax.legend()

    plt.tight_layout()
    plt.savefig("./figures/E1_state_equation.pdf", dpi=300)
    plt.show()
```

#### 预期结果
- PV/(NT) 在 cosine 的"稳态"阶段近似恒定（每个规模一个值）
- k_eff(N) 随 N 增大趋近常数 k₀，修正项按 N^{-1/3} 衰减
- R² > 0.95

#### 判断标准
- **成功**: PV/(NT) 在稳态阶段收敛（波动 <10%），k_eff(N) 的拟合 R² > 0.90
- **失败**: PV/(NT) 不收敛或 k_eff(N) 不遵循幂律修正 → P1 被证伪

---

### E2: WSD vs Cosine 热力学效率对比

#### 目标
验证预测 P2：WSD 的累计熵产低于 Cosine。

#### 为什么必须自己训练
Pythia 只有 Cosine schedule，OLMo-2 也是 Cosine（不是 WSD）。要做受控对比，必须在相同数据、相同规模上分别用两种 schedule 训练。

#### 训练配置

```yaml
# 共同配置
model_size: 70M (GPT-NeoX architecture, matching Pythia-70m)
data: The Pile (与 Pythia 一致)
total_tokens: 300B
batch_size: 2M tokens
weight_decay: 0.1
max_lr: 1.0e-3 (与 Pythia-70m 一致)
min_lr: 1.0e-4
warmup: 1,430 steps
checkpoint_interval: 200 steps  # 比 Pythia 密 5 倍，用于细粒度热力学追踪
seeds: [42, 123, 456]  # 3 个随机种子

# Run A: Cosine (对照组)
lr_schedule: cosine

# Run B: WSD
lr_schedule: wsd
stable_fraction: 0.78  # 78% 稳态
decay_type: linear

# Run C: WSD-Exponential (额外对照)
lr_schedule: wsd
stable_fraction: 0.78
decay_type: exponential
```

#### 分析
对每个 checkpoint 测量 S（谱熵）、T（lr 近似）、F = U - TS，然后计算：
- 熵产率: σ(t) = -(1/T) · dF/dt
- 累计熵产: ΔS_tot = ∫ σ(t) dt
- 热力学效率: η = |ΔF| / (|ΔF| + T̄ · ΔS_irr)

绘制 T-S 图（温度-熵相空间轨迹），比较两种 schedule 的轨迹形状。

#### 预计成本
3 seeds × 2 schedules × ~128 H100-hours = ~768 H100-hours

---

### E3: Mid-Training KWW 弛豫

#### 目标
验证预测 P3：mid-training 阶段的谱熵弛豫遵循 KWW 拉伸指数 φ(t) = exp[-(t/τ)^β]，β ∈ (0.5, 0.8)。

#### 两个数据源

**来源 A: OLMo-2 Stage 转换（已有数据，有混淆因素）**
- 用 OLMo-2-7B 的 Stage 1 → Stage 2 转换
- 在转换点前后的 checkpoints 上测量谱熵
- 混淆因素：LR schedule 同时改变（cosine → linear decay）

**来源 B: 自训 70M 模型（受控实验）**
```yaml
# 70M 模型，Stage 1 → Stage 2 受控实验
stage1:
  data: The Pile (subset)
  tokens: 200B
  lr_schedule: wsd (stable)
  max_lr: 1.0e-3

# 切换点：只换数据，不换 LR schedule
stage2:
  data: FineWeb-Edu + StarCoder
  tokens: 100B
  lr_schedule: wsd (继续 stable phase，然后 decay)
  max_lr: 1.0e-3 (先 warmup 到相同值)

checkpoint_interval: 100 steps  # Stage 2 更密集
```

#### KWW 拟合

```python
"""fit_kww.py
对 mid-training 谱熵弛豫做 KWW 拟合。
"""
from scipy.optimize import curve_fit

def kww(t, tau, beta):
    return np.exp(-(t / tau) ** beta)

def fit_kww_relaxation(steps, entropies, switch_step):
    """
    steps: 训练步数数组
    entropies: 对应的谱熵数组
    switch_step: 数据切换发生的步数
    """
    # 只取 switch 后的数据
    mask = steps >= switch_step
    t = steps[mask] - switch_step  # 从 0 开始
    S = entropies[mask]

    # 归一化
    S0 = S[0]
    S_inf = np.mean(S[-5:])  # 最后 5 个点的平均作为渐近值
    phi = (S - S_inf) / (S0 - S_inf + 1e-10)

    # 拟合
    popt, pcov = curve_fit(kww, t[1:], phi[1:],  # 跳过 t=0
                           p0=[t[-1]/3, 0.6],
                           bounds=([1, 0.1], [t[-1]*10, 1.0]))
    tau, beta = popt

    # 计算 R²
    phi_fit = kww(t[1:], *popt)
    ss_res = np.sum((phi[1:] - phi_fit) ** 2)
    ss_tot = np.sum((phi[1:] - np.mean(phi[1:])) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "tau": tau,
        "beta": beta,
        "r2": r2,
        "optimal_midtrain_tokens": 3 * tau,  # 95% 弛豫完成
    }
```

#### 预期结果
- β ≈ 0.55-0.65（玻璃样弛豫特征）
- R² > 0.95
- τ 随模型规模增大而增大

---

### E4: Gaussian Schedule 验证

#### 目标
验证预测 P4：从最小熵产原理推导的 Gaussian decay 优于 WSD。

#### 训练配置

```yaml
# 4 个 schedule 对比，70M 规模，3 seeds each
schedules:
  - cosine          # 基线 1
  - wsd_linear      # 基线 2
  - wsd_exponential # 基线 3
  - gaussian        # 我们的（从理论推导）

# Gaussian schedule 参数（从 E1 的状态方程推导）
gaussian:
  stable_fraction: 0.78
  min_lr_ratio: 0.01
  # tau_opt 由 boundary conditions 自动计算
```

#### 评估
- Final validation loss（主指标）
- 下游 benchmark（HellaSwag, ARC, PIQA, MMLU）
- 热力学效率 η_thermo
- 统计显著性：paired t-test across 3 seeds

---

### E5: ψ 监控指标验证

#### 目标
验证序参量 ψ 与下游 benchmark 性能的相关性远高于 loss。

#### 数据源
Pythia 全规模（70M-12B），利用已有的 benchmark 评估结果。

#### 分析流程

```python
"""analyze_monitoring.py
计算 ψ 与下游性能的相关性。
"""
import json
from scipy.stats import spearmanr

def load_benchmark_scores(pythia_repo_path: str, model_name: str, step: int):
    """从 Pythia 仓库加载 benchmark 分数。"""
    # benchmark 结果在 results/json/ 目录
    result_path = f"{pythia_repo_path}/results/json/{model_name}/step{step}.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return None

def compute_correlations(measurements: list, benchmarks: list):
    """
    计算各训练指标与下游性能的 Spearman 相关性。
    """
    # 提取指标
    losses = [m.get("training_loss", 0) for m in measurements]
    psis = [m["psi"] for m in measurements]
    entropies = [m["S"] for m in measurements]

    # 提取 benchmark 平均分
    avg_scores = []
    for b in benchmarks:
        if b is not None:
            scores = [v.get("acc", v.get("acc_norm", 0))
                      for k, v in b.items()
                      if isinstance(v, dict) and ("acc" in v or "acc_norm" in v)]
            avg_scores.append(np.mean(scores) if scores else 0)
        else:
            avg_scores.append(0)

    # 计算相关性
    r_loss, p_loss = spearmanr(losses, avg_scores)
    r_psi, p_psi = spearmanr(psis, avg_scores)
    r_entropy, p_entropy = spearmanr(entropies, avg_scores)

    return {
        "loss_correlation": {"r": r_loss, "p": p_loss},
        "psi_correlation": {"r": r_psi, "p": p_psi},
        "entropy_correlation": {"r": r_entropy, "p": p_entropy},
    }
```

#### 预期结果

| 指标 | 与下游性能相关性 (r) |
|------|-------------------|
| Training loss | ~0.3-0.4 |
| Spectral entropy S | ~0.7-0.8 |
| Order parameter ψ | **~0.9+** |

---

## 六、执行顺序与时间线

```
Week 1: 环境搭建 + 数据下载
├── Day 1-2: 安装依赖，搭建环境
├── Day 3-5: 下载 Pythia 70m/410m/2.8b checkpoints (~150 GB)
└── Day 5-7: 下载 Pythia benchmark 结果 + OLMo-2 stage 转换 checkpoints

Week 2-3: Phase 1 实验（零 Compute）
├── Day 8-10: E1 — 对 3 个规模的 ~75 个 checkpoint 做 SVD 测量
├── Day 11-12: E1 — 分析 PV/(NkT) + 拟合状态方程
├── Day 13-14: E5 — 计算 ψ 与 benchmark 相关性
├── Day 15: E3a — OLMo-2 stage 转换初步分析
├── Day 16-17: 补充下载更多规模 (160m, 1b, 6.9b)
└── Day 18-21: E1 扩展 — 6 个规模完整分析 + 论文核心图表

Week 4-5: Phase 2 实验（低 Compute）
├── Day 22-28: E2 — 训练 70M × 2 schedules × 3 seeds
├── Day 29-32: E3b — 训练 70M with mid-training data switch
├── Day 33-35: E4 — 训练 70M × Gaussian schedule × 3 seeds
└── Day 36-38: Phase 2 全部分析 + 图表

Week 6: 论文完善
├── Day 39-40: 填入所有 [tbd] 数值
├── Day 41-42: 绘制最终论文图表
└── Day 43-45: 论文修订 + 同步 Overleaf
```

---

## 七、输出文件结构

```
experiments/
├── EXPERIMENT_PLAN.md          # 本文档
├── checkpoints/                # 下载的 Pythia/OLMo checkpoints
│   ├── pythia-70m-deduped/
│   │   ├── step0/
│   │   ├── step1000/
│   │   └── ...
│   ├── pythia-410m-deduped/
│   └── pythia-2.8b-deduped/
├── scripts/
│   ├── download_pythia_checkpoints.py
│   ├── measure_thermodynamic_vars.py
│   ├── analyze_state_equation.py
│   ├── analyze_monitoring.py
│   ├── fit_kww.py
│   ├── train_schedule_comparison.py   # E2/E4 训练脚本
│   └── train_midtraining.py           # E3 训练脚本
├── results/
│   ├── E1/  # 状态方程测量结果 (JSON)
│   ├── E2/  # WSD vs Cosine 对比结果
│   ├── E3/  # KWW 弛豫拟合结果
│   ├── E4/  # Gaussian schedule 验证结果
│   └── E5/  # ψ 监控指标相关性
├── figures/
│   ├── E1_state_equation.pdf
│   ├── E1_keff_scaling.pdf
│   ├── E2_TS_diagram.pdf
│   ├── E2_entropy_production.pdf
│   ├── E3_kww_relaxation.pdf
│   ├── E4_schedule_comparison.pdf
│   └── E5_monitoring_correlation.pdf
└── logs/
    └── wandb/  # Phase 2 训练日志
```

---

## 八、风险与备选方案

| 风险 | 后果 | 备选方案 |
|------|------|---------|
| PV/(NT) 在 Pythia 上不收敛 | P1 被证伪 | 检查是否因为 Pythia 用 Adam 而非 AdamW 导致；用 OLMo-2（AdamW）验证 |
| 梯度方差数据不可用 | 无法计算精确温度 T | 用 lr 作为 proxy（论文中注明近似） |
| KWW β ≈ 1（简单指数） | 玻璃弛豫假说不成立 | 改为报告"指数弛豫"，讨论与 Winter & Janssen (2025) 的一致性 |
| Gaussian schedule 没有优势 | P4 被证伪 | 报告负结果，讨论最小熵产原理的适用条件 |
| Pythia 太旧（2023 架构） | Reviewer 质疑现代性 | 用 OLMo-2 补充验证 |

---

## 九、关键命令速查

```bash
# 下载 checkpoint
python scripts/download_pythia_checkpoints.py --scales 70m 410m 2.8b

# 测量单个 checkpoint
python scripts/measure_thermodynamic_vars.py --model pythia-70m-deduped --step 10000

# 批量测量
python scripts/measure_thermodynamic_vars.py --model pythia-70m-deduped --all_sampled

# 分析状态方程
python scripts/analyze_state_equation.py --results_dir ./results/E1/

# 拟合 KWW
python scripts/fit_kww.py --results_dir ./results/E3/

# 计算监控指标相关性
python scripts/analyze_monitoring.py --results_dir ./results/E5/
```
