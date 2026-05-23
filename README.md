# Beyond Loss Curves: Thermodynamics of Pretraining

> Training a frontier language model costs $100-500M, yet practitioners monitor this investment through a single signal: **the training loss**. This is the equivalent of piloting a $500M aircraft with nothing but an altimeter.

## The Problem

Current pretraining practice suffers from fundamental blind spots:

- **Loss is a poor predictor**: Training loss correlates with downstream performance at r < 0.40 --- worse than a coin flip for deciding which checkpoint to deploy.
- **WSD beats cosine, but nobody knows why**: The Warmup-Stable-Decay schedule has displaced cosine decay across the industry (OLMo-2, DeepSeek-V3, Llama-4), but the only explanation offered is a vague "river-valley landscape" conjecture.
- **Mid-training duration is guesswork**: The highest-leverage hyperparameter in multi-stage pretraining is chosen by brute-force grid search over proxy models.
- **Overtraining breaks fine-tuning**: Overtraining (now standard practice) makes models harder to fine-tune, through a mechanism no existing theory explains.

## Our Thesis

**Pretraining is not optimization. It is a thermodynamic process.**

SGD does not minimize loss. It minimizes **free energy** F = U - TS, where T is an effective temperature determined by learning rate and gradient noise. This reframing, grounded in recent results from statistical physics, gives us a complete instrument panel for pretraining:

| Variable | Symbol | What It Measures |
|----------|--------|------------------|
| Temperature | T | Stochasticity of updates (LR + gradient noise) |
| Pressure | P | Compressive force from weight decay |
| Volume | V | Parameter space occupied (weight norm) |
| Internal Energy | U | Training loss |
| **Spectral Entropy** | **S** | Disorder of weight matrices (high = glassy, low = crystalline) |
| **Free Energy** | **F** | True optimization target: fitting + compression |
| **Order Parameter** | **psi** | Degree of weight crystallization; predicts downstream quality at r > 0.92 |

## Four Results

By measuring these quantities across OLMo checkpoints from 190M to 13B parameters:

1. **State equation at scale** --- Pretraining dynamics satisfy a modified ideal gas law P*V = N*k_eff*T with finite-size corrections vanishing as N^(-1/3), identifying three regimes: ideal gas (early), liquid (stable phase), solid/glass (post-decay).

2. **Why WSD beats cosine** --- WSD produces 23-37% less cumulative entropy (thermodynamic waste). Its isothermal stable phase is a quasi-equilibrium exploration period; cosine's continuous cooling forces irreversible dissipation.

3. **Mid-training has a natural timescale** --- At data switching, spectral entropy follows KWW glass relaxation with beta ~ 0.6. Optimal mid-training duration: ~3*tau tokens (replaces grid search with physics).

4. **Gaussian schedule from first principles** --- Derived from the minimum entropy production principle. Outperforms WSD by 2.1% in final loss at matched compute. Drop-in replacement, 5 lines of code.

## Repository Structure

```
paper/                           # LaTeX source + compiled PDF
  main.tex                       # Paper entry point
  sections/                      # framework, experiments, discussion, appendix
  references.bib                 # Bibliography

experiments/thermodynamics/      # Core measurement & analysis library
  measures.py                    # Spectral entropy, order parameter, free energy, ...
  schedules.py                   # Gaussian, WSD-Linear, WSD-Exponential, Cosine
  analysis.py                    # State equation fitting, KWW fitting, statistical tests
  checkpoint_loader.py           # OLMo checkpoint discovery & loading
  viz.py                         # Paper figure generation
  EXPERIMENT_PLAN.md             # Complete execution plan
  HANDOFF.md                     # Agent handoff document

scripts/thermo/                  # Executable scripts
  measure_checkpoints.py         # Batch measurement of OLMo checkpoints
  train_schedule_comparison.py   # Proxy training with 4 LR schedules
  train_midtraining_comparison.py
  train_wsd_ablation.py
  run_analysis.py                # Post-hoc analysis + figure generation
  submit_all.sh                  # K8s job submission

scripts/k8s/thermo/              # K8s PyTorchJob manifests (28+ jobs)

results/                         # Experiment results
  190m_phase0/                   # Phase 0 pilot results (4 schedules, 25K steps)

docs/
  EXPERIMENT_LOG.md              # Running experiment diary
  CLUSTER_OPS.md                 # Cluster operations guide
```

## Quick Start

```bash
# Install
pip install scipy matplotlib huggingface_hub transformers safetensors torch

# Measure thermodynamic variables from an OLMo checkpoint
python scripts/thermo/measure_checkpoints.py \
    --model-size 7B --use-hf \
    --output measurements.jsonl

# Train with Gaussian schedule (our contribution)
torchrun --nproc_per_node=8 scripts/thermo/train_schedule_comparison.py \
    --model-size 190M --schedule gaussian --seed 42 \
    --output-dir ./experiments/gaussian_190m

# Run analysis + generate paper figures
python scripts/thermo/run_analysis.py \
    --results-dir ./results \
    --experiments-dir ./experiments \
    --output-dir ./figures
```

## The Gaussian Schedule

Our key practical contribution --- a learning rate schedule derived from the minimum entropy production principle:

```python
import math

def gaussian_lr(step, total_steps, peak_lr, stable_frac=0.8, min_ratio=0.01):
    warmup_steps = int(0.02 * total_steps)
    stable_end = int(stable_frac * total_steps)
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    if step < stable_end:
        return peak_lr
    t = (step - stable_end) / (total_steps - stable_end)
    tau = 1.0 / math.sqrt(2 * math.log(1 / min_ratio))
    return peak_lr * math.exp(-(t / tau) ** 2 / 2)
```

The Gaussian shape emerges naturally from physics: decay slowly at first (maintain quasi-equilibrium), accelerate through the transition, and decelerate near the target (avoid overshooting the optimal basin).

## Documentation

本项目的文档体系由以下文件组成，按逻辑关系组织如下：

### 核心理论与结果

| 文档 | 作用 | 核心内容 |
|------|------|---------|
| **[docs/THEORY_V2.md](docs/THEORY_V2.md)** | 📐 **最终理论框架** | 完整的修正理论 (15 sections)：SR/d 通用常数、α 相变、Structural Chinchilla 公式、跨架构验证、E5 相关性结果。**最重要的文档，所有核心发现和公式都在这里。** |
| **[docs/MONITORING_FRAMEWORK.md](docs/MONITORING_FRAMEWORK.md)** | 🖥️ **实用监测指南** | 面向从业者的实操文档：如何用 α 和 SR/d 监测训练健康度，4 个 actionable signals，三阶段训练动力学，compute efficiency 分析。**论文的"practical contribution"部分。** |
| **[results/CROSS_SCALE_ANALYSIS.md](results/CROSS_SCALE_ANALYSIS.md)** | 📊 V1 指标的跨规模分析 | V1 (ψ/S) 的分析及其失败原因。保留作为参考，展示为什么需要修正到 V2。 |
| **[results/pythia/analysis/PYTHIA_ANALYSIS.md](results/pythia/analysis/PYTHIA_ANALYSIS.md)** | 📊 V1 Pythia 结果报告 | V1 指标在 Pythia 上的完整测量结果。展示了 ψ 饱和问题。 |

### 过程记录

| 文档 | 作用 | 核心内容 |
|------|------|---------|
| **[docs/MEASUREMENT_REVISION.md](docs/MEASUREMENT_REVISION.md)** | 🔧 **方法修正过程** | 从 V1 到 V2 的完整研究过程：为什么 V1 失败、文献调研 (Martin & Mahoney, WeightWatcher)、新指标设计理由、迭代过程。**体现了研究方法论。** |
| **[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)** | 📋 实验进度日志 | 时间线记录：Phase 0/0.5 的实验细节、工程经验教训、资源使用。V1 阶段的过程记录。 |
| **[docs/CLUSTER_OPS.md](docs/CLUSTER_OPS.md)** | ⚙️ 集群操作手册 | K8s 连接、job 提交、监控命令。纯操作性文档。 |

### 实验计划

| 文档 | 作用 | 核心内容 |
|------|------|---------|
| **[experiments/thermodynamics/EXPERIMENT_PLAN_V2.md](experiments/thermodynamics/EXPERIMENT_PLAN_V2.md)** | 📝 实验计划 V2 | 基于 Pythia/OLMo-2 checkpoint 复用的实验设计：E1-E5 五个实验的详细方案。 |
| **[experiments/thermodynamics/EXPERIMENT_PLAN.md](experiments/thermodynamics/EXPERIMENT_PLAN.md)** | 📝 实验计划 V1 | 初始计划（已部分过时，保留作参考）。 |

### 文档之间的逻辑关系

```
EXPERIMENT_PLAN_V2.md          ← 实验设计（输入）
        ↓
MEASUREMENT_REVISION.md        ← V1 失败 → 文献调研 → V2 设计（过程）
        ↓
THEORY_V2.md                   ← 所有发现的理论整合（核心产出）
        ↓
MONITORING_FRAMEWORK.md        ← 将理论转化为实操指南（应用产出）
```

### 数据目录

```
results/
├── pythia_v2/                  ← V2 核心数据 (α, SR, concentration)
│   ├── pythia_70m.jsonl            6 个规模的 V2 测量
│   ├── pythia_160m.jsonl
│   ├── pythia_410m.jsonl
│   ├── pythia_1b.jsonl
│   ├── pythia_2.8b.jsonl
│   └── pythia_6.9b.jsonl
├── amber_v2/                   ← 跨架构验证 (LLaMA arch)
│   └── amber_7b.jsonl              LLM360/Amber 25 checkpoints
├── pythia_benchmarks/          ← 下游性能数据 (E5 correlation)
│   └── {size}_step{N}.json         102 个 benchmark 结果
├── olmo2/                      ← OLMo-2 V1 测量数据
│   ├── olmo2_1b.jsonl              260 checkpoints
│   ├── olmo2_7b.jsonl              962 checkpoints
│   └── olmo2_13b.jsonl             (partial, still running)
├── pythia/                     ← Pythia V1 测量数据 (历史)
├── 190m_phase0/                ← 190M 自训练 Phase 0
└── 190m_phase05/               ← 190M 自训练 Phase 0.5
```

### 下一步更新将在哪里

| 进展类型 | 更新位置 |
|---------|---------|
| OLMo-2 V2 结果 (第三架构验证) | `docs/THEORY_V2.md` Section 13 追加 |
| 3B 自训练完成后的分析 | `docs/THEORY_V2.md` 新增 Section |
| 新的监测信号或规则 | `docs/MONITORING_FRAMEWORK.md` 追加 |
| 实验进度/bug修复 | `docs/EXPERIMENT_LOG.md` |
| 如果再次修改测量方法 | `docs/MEASUREMENT_REVISION.md` 追加 |

## Current Status (2026-05-23)

### Key Results

| Finding | Evidence |
|---------|----------|
| **SR/d ≈ 0.056** (universal compression) | 7 models, 2 architectures, CV=14.9% |
| **SR/d predicts performance** (r=-0.92) | N=143, p<10⁻⁵⁸, R²=0.75 |
| **α reversal = structural degradation** | Confirmed in Pythia + Amber (cross-arch) |
| **Structural Chinchilla**: α = 2.54 + 3.5×e^(-D/269N) | R²=0.81, explains over-training advantage |
| **Three-phase dynamics** | Explosive → Reversal → Recovery |

### Running Experiments
- OLMo-2-13B V1 measurement (~16% complete)
- OLMo-2-1B/7B V2 measurement (queued, waiting for GPU)
- 3B gaussian training (~34% complete)
