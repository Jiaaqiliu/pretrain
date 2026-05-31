# MoE 谱测量代码使用指南

## 代码文件

| 文件 | 用途 |
|------|------|
| `experiments/thermodynamics/moe_measures.py` | 核心测量模块，支持 7 种 MoE 架构 |
| `scripts/measure_moe_olmoe.py` | OLMoE-1B-7B 批量 checkpoint 测量 |
| `scripts/measure_moe_cross_model.py` | 跨模型对比测量（分 3 个 Tier） |

## 支持的 MoE 架构

| 架构 | 代表模型 | 权重命名模式 |
|------|---------|-------------|
| `olmoe` | OLMoE-1B-7B | `model.layers.{L}.mlp.experts.{E}.{proj}.weight` |
| `mixtral` | Mixtral-8x7B/8x22B | `model.layers.{L}.block_sparse_moe.experts.{E}.{w}.weight` |
| `deepseek_v2` | DeepSeek-V2/V3 | 同 olmoe + `shared_experts` |
| `phi3_moe` | Phi-3.5-MoE | 同 mixtral |
| `dbrx` | DBRX | `transformer.blocks.{L}.ffn.experts.mlp.{E}.{w}.weight` |
| `qwen2_moe` | Qwen2-MoE, Qwen3 | 同 olmoe + `shared_expert` |
| `llama4_moe` | Llama 4 Scout/Maverick | `model.layers.{L}.feed_forward.experts.{E}.{w}.weight` |

自动检测架构，无需手动指定。

## 测量指标

### 逐专家 (ExpertSpectral)
- `alpha`: 幂律指数 α（ESD 的 P(λ) ~ λ^{-α}）
- `stable_rank`: SR = ||W||²_F / σ₁²
- `srd`: SR / d (归一化 stable rank)
- `spectral_entropy`: Shannon 谱熵
- `frobenius_norm`: ||W||_F
- `top10_sv`: 前 10 个奇异值

### 逐层 (LayerMoESpectral)
- `alpha_mean/std/min/max`: 跨专家 α 的统计量
- `srd_mean/std`: 跨专家 SR/d 的统计量
- `cross_expert_alignment`: 主导子空间余弦相似度均值 [0,1]
- `router_stable_rank/spectral_norm/srd`: 路由矩阵谱性质
- `shared_expert_alpha/srd`: 共享专家指标（如有）

### 全局 (MoECheckpointSpectral)
- `alpha_mean/std_across_experts`: 所有专家的 α 统计
- `alpha_attn/srd_attn`: Attention 层指标
- `alpha_moe/srd_moe`: MoE FFN 层指标
- `alpha_shared/srd_shared`: 共享专家指标
- `cross_expert_alignment_mean`: 跨层平均对齐度
- `router_srd_mean`: 路由矩阵平均 SR/d

## 快速使用

### 1. 测量单个模型

```bash
# 最简用法
python -m experiments.thermodynamics.moe_measures \
    allenai/OLMoE-1B-7B-0924 \
    -o results/olmoe_moe/test.jsonl

# 指定 GPU + 限制专家数（加速）
python -m experiments.thermodynamics.moe_measures \
    allenai/OLMoE-1B-7B-0924 \
    --device cuda \
    --max-experts 16 \
    -o results/olmoe_moe/test.jsonl

# 跳过 alignment 计算（更快）
python -m experiments.thermodynamics.moe_measures \
    mistralai/Mixtral-8x7B-v0.1 \
    --no-alignment \
    -o results/moe_cross_model/mixtral.jsonl

# 测量特定 revision
python -m experiments.thermodynamics.moe_measures \
    allenai/OLMoE-1B-7B-0924 \
    --revision step100000-tokens419B \
    --step 100000 \
    -o results/olmoe_moe/olmoe.jsonl
```

### 2. OLMoE 批量测量

```bash
# 10 个均匀分布的 checkpoint（推荐开始）
python scripts/measure_moe_olmoe.py --max-ckpts 10

# GPU 加速 + 断点续传
python scripts/measure_moe_olmoe.py --max-ckpts 25 --device cuda --resume

# 全量 244 checkpoints
python scripts/measure_moe_olmoe.py --max-ckpts 0 --device cuda --resume
```

### 3. 跨模型对比

```bash
# Tier 1: OLMoE + Phi-3.5-MoE + Mixtral-8x7B (CPU 可行)
python scripts/measure_moe_cross_model.py --tier 1

# Tier 2: + Mixtral-8x22B + DBRX + Qwen2-MoE (GPU)
python scripts/measure_moe_cross_model.py --tier 2 --device cuda

# 单个模型
python scripts/measure_moe_cross_model.py --model deepseek-ai/DeepSeek-V2 --device auto
```

## 输出格式

JSONL，每行一个 checkpoint 的完整测量结果。示例：

```json
{
  "model_name": "allenai/OLMoE-1B-7B-0924",
  "step": 100000,
  "revision": "step100000-tokens419B",
  "total_params": 6900000000,
  "active_params": 1300000000,
  "num_experts": 64,
  "top_k": 8,
  "hidden_dim": 2048,
  "alpha_mean": 3.45,
  "alpha_std_across_experts": 0.82,
  "srd_mean": 0.061,
  "alpha_attn": 2.89,
  "alpha_moe": 3.45,
  "cross_expert_alignment_mean": 0.73,
  "router_srd_mean": 0.42,
  "per_layer_summary": [
    {"layer": 0, "alpha_mean": 3.2, "alpha_std": 0.5, "srd_mean": 0.058, "alignment": 0.81, "router_srd": 0.45},
    ...
  ]
}
```

## 与现有 Dense 测量数据的对比

现有 dense 数据位于:
- `results/pythia_v2/` — Pythia 6 scales (70M-6.9B)
- `results/olmo2_v2/` — OLMo-2 4 scales (1B-32B)
- `results/mistral_v2/` — Mistral-7B
- `results/amber_v2/`, `results/k2_v2/` — Amber-7B, K2-65B

关键对比对:
- **OLMoE-1B-7B** (MoE, 1.3B active) vs **OLMo-2-1B** (dense, 1B) → 匹配 active params
- **Mixtral-8x7B** (MoE, 13B active) vs **Mistral-7B** (dense, 7B) → 同架构族
- **Phi-3.5-MoE** (MoE, 6.6B active) vs **Pythia-6.9B** (dense, 6.9B) → 匹配 active params

## 依赖

```bash
pip install torch transformers huggingface_hub numpy scipy
```
