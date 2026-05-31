# 前沿 MoE 模型全景（2023-2026）

## 开源 MoE 模型（100B+ 总参数）

| 模型 | 机构 | 总参数 | 激活参数 | 专家数 | 激活/token | 共享专家 | 路由方式 | 训练数据 | 发布 |
|------|------|--------|---------|--------|-----------|---------|---------|---------|------|
| DeepSeek-V2 | DeepSeek | 236B | 21B | 160+2 shared | 6+2 | ✅(2) | Top-6 细粒度 | 8.1T | 2024.05 |
| DeepSeek-V3 | DeepSeek | 671B | 37B | 256+1 shared | 8+1 | ✅(1) | 无辅助损失, Top-8 | 14.8T | 2024.12 |
| DeepSeek-R1 | DeepSeek | 671B | 37B | 256+1 shared | 8+1 | ✅(1) | 同V3 | 基于V3 | 2025.01 |
| DeepSeek-V4-Pro | DeepSeek | 1.6T | 49B | 384+1 shared | 6+1 | ✅(1) | Top-6 | - | 2026.04 |
| Mixtral 8x7B | Mistral | 46.7B | 12.9B | 8 | 2 | ❌ | Top-2 线性门控 | - | 2023.12 |
| Mixtral 8x22B | Mistral | 141B | 39B | 8 | 2 | ❌ | Top-2 线性门控 | - | 2024.04 |
| Qwen3-235B | Alibaba | 235B | 22B | 128 | 8 | ❌ | Global-batch LB | - | 2025 |
| Qwen3.5-397B | Alibaba | 397B | 17B | 512+1 shared | 10+1 | ✅(1) | Hybrid+GatedDeltaNet | - | 2026.02 |
| DBRX | Databricks | 132B | 36B | 16 | 4 | ❌ | Top-4 细粒度 | 12T | 2024.03 |
| Snowflake Arctic | Snowflake | 480B | 17B | 128 | 2 | Dense 10B+残差MoE | Top-2 | - | 2024.04 |
| Grok-1 | xAI | 314B | ~86B | 8 | 2 | ❌ | Top-2 | - | 2024.03 |
| Jamba 1.5 Large | AI21 | 398B | 94B | 16/MoE层 | 2 | ❌ | Top-2, 隔层MoE | - | 2024 |
| OLMoE-1B-7B | Allen AI | 6.9B | 1.3B | 64 | 8 | ❌ | Dropless token | 5T | 2024.09 |
| Llama 4 Scout | Meta | 109B | 17B | 16 | - | ❌ | 全层MoE | ~40T | 2025.04 |
| Llama 4 Maverick | Meta | 400B | 17B | 128 | - | ❌ | MoE+dense交替 | ~22T | 2025.04 |
| Phi-3.5-MoE | Microsoft | 42B | 6.6B | 16 | 2 | ❌ | Top-2 | 4.9T | 2024.08 |
| MiniMax-01 | MiniMax | 456B | 45.9B | 32 | Top-k | ❌ | 混合Lightning+Softmax attn | - | 2025.01 |
| Hunyuan-Large | Tencent | 389B | 52B | 1 shared+routed | 1+1 | ✅ | 混合专家路由 | - | 2024.11 |
| Kimi K2 | Moonshot | 1T | 32B | 384 | 8 | - | Top-8 细粒度 | 15.5T | 2025.07 |
| dots.llm1 | RedNote | 142B | 14B | 128+2 shared | 6+2 | ✅(2) | Top-6 细粒度 | 11.2T | 2025.06 |
| Nemotron 3 Super | NVIDIA | 120.6B | 12.7B | 128+1 shared | 6 | ✅(1) | LatentMoE | - | 2026 |
| Gemma 4-26B-A4B | Google | ~26B | ~4B | 128+1 shared | 8+1 | ✅(1) | Top-8 | - | 2026.04 |

## 商业/闭源模型（已知/推测使用 MoE）

| 模型 | 机构 | 已知/推测细节 |
|------|------|-------------|
| GPT-4 | OpenAI | 广泛推测 8x220B MoE (~1.8T 总参), top-2 路由 |
| Gemini 2.0/2.5 | Google | 确认 MoE, 具体配置未公开 |
| Claude Opus 5 | Anthropic | 推测 5T 参数 MoE, 未确认 |
| Grok-3/4/5 | xAI | 分别 3T/3T/6T MoE, 最大已公布模型 |

## 关键架构趋势

1. **细粒度 MoE 主导**: DeepSeek/Kimi/Qwen3.5 使用 128-512 个小专家而非 8 个大专家
2. **共享专家成为标配**: DeepSeek-V2 首创, 被 Qwen/Hunyuan/Nemotron/Gemma 采用
3. **无辅助损失路由**: DeepSeek-V3 的 auxiliary-loss-free 策略避免梯度冲突
4. **混合架构**: Jamba (SSM+Transformer+MoE), Snowflake (Dense+MoE 残差)
5. **参数量持续爆炸**: 从 2023 年的 ~50B 到 2026 年的 1.6T+

## 谱分析优先测量清单

### Tier 1（CPU 可行, <50B）
- `allenai/OLMoE-1B-7B-0924` — **244 个训练 checkpoint**, 最高优先
- `microsoft/Phi-3.5-MoE-instruct` — 42B, 16 experts
- `mistralai/Mixtral-8x7B-v0.1` — 46.7B, 经典 8 experts

### Tier 2（GPU 推荐, <200B）
- `mistralai/Mixtral-8x22B-v0.1` — 141B
- `databricks/dbrx-base` — 132B
- `Qwen/Qwen2-57B-A14B` — 57B, 带共享专家

### Tier 3（多 GPU, >200B）
- `deepseek-ai/DeepSeek-V2` — 236B, 细粒度+共享
- `deepseek-ai/DeepSeek-V3` — 671B

## 参考文献

- DeepSeek-V3 Technical Report: arxiv.org/abs/2412.19437
- Mixtral of Experts: arxiv.org/abs/2401.04088
- OLMoE: Open MoE: arxiv.org/abs/2409.02060
- Qwen3 Technical Report: arxiv.org/abs/2505.09388
- Llama 4 Blog: ai.meta.com/blog/llama-4-multimodal-intelligence/
- DBRX Blog: databricks.com/blog/introducing-dbrx-new-state-art-open-llm
- Kimi K2: intuitionlabs.ai/articles/kimi-k2-technical-deep-dive
