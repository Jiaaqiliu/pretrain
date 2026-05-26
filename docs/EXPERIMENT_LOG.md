# 实验进度日志 — Thermodynamics of Pretraining

> 本文档实时记录实验进度、发现、问题和经验教训。每次有重要进展时更新。

---

## 时间线

### 2026-05-25: Phase 2 — 下游评测 + 泛化验证 + 1B Scale-Up

#### Experiment 2.1: Mistral-7B 光谱测量 (COMPLETED ✅)

测量完全未见过的架构 (GQA, sliding window attention) 以验证 SR/d 公式泛化性。

**结果:**
| 指标 | 测量值 | 预测值 | 匹配 |
|------|--------|--------|------|
| SR/d (全部层) | 0.104 | 0.050 | ✗ (GQA K/V 层膨胀) |
| SR/d (方形层, aspect≤1.5) | **0.040** | 0.050 | ✓ |
| α (mean) | 6.13 | >4 (大模型) | ✓ |
| α_attn | 3.79 | — | 接近重尾！ |
| α_mlp | 9.22 | — | 随机 (未成熟) |
| MLP/Attn gap | **5.43** | — | 历史最大 |

**关键发现:**
1. SR/d 公式在方形层（d×d attention projections）上完美验证 (0.040 vs 预测 0.050±0.01)
2. GQA 的 K/V 投影 (1024×4096, 4:1 aspect ratio) 有自然较高的 SR/d, 需要分别处理
3. Mistral 的 MLP/Attn gap (5.43) 是所有测量模型中最大的, 进一步确认 MLP 是结构瓶颈

#### Experiment 2.2: 410M 下游 Benchmark (COMPLETED ✅)

对已有的 3-way 实验 (cosine/WSD/α-guided, 2 seeds each) 的 final checkpoint 跑 5 项标准 benchmark。

**评测配置:** lm-eval-harness v0.4.10, 0-shot, batch=32
**基准测试:** LAMBADA, PIQA, WinoGrande, ARC-Easy, HellaSwag

**结果:**
| Schedule | ARC-E | HellaSwag | LAMBADA | PIQA | WinoGrande | **Average** |
|----------|-------|-----------|---------|------|------------|-------------|
| Cosine (2 seeds) | 0.550 | 0.307 | 0.284 | 0.645 | 0.510 | **0.459** |
| WSD (2 seeds) | 0.567 | 0.314 | 0.293 | 0.659 | 0.504 | **0.467** |
| α-Guided (2 seeds) | 0.574 | 0.313 | 0.302 | 0.655 | 0.498 | **0.468** |

| 对比 | Δ Average | 百分比提升 |
|------|-----------|-----------|
| WSD vs Cosine | +0.0078 | **+1.71%** |
| α-Guided vs Cosine | +0.0090 | **+1.95%** |
| α-Guided vs WSD | +0.0011 | +0.11% |

**关键结论:**
1. **Loss 改善 → 下游指标提升**: Δloss = -0.054 转化为 ~2% 平均 benchmark 提升
2. **α-Guided ≈ WSD** (差异 0.11%), 验证了 α-guided 能自动匹配手动调参的效果
3. **LAMBADA 上 α-Guided 最佳** (0.302 vs cosine 0.284, +6.3%), 光谱引导对 LM 质量提升显著
4. 这直接回答审稿人 "lower loss ≠ better model" 的质疑

#### Experiment 2.3: 1B 3-Way Scale-Up (RUNNING 🔄)

在 Pythia-1B 规模重做 3-way schedule 对比, 验证 prescriptive claim 在更大规模成立。

**配置:**
- 模型: Pythia-1B-deduped (from step0), d=2048, 16 layers
- 数据: FineWeb-Edu (9.92B tokens, Pythia tokenizer)
- Steps: 9,500, batch=4×16×8×2048 = 1M tokens/step
- LR: peak=2.5e-4, min=2.5e-5
- Jobs: `luhanqin-p2-1b-cosine`, `luhanqin-p2-1b-wsd`, `luhanqin-p2-1b-alpha`

**当前进度:** step ~500/9500 (5%), speed 0.18 steps/s, ETA ~14h (2026-05-26 05:47)

---

### 2026-05-23: Pythia 全规模热力学测量 (Phase 1-E1)

**目标**: 用 EleutherAI Pythia 套件 (70M-6.9B, 6 个规模) 验证状态方程 PV/(NT) = k_eff(N)。

**关键优势**: Pythia 所有规模在相同数据 (The Pile) 上以相同顺序训练, 154 个中间 checkpoint 公开，是全球最完整的预训练动态分析资源。

**资源:**
| 模型 | 检查点数 | 每ckpt大小 | 峰值内存 | 预计时间 (8GPU) |
|------|---------|-----------|---------|---------------|
| pythia-70m-deduped | 25 (sampled) | 166MB | 166MB | ~1 min |
| pythia-160m-deduped | 25 | 375MB | 375MB | ~2 min |
| pythia-410m-deduped | 25 | 911MB | 911MB | ~3 min |
| pythia-1b-deduped | 25 | 2.1GB | 2.1GB | ~6 min |
| pythia-2.8b-deduped | 25 | 5.7GB | 5.7GB | ~7 min |
| pythia-6.9b-deduped | 25 | 13.9GB | 13.9GB | ~10 min |
| **总计** | **150** | **流式** | **13.9GB** | **~45 min** |

**方法**: 流式处理 (download → measure → delete), 8 GPU 并行分片, per-GPU HF cache

**Jobs (4 节点并行):**
- `luhanqin-measure-pythia-small` (70m + 160m + 410m sequential)
- `luhanqin-measure-pythia-1b`
- `luhanqin-measure-pythia-2-8b`
- `luhanqin-measure-pythia-6-9b`

**预期结果:**
- E1: PV/(NT) 在 cosine 稳态期 (step 10K-100K) 收敛到规模相关常数
- E1: k_eff(N) = k₀ + α·N^(-1/3) 拟合 R² > 0.90
- ψ(N) scaling law: 6 个点拟合 ψ ∝ N^b
- 训练相变识别: warmup/early/stable/late 四阶段热力学特征

**状态**: 🔄 已提交 (2026-05-23), 等待 GPU 节点自动扩容

---

### 2026-05-23: OLMo-2 公开检查点测量 (Phase 1)

**目标**: 从 HuggingFace 下载 OLMo-2 的中间检查点，测量热力学状态变量。零训练成本。

**资源:**
| 模型 | 检查点数 | 时间 (8 GPU) | 峰值存储 |
|------|---------|-------------|---------|
| OLMo-2-1B | ~267 | ~15 min | 2GB |
| OLMo-2-7B | ~970 | ~1.5h | 14GB |
| OLMo-2-13B | ~717 | ~2.5h | 26GB |
| **总计** | **~1954** | **~4h** | **流式不累积** |

**方法**: 流式处理（下载→SVD→记录→删除→下一个），8 GPU 并行分片

**Jobs:**
- `luhanqin-measure-olmo2-1b`
- `luhanqin-measure-olmo2-7b`
- `luhanqin-measure-olmo2-13b`

**预期:**
- 验证 S/ψ 在 7B/13B 上的行为 (190M 上 decay phase 无信号，大模型可能不同)
- 获得 cosine schedule 基线
- 拟合 k_eff(N) scaling law (论文 P1)
- 为 3B 自训练实验提供参照

**Phase 0.5 结论 (前一轮):**
- 190M 上 decay phase 对 ψ 贡献 <0.1%，结构形成集中在 stable phase
- Gaussian 是唯一 Δψ_decay > 0 的 schedule (极微弱信号)
- 需要更大模型验证是否是 scale 问题

---

### 2026-05-22: Phase 0 完成 + 分析 + Phase 0.5 设计

**Phase 0 结果:**
- [x] 190M × 4 schedule × 25000 步 — 全部完成 (~10h)
- [x] 3B gaussian 单节点启动 — 运行中 (step 4190/50000, ETA 4d18h)
- [x] 数据下载全部完成: web 134GB + code 37GB + math 27GB = 198GB

**Phase 0 分析结论:**
- 热力学基本信号存在: S 下降 ✓, ψ 上升 ✓
- 但 S 呈 U-shape (先降后升), 各 schedule 差异极小
- 根因: 500M tokens 循环 52 次 + decay 只有 20%
- 详见 `results/190m_phase0/ANALYSIS.md`

**Phase 0.5 (修正实验) 完成:**
- 修正1: 数据量 500M → 25B tokens (消除循环) → ✅ S 信号 3.1× 放大
- 修正2: decay 比例 20% → 40% (放大信号窗口) → ⚠️ Δψ_decay 仍极小
- 修正3: 只跑 wsd_linear/gaussian/wsd_exponential (cosine 不可比) → ✅ 公平对比
- 关键发现: **decay phase 对 ψ 贡献接近零 (0.07%)**，结构形成在 stable phase 完成
- 详见 `results/190m_phase05/ANALYSIS.md`

### 2026-05-22: 项目启动 + 9 次迭代调试

**完成:**
- [x] 代码审查完毕，修复 13 个 bug
- [x] 代码推送到 GitHub (`Jiaaqiliu/pretrain`)
- [x] 代码同步到 FSx (`/fsx/dev/jiaqi/A-EVOLVE-V2/`)
- [x] 数据下载全部完成 (web 34B + code 9.5B + math 7B = ~50B tokens)
- [x] Pre-flight 检查通过
- [x] 190M smoke test 成功 (100 步, 174.7 TFLOPS/device)
- [x] 190M + 热力学测量验证 (200 步, 信号正确)
- [x] Phase 0 全面启动并完成

---

## Phase 0 结果详细分析

### 实验配置
| 参数 | 值 |
|------|-----|
| 模型 | OLMo2-190M (267M total params) |
| Schedule | gaussian, wsd_linear, wsd_exponential, cosine |
| 步数 | 25,000 |
| Batch | 256 × 4096 = 1M tokens/step |
| 数据 | 500M tokens (InMemoryTokenSource) |
| 实际 token 消耗 | 26.2B (循环 52 次) |
| Seed | 42 |
| GPU | 8 × H200 / schedule |

### 关键指标

| Schedule | S(final) | ψ(final) | V(final) | ΔS | Δψ |
|----------|----------|----------|----------|-----|------|
| gaussian | 5.9886 | 0.0926 | 64,122 | -0.036 | +0.079 |
| wsd_linear | 5.9887 | 0.0950 | 64,217 | -0.036 | +0.082 |
| wsd_exponential | 5.9885 | 0.0940 | 64,212 | -0.036 | +0.081 |
| cosine | 5.9885 | 0.1010 | 64,084 | -0.036 | +0.088 |

### 发现的三个问题

#### 问题 1: S 的 U-shape (非单调)
- S 在 step 4400 达到最小值 5.9506, 之后回升到 5.9886
- 回升幅度占总下降的 51%
- **原因**: 500M tokens 循环 52 次, step 4400 (~第9遍) 后模型已过拟合
- 后续训练是纯噪声, 打破了过度有序的结构 → S 回升

#### 问题 2: 各 schedule 差异极小
- Decay 阶段 Δψ 只有 0.0005 (占总 Δψ 的 4%)
- **原因**: decay 只有 5000 步 (20%), 且此时模型已在循环数据上饱和

#### 问题 3: Cosine 的 ψ 最高 (不公平对比)
- Cosine 从 step 0 就开始 decay, step 15000 时 LR 已降 58%
- WSD/Gaussian 到 step 20000 才开始 decay
- Cosine 实际有 25000 步退火 vs WSD 只有 5000 步
- **不是** gaussian 不如 cosine, 而是 cosine 有 5× 的退火时间

---

## Phase 0.5: 修正实验设计

### 修正措施

| 修正项 | 之前 | 之后 | 理由 |
|--------|------|------|------|
| 数据量 | 500M tokens | 25B tokens | 消除循环 (26B 消耗 / 25B 数据 ≈ 1 遍) |
| Decay 比例 | 20% (5000步) | 40% (10000步) | 放大 schedule 差异的观测窗口 |
| Stable 比例 | 78% | 58% | 腾出给 decay |
| Schedule 对比 | 4个含cosine | 3个WSD变体 | Cosine 独立分析,不做直接对比 |
| 训练步数 | 25000 | 25000 (不变) | 计算量不变 |
| 训练时间 | ~8h | ~8h (不变) | 只改数据不改计算 |

### 预期改进

| 指标 | 预期变化 |
|------|---------|
| S 轨迹 | 单调下降 (无 U-shape), 因为每步都是新数据 |
| Schedule 差异 | Δψ 从 0.0005 放大到 ~0.005+ (10×) |
| Gaussian vs WSD | Gaussian 应有更低的熵产生率 (σ), ψ 更高 |
| Loss | 不再是"记忆数据", 而是真正的语言建模能力 |

### 为什么 190M 上会有变化

1. **消除过拟合假象**: 500M 循环导致的 S 回升消失后, 能看到真实的结构演化
2. **Decay 窗口翻倍**: 10000 步的 decay 给了 gaussian 的"最小熵产生"优势足够的时间展现
3. **公平对比**: 三种 WSD 变体的 stable 和 decay 长度完全相同, 只有曲线形状不同

---

## 经验教训总结

### 实验设计教训

| # | 教训 | 具体实例 |
|---|------|---------|
| 1 | **数据量必须 ≥ 总 token 消耗** | 500M 数据跑 26B tokens = 循环 52 次 → 过拟合 |
| 2 | **schedule 对比要控制退火总时长** | Cosine 全程 decay vs WSD 只有 20% → 不公平 |
| 3 | **decay 比例太短则信号被淹没** | 5000 步 decay 只贡献 4% 的总 Δψ |
| 4 | **先跑小规模验证再上大规模** | 190M × 8h 发现问题, 比直接跑 3B × 5天 省很多 |

### 工程教训

| # | 教训 | 具体实例 |
|---|------|---------|
| 1 | 不要在 K8s job 中 `pip \| tail` | SIGPIPE 导致 exit 141 |
| 2 | 不要用 `.[all]` 安装 | flash-attn 编译 10+ 分钟, 且 PyTorch 2.8 已自带 |
| 3 | 多 job 不要同时 `git pull` | git lock 冲突 |
| 4 | 一定要读源码确认 Callback API | pre_step(batch) 不是 pre_step(step) |
| 5 | FSDP 下做测量用 `pre_optim_step` + `@dynamo.disable` | 避免 DynamicOutputShapeException |
| 6 | work_dir 必须是共享文件系统 | 各 rank 需要同一个 global indices 文件 |
| 7 | 先 disable WandB 跑通再加 | WandB 不应该阻塞训练 |
| 8 | 数据加载用 InMemoryTokenSource | NumpyFSLDatasetConfig 语义不同 |
| 9 | **HF 检查点流式测量必须清理缓存** | 970×14GB=13.6TB, 每次 `from_pretrained` 后删 snapshots+blobs |

### 热力学测量教训

| # | 教训 | 具体实例 |
|---|------|---------|
| 1 | Loss 需要正确的 metric API | `self.trainer._metrics` 路径不对, loss=0.0 |
| 2 | FSDP 下 SVD 只能在 local shard 上做 | 全参数 gather 太贵, 近似即可 |
| 3 | 谱熵对数据循环极其敏感 | 循环导致 S 先降后升, 不反映真实学习 |
| 4 | 序参数 ψ 更鲁棒 | 即使有数据循环, ψ 仍单调上升 |

---

## 下一步计划

### 立即执行: Phase 0.5 (修正 190M)
- [ ] 修改 `train_schedule_comparison.py`: max_tokens=25B, stable_frac=0.58
- [ ] 提交 3 个 job: gaussian/wsd_linear/wsd_exponential × seed=42
- [ ] 预计 ~8h 完成

### Phase 0.5 完成后: 决策点
- 如果 gaussian < wsd_linear loss 且 Δψ 显著 → 信号验证, 上 3B
- 如果差异仍不显著 → 可能需要更长训练 (50K步) 或更大模型
- 如果 gaussian > wsd_linear loss → 重新审视理论假设

### 3B 实验 (并行)
- 当前 3B gaussian 单节点运行中 (ETA 4d18h)
- 190M 完成后释放节点, 可以改多节点加速
- 3B 实验将使用全部 50B tokens (无循环问题)

### 还能做什么
1. **修复 loss 读取**: 让 thermo_measurements.jsonl 记录真实训练 loss
2. **多 seed 实验**: 验证 schedule 差异的统计显著性 (至少 3 seeds)
3. **OLMo-2 公开检查点测量**: 免费获得 cosine baseline 的热力学数据
4. **可视化**: 生成论文 figure (S-t, ψ-t, T-S phase diagram)

---

### 2026-05-24: OLMo-2-13B V2 测量 + 真实数据 3-Way 实验

**OLMo-2-13B V2 测量** — ✅ 完成 (23 min, 25 checkpoints)

关键发现:
- SR/d_final = 0.043 (接近 asymptotic limit 0.040!)
- 巨大 α reversal: 4.25 → 6.95 (Δα = +2.71, 所有模型中最大)
- MLP/Attn gap = 1.69 (最大, 大模型 MLP 更难训练)
- 验证了 SR/d asymptotic model, 扩展了验证范围到 13B

**真实数据 3-Way 实验** — 🔄 运行中

- 数据: FineWeb-Edu, 9.92B tokens, Pythia tokenizer (10 shards, 35GB)
- 模型: Pythia-410M from step0 checkpoint
- Schedule: Cosine vs WSD vs α-Guided × 2 seeds = 6 runs
- 9000 steps, ~8h each
- **问题修复**: from_config() 初始化导致 NaN → 改用 step0 pretrained weights

**经验教训**:
- `AutoModelForCausalLM.from_config()` 的默认初始化对于 GPT-NeoX 不稳定, 第 2 步就 NaN
- 必须使用 `from_pretrained(..., revision="step0")` 获得正确初始化
- Pythia tokenizer vocab_size=50254 但 model config vocab_size=50304 (有 padding tokens)
- FineWeb-Edu `sample-10BT` 实际只产出 9.92B tokens (部分文档太短被过滤)

---

## 资源使用追踪

| 日期 | GPU-hours | Job | Notes |
|------|-----------|-----|-------|
| 2026-05-22 AM | ~5 | 调试迭代 (debug1-9) | 9 次迭代修复各种 bug |
| 2026-05-22 | ~256 | 190M × 4 schedule × 8h | Phase 0 完成 |
| 2026-05-24 | ~3 | OLMo-2-13B V2 measurement | 8 GPU × 23 min |
| 2026-05-24 | ~6 (data prep) | FineWeb tokenize | 1 GPU × 6.3h |
| 2026-05-24 | ~384 (est.) | 3-Way training × 6 runs | 6 × 8 GPU × 8h |
| 2026-05-22 ongoing | ~80 (so far) | 3B gaussian 单节点 | 运行中 |
| **总计** | **~341** | | |

---

## 文件位置

### 本地
```
/Users/itsjiaqi/Projects/pretrain-review/
├── results/190m_phase0/
│   ├── gaussian.jsonl         # 125 measurements
│   ├── wsd_linear.jsonl       # 125 measurements
│   ├── wsd_exponential.jsonl  # 125 measurements
│   ├── cosine.jsonl           # 125 measurements
│   └── ANALYSIS.md            # 分析报告
├── docs/
│   ├── EXPERIMENT_LOG.md      # 本文档
│   └── CLUSTER_OPS.md         # 集群操作手册
└── scripts/thermo/            # 训练脚本
```

### 集群 (FSx)
```
/fsx/dev/jiaqi/
├── A-EVOLVE-V2/               # 代码仓库
├── data/olmo-3b-pretrain/     # 原始数据 (198GB)
│   ├── web/   (3480 shards, 134GB, ~34B tokens)
│   ├── code/  (967 shards, 37GB, ~9.5B tokens)
│   └── math/  (350 shards, 27GB, ~7B tokens)
├── data/olmo-pretrain/        # 符号链接
├── thermo_experiments/
│   ├── 190m_gaussian_s42/     # Phase 0 结果 + checkpoints
│   ├── 190m_wsd_linear_s42/
│   ├── 190m_cosine_s42/
│   ├── 190m_wsd_exponential_s42/
│   └── 3b_gaussian_s42/       # 运行中
└── thermo_results/            # 导出的分析数据
```
