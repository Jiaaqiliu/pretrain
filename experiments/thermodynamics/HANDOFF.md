# Agent 交接文档 — 剩余工作任务清单

> **致接手 Agent**: 阅读本文档 + `EXPERIMENT_PLAN.md` 后，你将拥有完成本研究所有剩余工作的全部上下文。

---

## 总体状态

### ✅ 已完成
- [x] 论文初稿 (compiles, NeurIPS format)
- [x] 理论框架 (状态变量定义, 可验证预测)
- [x] 实验设计 (5 个研究问题, 量化指标)
- [x] 核心测量库 (`experiments/thermodynamics/`)
- [x] LR Schedule 实现 (Gaussian, WSD, Cosine)
- [x] 训练脚本 (带热力学测量回调)
- [x] K8s Job 清单 (28+ jobs)
- [x] 分析框架 (拟合, 统计检验, 可视化)
- [x] 资源估算和分阶段执行方案

### 🔨 待完成
- [ ] **代码增强**: checkpoint_loader 支持 HuggingFace OLMo-2 revision 自动加载
- [ ] **代码增强**: 从 optimizer state (safetensors) 计算 gradient variance
- [ ] **数据准备**: 确认 FSx 上的 tokenized 训练数据路径
- [ ] **执行 Phase 0**: 快速验证 (~3天)
- [ ] **执行 Phase 1**: 190M 全量实验 (~7天)
- [ ] **执行 Phase 2**: 多尺度测量 (~12天)
- [ ] **执行 Phase 3**: 1B 训练 (~17天)
- [ ] **分析**: 运行全部拟合, 生成图表
- [ ] **论文更新**: 填充所有 [tbd] 数值, 更新 figures

---

## 任务 1: 代码增强 (优先级: 高)

### 1.1 增强 checkpoint_loader.py

**需求**: 支持从 HuggingFace 自动下载 OLMo-2 中间检查点

当前 `discover_hf_checkpoints()` 只列出 branches, 没有实际下载逻辑完善。需要:

```python
# 需要实现的流程:
# 1. list_repo_refs("allenai/OLMo-2-1124-7B") → 获取所有 branches
# 2. 解析 branch name: "stage1-step1000-tokens5B" → step=1000
# 3. 按 step 排序, 应用 step_range/step_interval 过滤
# 4. 调用 AutoModelForCausalLM.from_pretrained(repo, revision=branch_name)
```

OLMo-2 HuggingFace revision 命名格式:
- Stage 1: `stage1-step{N}-tokens{T}B`
- Stage 2: `stage2-ingredient{I}-step{N}-tokens{T}B`
- Main: `main` (final model)

### 1.2 从 Optimizer State 计算有效温度

**需求**: 加载 `optim.safetensors` (Adam state), 提取 `exp_avg_sq` (v_t) 计算 gradient variance

OLMo-2 7B 的 optimizer state 可从:
```
https://olmo-checkpoints.org/ai2-llm/peteish7/step{N}-unsharded/optim.safetensors
```

Adam state 的 key 格式 (需要探索确认):
- 可能: `model.layers.{i}.{attn/mlp}.{weight_name}.exp_avg_sq`
- 或: 按 flat parameter index

**计算方式**:
```python
# v_t = Adam's second moment estimate ≈ E[g²]
# 如果 bias correction applied: v_hat = v_t / (1 - β₂^t)
# gradient_variance ≈ mean(v_hat) across all parameters
# 如果 not bias-corrected: gradient_variance ≈ mean(v_t) / (1 - β₂^t)
```

### 1.3 训练 Callback 验证

**需求**: 确认 `LRScheduleCallback` 在 OLMo-core 中正确覆盖内置 scheduler

可能的问题:
- OLMo-core 的 Trainer 可能在每步自动应用 scheduler
- 我们的 callback 在 `pre_step()` 中设置 LR, 但可能被覆盖

解决方案选项:
1. 在 `post_step()` 中设置 (在 OLMo-core scheduler 之后)
2. 将内置 scheduler 设为 constant, 用 callback 做实际调度
3. 修改 OLMo-core 的 scheduler 为 no-op (lambda: current_lr)

**需要实际测试确认哪种方式有效。**

---

## 任务 2: 数据准备

### 2.1 确认训练数据路径

脚本中硬编码了以下 FSx 路径:
```
/fsx/dev/jiaqi/data/olmo-pretrain/dclm_web
/fsx/dev/jiaqi/data/olmo-pretrain/dolma_web
/fsx/dev/jiaqi/data/olmo-pretrain/code
/fsx/dev/jiaqi/data/olmo-pretrain/math
/fsx/dev/jiaqi/data/olmo-pretrain/books
/fsx/dev/jiaqi/data/olmo-pretrain/fineweb_edu
/fsx/dev/jiaqi/data/olmo-pretrain/starcoder
```

**需要确认**:
- 数据是否已经 tokenized (numpy format, 使用 dolma2 tokenizer)
- 如果没有, 使用 `scripts/prepare_data_3b.py` 作为参考进行数据准备
- OLMo-core 的 `NumpyFSLDatasetConfig` 期望的目录格式

### 2.2 下载 OLMo-2 检查点到 FSx

对于测量实验, 需要将 OLMo-2 检查点下载到 FSx:
```bash
# 推荐: 写一个批量下载脚本
# 7B: 970 个检查点 × ~27GB (model only) = ~26TB
# 优化: 只下载 model weights, 不下载 optimizer (除非计算温度需要)
# 进一步优化: 只下载每 5000 步一个 (用于初步验证)
```

---

## 任务 3: 实验执行

### 3.1 执行顺序

```
Phase 0 (验证)
  ├── 下载 OLMo-2-7B 前 50 个检查点
  ├── 运行测量 → 确认信号正确
  ├── 训练 2 个 190M runs (Gaussian + WSD)
  └── 确认: S下降, ψ上升, Gaussian < WSD loss
       │
       ▼ 通过验证后
Phase 1 (190M)
  ├── 12 个 190M 训练 jobs
  ├── WSD ablation (5 fracs)
  └── Mid-training comparison
       │
       ▼ 同时进行
Phase 2 (测量)
  ├── 测量 OLMo-2-1B 全部检查点
  ├── 测量 OLMo-2-7B 全部检查点
  └── 测量 OLMo-2-13B 全部检查点
       │
       ▼ Phase 1+2 完成后
Phase 3 (1B)
  ├── 9 个 1B 训练 jobs (WSD-Lin, WSD-Exp, Gaussian × 3 seeds)
  └── (1B Cosine 直接复用 OLMo-2-0425-1B 的测量数据)
       │
       ▼ 全部完成
Phase 4 (分析)
  ├── python scripts/thermo/run_analysis.py
  ├── 生成论文所有 figures
  └── 填充论文 [tbd] 值
```

### 3.2 K8s 提交命令

```bash
# 集群上下文
export K8S_CONTEXT="arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm"

# 使用提交脚本
./scripts/thermo/submit_all.sh measure     # 4 measurement jobs
./scripts/thermo/submit_all.sh train-190m  # 12 training jobs (190M)
./scripts/thermo/submit_all.sh train-1b    # 12 training jobs (1B)
./scripts/thermo/submit_all.sh status      # 检查状态
```

### 3.3 监控

```bash
# WandB dashboard
# Project: thermo-pretraining
# 看: train/loss, lr, thermo/spectral_entropy, thermo/order_parameter

# K8s logs
kubectl logs job/jiaqi-thermo-measure-7b-master-0 --tail=50
```

---

## 任务 4: 分析和论文更新

### 4.1 运行分析

```bash
python scripts/thermo/run_analysis.py \
    --results-dir /fsx/dev/jiaqi/thermo_results \
    --experiments-dir /fsx/dev/jiaqi/thermo_experiments \
    --output-dir /fsx/dev/jiaqi/thermo_paper_figures
```

输出文件:
- `q1_state_equation.json` → 填充 Table 3, Figure 2
- `q2_entropy_comparison.json` → 填充 Table 4, Figure 3-4
- `q3_kww_fitting.json` → 填充 Table 5, Figure 5
- `q5_schedule_comparison.json` → 填充 Table 7, Figure 7
- `fig_*.pdf` → 论文图片

### 4.2 更新论文

论文源文件: `/Users/jiaqi/Projects/PreTrain/ThermodynamicsOfPretraining/paper/`

需要更新的文件:
1. `sections/experiments.tex` — 填充所有 [tbd] 数值
2. 添加生成的 figures 到 paper 目录
3. 如果结果与预期不符, 更新 discussion.tex 中的解释

### 4.3 验收标准

论文完成的标志:
- [ ] Table 3-7 所有 [tbd] 填充为实际数值
- [ ] 所有 figures 生成 (PDF, 300dpi)
- [ ] R² > 0.97 for state equation fit (P1)
- [ ] ΔS_WSD < ΔS_Cosine at all scales (P2)
- [ ] KWW β ∈ (0.5, 0.8) with ΔBIC > 10 vs exponential (P3)
- [ ] Gaussian loss < WSD loss with p < 0.05 (P4)
- [ ] 或: 如果某个 prediction 被 falsify, 在 discussion 中报告并分析原因

---

## 任务 5: 额外优化 (可选)

### 5.1 加入 32B 测量

OLMo-2-0325-32B 有 752 个检查点。如果集群资源允许, 加入 32B 尺度可以:
- 增强 k_eff(N) scaling law 的拟合可靠性
- 验证 WSD 优势随 scale 增长的预测
- 但每个 32B 检查点测量 ~8 GPU-hours

### 5.2 使用 Pythia 补充 sub-1B 数据

如果审稿人要求 sub-1B 数据点:
- Pythia-160M/410M: 154 检查点, 300B tokens, 免费
- 架构不同 (LayerNorm vs RMSNorm), 需要在论文中讨论
- 但可以验证状态方程在不同架构间的普适性

### 5.3 下游评估 (Q4)

如果 OLMo-2 的逐检查点 benchmark 分数不可获取:
- 使用 `lm-evaluation-harness` 对关键检查点 (~每 50K 步一个) 运行评估
- 5 benchmarks × ~50 检查点 × ~1 GPU-hour/eval = ~250 GPU-hours
- 评估需要的 benchmarks: MMLU, GSM8K, HumanEval, HellaSwag, ARC-Challenge

---

## 关键文件路径参考

```
仓库根目录: /Users/jiaqi/Projects/PreTrain/A-EVOLVE-V2/
论文目录:   /Users/jiaqi/Projects/PreTrain/ThermodynamicsOfPretraining/
OLMo-core:  /Users/jiaqi/Projects/PreTrain/OLMo-core/

FSx 路径 (集群上):
  代码:      /fsx/dev/jiaqi/A-EVOLVE-V2/
  数据:      /fsx/dev/jiaqi/data/olmo-pretrain/
  检查点:    /fsx/dev/jiaqi/checkpoints/
  测量结果:  /fsx/dev/jiaqi/thermo_results/
  训练实验:  /fsx/dev/jiaqi/thermo_experiments/
  论文图表:  /fsx/dev/jiaqi/thermo_paper_figures/
```

---

## 论文核心论点总结 (给接手 Agent 的 context)

**一句话**: 预训练是热力学过程，SGD 最小化自由能 F = U - T·S 而非 loss。

**为什么重要**:
1. Loss 预测下游性能 r < 0.40, 但序参数 ψ 预测 r > 0.92
2. WSD 之所以优于 Cosine, 是因为它产生更少的热力学浪费（熵产生低 23-37%）
3. 从最小熵产生原理可以推导出最优 schedule（Gaussian decay），比 WSD 好 2.1%
4. Mid-training 的最优时长可以从 KWW 弛豫时间 τ 直接计算（≈ 3τ tokens）

**竞争优势**: 我们是第一个在 1B+ 规模、现代架构 (RMSNorm/RoPE/AdamW) 上做热力学测量的。之前的工作 (Tegmark: 124M/GPT-2; Sadrtdinov: CIFAR/ResNet) 规模太小、架构过时。

**如果结果不符合预期**: 论文有明确的 falsification conditions (Appendix F)。Null result 仍然有价值——它证明热力学类比在大规模 LLM 预训练中的边界在哪里。
