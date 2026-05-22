# Thermodynamics of Pretraining — 实验代码

> 论文: "Beyond Loss Curves: Thermodynamics of Pretraining"

## 快速开始

```bash
# 安装依赖
pip install scipy matplotlib huggingface_hub transformers safetensors

# 测量一个 OLMo-2-7B 检查点
python scripts/thermo/measure_checkpoints.py \
    --model-size 7B --use-hf \
    --output results.jsonl --step-range 1000,5000

# 训练 190M (Gaussian schedule)
torchrun --nproc_per_node=8 scripts/thermo/train_schedule_comparison.py \
    --model-size 190M --schedule gaussian --seed 42 \
    --output-dir ./output

# 分析 + 图表
python scripts/thermo/run_analysis.py \
    --results-dir ./results --experiments-dir ./output \
    --output-dir ./figures
```

## 文档

- **[EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md)** — 完整实验执行计划（含资源估算、分阶段方案、代码结构、输出格式）
- **[HANDOFF.md](./HANDOFF.md)** — Agent 交接文档（任务清单、验收标准、论文填充指引）

## 代码结构

```
experiments/thermodynamics/
├── measures.py          # 热力学状态变量 (S, ψ, V, T, F, σ)
├── schedules.py         # LR schedules (Gaussian, WSD, Cosine)
├── checkpoint_loader.py # OLMo 检查点发现 + 加载
├── analysis.py          # 拟合 (状态方程, KWW, 统计检验)
└── viz.py               # 论文图表生成

scripts/thermo/
├── measure_checkpoints.py          # 批量测量
├── train_schedule_comparison.py    # Q5 代理训练
├── train_midtraining_comparison.py # Q3 验证
├── train_wsd_ablation.py           # Appendix 消融
├── run_analysis.py                 # 后处理分析
└── submit_all.sh                   # K8s 提交脚本

scripts/k8s/thermo/
├── thermo_measure_*.yaml           # 测量 jobs (4个)
├── thermo_train_*.yaml             # 训练 jobs (24个)
├── thermo_ablation_*.yaml          # 消融 job
├── thermo_midtrain_*.yaml          # Mid-training job
└── gen_training_jobs.py            # YAML 生成器
```
