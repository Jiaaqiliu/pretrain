# TrainingEvolver AutoML 快速入门

## 这是什么

基于 MCGS 的 AutoML 框架，在 MLE-Bench 任务上搜索最佳 ML 配置。两种 mutation 策略：

- **规则驱动** — 预定义的 sweep（depth / n_estimators / model_type），快速、零 API 成本
- **LLM 驱动** — Claude Opus 4.7 (via Bedrock) 分析训练历史做上下文感知的变异

## 环境准备

### Venv（已安装）

```bash
# 仓库里已有 Python 3.11 venv，包含 mlebench + boto3 + sklearn/xgboost/lightgbm
ls /fsx/yisi/A-EVOLVE-V2/.venv/bin/python
/fsx/yisi/A-EVOLVE-V2/.venv/bin/python -c "import mlebench, boto3, xgboost, lightgbm"
```

### Bedrock 认证（LLM-driven 才需要）

```bash
# 确认能访问 Opus 4.7 inference profile
aws bedrock list-inference-profiles --region us-west-2 | grep opus-4-7
# 期望看到：us.anthropic.claude-opus-4-7
```

### 数据

```bash
# spaceship-titanic 已准备在 seed workspace 里
ls seed_workspaces/mle_automl/data/
# train.csv, test.csv, sample_submission.csv

# 其他竞赛需要先准备：
# mlebench prepare --competition <competition_id>
# cp ~/.cache/mlebench/competitions/<id>/prepared/public/*.csv seed_workspaces/mle_automl/data/
```

## 规则驱动 AutoML

### 4-cycle 快速 baseline

```bash
cd /fsx/yisi/A-EVOLVE-V2
PYTHONPATH=. WANDB_DISABLED=true /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_model_search_4cycle.py
```

测试 4 种模型类型，~2-3 分钟完成。

### 20-cycle 分阶段搜索

```bash
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 WANDB_DISABLED=true /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_advanced_search_20cycle.py
```

分 4 个阶段：model exploration → depth tuning → n_estimators tuning → 随机变异。~10 分钟。

**Spaceship-Titanic 实测 best: 0.81839**（LightGBM defaults）

## LLM 驱动 AutoML（Claude Opus 4.7）

### 烟雾测试（3 cycles, ~1-2 分钟）

验证 FE flags 能被 LLM 开关、dedup 工作正常：

```bash
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 WANDB_DISABLED=true /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_llm_smoke3.py
```

### 完整 20-cycle 实验

```bash
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 WANDB_DISABLED=true /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_llm_20cycle.py
```

策略：前 3 cycles 规则探索 XGBoost/LightGBM/RandomForest，后 17 cycles 由 LLM 基于当前最优做上下文感知变异。~8-10 分钟，成本 ~$0.06 Bedrock。

**Spaceship-Titanic 实测 best: 0.81839**（LLM 找到第二条等价路径：XGBoost + 强正则化）

## 工作原理

### MCGS Cycle

```
1. SELECT  — HybridSelector: 前 N 个 cycles 从 root，之后从当前最优
2. MUTATE  — Mutator.propose() 返回 WorkspaceMutation（一组 patches）
3. FORK    — 复制 workspace，apply patches
4. TRAIN   — Sklearn Backend 训练 ML 模型
5. PREDICT — 生成 submission.csv
6. GRADE   — mlebench 直接评分（本地 grader）
7. UPDATE  — backprop reward，更新 incumbent
```

### LLM Mutator 的 context

每次调用 LLM 时给它：
- **完整的 parent config**（从祖先链 patch 重建出的 YAML）
- **Top 10 历史**（包括 metric 和完整 config）
- **已尝试的配置 fingerprint 列表**（硬约束：不许重复）
- **Crashed configs**（metric=None 的，警告 LLM 别再尝试）
- **13 维搜索空间**（10 个超参数 + 3 个 FE flags）

输出严格的 JSON，包含 reasoning、operations 列表、description。

## 查看结果

```bash
# 搜索图（完整 DAG）
cat runs/mle-automl-llm-20cycles/mle_automl/evolution/mcgs_graph.json | jq '.nodes[0:5]'

# 最佳 config
cat runs/mle-automl-llm-20cycles/mle_automl/evolution/incumbent/model/config.yaml

# 每个 cycle 的 report
cat runs/mle-automl-llm-20cycles/mle_automl/evolution/reports/cycle_0010.json

# LLM 的 reasoning（从 stdout）
grep -E '\[LLM\]' /tmp/llm_20cycle.log | head -30
```

## 自定义

### 改变搜索的模型

编辑对应的 `drive_*.py`：

```python
mutator = MLModelTypeMutationProposer(
    model_types=("xgboost", "lightgbm", "random_forest", "catboost")
)
```

### 调整 LLM 的搜索空间

编辑 `agent_evolve/training/algorithms/mcgs/llm_mutation.py` 的 `PARAM_SPACE`：

```python
PARAM_SPACE = {
    "hyperparameters.n_estimators": [50, 100, 150, 200, 300, 500, 1000],
    ...
    # 加新的可搜索维度
    "hyperparameters.num_leaves": [15, 31, 63, 127],
}
```

### 添加新的 FE flag

1. 在 `agent_evolve/backends/feature_engineering.py` 的 `DEFAULT_FLAGS` 里加 flag
2. 在 `fit_transform` 里加对应的 gated 调用
3. 在 `llm_mutation.py` 的 `PARAM_SPACE` 里加 `"feature_engineering.flags.<new_flag>": [True, False]`
4. 在 `seed_workspaces/mle_automl/model/config.yaml` 的 `flags` 里加默认值

### 换模型（Opus 4.7 → Opus 4.6）

```python
LLMHyperparameterProposer(model_id="us.anthropic.claude-opus-4-6-v1")
```

注意：Opus 4.7 不接受 `temperature` 参数，proposer 里已经移除。

## 实测结果（spaceship-titanic）

| 方法 | Cycles | Best | 备注 |
|------|--------|------|------|
| 规则驱动 20-cycle | 20 | 0.81839 | LightGBM defaults 最优 |
| LLM-guided v1 | 5 | 0.81839 | 用 1/4 cycles 达到同分 |
| LLM-guided v3 | 20 | 0.81839 | 找到第二条等价路径（XGBoost + 正则化）|

0.81839 是该 workspace 单模型、无 CV 场景下的天花板。要突破需要 ensemble 或 CV，见 `AUTOML_IMPLEMENTATION.md` 的"局限与下一步"。

## 常见问题

### `ModuleNotFoundError: No module named 'mlebench'`

用的是错误的 Python。务必用 venv 里的：

```bash
/fsx/yisi/A-EVOLVE-V2/.venv/bin/python  # 对
python3                                   # 错（除非你装过 mlebench）
```

### `ValidationException: The provided model identifier is invalid`

Bedrock 需要完整的 inference profile ID：

```python
# 错
model_id = "claude-opus-4-7"

# 对
model_id = "us.anthropic.claude-opus-4-7"
```

### `ValidationException: temperature is deprecated for this model`

Opus 4.7 不接受 `temperature`。如果你 fork 了 proposer，记得从 `invoke_model` 的 body 里去掉它。

### LLM 提议的配置导致训练崩溃

例如 `target_encoding=True` 引起 train/test 列数不一致。backend 的 try/except 会吞掉异常让 cycle 继续（metric=None），v3 proposer 会把 crash 的配置暴露给 LLM，避免重复提议。

### 训练太慢

```python
# drive_*.py 里调整
TrainingEvolveConfig(
    trial_budget_seconds=300,  # 默认 600，改小让超时更快失败
)
```

或者减 cycles、减 n_estimators。

## 相关文档

- `AUTOML_IMPLEMENTATION.md` — 完整架构、设计决策、实验分析
- `examples/mle_automl_example/README.md` — 规则驱动用法详解
- `examples/mle_automl_example/README_LLM_GUIDED.md` — LLM 驱动用法、context 设计、成本
