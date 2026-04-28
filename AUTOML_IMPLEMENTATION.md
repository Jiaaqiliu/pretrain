# TrainingEvolver → AutoML Framework 实现文档

## 改造目标

将 TrainingEvolver 从 **LLM 训练框架** 改造为 **AutoML 框架**：
- 搜索传统 ML 模型（XGBoost / LightGBM / RandomForest）的超参数
- 直接在 MLE-Bench 任务上评估（Kaggle grader）
- 支持**规则驱动**和 **LLM 驱动**两种 mutation 策略

## 架构对比

### 原始架构（LLM 训练）

```
Workspace (LoRA rank, LR, data mix)
    ↓
MCGS Algorithm
    ↓
HF Trainer + PEFT Backend (训练 LLM)
    ↓
vLLM Evaluation (文本 → 评分)
```

### 新架构（AutoML）

```
Workspace (model_type, hyperparameters, FE flags)
    ↓
MCGS Algorithm
    ↓
Sklearn Backend (训练 XGBoost/LightGBM/RF)
    ↓
MLE-Bench Evaluation (submission.csv → Kaggle grader)
```

## 关键改动

### 1. Backend 层

**新增**: `agent_evolve/backends/sklearn_backend.py`

- `run_trial()` — 完整的训练 + 评估流程
- `_load_data()` — 加载 CSV 训练/测试数据
- `_apply_feature_engineering()` — 应用**可配置**的特征工程（见 #3）
- `_train_model()` — sklearn / xgboost / lightgbm 模型训练
- `_save_model()` — pickle 序列化 checkpoint
- `run_eval_plan()` — 在测试集上生成预测并保存为 `submission.csv`

### 2. Workspace 结构

**目录**: `seed_workspaces/mle_automl/`

```yaml
# model/config.yaml — 可变异的 seed config
model_type: xgboost
hyperparameters:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 1
  gamma: 0
  reg_alpha: 0
  reg_lambda: 1
  random_state: 42
feature_engineering:
  advanced: true
  fillna: median
  scale: false
  flags:                          # 新增 — 可被 LLM 变异
    passenger_id: true
    cabin_split: true
    spending_features: true
    age_groups: true
    family_features: true
    interactions: false           # 可搜索
    log_transform_spending: false # 可搜索
    target_encoding: false        # 可搜索
competition_id: spaceship-titanic
target_column: Transported
```

### 3. 可配置特征工程

**文件**: `agent_evolve/backends/feature_engineering.py`

每个特征工程组件由 `flags` dict 控制启用/禁用。`SpaceshipTitanicFeatureEngineer` 接受 flags 参数，backend 从 config 读取并传入：

```python
fe = create_feature_engineer("spaceship-titanic", flags=fe_config["flags"])
X_train = fe.fit_transform(X_train_raw, is_train=True, y=y_train)
X_test = fe.transform(X_test_raw)
```

新增的三个可搜索 flags：
- `interactions` — 创建 CryoSleep×HasSpending、Age×Spent、VIP×Spent、FamilySize×Spent
- `log_transform_spending` — 对所有消费列做 log1p
- `target_encoding` — 对 HomePlanet / Destination / CabinDeck 做平滑的目标编码

### 4. Mutation 策略

**文件**: `agent_evolve/training/algorithms/mcgs/ml_mutation.py`（规则驱动）+ `llm_mutation.py`（LLM 驱动）

#### 规则驱动（4 种 proposers）

1. **MLModelTypeMutationProposer** — 轮换模型类型
2. **MLLearningRateSweepProposer** — 学习率扫描
3. **MLDepthSweepProposer / MLNEstimatorsSweepProposer** — 超参数扫描
4. **MLHyperparameterMutationProposer** — 随机变异
5. **CombinedMutationProposer** — 按 cycle 组合多个 mutator（分阶段）

#### LLM 驱动（Claude Opus 4.7 via Bedrock）

**文件**: `agent_evolve/training/algorithms/mcgs/llm_mutation.py`

- **LLMHyperparameterProposer** — 主 proposer，覆盖超参数 + FE flags
- **LLMFeatureEngineeringProposer** — 兼容别名（转发到 LLMHyperparameterProposer）

关键设计：
- **13 维搜索空间**（10 个超参数 + 3 个 FE flags），离散化的候选值集
- **完整 config 历史** —— LLM 看到祖先链 patch 重建出的完整 YAML，不只是描述
- **Fingerprint 去重** —— 所有非 root 节点（含 crash 节点）的配置指纹进入 `tried_configs`，LLM 提议重复时硬 retry（最多 3 次）
- **Crash 分离呈现** —— 训练失败（metric=None）的配置单独列出，防止 LLM 再次提议同一个会 crash 的组合
- **噪声警告** —— prompt 中明确告诉 LLM test set 只有 4277 行，< 0.002 的差异在噪声范围内

### 5. Benchmark 适配

**修改**: `agent_evolve/benchmarks/mle_bench/mle_bench.py`

- `_grade_submission()` — 直接调用 `mlebench.registry.registry.get_competition(comp_id).grader(...)` 评分
- `parse_metrics()` — 从 `metrics.json` 读出 `mle_bench_score` 作为 primary metric

### 6. 注册 Backend

**修改**: `agent_evolve/training/registries.py`

```python
TRAINING_BACKENDS = {
    "h200_single_node": "...",
    "sklearn_backend": "agent_evolve.backends.sklearn_backend.SklearnBackend",
}
```

## 实验结果（spaceship-titanic）

### Baselines 对比

| 方法 | Cycles | Best Score | 备注 |
|------|--------|-----------|------|
| 规则驱动（phased sweep） | 20 | **0.81839** | LightGBM defaults 最优 |
| LLM-guided v1（初版） | 5 | **0.81839** | 用 1/4 的 cycles 达到同分 |
| LLM-guided v3（优化版） | 20 | **0.81839** | 找到第二条等价路径（XGBoost + 正则化） |

**结论**：0.81839 是该 workspace 单模型、无 CV 场景下的天花板。LLM 的价值体现在**路径多样性**和**更快收敛**，而不是更高分数。要突破上限需要 ensemble / CV，见"局限与下一步"。

### v3 关键发现

在 20-cycle v3 运行中，LLM 独立找到了一个**不同参数空间**下达到 baseline 的配置：

| Rank | Metric | Config | 来源 |
|------|--------|--------|------|
| 1 | 0.81839 | LightGBM defaults | 规则（cycle 2） |
| 1 | 0.81839 | XGBoost + min_child_weight=3 + reg_lambda=3 + gamma=0.1 | **LLM** |
| 3 | 0.81724 | LightGBM + interactions=True | LLM |
| 3 | 0.81724 | XGBoost + interactions=True | LLM |

### v1 → v3 优化总结

| 问题 | v1 表现 | v3 修复 |
|------|--------|---------|
| LLM 看不到完整 config | 只看到 `"Switch to lightgbm"` | 从 ancestor patches 重建完整 YAML |
| 重复提议 | cycles 18-20 提同一个 config 3 次 | fingerprint + 最多 3 次 retry |
| 陷入 slow-learning loop | 8 个类似 `lr=0.01 n_est=1000` | prompt 加 noise warning + "unexplored regions" hint |
| Crash 配置被重复提 | `target_encoding` bug 后 LLM 提了 10+ 次 | 把 crashed nodes 单独列给 LLM |
| FE 从未被搜索 | 所有 mutations 只动超参数 | 搜索空间加入 3 个 FE flags |
| Mutation 多样性 | 2-3 种类型 | 8+ 种（FE / 正则化 / 不同模型） |

## 文件清单

### 新增文件

```
agent_evolve/backends/sklearn_backend.py
agent_evolve/backends/feature_engineering.py
agent_evolve/training/algorithms/mcgs/ml_mutation.py
agent_evolve/training/algorithms/mcgs/llm_mutation.py
seed_workspaces/mle_automl/
  ├── manifest.yaml
  ├── model/config.yaml
  ├── train/ensemble.yaml
  ├── data/paths.yaml
  └── eval/competition.yaml
examples/mle_automl_example/
  ├── drive_model_search_4cycle.py           # 规则驱动，4 cycles
  ├── drive_advanced_search_20cycle.py       # 规则驱动，20 cycles 分阶段
  ├── drive_llm_test_5cycle.py               # LLM 烟雾测试
  ├── drive_llm_smoke3.py                    # LLM v3 烟雾测试（FE/dedup 验证）
  ├── drive_llm_20cycle.py                   # LLM 主实验
  ├── create_ensemble_submission.py
  ├── run_automl_search.sh
  └── run_full_optimization.sh
```

### 修改文件

```
agent_evolve/training/registries.py           # 注册 sklearn_backend
agent_evolve/benchmarks/mle_bench/mle_bench.py # 添加 mlebench grading
```

## 局限与下一步

**0.81839 这个上限不是 LLM proposer 的问题，是 workspace 层面的设计限制**：

1. **无 CV**：评估是一次性 train/test split，±0.003 都在噪声范围内，LLM 没办法判断小幅提升是真改进还是噪声。下一步可以把 backend 改成 stratified k-fold，metric 用 CV 均值。
2. **无 ensemble**：backend 一次训一个模型。要突破 0.82+ 在 spaceship-titanic 上几乎必须 ensemble（voting/stacking）。需要加一个 `train/ensemble.yaml` 层和相应的 backend 逻辑。
3. **FE flags 偏激进**：`interactions` 里的 `CryoSleep×HasSpending` 可能带来数据泄漏，所有 FE 组合都降了分。更保守的特征（count encoding、binned age）可能更稳。
4. **固定 FE 空间**：目前三个 flag 是手写的离散选项，LLM 只能开关不能创造。下一步可以让 LLM 生成 Python FE 代码片段（需要代码沙盒）。

## 运行命令

```bash
# 规则驱动 20-cycle
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 \
  /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_advanced_search_20cycle.py

# LLM-guided 20-cycle（Opus 4.7 via Bedrock）
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 \
  /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_llm_20cycle.py

# LLM 3-cycle 烟雾测试（验证 FE flags + dedup 生效）
PYTHONPATH=/fsx/yisi/A-EVOLVE-V2 \
  /fsx/yisi/A-EVOLVE-V2/.venv/bin/python \
  examples/mle_automl_example/drive_llm_smoke3.py
```

## 核心洞察

1. **Backend 是关键抽象** — 只需替换 Backend，框架其他部分（MCGS / Workspace / Benchmark）都可以复用。
2. **Mutator 接口已经通用** — 加 LLM-driven mutation 不需要改框架，只需要实现 `.propose(parent, graph)`。
3. **LLM 提议的真实价值** — 不是"找到更好的 hyperparameter"，而是在**相同上限下**探索**多样的路径**，并在大搜索空间里比 grid search 收敛更快。
4. **工程细节决定成败** — config 完整呈现、fingerprint 去重、crash 可见性、noise warning 这几项中任何一个做错，LLM 都会陷入 loop。
