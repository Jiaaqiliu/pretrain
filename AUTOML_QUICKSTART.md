# TrainingEvolver AutoML 快速入门

## 🎯 这是什么？

**TrainingEvolver AutoML** 是一个基于 MCGS 的自动机器学习框架，用于在 MLE-Bench 任务上搜索最佳 ML 模型配置。

### 核心特点
- ✅ 不需要 LLM，直接优化传统 ML 模型（XGBoost、LightGBM、RandomForest）
- ✅ 使用 MCGS 智能搜索超参数空间
- ✅ 自动在 MLE-Bench 上评估（Kaggle 分数）
- ✅ 快速：每个 cycle 约 10 分钟

## 🚀 5 分钟快速开始

### 1. 安装依赖

```bash
pip install mle-bench scikit-learn xgboost lightgbm pandas numpy pyyaml
```

### 2. 准备数据

```bash
# 选择一个 MLE-Bench 竞赛
COMPETITION_ID="spaceship-titanic"

# 准备数据
mlebench prepare --competition $COMPETITION_ID

# 复制到 workspace
cp ~/.cache/mlebench/competitions/$COMPETITION_ID/prepared/public/*.csv \
   seed_workspaces/mle_automl/data/
```

### 3. 配置竞赛

```bash
# 编辑配置文件
vim seed_workspaces/mle_automl/model/config.yaml

# 修改这两行：
competition_id: spaceship-titanic  # 你的竞赛 ID
target_column: Transported         # 目标列名
```

### 4. 运行搜索

```bash
# 运行 4-cycle 搜索（测试 4 种模型配置）
bash examples/mle_automl_example/run_automl_search.sh

# 监控进度
tail -f runs/mle-automl-search/logs/run.log
```

### 5. 查看结果

```bash
# 查看最佳配置
cat runs/mle-automl-search/mle_automl/evolution/incumbent/model/config.yaml

# 查看搜索图
cat runs/mle-automl-search/mle_automl/evolution/mcgs_graph.json | jq

# 查看每个 cycle 的报告
cat runs/mle-automl-search/mle_automl/evolution/reports/cycle_0000.json | jq
```

## 📊 工作原理

```
每个 MCGS Cycle:

1. SELECT  → 选择父配置（如 XGBoost, lr=0.1）
2. MUTATE  → 变异（切换到 LightGBM）
3. TRAIN   → 训练 ML 模型（sklearn/xgboost）
4. PREDICT → 生成 submission.csv
5. GRADE   → mlebench grade → Kaggle 分数
6. REWARD  → MCGS 更新图，晋升最佳配置
```

## 🎨 自定义搜索

### 改变搜索的模型

编辑 `examples/mle_automl_example/drive_model_search_4cycle.py`:

```python
mutator=MLModelTypeMutationProposer(
    model_types=("xgboost", "lightgbm", "random_forest", "catboost")
)
```

### 搜索学习率

```python
from agent_evolve.training.algorithms.mcgs.ml_mutation import MLLearningRateSweepProposer

mutator=MLLearningRateSweepProposer(
    learning_rates=(0.01, 0.05, 0.1, 0.2, 0.3)
)
```

### 随机超参数搜索

```python
from agent_evolve.training.algorithms.mcgs.ml_mutation import MLHyperparameterMutationProposer

mutator=MLHyperparameterMutationProposer(mutation_rate=0.3)
```

## 📁 文件结构

```
agent_evolve/
├── backends/sklearn_backend.py              # AutoML Backend
├── benchmarks/mle_bench/mle_bench.py        # MLE-Bench 评分
└── training/algorithms/mcgs/ml_mutation.py  # Mutation 策略

seed_workspaces/mle_automl/
├── manifest.yaml                            # Workspace 定义
├── model/config.yaml                        # 模型配置（可变异）
└── model/feature_engineering.yaml           # 特征工程（可变异）

examples/mle_automl_example/
├── README.md                                # 详细使用指南
├── drive_model_search_4cycle.py             # 运行脚本
└── run_automl_search.sh                     # 一键启动

AUTOML_IMPLEMENTATION.md                     # 完整技术文档
```

## 💡 常见任务

### 批量运行多个竞赛

```bash
for comp in spaceship-titanic house-prices digit-recognizer; do
    mlebench prepare --competition $comp
    # 更新 config.yaml
    # 运行搜索
done
```

### 添加新模型类型

在 `agent_evolve/backends/sklearn_backend.py` 中：

```python
elif model_type == "catboost":
    import catboost
    model = catboost.CatBoostClassifier(**hyperparams)
```

### 调试评分

```bash
# 手动评分一个 submission
mlebench grade \
    --competition spaceship-titanic \
    --submission path/to/submission.csv
```

## 🐛 故障排除

### 找不到数据
```bash
# 确认数据已准备
ls ~/.cache/mlebench/competitions/spaceship-titanic/prepared/public/
# 应该看到 train.csv, test.csv
```

### 评分失败
```bash
# 检查 submission 格式
head runs/mle-automl-search/nodes/node-*/workspace/evolution/eval/ml_training/test/submission.csv

# 手动评分查看详细错误
mlebench grade --competition <comp_id> --submission <path>
```

### 训练太慢
- 减少 `n_estimators`（如 100 → 50）
- 使用更简单的模型（RandomForest → LogisticRegression）
- 减少训练数据（采样）

## 📚 文档

- **详细使用**: `examples/mle_automl_example/README.md`
- **技术文档**: `AUTOML_IMPLEMENTATION.md`
- **MLE-Bench**: https://github.com/openai/mle-bench

## 🎯 预期结果

运行 4 cycles 后：

```
=== Final Result ===
cycles_completed: 4
best_metric: 0.7956          ← Kaggle 分数
incumbent: LightGBM          ← 最佳模型

Top 4 configurations:
  1. LightGBM    (0.7956) ✓
  2. XGBoost     (0.7823)
  3. XGBoost-v2  (0.7890)
  4. RandomForest (0.7701)
```

## ✅ 就这么简单！

**3 条命令即可开始**：

```bash
pip install mle-bench scikit-learn xgboost lightgbm
mlebench prepare --competition spaceship-titanic
bash examples/mle_automl_example/run_automl_search.sh
```

Happy AutoML! 🚀
