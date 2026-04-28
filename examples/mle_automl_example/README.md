# TrainingEvolver as AutoML Framework for MLE-Bench

## 🎯 概述

这个示例展示了如何将 **TrainingEvolver 作为 AutoML 框架**使用，通过 MCGS 搜索找到传统 ML 模型在 MLE-Bench 任务上的最佳配置。

### 与标准用法的区别

| 标准 TrainingEvolver | AutoML TrainingEvolver |
|---------------------|----------------------|
| 训练 LLM（调整 LR、LoRA rank） | 训练 ML 模型（调整超参数） |
| Backend: HF Trainer + PEFT | Backend: Sklearn/XGBoost |
| 评估: vLLM 推理 | 评估: 预测 + MLE-Bench 评分 |
| 数据: {prompt, completion} | 数据: 表格/图像/文本（竞赛数据） |

## 🔧 工作原理

每个 MCGS cycle：

```
1. SELECT: 选择父配置（model_type + hyperparameters）
2. MUTATE: 变异配置
   - 切换模型类型（XGBoost → RandomForest）
   - 调整超参数（learning_rate, n_estimators, max_depth）
   - 修改特征工程（scaling, encoding）
3. TRAIN: 训练 ML 模型
   - 加载训练数据（train.csv）
   - 特征工程
   - 训练模型（sklearn/xgboost/lightgbm）
   - 保存模型（pickle）
4. EVALUATE: 在测试集上评估
   - 加载测试数据（test.csv）
   - 生成预测
   - 保存 submission.csv
   - 用 mlebench grade 评分
5. REWARD: MCGS 计算 reward
   - 基于 Kaggle 分数
   - 考虑训练时间
6. PROMOTE: 更新 incumbent
```

## 📦 安装依赖

```bash
# 基础依赖
pip install scikit-learn xgboost lightgbm pandas numpy pyyaml

# MLE-Bench
pip install mle-bench

# 可选：更快的模型
pip install catboost
```

## 🚀 快速开始

### 步骤 1: 准备 MLE-Bench 数据

```bash
# 选择一个竞赛
COMPETITION_ID="spaceship-titanic"

# 准备数据（下载并拆分）
mlebench prepare --competition $COMPETITION_ID

# 查看数据位置
ls ~/.cache/mlebench/competitions/$COMPETITION_ID/prepared/public/
# 应该看到: train.csv, test.csv, sample_submission.csv
```

### 步骤 2: 配置 Workspace

```bash
cd /home/ec2-user/fsx/yisi/A-EVOLVE-V2

# 更新竞赛 ID
vim seed_workspaces/mle_automl/model/config.yaml
# 修改: competition_id: spaceship-titanic

# 复制数据到 workspace
cp ~/.cache/mlebench/competitions/$COMPETITION_ID/prepared/public/*.csv \
   seed_workspaces/mle_automl/data/

# 更新目标列名（根据竞赛调整）
# 对于 spaceship-titanic: target_column: Transported
vim seed_workspaces/mle_automl/model/config.yaml
```

### 步骤 3: 运行 AutoML 搜索

```bash
# 方式 1: 使用脚本（推荐）
bash examples/mle_automl_example/run_automl_search.sh

# 方式 2: 直接运行
python examples/mle_automl_example/drive_model_search_4cycle.py

# 监控进度
tail -f runs/mle-automl-search/logs/run.log
```

### 步骤 4: 查看结果

```bash
# 查看 MCGS 图
cat runs/mle-automl-search/mle_automl/evolution/mcgs_graph.json | jq

# 查看每个 cycle 的报告
cat runs/mle-automl-search/mle_automl/evolution/reports/cycle_0000.json | jq

# 查看最佳配置
cat runs/mle-automl-search/mle_automl/evolution/incumbent/model/config.yaml

# 查看最佳模型
ls runs/mle-automl-search/nodes/node-*/workspace/checkpoints/models/
```

## 📊 示例输出

运行完成后，你会看到类似这样的输出：

```
=== Final Result ===
cycles_completed: 4
incumbent_node_id: node-e417bd2b8b
best_metric: 0.7956

topk results:
  topk: node=node-e417bd2b8b branch=1 metric=0.7956 reward=0.7850
  topk: node=node-34674e10b7 branch=0 metric=0.7823 reward=0.7720
  topk: node=node-4ca27d34f7 branch=2 metric=0.7701 reward=0.7598
  topk: node=node-e41245c00c branch=3 metric=0.7434 reward=0.7330

Best configuration: LightGBM with learning_rate=0.05, n_estimators=150
```

## 🎨 自定义搜索策略

### 1. 改变搜索的模型类型

编辑 `drive_model_search_4cycle.py`:

```python
algo = MCGSSearch(
    mutator=MLModelTypeMutationProposer(
        model_types=("xgboost", "catboost", "lightgbm")  # 改为你想要的模型
    ),
    selector=RootFanoutSelector(fanout=3),  # 调整 fanout
)
```

### 2. 搜索学习率

```python
from agent_evolve.training.algorithms.mcgs.ml_mutation import MLLearningRateSweepProposer

algo = MCGSSearch(
    mutator=MLLearningRateSweepProposer(
        learning_rates=(0.001, 0.01, 0.05, 0.1, 0.2)
    ),
)
```

### 3. 随机超参数搜索

```python
from agent_evolve.training.algorithms.mcgs.ml_mutation import MLHyperparameterMutationProposer

algo = MCGSSearch(
    mutator=MLHyperparameterMutationProposer(mutation_rate=0.3),
)
```

### 4. 多种 mutation 策略组合

```python
class CombinedMutationProposer:
    def __init__(self):
        self.model_mutator = MLModelTypeMutationProposer()
        self.hyperparam_mutator = MLHyperparameterMutationProposer()
        self.counter = 0
    
    def propose(self, parent, graph):
        # 前 3 个 cycle 尝试不同模型，后续优化超参数
        if self.counter < 3:
            mutation = self.model_mutator.propose(parent, graph)
        else:
            mutation = self.hyperparam_mutator.propose(parent, graph)
        self.counter += 1
        return mutation

algo = MCGSSearch(mutator=CombinedMutationProposer())
```

## 🔍 调试技巧

### 查看训练日志

```bash
# 每个 node 的训练细节
cat runs/mle-automl-search/nodes/node-*/workspace/evolution/eval/ml_training/test/metrics.json
```

### 查看生成的 submission

```bash
# 每个 node 的预测
cat runs/mle-automl-search/nodes/node-*/workspace/evolution/eval/ml_training/test/submission.csv
```

### 手动评分一个 submission

```bash
mlebench grade \
    --competition spaceship-titanic \
    --submission runs/mle-automl-search/nodes/node-xxx/workspace/evolution/eval/ml_training/test/submission.csv
```

## 📈 扩展到多个竞赛

### 批量运行

```python
# examples/mle_automl_example/batch_run.py

competitions = [
    "spaceship-titanic",
    "house-prices-advanced-regression-techniques",
    "digit-recognizer",
]

for comp_id in competitions:
    # 1. 准备数据
    subprocess.run(["mlebench", "prepare", "--competition", comp_id])
    
    # 2. 更新 workspace
    update_workspace_config(comp_id)
    
    # 3. 运行 AutoML
    evolver = TrainingEvolver(
        workspace=f"seed_workspaces/mle_automl_{comp_id}",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
    )
    result = evolver.run(cycles=10)
    
    # 4. 保存结果
    save_results(comp_id, result)
```

## 🎓 进阶用法

### 1. 添加特征工程搜索

在 `model/feature_engineering.yaml` 中定义可变异的特征工程策略：

```yaml
fillna: mean
scale: true
polynomial_features: true
degree: 2
```

然后创建 mutation 策略变异这些配置。

### 2. 集成 Ensemble 策略

```yaml
# train/ensemble.yaml
enabled: true
strategy: voting
models:
  - xgboost
  - random_forest
  - lightgbm
weights: [0.4, 0.3, 0.3]
```

### 3. 添加 CV 评估

修改 `SklearnBackend._train_model()`:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
cv_score = scores.mean()
# 使用 CV 分数作为中间反馈
```

## 📝 与 MLEvolve 的对比

| 特性 | MLEvolve | 这个 AutoML 框架 |
|------|----------|----------------|
| **方法** | Agent 系统（LLM 生成代码） | 直接搜索 ML 超参数 |
| **搜索空间** | 代码空间（无限） | 超参数空间（有限） |
| **评估时间** | 长（需要执行代码） | 快（直接训练模型） |
| **可解释性** | 低（代码复杂） | 高（超参数清晰） |
| **上限** | 高（可以做任何事） | 中（受限于模型类型） |
| **成本** | 高（大量 LLM 调用） | 低（无 LLM 调用） |
| **适用场景** | 复杂任务、新颖方法 | 标准表格任务 |

**结论**: 
- 如果任务是**标准表格数据**，用这个 AutoML 框架更快更便宜
- 如果任务需要**创新性方法**（如 CV、NLP），用 MLEvolve 更有潜力

## 🚀 性能优化

### 并行评估

```python
config = TrainingEvolveConfig(
    max_cycles=10,
    parallel_trials=4,  # 并行运行 4 个 trial
)
```

### 早停

```python
# 在 SklearnBackend 中添加早停
model = XGBClassifier(
    **hyperparams,
    early_stopping_rounds=10,
    eval_set=[(X_val, y_val)],
)
```

### 缓存特征工程

```python
# 缓存已处理的特征，避免重复计算
feature_cache = {}

def _apply_feature_engineering(X_train, X_test, config):
    cache_key = hash(frozenset(config.items()))
    if cache_key in feature_cache:
        return feature_cache[cache_key]
    # ... 处理特征
    feature_cache[cache_key] = (X_train_processed, X_test_processed)
    return feature_cache[cache_key]
```

## 📚 参考资料

- **MLE-Bench**: https://github.com/openai/mle-bench
- **TRAINDESIGN.md**: ../../TRAINDESIGN.md
- **Sklearn**: https://scikit-learn.org
- **XGBoost**: https://xgboost.readthedocs.io
- **LightGBM**: https://lightgbm.readthedocs.io

## 🐛 常见问题

### Q: 报错 "Unknown backend: sklearn_backend"

A: 确保 `agent_evolve/training/registries.py` 中已注册 sklearn_backend

### Q: MLE-Bench 评分失败

A: 
1. 检查 submission.csv 格式是否正确
2. 检查 competition_id 是否匹配
3. 手动运行 `mlebench grade` 查看详细错误

### Q: 模型训练时间太长

A: 
- 减少 `n_estimators`
- 减少训练数据量（采样）
- 使用更简单的模型（logistic_regression）

### Q: 想搜索图像/NLP 任务

A: 当前框架主要针对表格数据。对于图像/NLP：
- 需要修改 `_load_data()` 加载图像/文本
- 需要添加相应的模型（CNN、Transformer）
- 考虑使用标准 TrainingEvolver（训练 LLM）

## ✅ 总结

你现在可以：

1. ✅ 用 TrainingEvolver 作为 AutoML 框架
2. ✅ 在 MLE-Bench 任务上搜索最佳 ML 配置
3. ✅ 通过 MCGS 自动优化超参数
4. ✅ 不需要任何 LLM，直接训练传统 ML 模型

**下一步**：在更多 MLE-Bench 竞赛上运行，积累经验，调优搜索策略！
