# TrainingEvolver → AutoML Framework 改造文档

## 🎯 改造目标

将 TrainingEvolver 从 **LLM 训练框架** 改造为 **AutoML 框架**，使其能够：
- 搜索传统 ML 模型的最佳超参数
- 直接在 MLE-Bench 任务上优化
- 不涉及任何 LLM 训练

## 📊 架构对比

### 原始架构（LLM 训练）

```
┌─────────────┐
│  Workspace  │  LLM 训练配置（LR, LoRA rank, data）
└──────┬──────┘
       │
   ┌───▼──────────────┐
   │ MCGS Algorithm   │  搜索 LLM 训练超参数
   └───┬──────────────┘
       │
   ┌───▼──────────────┐
   │ HF Trainer       │  训练 LLM + LoRA
   │ Backend          │  → 微调后的模型
   └───┬──────────────┘
       │
   ┌───▼──────────────┐
   │ vLLM Evaluation  │  推理 → 文本 → 评分
   │                  │
   └──────────────────┘
```

### 新架构（AutoML）

```
┌─────────────┐
│  Workspace  │  ML 模型配置（model_type, hyperparameters）
└──────┬──────┘
       │
   ┌───▼──────────────┐
   │ MCGS Algorithm   │  搜索 ML 超参数
   └───┬──────────────┘
       │
   ┌───▼──────────────┐
   │ Sklearn          │  训练 XGBoost/RandomForest/LightGBM
   │ Backend          │  → 训练好的 ML 模型
   └───┬──────────────┘
       │
   ┌───▼──────────────┐
   │ MLE-Bench        │  预测 → submission.csv → Kaggle 评分
   │ Evaluation       │
   └──────────────────┘
```

## 🔧 关键改动

### 1. Backend 层

**文件**: `agent_evolve/backends/sklearn_backend.py` (新增)

**核心功能**:
- `run_trial()`: 完整的训练+评估流程
- `_load_data()`: 加载 CSV 数据
- `_apply_feature_engineering()`: 特征工程
- `_train_model()`: 训练 sklearn/xgboost/lightgbm 模型
- `_save_model()`: 保存为 pickle
- `run_eval_plan()`: 预测 + 生成 submission.csv

**关键代码**:
```python
class SklearnBackend:
    def run_trial(self, workspace, node, budget, benchmark):
        # 1. 加载配置
        config = self._load_ml_config(workspace)
        
        # 2. 加载数据
        X_train, y_train, X_test = self._load_data(workspace, config)
        
        # 3. 训练模型
        model = self._train_model(config, X_train, y_train)
        
        # 4. 保存模型
        checkpoint = self._save_model(workspace, node.node_id, model, config)
        
        # 5. 评估
        result_dir = benchmark.evaluate(workspace, checkpoint, self, "test")
        
        return TrainingTrialResult(...)
```

### 2. Workspace 结构

**目录**: `seed_workspaces/mle_automl/`

**与原始 workspace 的差异**:

| 原始（LLM） | 新（AutoML） | 说明 |
|------------|-------------|------|
| `model/base.yaml` | `model/config.yaml` | 模型配置 |
| `model/adapter.yaml` (LoRA 配置) | `model/feature_engineering.yaml` | 特征工程 |
| `train/optimizer.yaml` (LR, betas) | - | 不需要（ML 模型无 optimizer） |
| `train/pipeline.yaml` (SFT, RL stages) | `train/ensemble.yaml` | 集成策略 |
| `data/sources.yaml` (训练数据) | `data/paths.yaml` | MLE-Bench 数据路径 |

**核心配置文件**:

```yaml
# model/config.yaml
model_type: xgboost
hyperparameters:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
competition_id: spaceship-titanic
target_column: Transported
```

### 3. Mutation 策略

**文件**: `agent_evolve/training/algorithms/mcgs/ml_mutation.py` (新增)

**三种 mutator**:

1. **MLModelTypeMutationProposer**: 轮换模型类型
   ```python
   model_types = ("xgboost", "lightgbm", "random_forest")
   # 每次选择下一个模型类型
   ```

2. **MLLearningRateSweepProposer**: 学习率扫描
   ```python
   learning_rates = (0.01, 0.05, 0.1, 0.2)
   # 类似 LRBagMutationProposer
   ```

3. **MLHyperparameterMutationProposer**: 随机超参数变异
   ```python
   # 随机改变 n_estimators, max_depth, subsample 等
   ```

### 4. Benchmark 适配

**文件**: `agent_evolve/benchmarks/mle_bench/mle_bench.py` (修改)

**新增功能**:
- `_grade_submission()`: 调用 `mlebench grade` 评分
- `_parse_mlebench_output()`: 解析评分结果

**工作流程**:
```python
def evaluate(workspace, checkpoint, backend, split):
    # 1. Backend 生成 submission.csv
    result_dir = backend.run_eval_plan(...)
    
    # 2. 如果是 sklearn backend，调用 mlebench grader
    if isinstance(backend, SklearnBackend):
        self._grade_submission(workspace, result_dir, split)
    
    return result_dir
```

### 5. 注册 Backend

**文件**: `agent_evolve/training/registries.py` (修改)

```python
TRAINING_BACKENDS = {
    "h200_single_node": "...",
    "sklearn_backend": "agent_evolve.backends.sklearn_backend.SklearnBackend",  # 新增
}
```

## 📋 使用流程

### 完整工作流程

```bash
# 1. 准备 MLE-Bench 数据
mlebench prepare --competition spaceship-titanic

# 2. 配置 workspace
vim seed_workspaces/mle_automl/model/config.yaml
# 设置: competition_id, target_column

# 3. 复制数据
cp ~/.cache/mlebench/competitions/spaceship-titanic/prepared/public/*.csv \
   seed_workspaces/mle_automl/data/

# 4. 运行 AutoML 搜索
python examples/mle_automl_example/drive_model_search_4cycle.py

# 5. 查看结果
cat runs/mle-automl-search/mle_automl/evolution/mcgs_graph.json
```

### MCGS 搜索过程（单个 cycle）

```python
# Cycle 0: 测试 XGBoost
config = {
    "model_type": "xgboost",
    "hyperparameters": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
}
→ 训练 → Kaggle 分数: 0.7823

# Cycle 1: 测试 LightGBM
config = {
    "model_type": "lightgbm",
    "hyperparameters": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
}
→ 训练 → Kaggle 分数: 0.7956 ← 最佳！

# Cycle 2: 测试 RandomForest
config = {
    "model_type": "random_forest",
    "hyperparameters": {"n_estimators": 100, "max_depth": 10}
}
→ 训练 → Kaggle 分数: 0.7701

# Cycle 3: 优化 XGBoost 超参数
config = {
    "model_type": "xgboost",
    "hyperparameters": {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.05}
}
→ 训练 → Kaggle 分数: 0.7890

# 最终: LightGBM 获胜，成为 incumbent
```

## 🎨 扩展性

### 添加新模型类型

```python
# 在 sklearn_backend.py 的 _train_model() 中添加

elif model_type == "catboost":
    import catboost
    model = catboost.CatBoostClassifier(**hyperparams)
    model.fit(X_train, y_train, verbose=False)
```

### 添加新特征工程策略

```python
# 在 feature_engineering.yaml 中定义
feature_engineering:
  pca: true
  n_components: 10
  
# 在 _apply_feature_engineering() 中实现
if fe_config.get("pca"):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=fe_config["n_components"])
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
```

### 添加集成学习

```python
# 在 train/ensemble.yaml 中配置
enabled: true
strategy: voting
models: [xgboost, random_forest, lightgbm]

# 在 Backend 中实现
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ("xgb", xgb_model),
    ("rf", rf_model),
    ("lgb", lgb_model),
], voting="soft")
ensemble.fit(X_train, y_train)
```

## 📊 性能对比

### vs 标准 AutoML 工具

| 工具 | 搜索策略 | 并行度 | 可扩展性 | 记忆机制 |
|------|---------|--------|---------|---------|
| **TrainingEvolver (AutoML)** | MCGS (tree search) | 中 | 高 | ✅ (graph + memory) |
| Auto-sklearn | Bayesian Optimization | 低 | 中 | ✅ (metalearning) |
| TPOT | Genetic Programming | 高 | 中 | ❌ |
| H2O AutoML | Random + Grid Search | 高 | 低 | ❌ |
| AutoGluon | Ensemble + Bagging | 高 | 中 | ❌ |

**优势**:
- ✅ MCGS 比随机搜索更高效
- ✅ Memory layer 避免重复失败
- ✅ Graph 结构保留搜索历史
- ✅ 可自定义 mutation 策略

**劣势**:
- ⚠️ 需要手动设计 mutation 策略
- ⚠️ 并行度不如 genetic programming
- ⚠️ 暂无 meta-learning

### vs MLEvolve

| 维度 | MLEvolve | TrainingEvolver (AutoML) |
|------|----------|-------------------------|
| **搜索空间** | 代码空间（无限） | 超参数空间（有限） |
| **评估速度** | 慢（需执行代码） | 快（直接训练） |
| **资源消耗** | 高（LLM API + 代码执行） | 低（仅 ML 训练） |
| **适用任务** | 所有类型 | 主要是表格数据 |
| **可解释性** | 低 | 高 |
| **上限** | 高 | 中 |

**使用建议**:
- **表格数据任务**: 优先用 TrainingEvolver (AutoML)
- **图像/NLP/复杂任务**: 考虑 MLEvolve
- **混合**: 先用 AutoML 建立 baseline，再用 MLEvolve 探索新方法

## 🚀 未来改进

### 1. 添加神经网络支持

```python
# 支持 PyTorch 模型
elif model_type == "neural_network":
    import torch.nn as nn
    model = nn.Sequential(...)
    # 训练逻辑
```

### 2. Meta-Learning

```python
# 从历史任务学习，快速初始化新任务
meta_memory = MetaMemory()
initial_config = meta_memory.suggest_config(new_task_features)
```

### 3. 多目标优化

```python
# 同时优化准确率和推理速度
reward = alpha * accuracy - beta * inference_time
```

### 4. 自动特征工程

```python
# MCGS 搜索特征工程策略
class FeatureEngineeringMutationProposer:
    def propose(self, parent, graph):
        # 变异特征变换、选择、创建策略
        ...
```

## 📚 代码清单

### 新增文件

```
agent_evolve/backends/sklearn_backend.py                     # Sklearn backend
agent_evolve/training/algorithms/mcgs/ml_mutation.py         # ML mutation 策略
seed_workspaces/mle_automl/                                  # AutoML workspace
├── manifest.yaml
├── model/
│   ├── config.yaml
│   └── feature_engineering.yaml
├── train/ensemble.yaml
├── data/paths.yaml
└── eval/competition.yaml
examples/mle_automl_example/                                 # 示例代码
├── drive_model_search_4cycle.py
├── run_automl_search.sh
└── README.md
```

### 修改文件

```
agent_evolve/training/registries.py                          # 注册 sklearn_backend
agent_evolve/benchmarks/mle_bench/mle_bench.py               # 添加 mlebench grading
```

## ✅ 验证清单

运行以下命令验证安装：

```bash
# 1. 检查 backend 注册
python -c "from agent_evolve.training.registries import resolve_backend; print(resolve_backend('sklearn_backend'))"

# 2. 检查 workspace 结构
ls seed_workspaces/mle_automl/model/config.yaml

# 3. 检查示例脚本
python examples/mle_automl_example/drive_model_search_4cycle.py --help

# 4. 运行 smoke test
# (需要先准备 MLE-Bench 数据)
```

## 🎓 总结

通过这次改造，我们成功地将 TrainingEvolver 从：

**LLM 训练框架** 
↓
**通用优化框架**
↓
**AutoML 框架**

**关键洞察**:
1. **Backend 是关键抽象**: 只需替换 Backend，框架其他部分（MCGS、Workspace、Benchmark）都可以复用
2. **MCGS 是通用搜索算法**: 不仅可以搜索 LLM 训练超参数，也可以搜索 ML 超参数
3. **Workspace 是配置 DNA**: 通过改变 workspace 结构，可以适配不同类型的任务

**适用场景**:
- ✅ 表格数据竞赛（Kaggle）
- ✅ 快速 baseline 建立
- ✅ 超参数优化
- ✅ 模型选择

**下一步**: 在更多 MLE-Bench 任务上测试，积累经验，持续改进搜索策略！
