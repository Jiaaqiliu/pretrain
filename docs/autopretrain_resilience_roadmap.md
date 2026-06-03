# AutoPretrain: World-Class Autonomous Training Framework
## 设计文档 & 实施路线图

**愿景**：从零打造一个工业级自主训练编排框架，能够持续自主运行数天至数周，自主诊断所有故障，零Bug长期稳定运行。

**设计哲学**：推倒重来，不受旧代码约束。取各家之长，超越现有开源实现。

---

## 一、调研结论：顶级开源项目的核心机制

### 1.1 Ray Train（ray-project/ray）
| 机制 | 实现方式 |
|------|----------|
| 故障检测 | Worker heartbeat + health check timeout + `WorkerGroupPollStatus` |
| 自动恢复 | `FailurePolicy.make_decision()` → RETRY/RAISE/NOOP |
| 状态机 | `INITIALIZING → SCHEDULING → RUNNING → RESTARTING → FINISHED/ERRORED` |
| Worker/Controller分离 | worker_group_failures 和 controller_failures 独立计数 |
| 恢复模式 | Gang restart: 任何worker死 → 全组重启 → 从latest checkpoint resume |

**关键借鉴**：正式状态机 + FailurePolicy抽象 + 分离计数

### 1.2 Determined AI（determined-ai/determined）
| 机制 | 实现方式 |
|------|----------|
| Heartbeat | Idle Watcher (5s轮询) + 可配置timeout |
| 日志诊断 | Log Pattern regex → 匹配CUDA OOM等可触发EXCLUDE_NODE |
| 问题节点 | `GetBlockedNodes()` — 自动排除曾导致故障的节点 |
| 非可重试错误 | 明确列表（如sbatch failed）直接跳过重试 |
| 状态持久化 | PostgreSQL存储restarts + runID |

**关键借鉴**：问题节点黑名单 + 非可重试错误列表 + 动态Idle Watcher

### 1.3 MosaicML Composer
| 机制 | 实现方式 |
|------|----------|
| NaN检测 | `NaNMonitor` — fast-fail (RuntimeError) |
| OOM诊断 | `OOMObserver` — memory snapshot + flamegraph |
| Early Stop | `ThresholdStopper` — metric-based |
| 设计哲学 | **rich observability + fast-fail → 外部orchestrator负责恢复** |

**关键借鉴**：训练脚本只检测+快速失败，恢复交给外部Agent（分层设计）

### 1.4 torchtitan（Meta/PyTorch）
| 机制 | 实现方式 |
|------|----------|
| Elastic | `torchrun` elastic agent — node死亡后全组重启 |
| Timeout | 首次成功step后收紧timeout → 更快检测hang |
| DCP | 3种async模式（sync, gloo async, process-based + pinned mem） |
| 极简 | 只做checkpoint + elastic launcher，复杂逻辑在外部 |

**关键借鉴**：动态timeout收紧 + DCP async checkpoint

### 1.5 DeepSpeed（Microsoft）
| 机制 | 实现方式 |
|------|----------|
| Elastic | `DSElasticAgent` + heartbeat liveness |
| Batch调整 | GPU数变化时自动计算有效batch size |
| 事件分类 | Scaling event ≠ Failure event（不消耗restart budget） |
| Universal Ckpt | 任意并行度恢复（topology reshaping） |

**关键借鉴**：区分scaling/failure事件 + topology-agnostic checkpoint

### 1.6 Kubeflow + Volcano
| 机制 | 实现方式 |
|------|----------|
| Reconcile | retrieve → clear stale → reconcile runtime → sync status → check deadline |
| Deadline | 从创建时间计算过期 → 超期mark failed + delete |
| Event-Action | Volcano的task-level event-action映射 (PodEvicted → RestartJob) |
| Gang | `minAvailable` 确保所有worker同时调度 |

**关键借鉴**：Deadline enforcement + Event-Action mapping

### 1.7 NVIDIA NeMo Resiliency
| 机制 | 实现方式 |
|------|----------|
| Straggler | GPU性能分数 (relative 0.7 + individual 0.7 阈值) |
| Fault Tolerance | rank heartbeat + 动态timeout计算 + safety_factor |
| 抢占 | SIGTERM → rank-0 `_interrupted` → broadcast → graceful save |
| Auto-Resume | `exp_manager` 多层checkpoint发现 |

**关键借鉴**：Straggler detection阈值 + 动态timeout + NeMo的exp_manager模式

### 1.8 OLMo-core（我们的训练引擎）
| 机制 | 实现方式 |
|------|----------|
| StabilityMonitor | rolling spike detection (6σ), checkpoint-able state |
| SkipStepOptimizer | loss/grad spike → step_factor=0.0, 无host-device sync |
| Checkpoint原子性 | metadata文件最后写入 → partial write不被加载 |
| Ephemeral Ckpt | 高频临时ckpt，只保留最新1个 |
| Async Ckpt | 独立CPU process group + thread pool |
| Signal | SIGTERM/SIGINT → cancel_run() → graceful shutdown |

**关键借鉴**：已有强大内置防护，Agent只需补齐外部编排能力

### 1.9 AutoTrain（我们之前的框架）
| 机制 | 实现方式 |
|------|----------|
| Protocol接口 | ComputeBackend / TrainingBackend / MetricStore / JobManager |
| Orchestrator | async event loop + phase management + budget control |
| Monitor | 6种统计异常检测（z-score based） |
| Action系统 | confidence-based auto-execution（>0.7才执行） |
| 分层架构 | Substrate → Execution → Brain |

**关键借鉴**：Protocol设计 + Orchestrator模式 + Monitor异常检测 + 可以大量复用

---

## 二、架构设计决策

### 核心设计原则（综合所有调研）

| # | 原则 | 来源 |
|---|------|------|
| 1 | **训练脚本fast-fail + Agent恢复** | Composer, torchtitan |
| 2 | **正式状态机** | Ray Train |
| 3 | **Heartbeat + 动态timeout** | NeMo, Determined |
| 4 | **Scaling ≠ Failure** | DeepSpeed |
| 5 | **Atomic checkpoint** | OLMo-core |
| 6 | **问题节点黑名单** | Determined |
| 7 | **Event-Action映射** | Volcano |
| 8 | **Deadline enforcement** | Kubeflow |
| 9 | **All-or-nothing restart** | TorchElastic |
| 10 | **Confidence-based auto-action** | AutoTrain |
| 11 | **Protocol接口可插拔** | AutoTrain |
| 12 | **Complete audit trail** | All |

---

## 三、新框架架构（推倒重来）

### 3.1 总体分层

```
autopretrain/
├── core/                    # 核心类型和接口协议
│   ├── types.py             # JobState, Anomaly, Action, MetricSnapshot等
│   ├── protocols.py         # ComputeBackend, TrainingEngine等Protocol
│   └── config.py            # 全局配置
│
├── engine/                  # 训练引擎适配层（对接OLMo-core）
│   ├── olmo_adapter.py      # OLMo-core训练脚本生成 + metrics读取
│   ├── script_templates/    # Jinja2模板：生成训练脚本
│   └── callbacks.py         # 注入训练脚本的自定义callback（heartbeat, metric report）
│
├── compute/                 # 计算后端（K8s, SLURM, local）
│   ├── base.py              # ComputeBackend ABC
│   ├── kubernetes.py        # K8s Job/PyTorchJob 管理
│   ├── slurm.py             # SLURM sbatch/squeue
│   └── local.py             # 本地torchrun（开发用）
│
├── orchestrator/            # 核心编排引擎（大脑）
│   ├── state_machine.py     # 正式Job状态机（FSM）
│   ├── orchestrator.py      # 主event loop
│   ├── scheduler.py         # 多trial调度（优先级、并发控制）
│   └── budget.py            # GPU-hour预算管理
│
├── resilience/              # 容错与自愈层
│   ├── diagnoser.py         # 多层故障诊断（log regex + K8s events + correlations）
│   ├── recovery.py          # 恢复策略选择（per-failure-type + adaptive）
│   ├── heartbeat.py         # Heartbeat liveness probe（FSx文件协议）
│   ├── checkpoint_manager.py # Checkpoint发现、验证、恢复
│   ├── node_exclusion.py    # 问题节点黑名单
│   └── deadline.py          # Deadline enforcement
│
├── monitor/                 # 实时监控与异常检测
│   ├── metrics_collector.py # 从FSx/logs收集实时metrics
│   ├── anomaly_detector.py  # 统计异常检测（loss spike, divergence, throughput drop等）
│   ├── throughput_monitor.py # 吞吐量baseline + straggler检测
│   └── health_checker.py    # 综合健康评估（heartbeat + metrics + K8s status）
│
├── search/                  # 数据配比搜索（MCGS核心算法）
│   ├── mixture.py           # DataMixture（概率单纯形）
│   ├── mutator.py           # 6种mutation策略
│   ├── mcgs.py              # Monte Carlo Graph Search
│   ├── reward.py            # 奖励计算
│   └── transfer.py          # Proxy→Full scale transfer验证
│
├── eval/                    # 评估系统
│   ├── harness.py           # 快速eval（ARC-Easy, PIQA等）
│   ├── downstream.py        # 下游任务eval（Kaggle/reasoning）
│   └── comparator.py        # 多trial结果对比分析
│
├── store/                   # 持久化存储
│   ├── event_log.py         # Event/Decision audit trail (JSONL)
│   ├── metric_store.py      # 时序指标存储
│   └── experiment_db.py     # 实验历史（SQLite/JSON）
│
├── notify/                  # 通知系统
│   ├── base.py              # Notifier接口
│   ├── slack.py             # Slack webhook
│   └── console.py           # Console logging
│
└── cli.py                   # 命令行入口
```

### 3.2 核心状态机设计

```
                    ┌──────────────┐
                    │   CREATED    │
                    └──────┬───────┘
                           │ submit()
                           ▼
                    ┌──────────────┐
              ┌────→│   PENDING    │
              │     └──────┬───────┘
              │            │ K8s schedules pod
              │            ▼
              │     ┌──────────────┐
              │     │   RUNNING    │←────────────────┐
              │     └──┬───┬───┬───┘                 │
              │        │   │   │                     │
              │   done │   │   │ failure             │ retry
              │        │   │   │                     │
              │        ▼   │   ▼                     │
              │ ┌─────────┐│ ┌──────────────┐       │
              │ │COMPLETED││ │  DIAGNOSING  │       │
              │ └─────────┘│ └──────┬───────┘       │
              │            │        │               │
              │    timeout │        │ diagnosed     │
              │            │        ▼               │
              │            │ ┌──────────────┐       │
              │            │ │  RECOVERING  │───────┘
              │            │ └──────┬───────┘
              │            │        │
              │            ▼        │ non-retryable / max retries
              │     ┌──────────┐    │
              │     │ TIMED_OUT│    ▼
              │     └──────────┘ ┌──────────────┐
              │                  │   TERMINAL   │
              └──────────────────│  (FAILED /   │
                 (wait+retry)    │  ALERT_HUMAN)│
                                 └──────────────┘
```

**状态转换规则**：
- `PENDING → RUNNING`: K8s pod进入Running phase
- `RUNNING → COMPLETED`: Job条件为Complete
- `RUNNING → DIAGNOSING`: Job失败 OR heartbeat超时 OR deadline超时
- `DIAGNOSING → RECOVERING`: 故障已分类，恢复策略已选择
- `RECOVERING → PENDING`: 可重试故障，重新提交
- `RECOVERING → TERMINAL`: 不可重试 OR 超最大重试次数
- `RUNNING → TIMED_OUT`: 超deadline
- `TIMED_OUT → PENDING`: retry after timeout

### 3.3 Heartbeat协议（FSx文件）

训练脚本内置callback每30秒写入：
```json
// /fsx/.../heartbeat/{trial_id}.json
{
  "timestamp": 1748180000.0,
  "step": 2341,
  "loss": 2.847,
  "grad_norm": 0.42,
  "throughput_tps": 125000,
  "gpu_memory_pct": 73.2,
  "status": "training"
}
```

Agent检测规则：
- 2分钟无更新 → WARNING（log但不行动）
- 4分钟无更新 → STALE（发送通知）
- 6分钟无更新 → DEAD（触发DIAGNOSING状态）

### 3.4 Event-Action映射表

| Event | Condition | Action | Confidence |
|-------|-----------|--------|------------|
| NETWORK_DISCONNECT | log regex match | wait 30s → resubmit | 0.95 |
| OOM | CUDA OOM in logs | halve microbatch → resubmit | 0.9 |
| OOM (2nd) | already halved | enable grad_ckpt → resubmit | 0.85 |
| NCCL_TIMEOUT | NCCL timeout/watchdog | resubmit (same config) | 0.9 |
| SILENT_HANG | heartbeat dead 6min | force kill → resubmit | 0.85 |
| LOSS_DIVERGENCE | normalized_slope > 0.01 | rollback 1000 steps + reduce lr | 0.8 |
| NAN_DETECTED | NaN in heartbeat | rollback + reduce lr 50% | 0.9 |
| DATA_ERROR | FileNotFound / path error | **alert human** (non-retryable) | 1.0 |
| CODE_BUG | ImportError/SyntaxError | **alert human** (non-retryable) | 1.0 |
| PREEMPTION | node NotReady/Evicted | wait 60s → resubmit | 0.95 |
| STRAGGLER | throughput < 50% baseline | wait 5min → if persists, resubmit | 0.7 |
| NODE_REPEATED_FAIL | same node fails 2+ times | exclude node → resubmit | 0.9 |
| DEADLINE_EXCEEDED | runtime > max_duration | force stop → partial results | 0.95 |
| BUDGET_EXHAUSTED | GPU-hours > limit | graceful stop all trials | 1.0 |

---

## 四、实施计划（推倒重来）

### Phase 0: 核心骨架（Day 1）
**目标**：建立新的package结构，核心types和protocols

- [ ] 创建 `autopretrain/` package结构
- [ ] 定义 `core/types.py` — 所有核心数据类型
- [ ] 定义 `core/protocols.py` — 所有接口Protocol
- [ ] 迁移AutoTrain的MonitorConfig + 异常检测逻辑

### Phase 1: 状态机 + K8s后端（Day 1-2）
**目标**：能提交Job并正确跟踪状态

- [ ] `orchestrator/state_machine.py` — 正式FSM实现
- [ ] `compute/kubernetes.py` — K8s Job CRUD (从现有K8sClient重构)
- [ ] `store/event_log.py` — JSONL event logging
- [ ] `orchestrator/orchestrator.py` — 主event loop (async)

### Phase 2: Heartbeat + 故障诊断（Day 2-3）
**目标**：能检测所有已知故障类型并正确分类

- [ ] `engine/callbacks.py` — HeartbeatCallback (写入FSx)
- [ ] `resilience/heartbeat.py` — Heartbeat reader + timeout detection
- [ ] `resilience/diagnoser.py` — 多层诊断(regex + events + correlations)
- [ ] `resilience/recovery.py` — 恢复策略选择（event-action表）
- [ ] `resilience/node_exclusion.py` — 问题节点黑名单

### Phase 3: 实时监控 + 异常检测（Day 3-4）
**目标**：实时感知训练状态，主动发现问题

- [ ] `monitor/metrics_collector.py` — 从heartbeat文件收集metrics
- [ ] `monitor/anomaly_detector.py` — 6种统计异常检测
- [ ] `monitor/throughput_monitor.py` — throughput baseline + straggler
- [ ] `monitor/health_checker.py` — 综合健康评估
- [ ] `resilience/deadline.py` — Deadline enforcement

### Phase 4: 搜索算法集成（Day 4-5）
**目标**：MCGS搜索能通过新框架运行

- [ ] `search/` — 从现有代码迁移MCGS + mixture + mutator
- [ ] `engine/olmo_adapter.py` — 训练脚本生成（支持data mix参数化）
- [ ] `eval/harness.py` — 训练完成后自动eval
- [ ] `orchestrator/scheduler.py` — 多trial并发调度

### Phase 5: 高级特性（Day 5+）
- [ ] `notify/slack.py` — Slack通知
- [ ] `orchestrator/budget.py` — GPU-hour预算管理
- [ ] `store/experiment_db.py` — 实验历史分析
- [ ] Predictive failure detection
- [ ] LLM-guided diagnosis for UNKNOWN failures

---

## 五、KPI定义

| 指标 | 当前水平 | Phase 2完成 | 最终目标 |
|------|----------|-------------|----------|
| 自动恢复率 | ~50% | >85% | >95% |
| MTTR（恢复时间） | ~10min | <3min | <1min |
| 无人值守时间 | ~2h | >12h | >7天 |
| 故障分类准确率 | ~70% | >85% | >95% |
| GPU利用率 | ~70% | >85% | >95% |
| 虚假告警率 | N/A | <10% | <2% |

---

## 六、与现有代码的关系

### 保留 & 迁移
- `search/` ← `agent_evolve/model/algorithms/autopretrain/` (MCGS算法核心)
- `monitor/anomaly_detector.py` ← AutoTrain `agent/monitor.py` (统计检测逻辑)
- `core/types.py` ← AutoTrain `core/types.py` (type定义扩展)
- `compute/kubernetes.py` ← 现有 `K8sClient` (重构为async)
- 训练脚本模板 ← `mvp_3trial.py` (作为参考)

### 推倒重来
- `ExperimentAgent` — 用正式状态机 + async orchestrator替代
- `experiment_agent.py` 的 if/else 恢复逻辑 — 用event-action映射替代
- `script_generator.py` — 用Jinja2模板替代硬编码字符串
- A-EVOLVE-V2 的 backend/protocol 层 — 过度设计，简化

### 全新设计
- Heartbeat协议（FSx文件）
- 动态timeout机制
- 问题节点黑名单
- Deadline enforcement
- Budget management
- Audit trail system

---

## 七、自主进化能力（Self-Evolving Agent）

### 7.1 设计理念

框架不仅要能自愈已知问题，还要能**自主学习处理未知问题**。当遇到从未见过的错误时：

1. **SelfDiagnoser** 提取错误签名（去除时间戳/步数等变量）
2. 在"已学习修复"数据库中查找匹配
3. 如无匹配 → 运行启发式分析（语义理解错误类型）
4. 提出修复方案 → 安全验证（不能修改不安全字段）
5. 执行修复 → 观察结果
6. 如果成功 → **将新模式加入数据库**（下次直接匹配，无需重新分析）

这使框架具有"越用越聪明"的特性：
- 第一次遇到新错误：需要几分钟分析
- 第二次遇到相同错误：秒级匹配+修复
- 长期运行后：能处理99%的问题类型

### 7.2 已实现的模块

```
autopretrain/resilience/self_diagnoser.py
├── SelfDiagnoser class
│   ├── analyze_unknown_failure() — 分析未知故障
│   ├── validate_fix() — 安全验证修复方案
│   ├── learn_from_outcome() — 从结果学习
│   ├── _extract_signature() — 提取错误指纹
│   ├── _check_learned_fixes() — 查询知识库
│   └── _heuristic_analysis() — 启发式诊断
├── LearnedFix dataclass — 学习到的修复模式
└── Knowledge persistence — JSON文件持久化
```

### 7.3 未来扩展：LLM-Guided Diagnosis

对于启发式分析也无法解决的问题，可接入LLM：
```python
async def llm_analyze(self, logs: str, config: TrialConfig) -> RecoveryAction:
    prompt = f"""
    Training failed with these logs:
    {logs[-3000:]}

    Current config: {config}

    What is the most likely root cause? What config change would fix it?
    Respond with JSON: {{"diagnosis": "...", "fix": {{"field": "value"}}}}
    """
    response = await self.llm.complete(prompt)
    # Parse and validate response
    ...
```

---

## 八、多规模模型支持（Multi-Scale）

### 8.1 支持的模型规模

| 模型 | 参数量 | Factory | GPU需求 | Microbatch | 预计速度 |
|------|--------|---------|---------|------------|----------|
| OLMo2-190M | 190M | `olmo2_190M` | 1× H200 | 16 seq | ~500K tok/s |
| OLMo2-370M | 370M | `olmo2_370M` | 1× H200 | 16 seq | ~300K tok/s |
| OLMo2-760M | 760M | `olmo2_760M` | 2× H200 | 8 seq | ~200K tok/s |
| OLMo2-1B | 1.6B | `olmo2_1B` | 8× H200 | 4 seq | ~125K tok/s |
| OLMo2-3B | 3.3B | `olmo2_3B` | 8× H200 | 2 seq | ~60K tok/s |
| OLMo2-7B | 7B | `olmo2_7B` | 8× H200 | 1 seq | ~30K tok/s |
| OLMo2-13B | 13B | `olmo2_13B` | 16× H200 (2 nodes) | 1 seq | ~15K tok/s |
| OLMo2-32B | 32B | `olmo2_32B` | 32× H200 (4 nodes) | 1 seq | ~5K tok/s |

### 8.2 自动Scale配置

`OLMoAdapter` 会根据 `model_factory` 自动选择：
- GPU数量和内存请求
- 默认microbatch size
- FSDP分片策略
- 是否需要多节点
- 是否需要gradient checkpointing

### 8.3 Model Ladder（Proxy Transfer）

小模型实验→大模型验证的工作流：
1. 在190M/1B上快速搜索最优配置（<1小时/trial）
2. Top-3配置在3B上验证（~3小时/trial）
3. 最终winner在7B+上全量训练

这个工作流已在 `search/transfer.py` 中规划。

---

## 九、全参数搜索空间（不仅是数据配比）

### 9.1 人类顶级训练者会关注的所有变量

| 类别 | 参数 | 典型范围 | 影响 |
|------|------|----------|------|
| **数据配比** | web/code/math/academic比例 | 各0-100% | 决定模型能力分布 |
| **学习率** | peak lr | 1e-4 ~ 6e-3 | 过高→diverge, 过低→收敛慢 |
| **Warmup** | warmup steps | 100 ~ 5000 | 过短→初期不稳定 |
| **LR Schedule** | cosine/WSD/linear | - | WSD最新最优 |
| **Weight Decay** | weight_decay | 0.01 ~ 0.3 | 过大→underfitting |
| **Batch Size** | global tokens/step | 256K ~ 4M | 大batch→更稳定但lr需调整 |
| **Sequence Length** | context length | 2048 ~ 8192 | 影响内存和长文能力 |
| **Gradient Clip** | max_grad_norm | 0.5 ~ 2.0 | 过松→spike风险 |
| **Adam Betas** | β1, β2 | (0.9, 0.95)~(0.9, 0.99) | β2影响spike恢复速度 |
| **Z-loss** | z_loss_multiplier | 0 ~ 1e-4 | 防止logit divergence |
| **Dropout** | dropout rate | 0 ~ 0.1 | 小模型通常不用 |
| **Data Quality** | quality threshold | 0 ~ 1.0 | 过滤低质量数据 |
| **Repetition** | max_epoch per domain | 1 ~ 4 | Muennighoff上限 |
| **Curriculum** | 分阶段数据配比 | 2-3阶段 | 如Llama的pretrain→midtrain |

### 9.2 训练过程中的关键监控指标

| 指标 | 含义 | 异常阈值 | 应对策略 |
|------|------|----------|----------|
| CE Loss | 交叉熵损失 | z-score > 4σ | SkipStep / 降lr |
| Gradient Norm | 梯度范数 | > 100 或 z > 4σ | Clip / SkipStep |
| Throughput | tokens/sec | < 50% baseline | 检查straggler/数据瓶颈 |
| GPU Memory | CUDA内存占用 | > 90% | 预警OOM风险 |
| Learning Rate | 当前lr | 异常schedule | 检查scheduler配置 |
| Loss Variance | loss方差 | 持续增大 | 训练不稳定信号 |
| Spike Rate | spike频率 | > 5% | 数据质量/lr问题 |
| Tokens Seen | 已消耗tokens | vs budget | 进度跟踪 |
| Perplexity | exp(loss) | 对比baseline | 模型质量指标 |
| Val Loss | 验证集loss | > train loss 20%+ | 过拟合风险 |

### 9.3 未来支持的搜索维度

Phase 1（当前）：
- [x] 数据配比搜索（4维度: web/code/math/academic）

Phase 2（下一步）：
- [ ] 学习率搜索（log-uniform采样）
- [ ] Batch size搜索（powers of 2）
- [ ] Warmup比例搜索

Phase 3（进阶）：
- [ ] 完整超参数联合搜索（Bayesian + MCGS）
- [ ] Curriculum schedule搜索（多阶段配比转换点）
- [ ] 数据质量阈值搜索（Bitter Lesson axis）
- [ ] Scaling law拟合 + 预测最优配置

---

## 十、与顶级开源实现的关系

### 10.1 直接复用（站在巨人肩膀上）

| 能力 | 使用的开源实现 | 我们的封装 |
|------|---------------|-----------|
| 训练引擎 | OLMo-core Trainer + callbacks | `engine/olmo_adapter.py` |
| 梯度监控 | OLMo-core `StabilityMonitorCallback` | 直接使用，不重写 |
| Loss Spike跳过 | OLMo-core `SkipStepAdamW` | 直接使用 |
| GPU内存监控 | OLMo-core `GPUMemoryMonitorCallback` | 直接使用 |
| Checkpoint原子性 | OLMo-core `Checkpointer` (metadata-last) | 直接使用 |
| Async Checkpoint | OLMo-core async save (CPU process group) | 直接使用 |
| Signal处理 | OLMo-core SIGTERM/SIGINT handler | 直接使用 |
| K8s调度 | kubectl (标准工具) | async封装 |
| 弹性启动 | torchrun (PyTorch Elastic) | 作为launcher使用 |

### 10.2 借鉴设计（在其上改进）

| 设计模式 | 来源 | 我们的改进 |
|----------|------|-----------|
| 状态机 | Ray Train | 增加DIAGNOSING/RECOVERING状态 |
| Heartbeat | NeMo FaultToleranceCallback | FSx文件协议（更简单，跨pod） |
| 动态Timeout | NeMo/torchtitan | 首次step后自动收紧 |
| Node Exclusion | Determined AI | 自动学习哪些节点有问题 |
| Anomaly Detection | AutoTrain monitor.py | 增加straggler检测 |
| Event-Action表 | Volcano | 扩展为可学习的映射表 |
| Audit Trail | 所有框架 | 统一JSONL格式，可查询 |

### 10.3 全新创新

| 创新点 | 说明 |
|--------|------|
| SelfDiagnoser | 自学习的故障诊断：从成功修复中积累知识库 |
| 全参数搜索 | 不仅搜data mix，还搜lr/batch/warmup等 |
| Confidence-gated | 只有confidence > 0.7的action才自动执行 |
| Correlation分析 | 多trial同时失败→集群级问题（单独处理） |
| Budget-aware | GPU-hour预算管理，防止无限重试 |
| Multi-scale | 从190M到32B的统一管理 |

---

## 十一、目前已完成的实现

```
autopretrain/                          ✅ 已创建
├── __init__.py                        ✅
├── core/
│   ├── __init__.py                    ✅
│   ├── types.py                       ✅ (JobState, FailureType, Anomaly, Action, Budget等)
│   └── protocols.py                   ✅ (ComputeBackend, TrainingEngine, SelfDiagnoser等)
├── orchestrator/
│   ├── __init__.py                    ✅
│   ├── state_machine.py              ✅ (正式FSM + event logging)
│   └── orchestrator.py               ✅ (async event loop + 完整主循环)
├── resilience/
│   ├── __init__.py                    ✅
│   ├── diagnoser.py                   ✅ (多层诊断: heartbeat→code_bug→patterns→correlation)
│   ├── recovery.py                    ✅ (自适应恢复: 升级式OOM处理, backoff, node exclusion)
│   └── self_diagnoser.py             ✅ (自学习: signature提取→知识库→启发式→安全验证)
├── monitor/
│   ├── __init__.py                    ✅
│   └── anomaly_detector.py           ✅ (6种统计异常检测)
├── compute/
│   ├── __init__.py                    ✅
│   └── kubernetes.py                  ✅ (async kubectl封装)
├── engine/
│   ├── __init__.py                    ✅
│   └── olmo_adapter.py               ✅ (脚本生成+manifest生成+多模型scale支持)
├── store/
│   ├── __init__.py                    ✅
│   └── event_log.py                   ✅ (JSONL audit trail)
├── search/                            📋 待实现（从现有MCGS迁移）
├── eval/                              📋 待实现
└── notify/                            📋 待实现
```

---

## 十二、下一步立即推进

1. **Heartbeat Reader** — 实现FSx文件读取的HeartbeatReader
2. **集成测试** — 用本地mock验证整个orchestrator流程
3. **迁移MCGS** — 将现有搜索算法迁移到 `search/` 模块
4. **部署脚本** — 生成CPU pod部署manifest用于cluster上运行Agent
5. **首次真实运行** — 用新框架提交当前的3-trial MVP实验
