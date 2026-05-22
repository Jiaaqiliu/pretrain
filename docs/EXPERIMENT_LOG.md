# 实验进度日志 — Thermodynamics of Pretraining

> 本文档实时记录实验进度、发现、问题和经验教训。每次有重要进展时更新。

---

## 时间线

### 2026-05-22: 项目启动

**完成:**
- [x] 代码审查完毕，修复 6 个关键 bug（SIGPIPE、import 路径、FSDP 测量等）
- [x] 代码推送到 GitHub (`Jiaaqiliu/pretrain`)
- [x] 代码同步到 FSx (`/fsx/dev/jiaqi/A-EVOLVE-V2/`)
- [x] 数据下载进行中（3 个 job 并行）
- [ ] Pre-flight 检查（olmo-core + experiments 模块 + 数据路径）
- [ ] Phase 0 验证实验提交

**数据下载状态 (01:50 UTC):**
| Domain | 进度 | ETA |
|--------|------|-----|
| math | 6.81B / 7.0B (97%) | 完成 |
| code | 6.09B / 14B (43.5%) | ~4.4h |
| web | 7.58B / 38.5B (19.7%) | ~14.9h |

---

## 发现 & 经验教训

### Bug #1: SIGPIPE (pip | tail) 在所有 K8s Job 中

**问题**: `set -euxo pipefail` + `pip install ... | tail -5` → pip 写完输出后管道断裂 → SIGPIPE → exit 141 → job 失败。

**修复**: 改为 `pip install ... > /tmp/pip.log 2>&1 || true; tail -5 /tmp/pip.log`

**教训**: 在 Kubernetes Job 中永远不要把长输出的命令管道到 `tail`/`head`，而是先写文件再读。

### Bug #2: olmo_core 没有 `__version__` 属性

**问题**: `python -c "import olmo_core; print(olmo_core.__version__)"` 失败。

**原因**: ai2-olmo-core 2.5.0 不导出 `__version__`。

**修复**: 用 `from olmo_core.nn.transformer import TransformerConfig` 验证安装是否成功。

### Bug #3: 数据路径不一致

**问题**: 
- 数据下载到: `/fsx/dev/jiaqi/data/olmo-3b-pretrain/{web,code,math}/`
- 训练脚本期望: `/fsx/dev/jiaqi/data/olmo-pretrain/{dclm_web,dolma_web,code,math,books}/`

**解决方案**: 需要创建符号链接或修改训练脚本的 `DATA_PATHS`。

**教训**: 数据路径应该在一个配置文件中统一管理，而不是散落在多个脚本里硬编码。

### Bug #4: `NumpyFSLDatasetConfig` vs 简单 `.npy` 文件

**待验证**: OLMo-core 的 `NumpyFSLDatasetConfig` 期望特定格式（可能需要索引文件），而我们的数据准备脚本生成的是纯 `.npy` shard 文件。如果不兼容，需要用 OLMo-core 的 `InMemoryTokenSource` / `NumpyTokenSource` 代替。

---

## 关键决策

### 2026-05-22: 使用两阶段训练实现 mid-training data switch

**决策**: 放弃在 callback 中动态切换 data loader（OLMo-core 不支持），改为顺序执行两个 Trainer。

**权衡**: 
- 优点: 简单可靠，不依赖 Trainer 内部实现
- 缺点: WandB 会产生两个 run（stage1/stage2），需要在分析时合并

### 2026-05-22: Phase 0 先用已有数据

**决策**: Math 数据已足够 (~7B tokens)，可以先启动 190M 训练验证。不等 web/code 下载完毕。

**原因**: Phase 0 只需要验证管线正确性，不需要完整数据量。

---

## 实验矩阵

### Phase 0: 快速验证 (~3 天)
| Job | Status | Notes |
|-----|--------|-------|
| 190M gaussian s42 | 未提交 | 等待 preflight |
| 190M wsd_linear s42 | 未提交 | 等待 preflight |
| measure-7b (pilot) | 未提交 | 需 HF 下载 |

### Phase 1: 190M 全量 (~7 天)
| Schedule | s42 | s123 | s456 |
|----------|-----|------|------|
| cosine | - | - | - |
| wsd_linear | - | - | - |
| wsd_exponential | - | - | - |
| gaussian | - | - | - |

### Phase 2: 多尺度测量 (~12 天)
| Model | Status | Notes |
|-------|--------|-------|
| 190M | - | 需要先完成 Phase 1 |
| 1B | - | 公开检查点 |
| 7B | - | 公开检查点 |
| 13B | - | 公开检查点 |

### Phase 3: 1B 训练 (~17 天)
| Schedule | s42 | s123 | s456 |
|----------|-----|------|------|
| wsd_linear | - | - | - |
| wsd_exponential | - | - | - |
| gaussian | - | - | - |
| cosine | 复用 OLMo-2-0425-1B | - | - |

---

## 资源使用追踪

| 日期 | GPU-hours used | Job | Notes |
|------|---------------|-----|-------|
| 2026-05-22 | ~0.5 | code-sync, preflight | 轻量级 setup |
| - | - | - | - |

---

## 待解决问题

1. **数据路径对齐** — 训练脚本期望 `/fsx/dev/jiaqi/data/olmo-pretrain/`，数据在 `/fsx/dev/jiaqi/data/olmo-3b-pretrain/`
2. **`NumpyFSLDatasetConfig` 格式兼容性** — 需要验证 `.npy` shard 格式是否被 OLMo-core 原生数据加载器接受
3. **LRScheduleCallback 冲突** — 需要确认 `pre_step` 设置的 LR 不会被内置 scheduler 覆盖
4. **HuggingFace 检查点下载** — 需要确认集群可以访问 HF Hub (无 auth gating)
