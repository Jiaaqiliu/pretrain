# 实验进度日志 — Thermodynamics of Pretraining

> 本文档实时记录实验进度、发现、问题和经验教训。每次有重要进展时更新。

---

## 时间线

### 2026-05-22: 项目启动 + 9 次迭代调试

**完成:**
- [x] 代码审查完毕，修复 6 个关键 bug（SIGPIPE、import 路径、FSDP 测量等）
- [x] 代码推送到 GitHub (`Jiaaqiliu/pretrain`)
- [x] 代码同步到 FSx (`/fsx/dev/jiaqi/A-EVOLVE-V2/`)
- [x] 数据下载完成: math 7B, code 14B (进行中), web 38.5B (进行中)
- [x] Pre-flight 检查通过（olmo-core + experiments 模块 + 数据路径）
- [x] 数据符号链接创建 (`/fsx/dev/jiaqi/data/olmo-pretrain/` → `olmo-3b-pretrain/`)
- [x] 190M smoke test 成功 (100 步，无热力学测量, 174.7 TFLOPS/device)
- [x] 190M + 热力学测量验证 (200 步完成，谱熵/序参数信号正确)
- [x] **Phase 0 全面启动**: 4 schedule × 25000 步 × seed=42 (4 节点并行, ETA ~8h)

**正在运行的实验 (04:45 UTC):**
| Job | Schedule | Steps | ETA |
|-----|----------|-------|-----|
| luhanqin-thermo-190m-gaussian-s42 | gaussian | 25000 | ~8h |
| luhanqin-thermo-190m-wsd-linear-s42 | wsd_linear | 25000 | ~8h |
| luhanqin-thermo-190m-cosine-s42 | cosine | 25000 | ~8h |
| luhanqin-thermo-190m-wsd-exponential-s42 | wsd_exp | 25000 | ~8h |

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

### Bug #3: pip install olmo-core/.[all] 编译 flash-attn 耗时 10+ 分钟

**问题**: `pip install -e olmo-core/.[all]` 拉取 `flash-attn` 源码编译，极慢。

**解决**: 改为 `pip install -e olmo-core/`（不带 extras）。PyTorch 2.8 的内置 SDPA 已经包含 FlashAttention kernel（日志显示 `Using attention backend 'torch'`）。

**教训**: 如果镜像已有 PyTorch 2.8+，不需要单独安装 flash-attn。

### Bug #4: OLMo-core Callback API 与预期不同

**问题**: 假设 `pre_step(self, step, **kwargs)` 但实际 API 是 `pre_step(self, batch: Dict)`；假设 `post_step(self, step, **kwargs)` 但实际是 `post_step(self)`。

**正确 API** (从 OLMo-core 源码确认):
- `pre_step(self, batch: Dict[str, Any])` — 只接收 batch
- `post_step(self)` — 无参数
- `pre_optim_step(self)` — 在 forward-backward 后、optimizer step 前
- 步数通过 `self.step` 或 `self.trainer.global_step` 获取
- Trainer 通过 `self.trainer` 访问

**教训**: 一定要看源码确认 API，不要凭记忆。

### Bug #5: FSDP2 + torch.compile 下 SVD 报 DynamicOutputShapeException

**问题**: 在 `post_step()` 中对模型参数做 SVD，触发 `DynamicOutputShapeException: aten.nonzero.default`。

**原因**: FSDP2 使用 torch.compile，参数被 FakeTensor 包装，SVD 中的 `nonzero` 操作有动态 shape。

**修复**: 
1. 将测量移到 `pre_optim_step()` — 此时参数已在 forward-backward 后 unsharded
2. 添加 `@torch._dynamo.disable()` 装饰器
3. 使用 `get_local_tensor()` 获取本地 shard

**模式**: 参考 OLMo-core 自带的 `GAPMonitorCallback`，它在 `pre_optim_step` 中监控参数。

### Bug #6: NumpyFSLDatasetConfig 把每个 .npy 文件当作一个 instance

**问题**: 数据有 350 个 `.npy` 文件 → `dataset_size=27`，每个 epoch 只有 27 个 instance，导致训练循环在 epoch 间疯狂切换。

**原因**: `NumpyFSLDatasetConfig` 将每个文件视为一个"document"（一个 instance），而非将文件内的 token 切成多个 4096 序列。

**修复**: 改用 `InMemoryTokenSource`（加载 `.npy` 到内存）→ `ConcatAndChunkInstanceSource`（切成 4096 序列）→ `ComposableDataLoaderConfig`。

**教训**: OLMo-core 有两套数据 API:
- `NumpyFSLDatasetConfig`: 期望每个文件是一个完整 document
- `ComposableDataLoader (InMemoryTokenSource)`: 适合 flat token 数组

### Bug #7: ComposableDataLoader 的 work_dir 必须跨 rank 共享

**问题**: 用 `tempfile.mkdtemp()` 创建 work_dir，但每个 rank 生成不同的临时路径 → 非 rank-0 进程找不到 global indices 文件。

**修复**: 使用 FSx 上的共享目录作为 work_dir。

### Bug #8: CheckpointerCallback 的 ephemeral_save_interval 必须 < save_interval

**问题**: `CheckpointerCallback(save_interval=200, ephemeral_save_interval=200)` → `OLMoConfigurationError`

**修复**: 删除 `ephemeral_save_interval` 参数。

### Bug #9: WandBCallback 在 WANDB_API_KEY 不存在时硬报错

**问题**: 即使设置 `WANDB_MODE=disabled`，OLMo-core 的 WandBCallback 在初始化时检查环境变量。

**修复**: 代码中判断 `os.environ.get("WANDB_API_KEY")` 存在时才添加 WandB callback。

### Bug #11: WandB API Key 只有 20 字符，导致 OLMo-core WandBCallback 初始化失败

**问题**: K8s secret `wandb-secret-jiaqi` 中的 key 只有 20 字符（正常 40 字符），WandBCallback 初始化时连接失败 → 所有带 WandB 的 job 在 ~3 分钟时 fail。

**临时修复**: 设置 `WANDB_MODE=disabled`，热力学数据通过 JSONL 文件收集。

**后续**: 需要更新 K8s secret 为正确的 WandB API key。

**教训**: 
- 如果一个功能不是核心必需的（如 WandB 日志），不要让它阻塞训练
- 先用 WANDB_MODE=disabled 跑通训练，确认 stable 后再加 WandB

### Bug #12: 多个 PyTorchJob 同时 `git pull` 导致 git lock 冲突

**问题**: 4 个 job 同时启动，都执行 `git pull origin main`，git index.lock 冲突 → 第一个成功，后面的 fail。

**修复**: 
1. 把 `set -euxo pipefail` 改为 setup 阶段用 `set -uxo pipefail`（不退出），训练阶段再 `set -e`
2. `git pull origin main || true` — git pull 失败不影响训练（代码已在 FSx 上）

**教训**: 多 job 共享同一个 FSx git 仓库时，不要在启动脚本中做写操作（git pull 会写 .git/）。更好的方式是先用一个独立的 sync job 更新代码。

### Bug #13: 数据路径不一致

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
