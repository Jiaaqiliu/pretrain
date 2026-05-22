# 集群操作手册 — Thermodynamics Experiments on p5-llm

## 集群连接

### 前提条件

```bash
# 1. 安装 kubectl + aws cli
brew install kubectl awscli

# 2. 配置 kubeconfig (一次性)
aws eks update-kubeconfig \
  --region ap-south-1 \
  --name p5-llm \
  --alias p5-llm
```

### 集群信息

| 项目 | 值 |
|------|-----|
| 区域 | ap-south-1 |
| 集群名 | p5-llm |
| Context | `arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm` |
| 节点组 | trainer5 (H200 × 8 per node) |
| 命名空间 | default |
| 镜像 | `801953956576.dkr.ecr.ap-south-1.amazonaws.com/ads-foundation-model-training/verl-multiturn:1.0.2` |
| 存储 | FSx (PVC: `fsx`), 挂载在 `/fsx` |
| 代码路径 | `/fsx/dev/jiaqi/A-EVOLVE-V2/` |
| WandB Secret | `wandb-secret-jiaqi` (key: `WANDB_API_KEY`) |

### 快速验证连接

```bash
# 查看节点
kubectl get nodes --context arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm

# 查看 GPU 容量
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# 查看当前 job
kubectl get pytorchjobs -n default --context arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm
```

---

## 提交 Job

### Job 命名规范

所有 job 使用前缀 `luhanqin-thermo-*`（按 mentor 名字命名）。

生成脚本中的模板名为 `jiaqi-thermo-*`；提交前需批量替换或在新 job 中使用 `luhanqin` 前缀。

### 提交单个 Job

```bash
export CTX="arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm"

# 提交
kubectl apply -f scripts/k8s/thermo/thermo_train_190m_gaussian_s42.yaml -n default --context $CTX

# 查看状态
kubectl get pytorchjobs -n default --context $CTX | grep thermo

# 查看日志 (实时)
kubectl logs job/jiaqi-thermo-190m-gaussian-s42-master-0 -n default --context $CTX -f

# 查看最后 50 行
kubectl logs job/jiaqi-thermo-190m-gaussian-s42-master-0 -n default --context $CTX --tail=50
```

### 批量提交

```bash
# 使用提交脚本
./scripts/thermo/submit_all.sh measure      # 4 measurement jobs
./scripts/thermo/submit_all.sh train-190m   # 12 training jobs (190M)
./scripts/thermo/submit_all.sh train-1b     # 12 training jobs (1B)
./scripts/thermo/submit_all.sh all          # 全部 28 jobs

# Dry run (不执行)
./scripts/thermo/submit_all.sh all --dry-run

# 检查状态
./scripts/thermo/submit_all.sh status
```

### 删除/停止 Job

```bash
# 删除单个
kubectl delete pytorchjob jiaqi-thermo-190m-gaussian-s42 -n default --context $CTX

# 删除所有 thermo job
kubectl get pytorchjobs -n default --context $CTX -o name | grep thermo | xargs kubectl delete -n default --context $CTX
```

---

## 部署代码到 FSx

集群上的代码位于 `/fsx/dev/jiaqi/A-EVOLVE-V2/`。更新方式：

```bash
# 方法 1: 推送到 GitHub 后在集群上拉取
# 先推送本地代码
git push origin main

# 然后通过一个临时 pod 或在已有 job 中执行:
kubectl exec -it <pod-name> -- bash -c "cd /fsx/dev/jiaqi/A-EVOLVE-V2 && git pull"

# 方法 2: 创建一个一次性 sync job
kubectl apply -f - <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: luhanqin-code-sync
  namespace: default
spec:
  template:
    spec:
      nodeSelector:
        eks.amazonaws.com/nodegroup: trainer5
      restartPolicy: Never
      containers:
      - name: sync
        image: 801953956576.dkr.ecr.ap-south-1.amazonaws.com/ads-foundation-model-training/verl-multiturn:1.0.2
        command: ["/bin/bash", "-c"]
        args:
        - |
          cd /fsx/dev/jiaqi/A-EVOLVE-V2
          git fetch origin main
          git reset --hard origin/main
          echo "Code synced successfully"
        volumeMounts:
        - mountPath: /fsx
          name: fsx
      volumes:
      - name: fsx
        persistentVolumeClaim:
          claimName: fsx
EOF
```

---

## 监控

### WandB Dashboard

- Project: `thermo-pretraining`
- 关键指标:
  - `train/loss` — 训练损失
  - `lr` — 学习率
  - `thermo/spectral_entropy` — 谱熵 S
  - `thermo/order_parameter` — 序参数 ψ
  - `thermo/pv_over_nt` — 状态方程 P·V/(N·T)

### K8s 状态监控

```bash
# 查看所有 thermo job 状态
kubectl get pytorchjobs -n default --context $CTX | grep thermo

# 查看 pod 状态 (检查 GPU 分配)
kubectl get pods -n default --context $CTX -l job=jiaqi-thermo-190m-gaussian-s42

# 查看 pod 事件 (排查启动失败)
kubectl describe pod jiaqi-thermo-190m-gaussian-s42-master-0 -n default --context $CTX

# 查看 GPU 使用率 (需要 nvidia-smi)
kubectl exec -it jiaqi-thermo-190m-gaussian-s42-master-0 -n default --context $CTX -- nvidia-smi
```

### 实时日志监控

```bash
# 跟踪训练日志
kubectl logs -f job/jiaqi-thermo-190m-gaussian-s42-master-0 -n default --context $CTX

# 查看热力学测量输出
kubectl exec -it jiaqi-thermo-190m-gaussian-s42-master-0 -- \
  tail -20 /fsx/dev/jiaqi/thermo_experiments/190m_gaussian_s42/thermo_measurements.jsonl
```

---

## 常见问题排查

### Job 立即失败 (SIGPIPE / exit 141)

**原因**: `pip install ... | tail` 在 `set -euxo pipefail` 下管道断裂。

**修复**: 已改为 `pip install ... > /tmp/pip.log 2>&1 || true; tail -5 /tmp/pip.log`。

### OOM (Out of Memory)

**症状**: Pod 被 kill，`kubectl describe pod` 显示 OOMKilled。

**修复**: 增加 resources.requests.memory，或减小 `rank_microbatch_size`。

### NCCL Timeout

**症状**: 训练卡在某步不动，最终报 NCCL timeout。

**排查**:
```bash
# 检查所有 worker 是否都在运行
kubectl get pods -l job=<job-name> -n default --context $CTX
# 如果有 worker pending/failed，修复后重新提交
```

### 数据路径不存在

**症状**: `RuntimeError: No data paths found`

**修复**: 确认 FSx 上的数据路径：
```bash
kubectl exec -it <pod> -- ls /fsx/dev/jiaqi/data/olmo-pretrain/
```

### Import 失败

**症状**: `ModuleNotFoundError: No module named 'experiments'`

**原因**: 工作目录不在项目根。

**修复**: 脚本中已有 `sys.path.insert(0, ...)` 处理，确保 `/fsx/dev/jiaqi/A-EVOLVE-V2/` 是最新代码。

---

## 实验进度管理

### Phase 0: 快速验证 (~3 天)

```bash
# 1. 提交测量: OLMo-7B 前 50 个检查点
kubectl apply -f scripts/k8s/thermo/thermo_measure_7b.yaml -n default --context $CTX

# 2. 提交 2 个 190M 训练 (Gaussian vs WSD)
kubectl apply -f scripts/k8s/thermo/thermo_train_190m_gaussian_s42.yaml -n default --context $CTX
kubectl apply -f scripts/k8s/thermo/thermo_train_190m_wsd_linear_s42.yaml -n default --context $CTX
```

**验证条件**:
- [ ] S 随训练单调下降
- [ ] ψ 随训练单调上升
- [ ] P·V/(N·T) 在 stable phase 大致收敛
- [ ] Gaussian 最终 loss < WSD-Linear 最终 loss

### Phase 1: 190M 全量 (~7 天)

```bash
./scripts/thermo/submit_all.sh train-190m
```

### Phase 2: 多尺度测量 (~12 天, 与 Phase 1 并行)

```bash
./scripts/thermo/submit_all.sh measure
```

### Phase 3: 1B 训练 (~17 天, Phase 1+2 完成后)

```bash
./scripts/thermo/submit_all.sh train-1b
```

### Phase 4: 分析

```bash
python scripts/thermo/run_analysis.py \
  --results-dir /fsx/dev/jiaqi/thermo_results \
  --experiments-dir /fsx/dev/jiaqi/thermo_experiments \
  --output-dir /fsx/dev/jiaqi/thermo_paper_figures
```

---

## 资源估算

| 阶段 | GPU-hours | 节点 × 天 (32 GPU) |
|------|-----------|---------------------|
| Phase 0 | ~500 | ~1 |
| Phase 1 | ~3,100 | ~4 |
| Phase 2 | ~3,600 | ~5 |
| Phase 3 | ~13,200 | ~17 |
| **Total** | **~20,400** | **~30 天 (含并行)** |
