# nemo_mas_reasoning_example

End-to-end driver for the `nemo_mas` algorithm on the Nemotron Reasoning
workspace. Mirrors [examples/nemo_reasoning_example/](../nemo_reasoning_example/)
which drives the MCGS algorithm on the same benchmark, so you can
compare the two search strategies head-to-head on identical training DNA.

## Files

- [drive_nemo_mas.py](drive_nemo_mas.py) — the only entry point. One
  file, three modes, three backend choices.

Everything else the driver needs lives elsewhere:

- **Algorithm:**
  [agent_evolve/model/algorithms/nemo_mas/](../../agent_evolve/model/algorithms/nemo_mas/)
- **Workspace (training DNA):**
  [seed_workspaces/nemo_mas_reasoner/](../../seed_workspaces/nemo_mas_reasoner/)
- **Shared platform tools:**
  [seed_workspaces/_common_model/tools/](../../seed_workspaces/_common_model/tools/)
- **Training stages:**
  [agent_evolve/model/runners/stages/](../../agent_evolve/model/runners/stages/)
  (sft, rl, teacher_distill, solver_distill, data_merge, eval) — never
  re-scaffold runner code into the workspace.

## Modes

The driver's `--mode` flag controls which backend tools the orchestrator
actually calls. AWS Bedrock is needed for the orchestrator + workers in
both `demo` and `real`; this EC2 host's instance role already covers it.

| `--mode` | Bedrock | GPUs | What happens |
|---|---|---|---|
| `dry-run` (default) | stubbed | none | Monkey-patches `BedrockAgent` with a scripted stub. Exercises the workspace + memory + tool-wiring loop in seconds. |
| `demo` | yes | none | Orchestrator + workers run for real on Bedrock; compute-bound tools (`run_eval`, `launch_training`, `call_teacher_model`) return deterministic fake outputs via `demo_compute_handlers()`. |
| `real` | yes | yes | `launch_training` dispatches through `backend.run_trial` → `StageRegistry` → `agent_evolve/model/runners/stages/*.py`. Depending on `--backend`, compute lands on the local machine or a k8s cluster. |

## Backends (only meaningful with `--mode real`)

| `--backend` | Where training runs | Eval | Fallback on failure |
|---|---|---|---|
| `local` (default) | Local GPUs via torchrun subprocess | local vLLM | n/a |
| `k8s` | k8s `batch/v1` Jobs, capped at 2 concurrent (`k8s_queue_budget=2`) | local vLLM (eval is not cloudified today) | **None** — `local_enabled=False`. Missing kubeconfig / `kubernetes` pkg raises at backend construction. |

K8s routing only applies to DDP stages (SFT / RL). The `eval` step runs
on the host that started the driver — see
[elastic/backend.py](../../agent_evolve/backends/tinkerlite/elastic/backend.py)
docstring: *"eval keeps running on the caller's machine — cloudifying
it is a follow-up."*

Env vars read in `--backend k8s` (all optional; defaults in parens):

```
AE_K8S_NAMESPACE       (a-evolve)
AE_K8S_IMAGE           (a-evolve/trainer:latest)
AE_K8S_PVC             (fsx-zzsamshi)
AE_K8S_NODE_LABEL=1    (set 1 to add nodeSelector nvidia.com/gpu.product=H200)
AE_K8S_QUEUE_TIMEOUT   (900 seconds)
```

The pod runs as uid/gid 1000 (`runAsUser/runAsGroup/fsGroup=1000`) so
FSx outputs land as `ec2-user:ec2-user` instead of `root:root`. Override
via `K8sTinkerLiteBackend(k8s_run_as_uid=None, ...)` if you want the
image's default user.

## Debug harness

All three additions land automatically when the driver starts; see
commit `d727199` for the design.

- **Logging.** `NEMO_MAS_LOG={DEBUG,INFO,WARN}` routes `BedrockAgent`
  (root logger), orchestrator, spawner, and scheduler logs to stderr.
  Noisy deps (`botocore`, `urllib3`, `s3transfer`) are suppressed to
  WARN.
- **`--trace-dir PATH`.** Writes one JSONL per agent per cycle:
  `<trace-dir>/cycle_<NNNN>/agent_<id>.jsonl`. Each line is one event
  (`start`, `message`, `turn`, `done`) with the Bedrock assistant
  message, tool uses, stop reason, and token usage. Diagnosing a hang:
  *`ls -latr` → newest mtime = active agent; last line `event=turn`
  with `tool_uses=[launch_training]` and no follow-up = orchestrator
  blocked inside `backend.run_trial`.*
- **Heartbeat.** One line every 30 s:
  `[heartbeat] cycle=N elapsed=MM:SS records=K last_record=rec_xxxx`
  (reads `<workspace>/memory/records.jsonl` each tick).
- **Trace viewer.** `trace_viewer.py` is a stdlib-only web UI for a trace
  directory plus its sibling `memory/records.jsonl`. Example:
  `python examples/nemo_mas_reasoning_example/trace_viewer.py --trace-dir /fsx/zzsamshi/a-evolve/runs/nemo-mas-marathon/trace --port 7889 --host 127.0.0.1`.
  Run-local copies under `runs/` should stay as thin wrappers because
  `/runs/` is git-ignored.

## Typical invocations

Dry-run (no Bedrock, no GPU, exercises prompts / skills / tools only):

```bash
PYTHONPATH=/fsx/zzsamshi/a-evolve \
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
  examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
  --cycles 1 --mode dry-run
```

Demo mode (Bedrock live, compute stubbed, AWS role picks up creds):

```bash
NEMO_MAS_LOG=INFO \
PYTHONPATH=/fsx/zzsamshi/a-evolve \
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
  examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
  --cycles 1 --mode demo \
  --trace-dir /fsx/zzsamshi/a-evolve/runs/nemo-mas-demo/trace
```

Real mode, local GPUs (8× H200, eval-only smoke if all train stages are
`enabled: false` in [train/pipeline.yaml](../../seed_workspaces/nemo_mas_reasoner/train/pipeline.yaml)):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 AE_TRAIN_DDP=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true \
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 VLLM_USE_FLASHINFER_MOE_FP8=0 \
VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
NEMO_MAS_LOG=INFO \
PYTHONPATH=/fsx/zzsamshi/a-evolve \
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
  examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
  --cycles 1 --mode real --backend local \
  --trial-budget-seconds 3600 \
  --trace-dir /fsx/zzsamshi/a-evolve/runs/nemo-mas-real-local/trace \
  --work-dir /fsx/zzsamshi/a-evolve/runs/nemo-mas-real-local
```

Real mode, k8s (cap 2 concurrent Jobs, no local fallback — requires
`pip install kubernetes` + a configured `~/.kube/config` + an FSx PVC
in the target namespace):

```bash
AE_TRAIN_DDP=1 \
AE_K8S_NAMESPACE=ads-evolve AE_K8S_PVC=fsx-pvc \
AE_K8S_IMAGE=801953956576.dkr.ecr.ap-southeast-3.amazonaws.com/zzsamshi/a-evolve:latest \
AE_K8S_NODE_LABEL=0 AE_K8S_QUEUE_TIMEOUT=900 \
NEMO_MAS_LOG=INFO \
PYTHONPATH=/fsx/zzsamshi/a-evolve \
/fsx/zzsamshi/nemotron-auto-research/.venv/bin/python \
  examples/nemo_mas_reasoning_example/drive_nemo_mas.py \
  --cycles 1 --mode real --backend k8s \
  --trial-budget-seconds 3600 \
  --trace-dir /fsx/zzsamshi/a-evolve/runs/nemo-mas-real-k8s/trace \
  --work-dir /fsx/zzsamshi/a-evolve/runs/nemo-mas-real-k8s
```

While a k8s run is live:

```bash
kubectl get jobs -n ads-evolve -w            # at most 2 Running+Pending
kubectl logs <pod> -n ads-evolve --tail 10   # pod-side training log
tail -F /fsx/zzsamshi/a-evolve/runs/nemo-mas-real-k8s/driver.log
ls -latr /fsx/zzsamshi/a-evolve/runs/nemo-mas-real-k8s/trace/cycle_0001/
```

## Teacher distillation — which model?

`synth_generate` has two provider backends, selected in
[data/generators/teacher_distill.yaml](../../seed_workspaces/nemo_mas_reasoner/data/generators/teacher_distill.yaml)::`teacher_provider`:

| provider | Where it runs | Default model |
|---|---|---|
| `vllm_local` | Local GPU via vLLM, subprocess-isolated so CUDA state frees before SFT | `/fsx/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` (TP=8) |
| `bedrock` (**default**) | AWS Bedrock Converse, in-process, no GPU | `us.anthropic.claude-sonnet-4-6` (us-west-2) |

Output JSONL shape is identical either way, so SFT consumes `kept` rows
unchanged via `data/sources.yaml`. Flip back to the 120B path by setting
`teacher_provider: vllm_local` in the workspace YAML.

On Sonnet 4.6: ~10 s per 900-token trace end-to-end (prompt + response +
retry wrapper), versus minutes per trace on a local 120B at TP=8. Note
that `teacher_n_tokens` uses Bedrock's tokenizer, not Nemotron's — if
you need a hard floor calibrated for Nemotron tokens, gate it in a
downstream filter, not in `min_tokens`.

## Workspace contract

[seed_workspaces/nemo_mas_reasoner/](../../seed_workspaces/nemo_mas_reasoner/)
follows the same layout MCGS uses:

```
manifest.yaml              evolvable_layers / protected_layers / artifact_layers
prompts/                   orchestrator + 4 role prompts + benchmark_reference
skills/<role>/*.md         lazy-loaded playbooks per role
tools/<role>.yaml          per-role tool schemas (platform tools inherited
                           from ../_common_model/tools/)
model/base.yaml            Nemotron-3-Nano-30B base model path
model/adapter.yaml         LoRA rank/alpha/target_modules + optional seed_adapter_path
train/pipeline.yaml        [solver_distill, teacher_distill, data_merge,
                           sft_warmup, rl_gspo] — flip `enabled` per cycle
train/{optimizer,batching,loss}.yaml
data/sources.yaml          SFT / eval source paths
data/generators/           teacher_distill.yaml (sonnet by default)
data/recipes/default.yaml  dedup + filter thresholds
eval/local_splits.yaml     named splits including kaggle_dev_local (951 rows)
eval/kaggle_eval.yaml      Kaggle inference contract (temp, top_p, max_tokens)
```

Artifact dirs (`memory/`, `evolution/`, `checkpoints/`, `logs/`) are
created at run time and are **not** checked in. Delete them for a clean
slate; don't delete `manifest.yaml`, `prompts/`, `skills/`, `tools/`,
`model/`, `train/`, `eval/`, `data/` — those are the seed DNA.

## Where to read next

- [agent_evolve/model/algorithms/nemo_mas/README.md](../../agent_evolve/model/algorithms/nemo_mas/README.md)
  — algorithm design + per-cycle flow.
- [TRAINDESIGN.md](../../TRAINDESIGN.md) §4.5 — how `nemo_mas` fits
  alongside `mcgs` and `a_evolve_training_multi`.
- [INTEGRATION.md](../../INTEGRATION.md) — the extension recipe
  (`@register_stage`, `@register_adapter`, new benchmarks).
