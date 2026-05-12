# k8s infrastructure — nemo_mas_reasoner workspace

Two canonical job shapes, one CLI.

## Quick start

```bash
cd seed_workspaces/nemo_mas_reasoner/k8s

# Training: 1 pod × 1 GPU, uses train/recipes/huikang.yaml by default
export WANDB_API_KEY=<your-key>   # or use --no-wandb
./submit.sh train --name w7                       # defaults = recipe file
./submit.sh train --name w7_lrsweep --lr 3e-4     # override just lr

# Eval: 1 pod × 1 GPU on balanced_dev726 (default). Run multiple in parallel:
./submit.sh eval --adapter /path/to/adapter_dir --name w7_step200
./submit.sh eval --adapter /path/to/step_150 --name w7_step150  # in parallel
# Scheduler bin-packs across nodes; each pod claims 1 GPU.
```

## Where the recipe lives

`seed_workspaces/nemo_mas_reasoner/train/recipes/huikang.yaml`

- Single source of truth for lr, optimizer, scheduler, LoRA spec, batching,
  MoE tying toggle, CCE toggle, mamba fast path, etc.
- `k8s/entries/train_unsloth.py` loads it via `RECIPE_PATH` env var.
- `submit.sh train --recipe NAME` reads `train/recipes/NAME.yaml`
- Any env var (LR, NUM_STEPS, SAVE_EVERY, SEED) overrides the recipe value
  for quick sweeps without editing the YAML.

To try a new recipe: copy `huikang.yaml`, edit, then `./submit.sh train --recipe newname`.

## Outputs

| command | where | what |
|---|---|---|
| `train` | `artifacts/sft/<name>/step_{50,100,...}/` | periodic LoRA adapters |
|         | `artifacts/sft/<name>/final/`              | final adapter + key-renamed for Kaggle submission |
|         | `artifacts/sft/<name>/train.log`           | step-by-step loss/grad log |
| `eval`  | `artifacts/eval/<name>/metrics.json`       | overall + per-domain accuracy |
|         | `artifacts/eval/<name>/predictions.jsonl`  | per-row model output + correctness |

## Job shapes

### `train_1gpu.yaml` — proven, reliable
- 1 pod, 1 GPU, pinned to a single node via `nodeName`
- Runs `k8s/entries/train_unsloth.py` — huikang's full recipe:
  - LoRA r=32 α=32, 9 target modules incl. `lm_head`
  - AdamW β=(0.9,0.95), eps=1e-8, wd=0, grad_clip=1e9
  - linear LR decay from `--lr` to 0, no warmup
  - bf16 base, fp32 LoRA, fp32 MoE router (asserted)
  - Mamba fast path forced on, CCE patched forward
  - MoE expert weight tying (Tinker convention)
- One epoch on 14718 rows ≈ 460 steps ≈ **~15 h on one H200**

### `eval_1gpu.yaml` — proven, reliable (default)
- 1 pod, 1 GPU, `tensor_parallel_size=1` (override with `--tp N` if ever needed)
- Runs `agent_evolve.model.runners.eval_worker`
- Eval dataset: `balanced_dev726` (canonical path in `local_splits.yaml`)
- ~75 min per 726-row eval (model load + inference)
- Validated: `submit.sh eval` on huikang_step50 — **36.91%** on k8s TP=8 vs **37.33%** on host (within 0.42pp, confirms k8s eval pipeline is correct)
- Nemotron-3-Nano MoE is decode-bound; TP=8 gave essentially 0 speedup over TP=1. Stick with TP=1 + parallelize across nodes.

## Image

```bash
cd k8s/image
./build_and_push.sh unsloth-v6    # builds Dockerfile.unsloth, pushes to ECR
```

Image contents (key pins):
- torch 2.10.0+cu128, torchvision 0.25.0
- transformers 4.57.6, peft 0.19.1, accelerate 1.13.0, datasets 4.3.0
- trl 0.24.0, unsloth 2026.5.2, unsloth_zoo 2026.5.1
- cut-cross-entropy, bitsandbytes 0.49.2, wandb
- causal-conv1d v1.6.1, mamba-ssm v2.3.1 (both built from source)

Eval uses the platform's `:kernels` image (`agent_evolve/backends/tinkerlite/elastic/k8s/image/Dockerfile`) — unchanged.

## Why not 8-pod sweep / DP-eval?

See `../legacy/README.md`. We proved 8×1-GPU Unsloth on one node works
(capacity ≈ 82 GB per pod, plenty of headroom), but in practice we submit
one job at a time so that sweep pattern isn't wired into `submit.sh`. The
pattern is easy to re-add if needed — write a loop around `./submit.sh
train`.

DP=8 eval hit vLLM 0.19 gotchas that made the TP=8 + single-pod path the
clean choice.

## Canonical data paths

- Train: `/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/train/huikang_14718.jsonl`
- Dev:   `/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/dev/balanced_dev726.csv`

See `../eval/local_splits.yaml` for the workspace's split registry (points at those absolute paths).
