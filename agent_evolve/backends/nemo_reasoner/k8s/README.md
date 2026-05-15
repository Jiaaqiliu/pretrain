# k8s infrastructure — nemo_reasoner backend

Benchmark-specific k8s glue for `nemo_reasoner` (NVIDIA Nemotron Model Reasoning Kaggle Challenge). Lives at platform level so seed workspaces (and their per-cycle forks) don't carry a copy.

Two canonical job shapes, one CLI.

## Contract

Every path is **explicit**. Callers pass fork-rooted absolute paths. No implicit workspace resolution — multiple forks can submit in parallel without colliding.

```bash
BACKEND=/fsx/zzsamshi/a-evolve/agent_evolve/backends/nemo_reasoner
FORK=/path/to/fork/workspace        # agent harness exports this

# Training
$BACKEND/k8s/submit.sh train \
    --train-recipe $FORK/recipes/train/default.yaml \
    --data-recipe  $FORK/recipes/data/default_data.yaml \
    --out          $FORK/artifacts/sft/w7 \
    --name         w7

# Eval (multiple in parallel across nodes)
$BACKEND/k8s/submit.sh eval \
    --adapter $FORK/artifacts/sft/w7/step_200 \
    --out     $FORK/artifacts/eval/w7_step200 \
    --name    w7_step200
```

Overrides:
- `--lr`, `--steps`, `--save-every`, `--seed` override the corresponding recipe values via env vars.
- `--wandb | --no-wandb` toggles logging (requires `$WANDB_API_KEY` when enabled).

## Recipe anchor

`seed_workspaces/nemo_mas_reasoner/recipes/train/default.yaml` is the seed training recipe — single source of truth for lr, optimizer, scheduler, LoRA spec, batching, MoE tying toggle, CCE toggle, mamba fast path. New recipes are sibling YAMLs written by the planner agent; pass the absolute path via `--train-recipe`.

## Outputs

All outputs land under `--out` (caller-supplied, typically `$FORK/artifacts/{sft,eval}/<name>/`).

| command | file                          | what |
|---------|-------------------------------|------|
| `train` | `step_{50,100,...}/`          | periodic LoRA adapters |
|         | `final/`                      | final adapter + Kaggle-renamed lm_head keys |
|         | `train.log`                   | step-by-step loss/grad log |
| `eval`  | `metrics.json`                | overall + per-domain accuracy |
|         | `predictions.jsonl`           | per-row model output + correctness |

## Job shapes

### `train_1gpu.yaml` — proven, reliable
- 1 pod, 1 GPU (`nvidia.com/gpu: 1`), optional `nodeName` pin
- Runs `entries/train_unsloth.py` — the full default recipe:
  - LoRA r=32 α=32, 9 target modules incl. `lm_head`
  - AdamW β=(0.9,0.95), eps=1e-8, wd=0, grad_clip=1e9
  - linear LR decay from `--lr` to 0, no warmup
  - bf16 base, fp32 LoRA, fp32 MoE router (asserted)
  - Mamba fast path forced on, CCE patched forward
  - MoE expert weight tying (Tinker convention)
- One epoch on 14718 rows ≈ 460 steps ≈ **~15 h on one H200**

### `eval_1gpu.yaml` — proven, reliable (default TP=1)
- 1 pod, `${GPU_COUNT}` GPUs (`--tp N` sets it), `tensor_parallel_size=N`
- Runs `agent_evolve.model.runners.eval_worker`
- Dataset: `balanced_dev726`
- ~75 min per 726-row eval (TP=1 on H200; decode-bound)
- TP=8 validated to match TP=1 within 0.42pp (36.91% vs 37.33% on huikang_step50) but gave ~0x speedup. Prefer TP=1 + parallel runs across nodes.

## Image

```bash
cd image
./build_and_push.sh unsloth-v6    # builds Dockerfile.unsloth, pushes to ECR
```

Image contents (key pins):
- torch 2.10.0+cu128, torchvision 0.25.0
- transformers 4.57.6, peft 0.19.1, accelerate 1.13.0, datasets 4.3.0
- trl 0.24.0, unsloth 2026.5.2, unsloth_zoo 2026.5.1
- cut-cross-entropy, bitsandbytes 0.49.2, wandb
- causal-conv1d v1.6.1, mamba-ssm v2.3.1 (both built from source)

Eval uses the platform's `:kernels` image (`agent_evolve/backends/tinkerlite/elastic/k8s/image/Dockerfile`).

## Canonical data paths

- Train: `/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/train/default_14718.jsonl`
- Dev:   `/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/dev/balanced_dev726.csv`

See `seed_workspaces/nemo_mas_reasoner/eval/local_splits.yaml` for the split registry.
