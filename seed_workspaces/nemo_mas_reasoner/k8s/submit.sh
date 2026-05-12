#!/usr/bin/env bash
# Canonical CLI for submitting train + eval jobs to k8s.
#
# Usage:
#   submit.sh train --name <run>  [--lr 2e-4] [--steps 460] [--save-every 50] \
#                                 [--node <node-name>] [--wandb|--no-wandb]
#   submit.sh eval  --adapter <path> --name <out_name> [--node <node-name>]
#
# Examples:
#   ./submit.sh train --name w7 --lr 2e-4
#   ./submit.sh eval --adapter artifacts/sft/w7/step_200 --name w7_step200
#
# Outputs:
#   train → /fsx/.../artifacts/sft/${name}/step_{N}/ + /final/
#   eval  → /fsx/.../artifacts/eval/${name}/metrics.json + predictions.jsonl
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WS_ROOT="$(cd "$HERE/.." && pwd)"
ARTIFACTS="$WS_ROOT/artifacts"
CANONICAL_TRAIN_DATA=/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/train/huikang_14718.jsonl
CANONICAL_DEV_CSV=/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/dev/balanced_dev726.csv
DEFAULT_TRAIN_IMAGE=801953956576.dkr.ecr.ap-southeast-3.amazonaws.com/zzsamshi/a-evolve:unsloth-v5
# Reference the wandb key from memory/reference_wandb_nemo_mas.md.
WANDB_API_KEY_DEFAULT="${WANDB_API_KEY:-}"

usage() {
  cat <<EOF
Usage: $0 <subcommand> [--option value]...

Subcommands:
  train   launch 1-pod × 1-GPU Unsloth training
          --name RUN_NAME              (required) e.g. w7
          --recipe NAME                default huikang (reads train/recipes/NAME.yaml)
          --lr FLOAT                   override recipe lr (default from recipe)
          --steps INT                  override recipe num_steps
          --save-every INT             override recipe save_every
          --seed INT                   override recipe seed
          --node NODENAME              default: pick a free H200 node
          --image REF                  default $DEFAULT_TRAIN_IMAGE
          --wandb|--no-wandb           default --wandb (requires WANDB_API_KEY env)

  eval    launch vLLM eval on balanced_dev726 (default: 1 GPU, TP=1)
          --adapter PATH               (required) path to adapter dir (LoRA safetensors)
          --name OUT_NAME              (required) subdir under artifacts/eval/
          --tp N                       tensor_parallel_size (default 1).
                                       Note: on Nemotron-3-Nano-MoE, TP=8 gave ~0x speedup
                                       vs TP=1 (decode-bound). Stick with TP=1 and run
                                       multiple evals in parallel across nodes.
          --node NODENAME              default: k8s scheduler picks (bin-packs nicely)
EOF
}

die() { echo "error: $*" >&2; exit 2; }

pick_free_node() {
  # Pick an H200 node with 0 GPUs currently in use.
  for n in $(kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-H200 -o jsonpath='{.items[*].metadata.name}'); do
    used=$(kubectl describe node "$n" 2>/dev/null | grep -E "^\s+nvidia.com/gpu\s+" | head -1 | awk '{print $2}')
    if [[ "$used" == "0" ]]; then echo "$n"; return; fi
  done
  die "no free H200 node found"
}

sub="${1:-}"; shift || { usage; exit 0; }

case "$sub" in
  train)
    NAME=""; RECIPE="huikang"; LR=""; STEPS=""; SAVE_EVERY=""; SEED=""
    NODE=""; IMAGE="$DEFAULT_TRAIN_IMAGE"; WANDB="true"
    while (("$#")); do
      case "$1" in
        --name) NAME="$2"; shift 2;;
        --recipe) RECIPE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --save-every) SAVE_EVERY="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --node) NODE="$2"; shift 2;;
        --image) IMAGE="$2"; shift 2;;
        --wandb) WANDB="true"; shift;;
        --no-wandb) WANDB="false"; shift;;
        *) die "unknown flag: $1";;
      esac
    done
    [[ -z "$NAME" ]] && die "--name required"
    RECIPE_PATH="$WS_ROOT/train/recipes/${RECIPE}.yaml"
    [[ ! -f "$RECIPE_PATH" ]] && die "recipe not found: $RECIPE_PATH"
    OUTDIR="$ARTIFACTS/sft/$NAME"
    mkdir -p "$OUTDIR"
    if [[ "$WANDB" == "true" && -z "$WANDB_API_KEY_DEFAULT" ]]; then
      die "--wandb requested but WANDB_API_KEY not in env; either export it or use --no-wandb"
    fi

    K8S_NAME=$(echo "$NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
    export JOB_NAME="ne-train-$K8S_NAME"
    export IMAGE NODE_NAME="$NODE" OUTPUT_DIR="$OUTDIR"
    export RECIPE_PATH RUN_NAME="$NAME"
    # Empty string tells the entry script to use the recipe value.
    export LR NUM_STEPS="$STEPS" SAVE_EVERY SEED
    export WANDB_DISABLED=$([ "$WANDB" == "true" ] && echo "false" || echo "true")
    export WANDB_API_KEY="$WANDB_API_KEY_DEFAULT"
    echo "[submit] train $NAME  recipe=$RECIPE  node=$NODE  out=$OUTDIR"
    echo "[submit] overrides: lr=${LR:-<recipe>} steps=${STEPS:-<recipe>} save_every=${SAVE_EVERY:-<recipe>} seed=${SEED:-<recipe>}"
    if [[ -n "$NODE" ]]; then export NODE_NAME="$NODE"; envsubst < "$HERE/jobs/train_1gpu.yaml" | kubectl apply -f -
    else envsubst < "$HERE/jobs/train_1gpu.yaml" | grep -v 'nodeName:' | kubectl apply -f -; fi
    echo "[submit] watch: kubectl logs -f job/$JOB_NAME"
    ;;

  eval)
    ADAPTER=""; NAME=""; NODE=""; TP=1
    while (("$#")); do
      case "$1" in
        --adapter) ADAPTER="$2"; shift 2;;
        --name) NAME="$2"; shift 2;;
        --node) NODE="$2"; shift 2;;
        --tp) TP="$2"; shift 2;;
        *) die "unknown flag: $1";;
      esac
    done
    [[ -z "$ADAPTER" ]] && die "--adapter required"
    [[ -z "$NAME" ]] && die "--name required"
    [[ ! -d "$ADAPTER" ]] && die "adapter dir not found: $ADAPTER"
    [[ "$TP" =~ ^[0-9]+$ ]] || die "--tp must be an integer"
    OUTDIR="$ARTIFACTS/eval/$NAME"
    mkdir -p "$OUTDIR"
    CFG="$OUTDIR/.eval_config.json"
    cat > "$CFG" <<JSON
{
  "plan": {
    "benchmark_name": "nemo_reasoner",
    "split": "balanced_dev726",
    "checkpoint": {"name": "$NAME", "path": "$ADAPTER", "kind": "adapter", "metadata": {}},
    "config_path": "$CANONICAL_DEV_CSV",
    "output_dir": "$OUTDIR",
    "generation_config": {
      "engine": "vllm",
      "model_path": "/fsx/models/Nemotron-3-Nano-30B-A3B-BF16",
      "temperature": 0.0, "top_p": 1.0,
      "max_tokens": 7680, "max_model_len": 8192, "max_num_seqs": 64, "max_lora_rank": 32,
      "tensor_parallel_size": $TP, "data_parallel_size": 1,
      "gpu_memory_utilization": 0.85, "seed": 0, "enforce_eager": true,
      "dev_path": "$CANONICAL_DEV_CSV",
      "primary_metric_name": "overall_accuracy"
    },
    "metadata": {"source": "submit.sh eval", "tp": $TP}
  },
  "workspace_root": "$WS_ROOT",
  "benchmark_name": "nemo_reasoner",
  "split": "balanced_dev726",
  "out_result_path": "$OUTDIR/.eval_result.json"
}
JSON
    # k8s Job names must be RFC-1123 compliant — underscores not allowed.
    K8S_NAME=$(echo "$NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
    export JOB_NAME="ne-eval-$K8S_NAME"
    export EVAL_CONFIG_PATH="$CFG" NODE_NAME="$NODE" GPU_COUNT="$TP"
    echo "[submit] eval $NAME  tp=$TP  node=$NODE  adapter=$ADAPTER  out=$OUTDIR"
    if [[ -n "$NODE" ]]; then export NODE_NAME="$NODE"; envsubst < "$HERE/jobs/eval_1gpu.yaml" | kubectl apply -f -
    else envsubst < "$HERE/jobs/eval_1gpu.yaml" | grep -v 'nodeName:' | kubectl apply -f -; fi
    echo "[submit] watch: kubectl logs -f job/$JOB_NAME"
    ;;

  ""|-h|--help|help) usage;;
  *) die "unknown subcommand: $sub (try: train, eval)";;
esac
