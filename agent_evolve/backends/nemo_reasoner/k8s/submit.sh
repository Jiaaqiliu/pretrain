#!/usr/bin/env bash
# Canonical CLI for submitting train + eval jobs to k8s — benchmark = nemo_reasoner.
#
# Contract: every path is explicit. Callers (agents, orchestrators) pass
# fork-rooted absolute paths. No implicit workspace resolution — this lets
# multiple forked workspaces submit jobs without colliding.
#
# Usage:
#   submit.sh train  --train-recipe PATH --data-recipe PATH --out DIR --name NAME
#                    [--lr FLOAT] [--steps INT] [--save-every INT] [--seed INT]
#                    [--node NODENAME] [--image REF] [--wandb|--no-wandb]
#
#   submit.sh eval   --adapter PATH --out DIR --name NAME
#                    [--tp INT] [--node NODENAME]
#
# Examples:
#   FORK=/path/to/fork/workspace
#   ./submit.sh train \
#       --train-recipe $FORK/recipes/train/default.yaml \
#       --data-recipe  $FORK/recipes/data/default_data.yaml \
#       --out          $FORK/artifacts/sft/w7 \
#       --name         w7
#
#   ./submit.sh eval \
#       --adapter $FORK/artifacts/sft/w7/step_200 \
#       --out     $FORK/artifacts/eval/w7_step200 \
#       --name    w7_step200
#
# Outputs:
#   train → $OUT/step_{N}/, $OUT/final/, $OUT/train.log
#   eval  → $OUT/metrics.json, $OUT/predictions.jsonl, $OUT/eval.log
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
# Benchmark scorer reads its split registry + kaggle_eval.yaml from the
# nemo_mas_reasoner seed workspace. ``workspace_root`` in the eval plan
# must point here, not at the repo root.
SEED_WORKSPACE="$REPO_ROOT/seed_workspaces/nemo_mas_reasoner"
CANONICAL_DEV_CSV=/fsx/zzsamshi/nemotron-auto-research/data/nemo_reasoner/dev/balanced_dev726.csv
DEFAULT_TRAIN_IMAGE=801953956576.dkr.ecr.ap-southeast-3.amazonaws.com/zzsamshi/a-evolve:unsloth-v5
WANDB_API_KEY_DEFAULT="${WANDB_API_KEY:-}"

usage() {
  cat <<EOF
Usage: $0 <subcommand> [--option value]...

Subcommands:
  train   launch 1-pod × 1-GPU Unsloth training
          --train-recipe PATH          (required) absolute path to recipes/train/<name>.yaml
          --data-recipe PATH           (optional) absolute path to recipes/data/<name>.yaml
                                       (informational; training data path lives inside train recipe)
          --out DIR                    (required) absolute path to artifacts/sft/<run_name>/
          --name RUN_NAME              (required) e.g. w7 — used for job name + wandb name
          --lr FLOAT                   override recipe lr
          --steps INT                  override recipe num_steps
          --save-every INT             override recipe save_every
          --seed INT                   override recipe seed
          --node NODENAME              default: k8s scheduler picks
          --image REF                  default $DEFAULT_TRAIN_IMAGE
          --wandb|--no-wandb           default --wandb (requires WANDB_API_KEY env)

  eval    launch vLLM eval on balanced_dev726 (default: 1 GPU, TP=1)
          --adapter PATH               (required) absolute path to adapter dir
          --out DIR                    (required) absolute path to artifacts/eval/<run_name>/
          --name RUN_NAME              (required) subdir leaf; used for job name
          --tp N                       tensor_parallel_size (default 1).
                                       Note: Nemotron-3-Nano-MoE is decode-bound — TP=8 gave ~0x
                                       speedup vs TP=1. Prefer TP=1 + parallel runs across nodes.
          --node NODENAME              default: k8s scheduler picks (bin-packs)
EOF
}

die() { echo "error: $*" >&2; exit 2; }

sub="${1:-}"; shift || { usage; exit 0; }

case "$sub" in
  train)
    TRAIN_RECIPE=""; DATA_RECIPE=""; OUTDIR=""; NAME=""
    LR=""; STEPS=""; SAVE_EVERY=""; SEED=""
    NODE=""; IMAGE="$DEFAULT_TRAIN_IMAGE"; WANDB="true"
    while (("$#")); do
      case "$1" in
        --train-recipe) TRAIN_RECIPE="$2"; shift 2;;
        --data-recipe)  DATA_RECIPE="$2"; shift 2;;
        --out)          OUTDIR="$2"; shift 2;;
        --name)         NAME="$2"; shift 2;;
        --lr)           LR="$2"; shift 2;;
        --steps)        STEPS="$2"; shift 2;;
        --save-every)   SAVE_EVERY="$2"; shift 2;;
        --seed)         SEED="$2"; shift 2;;
        --node)         NODE="$2"; shift 2;;
        --image)        IMAGE="$2"; shift 2;;
        --wandb)        WANDB="true"; shift;;
        --no-wandb)     WANDB="false"; shift;;
        *) die "unknown flag: $1";;
      esac
    done
    [[ -z "$TRAIN_RECIPE" ]] && die "--train-recipe required (absolute path)"
    [[ -z "$OUTDIR" ]]       && die "--out required (absolute path)"
    [[ -z "$NAME" ]]         && die "--name required"
    [[ ! -f "$TRAIN_RECIPE" ]] && die "train recipe not found: $TRAIN_RECIPE"
    [[ -n "$DATA_RECIPE" && ! -f "$DATA_RECIPE" ]] && die "data recipe not found: $DATA_RECIPE"
    [[ "$OUTDIR" = /* ]]     || die "--out must be an absolute path: $OUTDIR"
    mkdir -p "$OUTDIR"
    if [[ "$WANDB" == "true" && -z "$WANDB_API_KEY_DEFAULT" ]]; then
      die "--wandb requested but WANDB_API_KEY not in env; either export it or use --no-wandb"
    fi

    K8S_NAME=$(echo "$NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
    export JOB_NAME="ne-train-$K8S_NAME"
    export IMAGE NODE_NAME="$NODE" OUTPUT_DIR="$OUTDIR"
    export RECIPE_PATH="$TRAIN_RECIPE" RUN_NAME="$NAME"
    export LR NUM_STEPS="$STEPS" SAVE_EVERY SEED
    export WANDB_DISABLED=$([ "$WANDB" == "true" ] && echo "false" || echo "true")
    export WANDB_API_KEY="$WANDB_API_KEY_DEFAULT"
    echo "[submit] train $NAME"
    echo "[submit]   train_recipe=$TRAIN_RECIPE"
    echo "[submit]   data_recipe=${DATA_RECIPE:-<none>}"
    echo "[submit]   out=$OUTDIR  node=${NODE:-<scheduler>}"
    echo "[submit]   overrides: lr=${LR:-<recipe>} steps=${STEPS:-<recipe>} save_every=${SAVE_EVERY:-<recipe>} seed=${SEED:-<recipe>}"
    if [[ -n "$NODE" ]]; then envsubst < "$HERE/jobs/train_1gpu.yaml" | kubectl apply -f -
    else envsubst < "$HERE/jobs/train_1gpu.yaml" | sed '/nodeSelector:/,/kubernetes.io\/hostname:/d' | kubectl apply -f -; fi
    echo "[submit] watch: kubectl logs -f job/$JOB_NAME"
    ;;

  eval)
    ADAPTER=""; OUTDIR=""; NAME=""; NODE=""; TP=1
    while (("$#")); do
      case "$1" in
        --adapter) ADAPTER="$2"; shift 2;;
        --out)     OUTDIR="$2"; shift 2;;
        --name)    NAME="$2"; shift 2;;
        --node)    NODE="$2"; shift 2;;
        --tp)      TP="$2"; shift 2;;
        *) die "unknown flag: $1";;
      esac
    done
    [[ -z "$ADAPTER" ]]      && die "--adapter required"
    [[ -z "$OUTDIR" ]]       && die "--out required (absolute path)"
    [[ -z "$NAME" ]]         && die "--name required"
    [[ ! -d "$ADAPTER" ]]    && die "adapter dir not found: $ADAPTER"
    [[ "$OUTDIR" = /* ]]     || die "--out must be an absolute path: $OUTDIR"
    [[ "$TP" =~ ^[0-9]+$ ]]  || die "--tp must be an integer"
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
  "workspace_root": "$SEED_WORKSPACE",
  "benchmark_name": "nemo_reasoner",
  "split": "balanced_dev726",
  "out_result_path": "$OUTDIR/.eval_result.json"
}
JSON
    K8S_NAME=$(echo "$NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
    export JOB_NAME="ne-eval-$K8S_NAME"
    export EVAL_CONFIG_PATH="$CFG" NODE_NAME="$NODE" GPU_COUNT="$TP"
    echo "[submit] eval $NAME  tp=$TP  adapter=$ADAPTER  out=$OUTDIR  node=${NODE:-<scheduler>}"
    if [[ -n "$NODE" ]]; then envsubst < "$HERE/jobs/eval_1gpu.yaml" | kubectl apply -f -
    else envsubst < "$HERE/jobs/eval_1gpu.yaml" | sed '/nodeSelector:/,/kubernetes.io\/hostname:/d' | kubectl apply -f -; fi
    echo "[submit] watch: kubectl logs -f job/$JOB_NAME"
    ;;

  ""|-h|--help|help) usage;;
  *) die "unknown subcommand: $sub (try: train, eval)";;
esac
