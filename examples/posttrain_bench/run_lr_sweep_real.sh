#!/usr/bin/env bash
# Real 4-cycle LR sweep for posttrain_bench SFT inside Docker.
# Uses DeepSeek-R1-Distill-Qwen-1.5B, full-param SFT, 8x H200.
#
# To re-run cleanly:
#   rm -rf runs/posttrain-lr-sweep-real

set -euo pipefail

AE="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$AE"
mkdir -p runs/posttrain-lr-sweep-real/logs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTHONPATH="$AE" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  WANDB_DISABLED=true \
  NCCL_NVLS_ENABLE=0 \
  python "$AE/examples/posttrain_bench/drive_lr_sweep_real.py" \
  2>&1 | tee runs/posttrain-lr-sweep-real/logs/run.log

echo "exit=$?"
echo "log: $AE/runs/posttrain-lr-sweep-real/logs/run.log"
