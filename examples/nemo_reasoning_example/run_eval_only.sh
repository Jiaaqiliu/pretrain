#!/usr/bin/env bash
# Eval-only cycle: scores an existing LoRA adapter on the 951-row Kaggle dev
# split via vLLM. The seed workspace's ``model/adapter.yaml::seed_adapter_path``
# picks which adapter to evaluate (default: ../nemotron-auto-research E-28).
#
# If seed_adapter_path is commented out in adapter.yaml, this cycle would
# trigger real training instead — UNcomment the two seed_adapter_* lines in
# ``seed_workspaces/nemotron_reasoner/model/adapter.yaml`` before running
# this, or the eval-only path won't activate.
#
# Wallclock: ~10 min on 1x H200. vLLM load ~2 min + 951-row generation ~8 min.
#
# To re-run cleanly:
#   rm -rf runs/eval-only/nemotron_reasoner runs/eval-only/nodes

set -euo pipefail

AE=/fsx/zzsamshi/a-evolve
NEMOTRON_VENV=/fsx/zzsamshi/nemotron-auto-research/.venv

cd "$AE"
mkdir -p runs/eval-only/logs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  PYTHONPATH="$AE" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  WANDB_DISABLED=true \
  "$NEMOTRON_VENV/bin/python" \
  "$AE/examples/nemo_reasoning_example/drive_eval_only.py" \
  > runs/eval-only/logs/run.log 2>&1 &

echo "pid=$!"
echo "tail: tail -f $AE/runs/eval-only/logs/run.log"
