#!/usr/bin/env bash
# 4-cycle real-SFT LR sweep on the Kaggle Nemotron-Reasoning benchmark.
#
# Each cycle forks a fresh sibling from root, trains a rank-16 LoRA adapter
# from /fsx/models/Nemotron-3-Nano-30B-A3B-BF16 at one of
# [1e-4, 5e-5, 3e-5, 1e-5], then evaluates on the 951-row dev split.
# MCGS selects the best-of-4 as incumbent. Wallclock ~2h 54m on one H200.
#
# Notes:
#   * Uses the nemotron-auto-research venv because the a-evolve default venv
#     lacks peft / transformers / accelerate / vllm / mamba-ssm shim.
#   * PYTHONPATH is set so `import agent_evolve` resolves to this repo while
#     the nemotron venv's site-packages provides the training stack.
#   * HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE keep `trust_remote_code` from
#     stalling on a network fetch.
#   * WANDB_DISABLED=true is belt-and-braces; TrainingArguments already sets
#     report_to="none".
#
# To re-run cleanly:
#   rm -rf runs/lr-sweep-4cycle/nemotron_reasoner runs/lr-sweep-4cycle/nodes
#   mv runs/lr-sweep-4cycle/logs/run.log runs/lr-sweep-4cycle/logs/run.$(date +%s).log

set -euo pipefail

AE=/fsx/zzsamshi/a-evolve
NEMOTRON_VENV=/fsx/zzsamshi/nemotron-auto-research/.venv

cd "$AE"
mkdir -p runs/lr-sweep-4cycle/logs

CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$AE" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  WANDB_DISABLED=true \
  "$NEMOTRON_VENV/bin/python" \
  "$AE/examples/nemo_reasoning_example/drive_lr_sweep_4cycle.py" \
  > runs/lr-sweep-4cycle/logs/run.log 2>&1 &

echo "pid=$!"
echo "tail: tail -f $AE/runs/lr-sweep-4cycle/logs/run.log"
