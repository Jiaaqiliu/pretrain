#!/usr/bin/env bash
# Smoke test: validate MCGS wiring for posttrain_bench pipeline.
# No GPU needed. Runs in a few seconds.
#
# To re-run cleanly:
#   rm -rf runs/posttrain-lr-sweep-smoke

set -euo pipefail

AE="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$AE"
mkdir -p runs/posttrain-lr-sweep-smoke/logs

PYTHONPATH="$AE" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  WANDB_DISABLED=true \
  python "$AE/examples/posttrain_bench/drive_lr_sweep_smoke.py" \
  2>&1 | tee runs/posttrain-lr-sweep-smoke/logs/run.log

echo "smoke exit=$?"
echo "log: $AE/runs/posttrain-lr-sweep-smoke/logs/run.log"
