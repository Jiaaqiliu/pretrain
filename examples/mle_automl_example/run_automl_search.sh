#!/usr/bin/env bash
# Run AutoML search on MLE-Bench task using TrainingEvolver

set -euo pipefail

AE=/home/ec2-user/fsx/yisi/A-EVOLVE-V2
VENV=/home/ec2-user/.venv  # Update to your venv

cd "$AE"
mkdir -p runs/mle-automl-search/logs

# Prepare MLE-Bench data first
echo "=== Preparing MLE-Bench Data ==="
COMPETITION_ID="spaceship-titanic"  # Change to your competition
mlebench prepare --competition "$COMPETITION_ID" || echo "Data may already be prepared"

# Copy prepared data to workspace
echo "=== Copying Data to Workspace ==="
MLEBENCH_DATA=~/.cache/mlebench/competitions/$COMPETITION_ID/prepared/public
WORKSPACE_DATA="$AE/seed_workspaces/mle_automl/data"

if [ -d "$MLEBENCH_DATA" ]; then
    cp "$MLEBENCH_DATA/train.csv" "$WORKSPACE_DATA/" || true
    cp "$MLEBENCH_DATA/test.csv" "$WORKSPACE_DATA/" || true
    echo "  ✓ Data copied to $WORKSPACE_DATA"
else
    echo "  ⚠ MLE-Bench data not found at $MLEBENCH_DATA"
    echo "  Run: mlebench prepare --competition $COMPETITION_ID"
    exit 1
fi

# Update workspace config with competition ID
echo "=== Updating Workspace Config ==="
sed -i "s/competition_id:.*/competition_id: $COMPETITION_ID/" \
    "$AE/seed_workspaces/mle_automl/model/config.yaml"

# Run AutoML search
echo "=== Running AutoML Search ==="
PYTHONPATH="$AE" \
  WANDB_DISABLED=true \
  "$VENV/bin/python" \
  "$AE/examples/mle_automl_example/drive_model_search_4cycle.py" \
  > runs/mle-automl-search/logs/run.log 2>&1 &

echo "pid=$!"
echo "tail: tail -f $AE/runs/mle-automl-search/logs/run.log"
echo ""
echo "After completion, check best model at:"
echo "  $AE/runs/mle-automl-search/mle_automl/evolution/incumbent/"
