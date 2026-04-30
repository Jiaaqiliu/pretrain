#!/bin/bash
# Full optimization pipeline for MLE-Bench AutoML
#
# This script runs the complete optimization workflow:
# 1. 20-cycle advanced search with feature engineering
# 2. Create ensemble from top-5 models
# 3. Compare results

set -e  # Exit on error

echo "============================================================"
echo "=== MLE-Bench AutoML - Full Optimization Pipeline ==="
echo "============================================================"
echo ""
echo "This will run:"
echo "  1. 20-cycle advanced search (~20-30 minutes)"
echo "  2. Create top-5 ensemble"
echo "  3. Compare results"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Step 1: Run 20-cycle advanced search
echo ""
echo "===================================================="
echo "Step 1: Running 20-cycle advanced search..."
echo "===================================================="
echo ""

python examples/mle_automl_example/drive_advanced_search_20cycle.py

# Check if search completed
if [ $? -ne 0 ]; then
    echo "Error: 20-cycle search failed"
    exit 1
fi

echo ""
echo "✓ 20-cycle search completed"
echo ""

# Step 2: Create ensemble
echo "===================================================="
echo "Step 2: Creating ensemble from top-5 models..."
echo "===================================================="
echo ""

python examples/mle_automl_example/create_ensemble_submission.py \
    --graph runs/mle-automl-advanced-20cycles/mle_automl/evolution/mcgs_graph.json \
    --k 5 \
    --strategy voting \
    --output runs/mle-automl-advanced-20cycles/ensemble_submission.csv

if [ $? -ne 0 ]; then
    echo "Warning: Ensemble creation had issues, but continuing..."
fi

echo ""
echo "✓ Ensemble created"
echo ""

# Step 3: Summary
echo "===================================================="
echo "=== Optimization Complete! ==="
echo "===================================================="
echo ""
echo "Results saved to:"
echo "  - Graph: runs/mle-automl-advanced-20cycles/mle_automl/evolution/mcgs_graph.json"
echo "  - Reports: runs/mle-automl-advanced-20cycles/mle_automl/evolution/reports/"
echo "  - Ensemble: runs/mle-automl-advanced-20cycles/ensemble_submission.csv"
echo ""
echo "To view results:"
echo "  cat runs/mle-automl-advanced-20cycles/mle_automl/evolution/reports/cycle_0020.json | jq"
echo ""
echo "To compare with baseline (4-cycle):"
echo "  # Baseline: 0.77816"
echo "  # Check new score in the final report"
echo ""

exit 0
