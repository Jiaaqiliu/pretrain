#!/bin/bash
# Submit all thermodynamics experiment jobs to K8s cluster.
#
# Usage:
#   # Submit measurement jobs only:
#   ./scripts/thermo/submit_all.sh measure
#
#   # Submit training jobs only (190M):
#   ./scripts/thermo/submit_all.sh train-190m
#
#   # Submit training jobs only (1B):
#   ./scripts/thermo/submit_all.sh train-1b
#
#   # Submit everything:
#   ./scripts/thermo/submit_all.sh all
#
#   # Dry run (show commands without executing):
#   ./scripts/thermo/submit_all.sh all --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="${SCRIPT_DIR}/../k8s/thermo"
CONTEXT="arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm"
NAMESPACE="default"

DRY_RUN=false
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN MODE]"
fi

apply_job() {
    local yaml_file="$1"
    local job_name=$(basename "$yaml_file" .yaml)
    echo "  Submitting: $job_name"
    if [ "$DRY_RUN" = false ]; then
        kubectl apply -f "$yaml_file" -n "$NAMESPACE" --context "$CONTEXT"
    else
        echo "    kubectl apply -f $yaml_file -n $NAMESPACE --context $CONTEXT"
    fi
}

submit_measurement() {
    echo ""
    echo "===== MEASUREMENT JOBS (Q1-Q4) ====="
    echo "Submitting checkpoint measurement jobs for all model sizes..."
    echo ""

    for size in 190m 1b 7b 13b; do
        apply_job "${K8S_DIR}/thermo_measure_${size}.yaml"
    done

    echo ""
    echo "4 measurement jobs submitted."
    echo "Monitor: kubectl get pytorchjobs -n $NAMESPACE --context $CONTEXT | grep thermo-measure"
}

submit_train_190m() {
    echo ""
    echo "===== TRAINING JOBS: 190M (Q5) ====="
    echo "4 schedules × 3 seeds = 12 jobs (1 node × 8 GPUs each)"
    echo "Estimated: 12 × ~32h = ~384 GPU-hours total"
    echo ""

    for yaml in "${K8S_DIR}"/thermo_train_190m_*.yaml; do
        apply_job "$yaml"
    done

    echo ""
    echo "12 training jobs (190M) submitted."
}

submit_train_1b() {
    echo ""
    echo "===== TRAINING JOBS: 1B (Q5) ====="
    echo "4 schedules × 3 seeds = 12 jobs (2 nodes × 8 GPUs each)"
    echo "Estimated: 12 × ~128h = ~1536 GPU-hours total"
    echo ""

    for yaml in "${K8S_DIR}"/thermo_train_1b_*.yaml; do
        apply_job "$yaml"
    done

    echo ""
    echo "12 training jobs (1B) submitted."
}

case "${1:-help}" in
    measure)
        submit_measurement
        ;;
    train-190m)
        submit_train_190m
        ;;
    train-1b)
        submit_train_1b
        ;;
    train)
        submit_train_190m
        submit_train_1b
        ;;
    all)
        submit_measurement
        submit_train_190m
        submit_train_1b
        echo ""
        echo "===== TOTAL ====="
        echo "28 jobs submitted (4 measurement + 24 training)"
        echo ""
        echo "Monitor all:"
        echo "  kubectl get pytorchjobs -n $NAMESPACE --context $CONTEXT | grep thermo"
        echo ""
        echo "Once measurement and training complete, run analysis:"
        echo "  python scripts/thermo/run_analysis.py \\"
        echo "    --results-dir /fsx/dev/jiaqi/thermo_results \\"
        echo "    --experiments-dir /fsx/dev/jiaqi/thermo_experiments \\"
        echo "    --output-dir /fsx/dev/jiaqi/thermo_paper_figures"
        ;;
    status)
        echo "Job status:"
        kubectl get pytorchjobs -n "$NAMESPACE" --context "$CONTEXT" | grep thermo || echo "  No thermo jobs found"
        ;;
    logs)
        # Show logs for a specific job
        JOB_NAME="${2:-}"
        if [ -z "$JOB_NAME" ]; then
            echo "Usage: $0 logs <job-name>"
            exit 1
        fi
        kubectl logs "job/$JOB_NAME-master-0" -n "$NAMESPACE" --context "$CONTEXT" --tail=50
        ;;
    help|*)
        echo "Usage: $0 {measure|train-190m|train-1b|train|all|status|logs} [--dry-run]"
        echo ""
        echo "Commands:"
        echo "  measure    - Submit measurement jobs (4 jobs, one per model size)"
        echo "  train-190m - Submit 190M proxy training jobs (12 jobs)"
        echo "  train-1b   - Submit 1B proxy training jobs (12 jobs)"
        echo "  train      - Submit all training jobs (24 jobs)"
        echo "  all        - Submit everything (28 jobs total)"
        echo "  status     - Check job status"
        echo "  logs       - View logs for a specific job"
        echo ""
        echo "Estimated compute:"
        echo "  Measurement:  ~4,000 GPU-hours (parallelizable)"
        echo "  Training 190M: ~768 GPU-hours (12 runs × 256 H100-hours / 4 per run)"
        echo "  Training 1B:   ~6,144 GPU-hours (12 runs × 2048 H100-hours / 4 per run)"
        echo "  Total:         ~10,912 GPU-hours"
        ;;
esac
