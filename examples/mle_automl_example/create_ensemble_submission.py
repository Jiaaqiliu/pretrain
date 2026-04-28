#!/usr/bin/env python3
"""Create ensemble submission from top-k models in MCGS graph.

This script loads the top k models from a completed search and creates
an ensemble prediction that typically outperforms any single model.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from agent_evolve.backends.ensemble import create_topk_ensemble


def main():
    parser = argparse.ArgumentParser(description="Create ensemble submission")
    parser.add_argument(
        "--graph",
        type=str,
        default="runs/mle-automl-advanced-20cycles/mle_automl/evolution/mcgs_graph.json",
        help="Path to MCGS graph JSON"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top models to ensemble"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="voting",
        choices=["voting", "averaging", "stacking"],
        help="Ensemble strategy"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="seed_workspaces/mle_automl/data/test.csv",
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ensemble_submission.csv",
        help="Output submission file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("=== Creating Ensemble Submission ===")
    print("=" * 60)
    print(f"Graph: {args.graph}")
    print(f"Top-k: {args.k}")
    print(f"Strategy: {args.strategy}")
    print()

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    print(f"  Test samples: {len(test_df)}")

    # Extract IDs
    id_col = "PassengerId"
    test_ids = test_df[id_col].values if id_col in test_df.columns else test_df.index.values
    X_test_raw = test_df.drop(columns=[id_col] if id_col in test_df.columns else [])

    # Load feature engineer from the best model's checkpoint
    # (Feature engineer must be fitted on training data)
    import json
    import pickle

    with open(args.graph) as f:
        graph = json.load(f)

    # Find best node
    nodes = graph.get("nodes", [])
    valid_nodes = [n for n in nodes if n.get("metric") is not None]
    if not valid_nodes:
        print("Error: No valid nodes found in graph")
        return 1

    best_node = max(valid_nodes, key=lambda n: n["metric"])
    checkpoint_info = best_node.get("checkpoint")

    if not checkpoint_info or not checkpoint_info.get("path"):
        print("Error: Best node has no checkpoint path")
        return 1

    best_checkpoint_path = checkpoint_info["path"]

    # Load feature engineer from checkpoint
    fe_path = Path(best_checkpoint_path) / "feature_engineer.pkl"
    if fe_path.exists():
        print(f"Loading feature engineer from {fe_path}")
        with open(fe_path, "rb") as f:
            fe = pickle.load(f)

        print(f"Applying advanced feature engineering...")
        X_test = fe.transform(X_test_raw)
        print(f"  Features: {X_test.shape[1]}")
    else:
        print("Warning: No feature engineer found, using basic FE")
        from sklearn.preprocessing import LabelEncoder
        X_test = X_test_raw.copy()

        # Basic FE
        numeric_cols = X_test.select_dtypes(include=['number']).columns
        categorical_cols = X_test.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0:
            X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].median())

        for col in categorical_cols:
            X_test[col] = X_test[col].fillna('Unknown')
            le = LabelEncoder()
            X_test[col] = le.fit_transform(X_test[col].astype(str))

    # Create ensemble
    print(f"\nCreating top-{args.k} ensemble with {args.strategy} strategy...")
    graph_path = Path(args.graph)
    ensemble = create_topk_ensemble(
        graph_path=graph_path,
        k=args.k,
        strategy=args.strategy
    )

    # Generate predictions
    print(f"\nGenerating ensemble predictions...")
    if args.strategy in ["voting", "averaging"]:
        predictions = ensemble.predict_proba(X_test)
        # Convert to binary predictions
        pred_labels = (predictions[:, 1] > 0.5).astype(bool)
    else:
        pred_labels = ensemble.predict(X_test)

    # Create submission
    submission_df = pd.DataFrame({
        id_col: test_ids,
        "Transported": pred_labels,
    })

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"\n✓ Ensemble submission saved to: {output_path}")
    print(f"  Samples: {len(submission_df)}")
    print(f"  Columns: {list(submission_df.columns)}")

    # Preview
    print(f"\nPreview (first 5 rows):")
    print(submission_df.head().to_string(index=False))

    # Grade if possible
    try:
        from mlebench.registry import registry
        from mlebench.utils import load_answers, read_csv

        print(f"\nGrading ensemble submission...")
        competition_id = "spaceship-titanic"
        competition = registry.get_competition(competition_id)
        submission_data = read_csv(output_path)
        answers = load_answers(competition.answers)

        score = competition.grader(submission_data, answers)
        print(f"  Ensemble score: {score:.5f}")

        # Compare with best single model
        with open(graph_path) as f:
            graph = json.load(f)

        nodes = graph.get("nodes", [])
        valid_nodes = [n for n in nodes if n.get("metric") is not None]
        if valid_nodes:
            best_single = max(valid_nodes, key=lambda n: n["metric"])
            best_single_score = best_single["metric"]
            improvement = score - best_single_score

            print(f"\nComparison:")
            print(f"  Best single model: {best_single_score:.5f}")
            print(f"  Ensemble:          {score:.5f}")
            print(f"  Improvement:       {improvement:+.5f} ({(improvement/best_single_score)*100:+.2f}%)")

    except Exception as e:
        print(f"\nCould not grade submission: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
