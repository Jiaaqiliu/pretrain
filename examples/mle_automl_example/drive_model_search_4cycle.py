"""4-cycle AutoML search on MLE-Bench using TrainingEvolver.

This demonstrates using TrainingEvolver as an AutoML framework:
- No LLM training involved
- MCGS searches over ML model hyperparameters
- Evaluates directly on Kaggle tasks from MLE-Bench

Each cycle:
  1. Select parent configuration (model type + hyperparameters)
  2. Mutate: change model type OR hyperparameters
  3. Train: fit ML model (XGBoost/RandomForest/LightGBM) on training data
  4. Evaluate: predict on test set, grade with MLE-Bench
  5. MCGS: compute reward, update graph, promote best configuration
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Disable wandb, HF offline mode not needed for sklearn
os.environ.setdefault("WANDB_DISABLED", "true")

AE = Path("/home/ec2-user/fsx/yisi/A-EVOLVE-V2")
sys.path.insert(0, str(AE))

from agent_evolve.model.algorithms.mcgs.ml_mutation import MLModelTypeMutationProposer  # noqa: E402
from agent_evolve.model.algorithms.mcgs.search import MCGSSearch  # noqa: E402
from agent_evolve.model.api import TrainingEvolver  # noqa: E402
from agent_evolve.model.types import TrainingEvolveConfig  # noqa: E402


class RootFanoutSelector:
    """Forces first `fanout` cycles to pick root as parent.

    Ensures we get independent siblings testing different models,
    rather than a chain from UCT.
    """

    def __init__(self, fanout: int = 4) -> None:
        self.fanout = fanout

    def select(self, graph, *, cycle: int):  # noqa: ARG002
        root = graph.root()
        assert root is not None
        direct_children = [
            n for n in graph.nodes.values() if n.parent_id == root.node_id
        ]
        if len(direct_children) < self.fanout:
            return root
        # Fallback: pick best child
        return max(
            direct_children,
            key=lambda n: (n.mean_reward, n.metric or float("-inf")),
        )


def main() -> int:
    """Run 4-cycle AutoML search.

    Tests 4 different model types/configurations:
    1. XGBoost
    2. LightGBM
    3. Random Forest
    4. XGBoost with different hyperparameters
    """
    print("=== AutoML Search on MLE-Bench ===")
    print("Using TrainingEvolver as AutoML framework")
    print("Searching over: XGBoost, LightGBM, Random Forest")
    print()

    # MCGS with model type rotation
    algo = MCGSSearch(
        mutator=MLModelTypeMutationProposer(
            model_types=("xgboost", "lightgbm", "random_forest", "xgboost")
        ),
        selector=RootFanoutSelector(fanout=4),
    )

    evolver = TrainingEvolver(
        workspace=AE / "seed_workspaces" / "mle_automl",
        benchmark="mle_bench",
        algorithm=algo,
        backend="sklearn_backend",
        config=TrainingEvolveConfig(
            smoke=False,  # Use real ML training
            max_cycles=4,
            trial_budget_seconds=600,  # 10 min per cycle (ML training is fast)
        ),
        work_dir=AE / "runs" / "mle-automl-search",
    )

    result = evolver.run(cycles=4)

    print("\n=== Final Result ===")
    print(f"cycles_completed: {result.cycles_completed}")
    print(f"incumbent_node_id: {result.incumbent_node_id}")
    print(f"best_metric: {result.best_metric}")
    print(f"graph_path: {result.graph_path}")
    print(f"report_path: {result.report_path}")

    for entry in result.topk:
        print(
            f"  topk: node={entry.node_id} branch={entry.branch_id} "
            f"metric={entry.metric:.4f} reward={entry.reward:.4f}"
        )

    print("\nBest configuration saved to:")
    from pathlib import Path
    print(f"  {Path(result.graph_path).parent / 'incumbent' / 'model' / 'config.yaml'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
