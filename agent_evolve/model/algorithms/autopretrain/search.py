"""ProxySearchLoop — main orchestrator for pretraining recipe search.

This is the top-level algorithm that:
1. Maintains a search graph of explored recipes (reusing MCGS infrastructure)
2. Selects parents via UCT
3. Mutates to produce candidate recipes
4. Launches proxy-scale training via OLMoCoreBackend
5. Evaluates checkpoints
6. Computes rewards and backpropagates
7. Tracks the Pareto frontier of {loss, downstream, compute}

The search runs at proxy scale (190M model, 5000 steps per trial) and
produces a ranked list of recipes for transfer verification at 3B scale.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .eval_harness import EvalResult, PretrainEvalHarness
from .mixture import CurriculumSchedule, DataMixture
from .mutator import MixtureMutator, MutationRecord
from .reward import PretrainRewardPolicy
from .workspace_bridge import WorkspaceBridge

logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    """A single node in the recipe search graph."""

    node_id: str
    parent_id: str | None
    curriculum: CurriculumSchedule
    mutation_record: MutationRecord | None = None

    # Results (filled after trial completes)
    eval_result: EvalResult | None = None
    reward: float | None = None

    # MCGS statistics
    visits: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "curriculum": self.curriculum.to_dict(),
            "eval_result": self.eval_result.to_dict() if self.eval_result else None,
            "reward": self.reward,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "mutation": {
                "type": self.mutation_record.mutation_type,
                "description": self.mutation_record.description,
                "reasoning": self.mutation_record.reasoning,
            } if self.mutation_record else None,
        }


@dataclass
class SearchConfig:
    """Configuration for the proxy search loop."""

    # Search parameters
    max_cycles: int = 50
    exploration_c: float = 1.4  # UCT exploration constant

    # Proxy model configuration
    proxy_model_factory: str = "olmo2_190M"
    proxy_steps_per_trial: int = 5000
    proxy_sequence_length: int = 4096
    proxy_batch_size: int = 64  # sequences per step (smaller for fast iteration)
    proxy_gpus: int = 8

    # Data configuration
    data_root: str = "/fsx/dev/jiaqi/data/olmo-3b-pretrain"

    # Eval configuration
    fast_eval: bool = True

    # Output
    output_dir: str = "./autopretrain_search"

    # Starting point
    initial_mixture: str = "olmo2_stage1"  # Reference mixture name


@dataclass
class ProxySearchLoop:
    """Main search loop for pretraining recipe optimization.

    Usage:
        config = SearchConfig(max_cycles=50)
        loop = ProxySearchLoop(config)
        results = loop.run()
        top_recipes = results.top_k(5)
    """

    config: SearchConfig
    mutator: MixtureMutator = field(default_factory=MixtureMutator)
    reward_policy: PretrainRewardPolicy = field(default_factory=PretrainRewardPolicy)
    eval_harness: PretrainEvalHarness = field(default_factory=PretrainEvalHarness)
    workspace_bridge: WorkspaceBridge | None = None
    _backend: Any = None

    # Search state
    nodes: dict[str, SearchNode] = field(default_factory=dict)
    root_id: str | None = None

    # History
    cycle_history: list[dict[str, Any]] = field(default_factory=list)

    def run(self) -> SearchResult:
        """Execute the full search loop.

        Returns:
            SearchResult with ranked recipes and analysis.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize root node
        self._initialize_root()

        logger.info(
            "Starting proxy search: %d cycles, model=%s, steps/trial=%d",
            self.config.max_cycles,
            self.config.proxy_model_factory,
            self.config.proxy_steps_per_trial,
        )

        for cycle in range(self.config.max_cycles):
            t0 = time.time()

            # 1. Select parent via UCT
            parent = self._select_parent()

            # 2. Mutate to produce candidate
            eval_feedback = parent.eval_result.to_dict() if parent.eval_result else None
            child_curriculum, mutation_record = self.mutator.mutate(
                parent.curriculum, eval_feedback, cycle
            )

            # 3. Create child node
            child_id = f"node_{cycle:03d}_{uuid.uuid4().hex[:6]}"
            child = SearchNode(
                node_id=child_id,
                parent_id=parent.node_id,
                curriculum=child_curriculum,
                mutation_record=mutation_record,
            )
            self.nodes[child_id] = child

            # 4. Run proxy trial
            logger.info(
                "[Cycle %d/%d] %s → %s | %s",
                cycle + 1, self.config.max_cycles,
                parent.node_id[:10], child_id[:10],
                mutation_record.description,
            )
            eval_result = self._run_trial(child)
            child.eval_result = eval_result

            # 5. Compute reward
            parent_eval = parent.eval_result
            reward = self.reward_policy.compute_reward(
                eval_result, child_curriculum, parent_eval
            )
            child.reward = reward
            self.reward_policy.update_bounds(eval_result.val_loss)

            # 6. Backpropagate
            self._backpropagate(child_id, reward)

            # 7. Log
            elapsed = time.time() - t0
            cycle_record = {
                "cycle": cycle,
                "node_id": child_id,
                "parent_id": parent.node_id,
                "mutation": mutation_record.description,
                "val_loss": eval_result.val_loss,
                "reward": reward,
                "elapsed_seconds": elapsed,
            }
            self.cycle_history.append(cycle_record)

            logger.info(
                "  → loss=%.4f, reward=%.3f, time=%.0fs",
                eval_result.val_loss, reward, elapsed,
            )

            # 8. Save checkpoint
            if (cycle + 1) % 10 == 0:
                self._save_checkpoint(output_dir, cycle)

        # Final save
        self._save_checkpoint(output_dir, self.config.max_cycles - 1)

        return self._build_result()

    def _initialize_root(self):
        """Create root node from initial mixture."""
        initial = DataMixture.from_reference(self.config.initial_mixture)
        root_curriculum = CurriculumSchedule.static(initial)
        root_id = "root"

        root = SearchNode(
            node_id=root_id,
            parent_id=None,
            curriculum=root_curriculum,
        )

        # Run initial evaluation
        root.eval_result = self._run_trial(root)
        root.reward = self.reward_policy.compute_reward(root.eval_result, root_curriculum)
        root.visits = 1
        root.total_reward = root.reward

        self.nodes[root_id] = root
        self.root_id = root_id
        logger.info("Root initialized: loss=%.4f, reward=%.3f", root.eval_result.val_loss, root.reward)

    def _select_parent(self) -> SearchNode:
        """Select parent node via UCT (Upper Confidence Bound for Trees)."""
        import math

        total_visits = sum(n.visits for n in self.nodes.values())

        best_node = None
        best_uct = -float("inf")

        for node in self.nodes.values():
            if node.visits == 0:
                continue

            exploitation = node.mean_reward
            exploration = self.config.exploration_c * math.sqrt(
                math.log(total_visits + 1) / node.visits
            )
            uct = exploitation + exploration

            if uct > best_uct:
                best_uct = uct
                best_node = node

        return best_node or self.nodes[self.root_id]

    def _run_trial(self, node: SearchNode) -> EvalResult:
        """Run a proxy-scale training trial for a given recipe.

        Pipeline:
        1. WorkspaceBridge creates a workspace directory from the curriculum
        2. OLMoCoreBackend translates workspace → script → launches training
        3. EvalHarness reads metrics from the completed trial
        """
        bridge = self._get_workspace_bridge()
        trial_dir = bridge.create_trial_workspace(
            node_id=node.node_id,
            curriculum=node.curriculum,
            model_factory=self.config.proxy_model_factory,
            max_steps=self.config.proxy_steps_per_trial,
            batch_size=self.config.proxy_batch_size,
            sequence_length=self.config.proxy_sequence_length,
        )

        backend = self._get_backend()
        if backend is not None:
            from agent_evolve.model.types import TrainingSearchNode, TrialBudget

            search_node = TrainingSearchNode(
                node_id=node.node_id,
                parent_id=node.parent_id,
                branch_id=0,
            )
            budget = TrialBudget(
                steps=self.config.proxy_steps_per_trial,
                seconds=3600.0,
            )

            # Create a minimal workspace object with .root attribute
            class _Workspace:
                def __init__(self, root): self.root = root
            workspace = _Workspace(trial_dir)

            trial_result = backend.run_trial(workspace, search_node, budget, benchmark=None)

            if trial_result.status == "success" and trial_result.train_metrics:
                loss = trial_result.train_metrics.get(
                    "final_loss", trial_result.train_metrics.get("loss", 4.0)
                )
                return EvalResult(
                    val_loss=float(loss),
                    val_perplexity=2 ** float(loss),
                    domain_losses={"overall": float(loss)},
                    composite_score=-float(loss),
                    checkpoint_path=trial_result.checkpoint.path if trial_result.checkpoint else "",
                    step=self.config.proxy_steps_per_trial,
                )
            else:
                logger.warning(
                    "Trial %s failed (status=%s): %s",
                    node.node_id, trial_result.status, trial_result.train_metrics.get("error", ""),
                )

        # Fallback: use eval_harness to read metrics from trial_dir
        return self.eval_harness.evaluate(
            checkpoint_path=str(trial_dir / "checkpoints"),
            step=self.config.proxy_steps_per_trial,
            trial_dir=trial_dir,
        )

    def _get_workspace_bridge(self) -> WorkspaceBridge:
        if self.workspace_bridge is None:
            self.workspace_bridge = WorkspaceBridge(
                trials_root=Path(self.config.output_dir) / "trials",
                data_root=self.config.data_root,
            )
        return self.workspace_bridge

    def _get_backend(self):
        """Lazy-initialize the OLMoCoreBackend.

        Uses mock=True if torchrun is not available (local dev without GPU).
        """
        if self._backend is None:
            try:
                import shutil
                from agent_evolve.backends.olmo_core import OLMoCoreBackend
                has_torchrun = shutil.which("torchrun") is not None
                self._backend = OLMoCoreBackend(
                    gpus_per_node=self.config.proxy_gpus,
                    num_nodes=1,
                    mock=not has_torchrun,
                )
                mode = "real" if has_torchrun else "mock"
                logger.info("OLMoCoreBackend initialized (%s, gpus=%d)", mode, self.config.proxy_gpus)
            except ImportError:
                logger.warning("OLMoCoreBackend not available, running in eval-only mode")
                self._backend = None
        return self._backend

    def _backpropagate(self, node_id: str, reward: float):
        """Backpropagate reward up the tree."""
        current = node_id
        while current is not None:
            node = self.nodes[current]
            node.visits += 1
            node.total_reward += reward
            current = node.parent_id

    def _save_checkpoint(self, output_dir: Path, cycle: int):
        """Save search state to disk."""
        state = {
            "cycle": cycle,
            "config": {
                "max_cycles": self.config.max_cycles,
                "proxy_model_factory": self.config.proxy_model_factory,
                "proxy_steps_per_trial": self.config.proxy_steps_per_trial,
            },
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "history": self.cycle_history,
        }
        path = output_dir / f"search_state_cycle_{cycle:03d}.json"
        path.write_text(json.dumps(state, indent=2, default=str))
        logger.info("Saved search state to %s", path)

    def _build_result(self) -> SearchResult:
        """Build final result with ranked recipes."""
        ranked = sorted(
            [n for n in self.nodes.values() if n.eval_result is not None],
            key=lambda n: n.eval_result.val_loss,
        )
        return SearchResult(
            ranked_nodes=ranked,
            total_cycles=len(self.cycle_history),
            best_loss=ranked[0].eval_result.val_loss if ranked else float("inf"),
            history=self.cycle_history,
        )


@dataclass
class SearchResult:
    """Final result of a proxy search run."""

    ranked_nodes: list[SearchNode]
    total_cycles: int
    best_loss: float
    history: list[dict[str, Any]]

    def top_k(self, k: int = 5) -> list[SearchNode]:
        """Return top-k recipes by validation loss."""
        return self.ranked_nodes[:k]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Search complete: {self.total_cycles} cycles, {len(self.ranked_nodes)} nodes",
            f"Best val loss: {self.best_loss:.4f}",
            "",
            "Top 5 recipes:",
        ]
        for i, node in enumerate(self.top_k(5)):
            lines.append(
                f"  {i+1}. {node.node_id}: loss={node.eval_result.val_loss:.4f}, "
                f"reward={node.reward:.3f}"
            )
            if node.mutation_record:
                lines.append(f"     Mutation: {node.mutation_record.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "best_loss": self.best_loss,
            "top_5": [n.to_dict() for n in self.top_k(5)],
            "history": self.history,
        }
