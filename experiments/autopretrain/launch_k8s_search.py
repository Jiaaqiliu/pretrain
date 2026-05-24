"""Launch proxy search on K8s cluster.

Orchestrates the MCGS search loop where each trial is a K8s PyTorchJob.
Unlike the single-process mock mode, this runs trials on the actual
H200 cluster and collects results asynchronously.

Architecture:
    This script (runs on devbox/laptop) → submits K8s jobs → polls for completion
    → reads metrics from FSx → decides next trial → repeat

Usage:
    python experiments/autopretrain/launch_k8s_search.py \
        --cycles 50 \
        --output-dir /fsx/dev/jiaqi/experiments/autopretrain_search_001

Prerequisites:
    - kubectl configured for the p5-llm cluster
    - FSx mounted or accessible
    - Data prepared at /fsx/dev/jiaqi/data/olmo-3b-pretrain/
"""

import argparse
import json
import logging
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_evolve.model.algorithms.autopretrain.search import ProxySearchLoop, SearchConfig, SearchNode
from agent_evolve.model.algorithms.autopretrain.mixture import CurriculumSchedule, DataMixture
from agent_evolve.model.algorithms.autopretrain.mutator import MixtureMutator
from agent_evolve.model.algorithms.autopretrain.eval_harness import PretrainEvalHarness, EvalResult
from agent_evolve.model.algorithms.autopretrain.reward import PretrainRewardPolicy
from agent_evolve.model.algorithms.autopretrain.workspace_bridge import WorkspaceBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

K8S_CONTEXT = "arn:aws:eks:ap-south-1:801953956576:cluster/p5-llm"
K8S_NAMESPACE = "default"
POLL_INTERVAL = 60  # seconds


def parse_args():
    parser = argparse.ArgumentParser(description="Launch MCGS proxy search on K8s")
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="/fsx/dev/jiaqi/experiments/autopretrain_search")
    parser.add_argument("--data-root", type=str, default="/fsx/dev/jiaqi/data/olmo-3b-pretrain")
    parser.add_argument("--steps-per-trial", type=int, default=5000)
    parser.add_argument("--initial-mix", type=str, default="olmo2_stage1")
    parser.add_argument("--dry-run", action="store_true", help="Generate workspaces without submitting")
    return parser.parse_args()


def submit_k8s_job(trial_id: str, workspace_dir: str, dry_run: bool = False) -> bool:
    """Submit a PyTorchJob for a single trial."""
    template_path = Path(__file__).parent / "k8s_proxy_trial.yaml"
    template = template_path.read_text()

    # Substitute variables
    manifest = template.replace("${TRIAL_ID}", trial_id).replace("${WORKSPACE_DIR}", workspace_dir)

    if dry_run:
        logger.info(f"  [DRY RUN] Would submit job: autopretrain-{trial_id}")
        return True

    result = subprocess.run(
        ["kubectl", "apply", "-f", "-", "-n", K8S_NAMESPACE, "--context", K8S_CONTEXT],
        input=manifest,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        logger.error(f"Failed to submit job {trial_id}: {result.stderr}")
        return False

    logger.info(f"  Submitted K8s job: autopretrain-{trial_id}")
    return True


def wait_for_job(trial_id: str, timeout: int = 7200) -> str:
    """Wait for a K8s PyTorchJob to complete. Returns 'Succeeded' or 'Failed'."""
    job_name = f"autopretrain-{trial_id}"
    start = time.time()

    while time.time() - start < timeout:
        result = subprocess.run(
            [
                "kubectl", "get", "pytorchjob", job_name,
                "-n", K8S_NAMESPACE, "--context", K8S_CONTEXT,
                "-o", "jsonpath={.status.conditions[-1:].type}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = result.stdout.strip()
        if status in ("Succeeded", "Failed"):
            return status
        time.sleep(POLL_INTERVAL)

    return "Timeout"


def collect_metrics(workspace_dir: str, steps: int) -> EvalResult:
    """Read metrics from a completed trial's workspace directory."""
    harness = PretrainEvalHarness(fast_mode=True)
    trial_dir = Path(workspace_dir)
    return harness.evaluate(
        checkpoint_path=str(trial_dir / "checkpoints"),
        step=steps,
        trial_dir=trial_dir,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AutoPretrain K8s Search Orchestrator")
    logger.info(f"  Cycles: {args.cycles}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Initialize search components
    bridge = WorkspaceBridge(
        trials_root=output_dir / "trials",
        data_root=args.data_root,
    )
    mutator = MixtureMutator(perturbation_scale=0.08)
    reward_policy = PretrainRewardPolicy()

    # Initialize with root
    initial = DataMixture.from_reference(args.initial_mix)
    root_curriculum = CurriculumSchedule.static(initial)

    nodes: dict[str, SearchNode] = {}
    cycle_history: list[dict] = []

    # Root trial
    logger.info("\n--- Root Trial ---")
    root_dir = bridge.create_trial_workspace(
        node_id="root",
        curriculum=root_curriculum,
        model_factory="olmo2_190M",
        max_steps=args.steps_per_trial,
        batch_size=64,
        sequence_length=4096,
    )

    if submit_k8s_job("root", str(root_dir), args.dry_run):
        if not args.dry_run:
            status = wait_for_job("root")
            logger.info(f"  Root job status: {status}")
            root_eval = collect_metrics(str(root_dir), args.steps_per_trial)
        else:
            root_eval = EvalResult(val_loss=3.5, val_perplexity=11.3)
    else:
        root_eval = EvalResult(val_loss=4.0, val_perplexity=16.0)

    root_node = SearchNode(
        node_id="root", parent_id=None, curriculum=root_curriculum,
        eval_result=root_eval, reward=reward_policy.compute_reward(root_eval, root_curriculum),
        visits=1, total_reward=reward_policy.compute_reward(root_eval, root_curriculum),
    )
    nodes["root"] = root_node
    logger.info(f"  Root: loss={root_eval.val_loss:.4f}")

    # MCGS Loop
    for cycle in range(args.cycles):
        logger.info(f"\n--- Cycle {cycle+1}/{args.cycles} ---")

        # Select parent (UCT)
        import math
        total_visits = sum(n.visits for n in nodes.values())
        best_node, best_uct = None, -float("inf")
        for node in nodes.values():
            if node.visits == 0:
                continue
            exploit = node.mean_reward
            explore = 1.4 * math.sqrt(math.log(total_visits + 1) / node.visits)
            uct = exploit + explore
            if uct > best_uct:
                best_uct, best_node = uct, node
        parent = best_node or root_node

        # Mutate
        feedback = parent.eval_result.to_dict() if parent.eval_result else None
        child_curriculum, mutation_record = mutator.mutate(parent.curriculum, feedback, cycle)

        node_id = f"cycle{cycle:03d}"
        logger.info(f"  Parent: {parent.node_id} → Child: {node_id}")
        logger.info(f"  Mutation: {mutation_record.description}")

        # Create workspace and submit
        trial_dir = bridge.create_trial_workspace(
            node_id=node_id,
            curriculum=child_curriculum,
            model_factory="olmo2_190M",
            max_steps=args.steps_per_trial,
            batch_size=64,
            sequence_length=4096,
        )

        if submit_k8s_job(node_id, str(trial_dir), args.dry_run):
            if not args.dry_run:
                status = wait_for_job(node_id, timeout=5400)
                eval_result = collect_metrics(str(trial_dir), args.steps_per_trial)
            else:
                import random
                eval_result = EvalResult(val_loss=3.0 + random.uniform(-0.5, 0.5), val_perplexity=8.0)
        else:
            eval_result = EvalResult(val_loss=5.0, val_perplexity=32.0)

        # Compute reward and backpropagate
        reward = reward_policy.compute_reward(eval_result, child_curriculum, parent.eval_result)
        reward_policy.update_bounds(eval_result.val_loss)

        child_node = SearchNode(
            node_id=node_id, parent_id=parent.node_id, curriculum=child_curriculum,
            mutation_record=mutation_record, eval_result=eval_result, reward=reward,
        )
        nodes[node_id] = child_node

        # Backpropagate
        current = node_id
        while current is not None:
            n = nodes[current]
            n.visits += 1
            n.total_reward += reward
            current = n.parent_id

        logger.info(f"  Loss: {eval_result.val_loss:.4f}, Reward: {reward:.3f}")

        # Save state periodically
        if (cycle + 1) % 5 == 0:
            state_path = output_dir / f"search_state_cycle_{cycle+1:03d}.json"
            state_path.write_text(json.dumps({
                "cycle": cycle + 1,
                "nodes": {nid: n.to_dict() for nid, n in nodes.items()},
                "history": cycle_history,
            }, indent=2, default=str))

        cycle_history.append({
            "cycle": cycle, "node_id": node_id, "loss": eval_result.val_loss, "reward": reward,
            "mutation": mutation_record.description,
        })

    # Final results
    ranked = sorted(
        [n for n in nodes.values() if n.eval_result],
        key=lambda n: n.eval_result.val_loss,
    )

    logger.info("\n" + "=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info(f"Best loss: {ranked[0].eval_result.val_loss:.4f}")
    logger.info("\nTop 5 recipes:")
    for i, n in enumerate(ranked[:5]):
        logger.info(f"  {i+1}. {n.node_id}: loss={n.eval_result.val_loss:.4f}")
        if n.mutation_record:
            logger.info(f"     {n.mutation_record.description}")

    # Save final results
    final_path = output_dir / "final_results.json"
    final_path.write_text(json.dumps({
        "top_5": [n.to_dict() for n in ranked[:5]],
        "total_cycles": args.cycles,
        "best_loss": ranked[0].eval_result.val_loss,
        "history": cycle_history,
    }, indent=2, default=str))
    logger.info(f"\nResults saved to {final_path}")


if __name__ == "__main__":
    main()
