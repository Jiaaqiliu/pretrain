#!/usr/bin/env python3
"""Evaluate the MAS (Multi-Agent System) agent on ARC-AGI-3."""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_evolve.harness.agents.arc.mas_agent import MASArcAgent
from agent_evolve.benchmarks.arc_agi3 import ArcAgi3Benchmark


def run_one(task, workspace, args, log_dir):
    """Run a single game in its own agent instance (thread-safe)."""
    agent = MASArcAgent(
        workspace_dir=str(workspace),
        max_actions=args.actions,
        use_wiki=not args.flat,
        thinking_effort=args.thinking,
        log_dir=log_dir,
    )
    traj = agent.solve(task)
    return task, traj


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAS agent on ARC-AGI-3")
    parser.add_argument("-g", "--game", help="Game ID filter (e.g. ft09)")
    parser.add_argument("-n", "--limit", type=int, default=25, help="Max games")
    parser.add_argument("-w", "--workers", type=int, default=12, help="Parallel workers")
    parser.add_argument("-a", "--actions", type=int, default=350, help="Max actions per game")
    parser.add_argument("--flat", action="store_true", help="Use flat memory instead of wiki")
    parser.add_argument("--thinking", default="", help="Thinking effort (low/medium/high/max)")
    parser.add_argument("--log-dir", default="mas_logs", help="Log directory")
    args = parser.parse_args()

    workspace = Path(__file__).parent.parent.parent / "seed_workspaces" / "arc-mas"

    bench = ArcAgi3Benchmark(
        max_actions_per_game=args.actions,
        game_filter=args.game,
    )

    tasks = bench.get_tasks(split="test", limit=args.limit)

    mode = "flat" if args.flat else "wiki"

    # Unique run ID: {mode}_{timestamp}
    run_id = f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(args.log_dir) / run_id)

    print(f"\n{'='*60}")
    print(f"  MAS Agent Evaluation")
    print(f"  Run ID:    {run_id}")
    print(f"  Games:     {len(tasks)}")
    print(f"  Workers:   {args.workers}")
    print(f"  Actions:   {args.actions}/game")
    print(f"  Knowledge: {mode}")
    print(f"  Thinking:  {args.thinking or 'disabled'}")
    print(f"  Log dir:   {log_dir}")
    print(f"{'='*60}\n")

    results = []
    completed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, task, workspace, args, log_dir): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            completed += 1
            try:
                task, traj = future.result()
                feedback = bench.evaluate(task, traj)
                result = json.loads(traj.output)
                entry = {
                    "game": task.id,
                    "levels": result.get("levels_completed", 0),
                    "total_levels": result.get("total_levels", 0),
                    "actions": result.get("total_actions", 0),
                    "agents": result.get("agents_spawned", 0),
                    "score": feedback.score,
                }
                results.append(entry)
                elapsed = time.time() - start
                print(
                    f"[{completed}/{len(tasks)}] {task.id.split('-')[0]}: "
                    f"{entry['levels']} levels, {entry['actions']} actions, "
                    f"score={entry['score']:.2f}  (wall: {elapsed:.0f}s)"
                )
            except Exception as e:
                logger.error("Game %s failed: %s", task.id, e)
                results.append({"game": task.id, "error": str(e)})

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"  RESULTS ({elapsed:.0f}s total)")
    print(f"{'='*60}")
    print(f"  {'Game':<20} {'Levels':>8} {'Actions':>8} {'Score':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")

    total_levels = 0
    total_actions = 0
    scores = []
    for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
        if "error" in r:
            print(f"  {r['game'].split('-')[0]:<20} {'ERROR':>8}")
        else:
            total_levels += r["levels"]
            total_actions += r["actions"]
            scores.append(r["score"])
            print(f"  {r['game'].split('-')[0]:<20} {r['levels']:>8} {r['actions']:>8} {r['score']:>7.2f}")

    avg = sum(scores) / len(scores) if scores else 0
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<20} {total_levels:>8} {total_actions:>8} {avg:>7.2f}")

    # Save results
    output = {
        "mode": mode, "workers": args.workers, "max_actions": args.actions,
        "elapsed": round(elapsed, 1), "games": results,
    }
    results_path = Path(log_dir) / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
