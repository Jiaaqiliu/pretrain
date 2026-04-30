#!/usr/bin/env python3
"""
Competition mode submission for ARC-AGI-3 Community Leaderboard.

Requirements (from https://github.com/arcprize/ARC-AGI-Community-Leaderboard):
- OperationMode.COMPETITION
- Single scorecard for all 25 games
- Each game make()'d only once
- No get_scorecard during run
- Scorecard URL submitted via PR

Usage:
  python submit_competition.py                    # full 25 games, 12 workers
  python submit_competition.py -w 8 -a 350       # custom workers/actions
  python submit_competition.py --dry-run -g ft09  # test with one game
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_one_game(agent_factory, env, game_id, max_actions):
    """Run a single game on a pre-created env. Thread-safe."""
    agent = agent_factory()
    try:
        result = agent.play_game_on_env(env, game_id, max_actions)
        return game_id, result
    except Exception as e:
        logger.error("Game %s failed: %s", game_id, e, exc_info=True)
        return game_id, {"game_id": game_id, "error": str(e), "levels_completed": 0, "total_actions": 0}


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Competition Submission")
    parser.add_argument("-w", "--workers", type=int, default=12, help="Parallel workers")
    parser.add_argument("-a", "--actions", type=int, default=350, help="Max actions per game")
    parser.add_argument("-g", "--game", help="Filter to specific game(s) for testing")
    parser.add_argument("--flat", action="store_true", help="Use flat memory instead of wiki")
    parser.add_argument("--thinking", default="", help="Thinking effort")
    parser.add_argument("--log-dir", default="mas_logs", help="Log directory")
    parser.add_argument("--dry-run", action="store_true", help="Use online mode instead of competition (for testing)")
    parser.add_argument("--api-key", default=os.environ.get("ARC_API_KEY", ""), help="ARC API key")
    args = parser.parse_args()

    from arc_agi import Arcade, OperationMode
    from agent_evolve.harness.agents.arc.mas_agent import MASArcAgent

    # Competition mode vs dry-run
    if args.dry_run:
        op_mode = OperationMode.ONLINE
        mode_label = "DRY-RUN (online)"
    else:
        op_mode = OperationMode.COMPETITION
        mode_label = "COMPETITION"

    run_id = f"competition_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(Path(args.log_dir) / run_id)
    workspace = Path(__file__).parent.parent.parent / "seed_workspaces" / "arc-mas"
    knowledge = "flat" if args.flat else "wiki"

    print(f"\n{'='*60}")
    print(f"  ARC-AGI-3 Competition Submission")
    print(f"  Mode:      {mode_label}")
    print(f"  Run ID:    {run_id}")
    print(f"  Workers:   {args.workers}")
    print(f"  Actions:   {args.actions}/game")
    print(f"  Knowledge: {knowledge}")
    print(f"  Log dir:   {log_dir}")
    print(f"{'='*60}\n")

    # Create Arcade in competition mode
    if args.api_key:
        arcade = Arcade(arc_api_key=args.api_key, operation_mode=op_mode)
    else:
        arcade = Arcade(operation_mode=op_mode)

    # Get all games
    envs_list = arcade.get_environments()
    all_game_ids = sorted(e.game_id for e in envs_list)
    logger.info("Discovered %d games", len(all_game_ids))

    # Filter if requested
    if args.game:
        filters = args.game.split(",")
        all_game_ids = [g for g in all_game_ids if any(g.startswith(f) for f in filters)]

    # Open single scorecard (competition mode requirement)
    tags = ["a-evolve-mas", knowledge, f"actions-{args.actions}"]
    card_id = arcade.open_scorecard(tags=tags)
    logger.info("Scorecard opened: %s", card_id)

    # Create all envs upfront (competition mode: make() only once per game)
    game_envs = {}
    for game_id in all_game_ids:
        try:
            env = arcade.make(game_id, scorecard_id=card_id)
            game_envs[game_id] = env
            logger.info("Created env for %s", game_id)
        except Exception as e:
            logger.error("Failed to create env for %s: %s", game_id, e)

    logger.info("Created %d game environments", len(game_envs))

    # Agent factory (creates fresh agent per game, thread-safe)
    def make_agent():
        return MASArcAgent(
            workspace_dir=str(workspace),
            max_actions=args.actions,
            use_wiki=not args.flat,
            thinking_effort=args.thinking,
            log_dir=log_dir,
        )

    # Run all games in parallel
    results = {}
    completed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one_game, make_agent, env, game_id, args.actions): game_id
            for game_id, env in game_envs.items()
        }

        for future in as_completed(futures):
            game_id = futures[future]
            completed += 1
            try:
                gid, result = future.result()
                results[gid] = result
                elapsed = time.time() - start
                lv = result.get("levels_completed", 0)
                ac = result.get("total_actions", 0)
                err = " ERROR" if "error" in result else ""
                print(f"[{completed}/{len(game_envs)}] {gid.split('-')[0]}: {lv} levels, {ac} actions{err}  (wall: {elapsed:.0f}s)")
            except Exception as e:
                logger.error("Game %s exception: %s", game_id, e)
                results[game_id] = {"error": str(e)}

    total_elapsed = time.time() - start

    # Close scorecard
    logger.info("Closing scorecard %s ...", card_id)
    try:
        scorecard = arcade.close_scorecard(card_id)
        if scorecard:
            logger.info("Scorecard closed successfully")
            print(f"\n{'='*60}")
            print(f"  SCORECARD")
            print(f"{'='*60}")
            print(json.dumps(scorecard.model_dump(), indent=2))
    except Exception as e:
        logger.error("Failed to close scorecard: %s", e)
        scorecard = None

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS ({total_elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  {'Game':<20} {'Levels':>8} {'Actions':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")

    total_levels = 0
    total_actions = 0
    for gid in sorted(results):
        r = results[gid]
        if "error" in r:
            print(f"  {gid.split('-')[0]:<20} {'ERROR':>8}")
        else:
            lv = r.get("levels_completed", 0)
            ac = r.get("total_actions", 0)
            total_levels += lv
            total_actions += ac
            print(f"  {gid.split('-')[0]:<20} {lv:>8} {ac:>8}")

    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<20} {total_levels:>8} {total_actions:>8}")

    # Save everything
    output = {
        "run_id": run_id,
        "mode": mode_label,
        "scorecard_id": card_id,
        "knowledge": knowledge,
        "workers": args.workers,
        "max_actions": args.actions,
        "total_elapsed": round(total_elapsed, 1),
        "total_levels": total_levels,
        "total_actions": total_actions,
        "games": results,
    }

    if not args.dry_run:
        scorecard_url = f"https://three.arcprize.org/scorecards/{card_id}"
        output["scorecard_url"] = scorecard_url
        print(f"\n  Scorecard URL: {scorecard_url}")

    results_path = Path(log_dir) / "competition_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {results_path}")

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  SUBMIT TO COMMUNITY LEADERBOARD")
        print(f"{'='*60}")
        print(f"  1. Fork https://github.com/arcprize/ARC-AGI-Community-Leaderboard")
        print(f"  2. Copy submissions/.example/ to submissions/a-evolve-mas/")
        print(f"  3. Edit submission.yaml with:")
        print(f"     scorecard_url: {scorecard_url}")
        print(f"  4. Open a PR")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
