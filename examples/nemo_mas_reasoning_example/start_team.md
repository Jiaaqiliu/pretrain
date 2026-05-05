# Running nemo_mas via Claude Code Agent Teams

Interactive front-end for the nemo_mas Quality-Plan loop. Replaces
`drive_nemo_mas.py` when you want to sit at the terminal and drive
cycles conversationally with human signoff. The Python headless driver
still works — Agent Teams is an alternate mode, not a replacement.

## Pre-flight

1. Agent Teams requires Claude Code v2.1.32+. Check with `claude --version`.
2. The repo-local `.claude/settings.json` sets
   `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` — no separate opt-in needed.
3. The `nemo_mas` MCP server is declared in `.mcp.json` at the repo root
   — Claude Code auto-starts it on session launch.
4. AWS creds must be in the shell (Bedrock for teammate models + training
   backend for `launch_training`).

## One-shot shell setup

```bash
cd /fsx/zzsamshi/a-evolve

# Where cycle artifacts + the shared ledger live. Use a fresh dir per run.
export NEMO_MAS_WORK_DIR=runs/nemo-mas-teams-v1
# manual (default): human signs every required checkpoint via the lead.
# auto: reviewer can sign once it has posted ready_to_sign.
export NEMO_MAS_CHECKPOINT_MODE=manual

# Optional: override the seed workspace path.
# export NEMO_MAS_SEED_WORKSPACE=seed_workspaces/nemo_mas_reasoner

# Optional: lift the per-run Kaggle submission cap (default 1).
# Alternatively edit <work_dir>/meta.json after start_iteration.
# export NEMO_MAS_KAGGLE_MAX_PER_RUN=2
```

## Launch the team

```bash
claude
```

Inside `claude`, paste this as the first prompt:

> Create a nemo_mas team. Spawn five teammates using the subagent types
> `nemo_mas_orchestrator`, `nemo_mas_planner`, `nemo_mas_data_worker`,
> `nemo_mas_trainer`, `nemo_mas_reviewer`. Call
> `mcp__nemo_mas__start_iteration` to fork the workspace and open
> cycle 1, then follow the orchestrator's system prompt.

The lead spawns the five teammates (one per subagent file in
`.claude/agents/nemo_mas_*.md`), the MCP server forks
`seed_workspaces/nemo_mas_reasoner/` to
`<work_dir>/cycles/0001/.fork_target/nodes/workspace/workspace/`, and
the orchestrator begins assigning tasks.

## Signing a checkpoint (manual mode)

When the reviewer posts `verdict=ready_to_sign` on a required slot, the
next `TaskCreated` event is blocked by the hook:

```
[nemo_mas] BLOCKED on required checkpoint cp_data_check (Training data format + eyeball check).
  state: pending_human
  requires_evidence: ['dataset_snapshot', 'data_audit_finding']
  evidence_counts: {'data_audit_finding': 1, 'dataset_snapshot': 1}
  depends_on: (none)
  last_review: verdict=ready_to_sign · cycle 0001 · evidence complete …
  ACTION: tell the lead `sign cp_data_check with refs=[...]` …
```

Read the evidence records via `mem_get` or just trust the reviewer,
then tell the lead:

> Sign cp_data_check with refs rec_ab12cd, rec_ef34gh.

The lead calls `mcp__nemo_mas__checkpoint_sign(slot_id="cp_data_check", refs=[...], role="human")`.
The slot transitions to `signed` and tasks unblock.

## Advancing to the next cycle

A cycle finishes when all required slots for this iteration are
`signed`. To start the next iteration:

> Call `mcp__nemo_mas__start_iteration` and begin cycle 2.

This bumps the counter in `<work_dir>/meta.json`, forks the seed into a
fresh `cycles/0002/...` dir, and swings the server's active workspace
pointer. The cross-cycle ledger at `<work_dir>/memory/records.jsonl`
persists, so breakthroughs and prior records remain visible.

## Stopping cleanly

Ask the lead:

> Clean up the team.

If any teammate is still running, shut them down first:

> Ask the trainer teammate to shut down.
> Ask the reviewer teammate to shut down. (and so on)

## Known limits — acknowledged by the port

- **No `/resume` for teammates.** If your `claude` session dies, the
  lead can be resumed but teammates are gone. Respawn them with the
  same prompt.
- **No headless marathon.** The Python driver (`drive_nemo_mas.py`)
  stays available for unattended multi-cycle runs.
- **One team per lead.** Open a second terminal for a second
  experiment.
- **Per-cycle fork is manual.** The lead must call `start_iteration`
  between cycles; if you forget, writes still land in the previous
  cycle's fork.
- **Session shutdown can be slow.** Teammates finish their current
  turn before exiting.

## Troubleshooting

- **`start_iteration` returns `seed workspace not found`** — export
  `NEMO_MAS_SEED_WORKSPACE` pointing at `seed_workspaces/nemo_mas_reasoner`
  (absolute path).
- **Tool calls return `NEMO_MAS_MEMORY_PATH is unset`** — you skipped
  `start_iteration`. Call it first.
- **Hook doesn't block when it should** — Claude Code only watches
  `.claude/` directories that had a settings file at session start. Run
  `/hooks` once to refresh, or restart `claude`.
- **Reviewer tries to sign in manual mode** — the MCP server rejects
  this with an error message. If you really want the reviewer signing,
  flip to `NEMO_MAS_CHECKPOINT_MODE=auto` before launching.
