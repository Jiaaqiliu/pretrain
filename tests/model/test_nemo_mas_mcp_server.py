"""Smoke tests for the nemo_mas MCP server.

Covers:
  * module imports and registers the expected tool surface,
  * ``start_iteration`` forks the workspace and updates env,
  * workspace-root rebinding: after a second ``start_iteration``, tool
    calls target the newer fork.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


EXPECTED_MEMORY_TOOLS = {"mem_write", "mem_get", "mem_search", "mem_recent"}
EXPECTED_CHECKPOINT_TOOLS = {
    "list_slots", "checkpoint_state",
    "checkpoint_review_suggest", "checkpoint_sign",
}
EXPECTED_ITERATION_TOOLS = {"start_iteration", "current_iteration"}
# Partial list — just the ones we're confident exist from backends.py.
EXPECTED_BACKEND_TOOLS = {
    "sample_jsonl", "write_jsonl", "mix_sources", "pack_submission",
    "kaggle_submit", "kaggle_fetch_score",
}


@pytest.fixture
def teams_env(tmp_path, monkeypatch):
    seed = REPO / "seed_workspaces" / "nemo_mas_reasoner"
    work_dir = tmp_path / "run"
    work_dir.mkdir(parents=True)
    monkeypatch.setenv("NEMO_MAS_WORK_DIR", str(work_dir))
    monkeypatch.setenv("NEMO_MAS_SEED_WORKSPACE", str(seed))
    monkeypatch.setenv("NEMO_MAS_CHECKPOINT_MODE", "manual")
    # Drop WORKSPACE_ROOT / MEMORY_PATH so start_iteration sets them.
    monkeypatch.delenv("NEMO_MAS_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("NEMO_MAS_MEMORY_PATH", raising=False)

    from agent_evolve.model.algorithms.nemo_mas.agent_teams import server as mcp_server
    mcp_server._State.invalidate()
    return {"seed": seed, "work_dir": work_dir, "mcp_server": mcp_server}


def _call(fn, **kwargs) -> dict:
    result = fn.fn(**kwargs) if hasattr(fn, "fn") else fn(**kwargs)
    return json.loads(result)


def test_tool_surface_registered(teams_env):
    """All four tool groups are present under the FastMCP registry."""
    mcp = teams_env["mcp_server"].mcp
    tools = mcp._tool_manager.list_tools()
    names = {t.name for t in tools}

    for expected in (EXPECTED_MEMORY_TOOLS, EXPECTED_CHECKPOINT_TOOLS,
                     EXPECTED_ITERATION_TOOLS, EXPECTED_BACKEND_TOOLS):
        missing = expected - names
        assert not missing, (
            f"missing tools: {missing}\nregistered: {sorted(names)}"
        )


def test_start_iteration_forks_workspace(teams_env):
    mcp = teams_env["mcp_server"]
    wd = teams_env["work_dir"]

    r = _call(mcp.start_iteration)
    assert r["ok"], r
    assert r["current_cycle"] == "0001"

    fork_path = Path(r["workspace_root"])
    assert fork_path.is_dir(), fork_path
    # Fork path convention matches the headless runtime.
    expected_prefix = wd / "cycles" / "0001" / ".fork_target"
    assert str(fork_path).startswith(str(expected_prefix)), (fork_path, expected_prefix)

    # Env vars are updated in-process so subsequent tool calls see them.
    assert os.environ["NEMO_MAS_WORKSPACE_ROOT"] == str(fork_path)
    assert os.environ["NEMO_MAS_MEMORY_PATH"].startswith(str(wd))
    # Memory lives OUTSIDE the fork so state accumulates across cycles.
    assert not os.environ["NEMO_MAS_MEMORY_PATH"].startswith(str(fork_path))


def test_start_iteration_twice_rebinds(teams_env):
    """Two successive iterations create distinct fork dirs."""
    mcp = teams_env["mcp_server"]

    r1 = _call(mcp.start_iteration)
    assert r1["ok"]
    fork1 = r1["workspace_root"]

    r2 = _call(mcp.start_iteration)
    assert r2["ok"]
    fork2 = r2["workspace_root"]

    assert fork1 != fork2
    assert r2["current_cycle"] == "0002"

    # current_iteration agrees with the latest start.
    r3 = _call(mcp.current_iteration)
    assert r3["current_cycle"] == "0002"
    assert r3["workspace_root"] == fork2


def test_mem_write_lands_in_fork_ledger(teams_env):
    """A write after start_iteration lands in <work_dir>/memory/records.jsonl."""
    mcp = teams_env["mcp_server"]

    _call(mcp.start_iteration)
    r = _call(mcp.mem_write,
              role="data_worker",
              kind="breakthrough",
              title="t",
              body="b",
              refs=[])
    # breakthrough needs ≥1 ref; expect a rejection — proves the
    # validator is wired.
    assert not r["ok"], r

    r = _call(mcp.mem_write,
              role="data_worker",
              kind="dataset_snapshot",
              title="t", body="b")
    assert r["ok"], r

    ledger = Path(os.environ["NEMO_MAS_MEMORY_PATH"])
    assert ledger.is_file()
    lines = ledger.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["kind"] == "dataset_snapshot"
    assert row["cycle_id"] == "0001"
    assert row["author"] == "data_worker"
