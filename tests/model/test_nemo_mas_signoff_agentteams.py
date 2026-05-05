"""Round-trip test for the Agent Teams checkpoint signoff flow.

Mirrors the real lifecycle the user asked about:

  1. Teammates write evidence records tagged with the target slot id.
  2. Reviewer posts ``verdict=ready_to_sign``.
  3. In MANUAL mode the fold reports ``pending_human`` and the
     ``TaskCreated`` hook exits 2 with a blocker message.
  4. The lead (role="human") calls ``checkpoint_sign``.
  5. The fold reports ``signed`` and the hook exits 0.
  6. AUTO mode skips step 3: the reviewer may call ``checkpoint_sign``
     directly with role="reviewer".

No Bedrock calls; no real workspace fork. Uses ``cp_data_check`` from
the real seed workspace ``checkpoints.yaml`` so slot ids match what
the hook script will look up in production.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Make nemo_mas importable when this test runs in isolation.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def teams_env(tmp_path, monkeypatch):
    """Mint a workspace + work_dir and wire env vars the way the MCP
    server + hooks expect, then clear the cached ``_State``.
    """
    seed = REPO / "seed_workspaces" / "nemo_mas_reasoner"
    # Point NEMO_MAS_WORKSPACE_ROOT at the real seed so load_slot_decls
    # returns the actual slots. The per-cycle fork is orthogonal to this
    # test — here we just want to exercise the MCP tool handlers against
    # an in-memory ledger.
    work_dir = tmp_path / "run"
    (work_dir / "memory").mkdir(parents=True)
    records_path = work_dir / "memory" / "records.jsonl"

    monkeypatch.setenv("NEMO_MAS_WORK_DIR", str(work_dir))
    monkeypatch.setenv("NEMO_MAS_WORKSPACE_ROOT", str(seed))
    monkeypatch.setenv("NEMO_MAS_MEMORY_PATH", str(records_path))
    monkeypatch.setenv("NEMO_MAS_SEED_WORKSPACE", str(seed))
    monkeypatch.setenv("NEMO_MAS_CHECKPOINT_MODE", "manual")

    # Import AFTER env is set so module-level lambdas pick up the right
    # paths. Clear any cached state from prior tests.
    from agent_evolve.model.algorithms.nemo_mas import mcp_server
    mcp_server._State.invalidate()

    return {
        "seed": seed,
        "work_dir": work_dir,
        "records_path": records_path,
        "mcp_server": mcp_server,
    }


def _call(fn, **kwargs) -> dict:
    """Invoke an MCP tool function. FastMCP wraps the decorated fn as a
    ``FunctionTool`` — the underlying callable is on ``fn.fn``."""
    result = fn.fn(**kwargs) if hasattr(fn, "fn") else fn(**kwargs)
    return json.loads(result)


# ── Signoff round-trip (manual mode) ────────────────────────────


def test_manual_signoff_roundtrip(teams_env, monkeypatch):
    """Evidence → review → pending_human → hook blocks → sign → unblocked."""
    mcp = teams_env["mcp_server"]
    slot = "cp_data_check"

    # 1. data_worker writes a dataset_snapshot tagged with the slot.
    r = _call(
        mcp.mem_write,
        role="data_worker",
        kind="dataset_snapshot",
        title="snapshot v1",
        body="rows=1000 sha=abc123",
        tags=[f"checkpoint:{slot}"],
    )
    assert r["ok"], r
    snapshot_id = r["id"]

    # 2. reviewer writes a data_audit_finding tagged with the slot.
    r = _call(
        mcp.mem_write,
        role="reviewer",
        kind="data_audit_finding",
        title="audit v1",
        body="format OK, coverage OK",
        tags=[f"checkpoint:{slot}"],
        refs=[snapshot_id],
    )
    assert r["ok"], r
    audit_id = r["id"]

    # 3. reviewer posts verdict=ready_to_sign.
    r = _call(
        mcp.checkpoint_review_suggest,
        slot_id=slot,
        verdict="ready_to_sign",
        reason="evidence complete, ready for human signoff",
        refs=[snapshot_id, audit_id],
    )
    assert r["ok"], r

    # 4. state → pending_human in manual mode.
    r = _call(mcp.checkpoint_state, slot_id=slot)
    assert r["ok"], r
    assert r["slot"]["state"] == "pending_human", r["slot"]
    assert r["slot"]["can_sign"] is True
    assert r["mode"] == "manual"

    # 5. TaskCreated hook blocks (exit 2) with a blocker message.
    hook = REPO / ".claude" / "hooks" / "nemo_mas_task_created.py"
    assert hook.is_file() and os.access(hook, os.X_OK), hook
    proc = subprocess.run(
        [sys.executable, str(hook)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ},
    )
    assert proc.returncode == 2, (
        f"hook should block on pending_human, got rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert slot in proc.stderr, proc.stderr
    assert "BLOCKED" in proc.stderr, proc.stderr

    # 6. reviewer in manual mode CANNOT sign — assert the MCP server
    # rejects it (plan item 7 of the signoff mapping).
    r = _call(
        mcp.checkpoint_sign,
        slot_id=slot,
        refs=[snapshot_id, audit_id],
        role="reviewer",
    )
    assert not r["ok"], r
    assert "manual" in r["error"].lower(), r["error"]

    # 7. lead signs as role="human".
    r = _call(
        mcp.checkpoint_sign,
        slot_id=slot,
        refs=[snapshot_id, audit_id],
        role="human",
        note="looks good",
    )
    assert r["ok"], r
    assert r["actor"] == "human:lead"
    assert r["mode"] == "manual"

    # 8. state → signed.
    r = _call(mcp.checkpoint_state, slot_id=slot)
    assert r["ok"], r
    assert r["slot"]["state"] == "signed", r["slot"]

    # 9. TaskCreated hook now exits 0 (unblocks). The next required
    # slot (cp_training_health) is still in pending — but pending alone
    # does not block task creation (teammates need to produce evidence),
    # so the hook returns 0.
    proc = subprocess.run(
        [sys.executable, str(hook)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ},
    )
    assert proc.returncode == 0, (
        f"hook should allow after signoff, got rc={proc.returncode}\n"
        f"stderr={proc.stderr}"
    )


# ── Signoff in auto mode ─────────────────────────────────────────


def test_auto_mode_reviewer_can_sign(teams_env, monkeypatch):
    """AUTO mode: reviewer signs directly after posting ready_to_sign."""
    monkeypatch.setenv("NEMO_MAS_CHECKPOINT_MODE", "auto")
    teams_env["mcp_server"]._State.invalidate()

    mcp = teams_env["mcp_server"]
    slot = "cp_data_check"

    r = _call(
        mcp.mem_write,
        role="data_worker",
        kind="dataset_snapshot",
        title="snapshot v1",
        body="rows=1000 sha=abc123",
        tags=[f"checkpoint:{slot}"],
    )
    snapshot_id = r["id"]

    r = _call(
        mcp.mem_write,
        role="reviewer",
        kind="data_audit_finding",
        title="audit v1",
        body="looks good",
        tags=[f"checkpoint:{slot}"],
        refs=[snapshot_id],
    )
    audit_id = r["id"]

    r = _call(
        mcp.checkpoint_review_suggest,
        slot_id=slot,
        verdict="ready_to_sign",
        reason="auto-mode approval",
        refs=[snapshot_id, audit_id],
    )
    assert r["ok"], r

    # reviewer signs directly — allowed in auto mode.
    r = _call(
        mcp.checkpoint_sign,
        slot_id=slot,
        refs=[snapshot_id, audit_id],
        role="reviewer",
    )
    assert r["ok"], r
    assert r["actor"] == "reviewer"

    r = _call(mcp.checkpoint_state, slot_id=slot)
    assert r["slot"]["state"] == "signed", r["slot"]


# ── Guardrails on signoff ────────────────────────────────────────


def test_sign_requires_evidence_coverage(teams_env):
    """Missing a required_evidence kind must fail with a clear error."""
    mcp = teams_env["mcp_server"]
    slot = "cp_data_check"

    # Only attach dataset_snapshot (cp_data_check also requires
    # data_audit_finding).
    r = _call(
        mcp.mem_write,
        role="data_worker",
        kind="dataset_snapshot",
        title="snapshot",
        body="…",
        tags=[f"checkpoint:{slot}"],
    )
    snap = r["id"]

    r = _call(
        mcp.checkpoint_sign,
        slot_id=slot,
        refs=[snap],
        role="human",
    )
    assert not r["ok"], r
    assert "data_audit_finding" in r["error"], r["error"]


def test_sign_rejects_unmet_dependency(teams_env):
    """cp_training_health depends on cp_data_check — reject early sign."""
    mcp = teams_env["mcp_server"]

    # Attach a training_run (the requires_evidence for cp_training_health).
    # training_run needs refs to both recipe_proposal and dataset_snapshot
    # per REF_RULES; build them first.
    r = _call(
        mcp.mem_write,
        role="data_worker",
        kind="dataset_snapshot",
        title="snapshot",
        body="…",
        tags=["checkpoint:cp_training_health"],
    )
    snap = r["id"]
    # recipe_proposal needs a ref to eval_report or data_gap.
    r = _call(
        mcp.mem_write,
        role="reviewer",
        kind="data_gap",
        title="gap",
        body="cov gap",
    )
    gap = r["id"]
    r = _call(
        mcp.mem_write,
        role="planner",
        kind="recipe_proposal",
        title="proposal",
        body="try lr=1e-5",
        refs=[gap],
    )
    prop = r["id"]
    r = _call(
        mcp.mem_write,
        role="trainer",
        kind="training_run",
        title="run",
        body="loss=0.5",
        tags=["checkpoint:cp_training_health"],
        refs=[prop, snap],
    )
    tr = r["id"]

    # Try to sign cp_training_health without signing cp_data_check first.
    r = _call(
        mcp.checkpoint_sign,
        slot_id="cp_training_health",
        refs=[tr],
        role="human",
    )
    assert not r["ok"], r
    assert "cp_data_check" in r["error"], r["error"]


# ── Role guard on mem_write ──────────────────────────────────────


def test_role_guard_rejects_wrong_kind(teams_env):
    """trainer cannot write checkpoint_review — only reviewer can."""
    mcp = teams_env["mcp_server"]
    r = _call(
        mcp.mem_write,
        role="trainer",
        kind="checkpoint_review",
        title="sneaky",
        body="trying to approve my own work",
        tags=["checkpoint:cp_data_check"],
        refs=[],
    )
    assert not r["ok"], r
    # The role-whitelist in schema.py fires: trainer has no
    # checkpoint_review permission.
    assert "checkpoint_review" in r["error"] or "allowed" in r["error"].lower()


def test_role_guard_rejects_unknown_role(teams_env):
    """orchestrator_auto / human / random strings cannot call mem_write."""
    mcp = teams_env["mcp_server"]
    for bad_role in ("orchestrator_auto", "human", "bogus"):
        r = _call(
            mcp.mem_write,
            role=bad_role,
            kind="breakthrough",
            title="x",
            body="y",
        )
        assert not r["ok"], (bad_role, r)


# ── Kaggle budget hook ────────────────────────────────────────────


def test_kaggle_budget_hook_blocks_after_cap(teams_env, monkeypatch):
    """Hook exits 2 once kaggle_submission_result count hits the cap."""
    mcp = teams_env["mcp_server"]

    # Seed a single kaggle_submission_result. The cap defaults to 1.
    # First build the upstream chain: recipe_proposal → training_run →
    # submission_artifact → kaggle_submission_result.
    r = _call(mcp.mem_write, role="reviewer", kind="data_gap",
              title="gap", body="x")
    gap = r["id"]
    r = _call(mcp.mem_write, role="planner", kind="recipe_proposal",
              title="p", body="p", refs=[gap])
    prop = r["id"]
    r = _call(mcp.mem_write, role="data_worker", kind="dataset_snapshot",
              title="ds", body="ds")
    snap = r["id"]
    r = _call(mcp.mem_write, role="trainer", kind="training_run",
              title="run", body="loss", refs=[prop, snap])
    tr = r["id"]
    r = _call(mcp.mem_write, role="trainer", kind="submission_artifact",
              title="art", body="zip", refs=[tr])
    art = r["id"]
    r = _call(mcp.mem_write, role="reviewer", kind="kaggle_submission_result",
              title="sub 1", body="pending", refs=[art])
    assert r["ok"], r

    hook = REPO / ".claude" / "hooks" / "nemo_mas_kaggle_budget.py"
    assert hook.is_file() and os.access(hook, os.X_OK), hook

    # With cap=1 (default) and 1 already logged, hook must block.
    proc = subprocess.run(
        [sys.executable, str(hook)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ},
    )
    assert proc.returncode == 2, (
        f"kaggle budget hook should block, got rc={proc.returncode}\n"
        f"stderr={proc.stderr}"
    )
    assert "budget exhausted" in proc.stderr.lower(), proc.stderr


# ── Slot metadata (sanity) ───────────────────────────────────────


def test_list_slots_matches_yaml(teams_env):
    """list_slots reflects the on-disk checkpoints.yaml exactly."""
    mcp = teams_env["mcp_server"]
    r = _call(mcp.list_slots)
    assert r["ok"], r

    yaml_path = teams_env["seed"] / "checkpoints.yaml"
    decl = yaml.safe_load(yaml_path.read_text())["checkpoints"]
    assert [s["id"] for s in r["slots"]] == [s["id"] for s in decl]
    # All required slots start in `pending` (no evidence yet).
    for s in r["slots"]:
        assert s["state"] == "pending", s
