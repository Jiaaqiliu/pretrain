#!/usr/bin/env python3
"""Minimal stdlib HTTP server for browsing nemo_mas trace JSONL files.

Layout:
  /                        → Quality Plan cockpit + chat
  /cycle/<NNNN>            → agents in a cycle
  /cycle/<NNNN>/<ID>       → pretty-printed trace for one agent
  /raw/<NNNN>/<ID>         → raw JSONL
  /leaderboard             → ranked eval runs (derived from records.jsonl)
  /run/<ID>                → single eval-run detail card
  POST /checkpoint/<id>/sign → human signoff for Quality Plan checkpoint
  POST /directive          → human → orchestrator chat message

Content sources:
  * Per-agent JSONL traces under <trace-dir>/cycle_NNNN/agent_*.jsonl
    drive the live feed, agent detail, sequence, and calls views.
  * The Quality Plan ledger, leaderboard, and chat thread are all derived
    from a single append-only memory store at
    <trace-dir>/../memory/records.jsonl — same file the backend writes.

Usage:
  python trace_viewer.py [--trace-dir PATH] [--port 7890] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import contextvars
import datetime
import html
import json
import os
import re
import secrets
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Shared checkpoint slot declarations + fold function live with the backend
# algorithm so the two sides never drift. Insert the repo root on sys.path so
# the import resolves no matter where the viewer is run from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from agent_evolve.model.algorithms.nemo_mas.checkpoints import (  # noqa: E402
    CHECKPOINT_MODE_AUTO,
    CHECKPOINT_MODE_MANUAL,
    FoldedSlot,
    evidence_refs_for_slot,
    fold_checkpoints,
    load_slot_decls,
)
from agent_evolve.model.algorithms.nemo_mas.orchestrator import (  # noqa: E402
    cycle_workspace_path,
)

# Multi-run discovery:
#   ``RUNS_ROOT`` is the parent directory that holds one subdirectory per
#   marathon run — each containing ``trace/`` and ``memory/`` siblings.
#   When the viewer starts with ``--trace-dir`` instead (legacy single-run
#   mode), we treat the parent of the trace dir as the "runs root" and
#   pin ``_CURRENT_RUN`` to that one run so nothing surprises the user.
RUNS_ROOT: Path = Path()       # set in main()
DEFAULT_RUN: str | None = None  # run name to serve on legacy paths (/train, /cycle/...)
LEGACY_PINNED: bool = False     # True when booted with --trace-dir (single-run mode)

# Per-request active run. Handler sets this from the URL; helpers below
# (trace_dir_for, memory_path_for, _cycle_dirs, _load_records, ...) read
# from it. A ContextVar keeps concurrent requests (ThreadingHTTPServer)
# from stepping on each other's state.
_ACTIVE_RUN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nemo_mas_active_run", default=None,
)

CHECKPOINT_MODE: str = CHECKPOINT_MODE_MANUAL  # set in main() from env var; fallback
                                                # only — real mode now comes from per-run
                                                # meta.json (see _mode_for_active_run).
ABOUT_URL = "https://github.com/A-EVO-Lab/a-evolve"

# Safety rail on POST body sizes. Directives and signoff notes are short;
# anything bigger is a client bug or abuse.
_MAX_POST_BYTES = 64 * 1024


# ── Run discovery ────────────────────────────────────────────────────

def _is_run_dir(p: Path) -> bool:
    """A "run dir" has a ``trace/`` subdir (memory/ is optional — pre-launch
    runs may not have written a record yet)."""
    return p.is_dir() and (p / "trace").is_dir()


def _is_run_live(name: str) -> bool:
    return _run_summary(name)["state"] == "live"


def _pick_default_run() -> str | None:
    """Run that the "Open default run →" button + legacy URLs should target.

    Preference order (so clicking the button always lands on the most
    useful thing the user is likely to want):

      1. Any run whose ``state`` is ``"live"`` — that's the one they're
         actively watching. If two are live, newest activity wins.
      2. Otherwise, the most recently-active run (finished or empty —
         matches the prior mtime-based default).

    Returns ``None`` when no runs exist on disk yet.
    """
    runs = list_runs()
    if not runs:
        return None
    summaries = [_run_summary(n) for n in runs]
    live = [s for s in summaries if s["state"] == "live"]
    if live:
        live.sort(key=lambda s: s["last_activity"], reverse=True)
        return live[0]["name"]
    summaries.sort(key=lambda s: s["last_activity"] or s["started_at"],
                   reverse=True)
    return summaries[0]["name"]


def list_runs() -> list[str]:
    """Every marathon run currently on disk under ``RUNS_ROOT``."""
    if not RUNS_ROOT.is_dir():
        return []
    names = []
    for p in sorted(RUNS_ROOT.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0):
        if _is_run_dir(p) and not p.name.startswith("."):
            names.append(p.name)
    return names


def active_run() -> str | None:
    """Whichever run the current request is scoped to (URL-derived), falling
    back to the CLI-pinned default when a legacy URL doesn't name a run."""
    run = _ACTIVE_RUN.get()
    if run:
        return run
    return DEFAULT_RUN


_REWRITE_PREFIXES = (
    "/cycle/", "/raw/", "/train", "/leaderboard",
    "/directive", "/checkpoint/", "/live-status.json", "/run/",
    "/record/",
)


def _scoped_href(href: str) -> str:
    """Rewrite an absolute in-app link so it keeps the active run prefix.

    Renderers in this file emit naked paths (``/cycle/0001``, ``/train``).
    When the user is viewing ``/runs/<name>/...``, each of those links
    must resolve back to the same run — otherwise clicks drop out of the
    scope and land on the default. This helper is idempotent and only
    fires when there's an active run; otherwise returns ``href`` as-is.
    """
    run = _ACTIVE_RUN.get()
    if not run:
        return href
    if href.startswith(f"/runs/{run}/") or href == f"/runs/{run}":
        return href
    for pref in _REWRITE_PREFIXES:
        if href == pref.rstrip("/") or href.startswith(pref):
            return f"/runs/{run}{href}"
    return href


def _apply_run_scope(html: str) -> str:
    """Rewrite every ``href='...'`` and ``action='...'`` in the rendered
    HTML to be run-scoped. Cheap regex pass — only runs when an active
    run is set (request scope), so the runs-index page stays untouched.
    """
    run = _ACTIVE_RUN.get()
    if not run:
        return html
    def repl(m: re.Match) -> str:
        attr, quote, url = m.group(1), m.group(2), m.group(3)
        # Emit only the opening bits; the closing quote was consumed by
        # the lookahead so we don't re-emit it (doubling the quote).
        return f"{attr}={quote}{_scoped_href(url)}"
    return re.sub(
        r"(href|action)=(['\"])(/[^'\"#]*)(?=\2|#|$)",
        repl, html,
    )


def trace_dir_for(run: str | None = None) -> Path:
    """Where ``cycle_NNNN/agent_*.jsonl`` files live for the given run."""
    name = run or active_run()
    if name is None:
        return Path()                   # unresolvable; callers treat as empty
    return RUNS_ROOT / name / "trace"


def memory_path_for(run: str | None = None) -> Path:
    """Where the typed-record memory lives for the given run."""
    name = run or active_run()
    if name is None:
        return Path()
    return RUNS_ROOT / name / "memory" / "records.jsonl"


def driver_log_path_for(run: str | None = None) -> Path:
    name = run or active_run()
    if name is None:
        return Path()
    return RUNS_ROOT / name / "driver.log"


def _mode_for_active_run() -> str:
    """Checkpoint mode for the scoped run, read from ``<run>/meta.json``.

    Falls back to the process-level ``CHECKPOINT_MODE`` (set from env at
    startup) if the file is missing or the value is unrecognized — so
    legacy runs launched before this file existed still render sensibly.
    """
    name = active_run()
    if name:
        meta_path = RUNS_ROOT / name / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            candidate = meta.get("checkpoint_mode")
            if candidate in (CHECKPOINT_MODE_AUTO, CHECKPOINT_MODE_MANUAL):
                return candidate
        except (OSError, json.JSONDecodeError):
            pass
    return CHECKPOINT_MODE

STYLE = """
:root {
  --aws-navy: #0f1924;
  --aws-navy-2: #162231;
  --aws-ink: #172033;
  --aws-muted: #5f6b7a;
  --aws-line: #d5dbe3;
  --aws-soft-line: #e8edf3;
  --aws-bg: #f6f8fb;
  --aws-panel: #ffffff;
  --aws-orange: #ff9900;
  --aws-orange-2: #ff6f00;
  --aws-blue: #0972d3;
  --aws-green: #2ea44f;
  --aws-red: #d13212;
  --aws-yellow: #f2b824;
  --shadow-sm: 0 1px 2px rgba(15, 25, 36, 0.08);
  --spacex-black: #03060a;
  --spacex-deep: #07101a;
  --spacex-panel: rgba(10, 18, 28, 0.86);
  --spacex-line: rgba(210, 228, 246, 0.34);
  --spacex-dim: #8a98a8;
  --spacex-ice: #f8fbff;
  --spacex-blue: #7fc8ff;
}
* { box-sizing: border-box; }
html { min-width: 320px; }
body {
  margin: 0;
  color: var(--aws-ink);
  background: var(--aws-bg);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.45;
}
a { color: #006ce0; text-decoration: none; }
a:hover { color: #004f9e; text-decoration: underline; }
h1, h2, h3 { color: #111827; letter-spacing: 0; }
h1 { margin: 0; font-size: clamp(22px, 2.4vw, 32px); line-height: 1.15; }
h2 { margin: 0; font-size: 18px; }
h3 { margin: 0; font-size: 15px; }
button, .button {
  align-items: center;
  background: #fff;
  border: 1px solid #b7c1ce;
  border-radius: 4px;
  color: #172033;
  cursor: pointer;
  display: inline-flex;
  font: inherit;
  font-weight: 650;
  gap: 7px;
  min-height: 34px;
  padding: 7px 12px;
  text-decoration: none;
  white-space: nowrap;
}
button:hover, .button:hover { background: #f3f6fa; text-decoration: none; }
.button.primary, button.primary {
  background: linear-gradient(180deg, var(--aws-orange), var(--aws-orange-2));
  border-color: #e36d00;
  color: #fff;
  box-shadow: 0 1px 0 rgba(255, 255, 255, 0.28) inset;
}
.button.primary:hover, button.primary:hover {
  color: #fff;
  filter: brightness(0.98);
}
.button.icon, button.icon {
  justify-content: center;
  min-width: 34px;
  padding: 7px;
}
.topbar {
  align-items: stretch;
  background:
    radial-gradient(circle at 55% -90%, rgba(255, 153, 0, 0.22), transparent 34%),
    linear-gradient(180deg, #111c29, #0b121b);
  border-bottom: 1px solid #263445;
  color: #f9fafb;
  display: flex;
  min-height: 66px;
  position: sticky;
  top: 0;
  z-index: 20;
}
.brand {
  align-items: center;
  border-right: 1px solid #324053;
  display: flex;
  gap: 12px;
  min-width: 292px;
  padding: 0 22px;
}
.brand-mark {
  background: var(--aws-orange);
  border-radius: 3px;
  box-shadow: 0 0 0 4px rgba(255, 153, 0, 0.14);
  height: 24px;
  position: relative;
  width: 24px;
}
.brand-mark::after {
  border-bottom: 2px solid #111c29;
  border-radius: 0 0 18px 18px;
  bottom: 5px;
  content: "";
  height: 8px;
  left: 5px;
  position: absolute;
  width: 14px;
}
.brand-title { font-size: 20px; font-weight: 800; letter-spacing: 0; line-height: 1; }
.brand-title .beta {
  background: var(--aws-orange);
  border-radius: 4px;
  color: #fff;
  font-size: 11px;
  margin-left: 6px;
  padding: 1px 5px 2px;
  vertical-align: 2px;
}
.brand-subtitle { color: #dbeafe; font-size: 12px; font-weight: 650; margin-top: 3px; }
.topnav {
  align-items: stretch;
  display: flex;
  flex: 1;
  gap: 2px;
  min-width: 0;
  overflow-x: auto;
  padding-left: 18px;
}
.topnav a {
  align-items: center;
  border-bottom: 3px solid transparent;
  color: #f3f5f8;
  display: flex;
  font-weight: 650;
  min-height: 66px;
  padding: 0 15px;
  text-decoration: none;
  white-space: nowrap;
}
.topnav a.active {
  border-bottom-color: var(--aws-orange);
  color: var(--aws-orange);
}
.top-actions {
  align-items: center;
  color: #d6dee8;
  display: flex;
  gap: 14px;
  padding: 0 18px;
  white-space: nowrap;
}
.avatar {
  align-items: center;
  background: #40506a;
  border-radius: 999px;
  color: #fff;
  display: inline-flex;
  font-size: 12px;
  font-weight: 800;
  height: 32px;
  justify-content: center;
  width: 32px;
}
.app-shell { display: grid; grid-template-columns: 292px minmax(0, 1fr); min-height: calc(100vh - 66px); }
.sidebar {
  background: #fff;
  border-right: 1px solid var(--aws-line);
  padding: 24px 18px;
}
.side-section { margin-bottom: 28px; }
.side-title {
  color: #4b5563;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.04em;
  margin: 0 0 10px;
  text-transform: uppercase;
}
.side-nav { display: grid; gap: 4px; }
.side-nav a {
  align-items: center;
  border-radius: 6px;
  color: #344054;
  display: flex;
  font-weight: 650;
  gap: 10px;
  min-height: 38px;
  padding: 8px 10px;
  text-decoration: none;
}
.side-nav a.active {
  background: #fff4e5;
  color: #cc5f00;
}
.side-icon {
  align-items: center;
  color: inherit;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 15px;
  height: 18px;
  justify-content: center;
  width: 18px;
}
.filter-row, .mini-row {
  align-items: center;
  color: #344054;
  display: flex;
  gap: 8px;
  justify-content: space-between;
  margin: 8px 0;
}
.dot { border-radius: 999px; display: inline-block; height: 9px; width: 9px; }
.dot.orange { background: var(--aws-orange-2); }
.dot.blue { background: var(--aws-blue); }
.dot.green { background: var(--aws-green); }
.dot.red { background: var(--aws-red); }
.dot.gray { background: #98a2b3; }
.badge {
  background: #eef2f6;
  border-radius: 999px;
  color: #344054;
  display: inline-flex;
  font-size: 12px;
  font-weight: 750;
  line-height: 1;
  min-width: 22px;
  padding: 5px 7px;
  justify-content: center;
}
.workspace {
  min-width: 0;
  padding: 28px 32px 44px;
}
.workspace-narrow { max-width: 1180px; }
.breadcrumbs {
  align-items: center;
  color: #667085;
  display: flex;
  flex-wrap: wrap;
  font-size: 13px;
  gap: 8px;
  margin-bottom: 14px;
}
.page-head {
  align-items: start;
  display: flex;
  gap: 18px;
  justify-content: space-between;
  margin-bottom: 22px;
}
.page-title-group { min-width: 0; }
.page-meta {
  align-items: center;
  color: #475467;
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  margin-top: 14px;
}
.status-pill {
  align-items: center;
  border: 1px solid;
  border-radius: 4px;
  display: inline-flex;
  font-size: 12px;
  font-weight: 750;
  line-height: 1;
  padding: 5px 8px;
}
.status-pill.running, .status-pill.pending_human, .status-pill.reopened {
  background: #fff7ed;
  border-color: #fed7aa;
  color: #c2410c;
}
.status-pill.signed, .status-pill.final, .status-pill.done {
  background: #ecfdf3;
  border-color: #abefc6;
  color: #067647;
}
.status-pill.pending_evidence, .status-pill.pending_pre_review, .status-pill.draft {
  background: #eff8ff;
  border-color: #b2ddff;
  color: #175cd3;
}
.status-pill.blocked, .status-pill.rejected, .status-pill.overdue {
  background: #fef3f2;
  border-color: #fecdca;
  color: #b42318;
}
.status-pill.idle {
  background: #f2f4f7;
  border-color: #e4e7ec;
  color: #475467;
}
.quick-actions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
.main-grid {
  align-items: start;
  display: grid;
  gap: 22px;
  grid-template-columns: minmax(0, 1fr) 282px;
}
.content-stack { display: grid; gap: 18px; min-width: 0; }
.right-rail { display: grid; gap: 18px; min-width: 0; }
.panel, .card {
  background: var(--aws-panel);
  border: 1px solid var(--aws-line);
  border-radius: 6px;
  box-shadow: var(--shadow-sm);
  min-width: 0;
}
.card { margin: 0.55rem 0; padding: 12px 14px; }
.panel-header {
  align-items: center;
  border-bottom: 1px solid var(--aws-soft-line);
  display: flex;
  justify-content: space-between;
  min-height: 48px;
  padding: 13px 16px;
}
.panel-body { padding: 16px; }
.panel-tight .panel-body { padding: 12px 14px; }
.objective {
  border-left: 4px solid var(--aws-orange);
}
.objective p { color: #344054; margin: 8px 0 0; }
.summary-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.metric-box {
  background: #fff;
  border: 1px solid var(--aws-soft-line);
  border-radius: 6px;
  min-height: 92px;
  padding: 13px;
}
.metric-label {
  color: #667085;
  font-size: 12px;
  font-weight: 750;
  margin-bottom: 8px;
}
.metric-value { color: #111827; font-size: 25px; font-weight: 800; line-height: 1; }
.metric-note { color: #667085; font-size: 12px; margin-top: 7px; }
.phase-track {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(10, minmax(86px, 1fr));
  overflow-x: auto;
  padding: 3px 0 2px;
}
.phase {
  min-width: 86px;
  position: relative;
}
.phase::before {
  background: var(--aws-line);
  content: "";
  height: 2px;
  left: 24px;
  position: absolute;
  right: -20px;
  top: 15px;
}
.phase:last-child::before { display: none; }
.phase-node {
  align-items: center;
  background: #fff;
  border: 2px solid var(--aws-line);
  border-radius: 999px;
  color: #667085;
  display: flex;
  font-size: 12px;
  font-weight: 850;
  height: 30px;
  justify-content: center;
  position: relative;
  width: 30px;
  z-index: 1;
}
.phase.signed .phase-node { background: var(--aws-green); border-color: var(--aws-green); color: #fff; }
.phase.active .phase-node, .phase.reopened .phase-node {
  background: #fff8eb;
  border-color: var(--aws-orange);
  color: #cc5f00;
}
.phase.pending .phase-node { background: #eff8ff; border-color: #98caff; color: #175cd3; }
.phase-title { color: #172033; font-size: 12px; font-weight: 800; margin-top: 8px; }
.phase-state { color: #667085; font-size: 11px; margin-top: 2px; }
.card-grid { display: grid; gap: 12px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.card-mini {
  border: 1px solid var(--aws-soft-line);
  border-radius: 6px;
  padding: 12px;
}
.card-mini strong { display: block; font-size: 13px; margin-bottom: 4px; }
.card-mini span { color: #667085; font-size: 12px; }
.checkpoint-card {
  display: grid;
  gap: 16px;
  grid-template-columns: minmax(0, 1.15fr) minmax(260px, 0.85fr);
}
.metric-table {
  border: 1px solid var(--aws-soft-line);
  border-radius: 6px;
  overflow: hidden;
}
.metric-row {
  align-items: center;
  display: grid;
  grid-template-columns: minmax(120px, 1fr) 88px 88px 74px;
  min-height: 38px;
}
.metric-row:not(:last-child) { border-bottom: 1px solid var(--aws-soft-line); }
.metric-row span { padding: 8px 10px; }
.metric-row .delta-pos { color: #067647; font-weight: 800; }
.metric-row .delta-neg { color: #b42318; font-weight: 800; }
.callout {
  background: #fff8eb;
  border: 1px solid #fed7aa;
  border-radius: 6px;
  color: #7a3f00;
  padding: 12px;
}
.callout.red {
  background: #fff4f2;
  border-color: #fda29b;
  color: #912018;
}
.actions-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.timeline { display: grid; gap: 10px; }
.timeline-row {
  align-items: center;
  display: grid;
  gap: 12px;
  grid-template-columns: 74px 138px minmax(0, 1fr) auto;
  min-height: 42px;
}
.timeline-time {
  color: #344054;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}
.agent-label {
  border-radius: 4px;
  display: inline-flex;
  font-size: 12px;
  font-weight: 750;
  padding: 4px 7px;
}
.agent-label.orchestrator { background: #e0f2fe; color: #075985; }
.agent-label.reviewer { background: #ecfdf3; color: #067647; }
.agent-label.data_worker { background: #fff4e5; color: #c2410c; }
.agent-label.planner { background: #eef2ff; color: #3538cd; }
.agent-label.trainer { background: #fef3f2; color: #b42318; }
.agent-label.unknown { background: #f2f4f7; color: #475467; }
.artifact-grid { display: grid; gap: 10px 14px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.artifact {
  align-items: center;
  border: 1px solid var(--aws-soft-line);
  border-radius: 6px;
  color: #344054;
  display: grid;
  gap: 10px;
  grid-template-columns: 22px minmax(0, 1fr) auto;
  min-height: 40px;
  padding: 8px 10px;
}
.artifact-name { font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.artifact-type { color: #667085; font-size: 12px; white-space: nowrap; }
.tabbar {
  align-items: center;
  border-bottom: 1px solid var(--aws-line);
  display: flex;
  gap: 18px;
  overflow-x: auto;
  padding: 0 14px;
}
.tabbar a {
  border-bottom: 2px solid transparent;
  color: #344054;
  font-weight: 750;
  padding: 14px 0 12px;
  text-decoration: none;
  white-space: nowrap;
}
.tabbar a.active { border-bottom-color: var(--aws-orange); color: #cc5f00; }
.data-table { border-collapse: collapse; width: 100%; }
.data-table th, .data-table td {
  border-bottom: 1px solid var(--aws-soft-line);
  padding: 10px 12px;
  text-align: left;
  vertical-align: middle;
}
.data-table th {
  background: #fbfcfe;
  color: #475467;
  font-size: 12px;
  font-weight: 800;
}
.data-table td { color: #344054; }
.rail-list { display: grid; gap: 11px; }
.rail-row {
  align-items: center;
  display: flex;
  gap: 10px;
  justify-content: space-between;
}
.rail-row span:first-child { color: #475467; }
.rail-row strong { color: #111827; }
.progress-bar {
  background: #e4e7ec;
  border-radius: 999px;
  height: 6px;
  overflow: hidden;
  width: 118px;
}
.progress-fill { background: var(--aws-orange-2); height: 100%; }
.team-member {
  display: grid;
  gap: 2px;
  grid-template-columns: 18px minmax(0, 1fr);
}
.team-member strong { display: block; font-size: 13px; }
.team-member span { color: #667085; display: block; font-size: 12px; }
.collapse-link { margin-top: 26px; }
.meta { color: #667085; font-size: 0.87rem; }
.nav { color: #667085; font-size: 0.9rem; margin-bottom: 1rem; }
.event-start { border-left: 3px solid var(--aws-blue); }
.event-message { border-left: 3px solid #98a2b3; }
.event-turn { border-left: 3px solid var(--aws-green); }
.event-done { border-left: 3px solid #7c3aed; }
pre {
  background: #0f172a;
  border-radius: 4px;
  color: #e5e7eb;
  font-size: 0.85rem;
  line-height: 1.5;
  margin: 0.35rem 0;
  overflow-x: auto;
  padding: 0.75rem;
  white-space: pre-wrap;
  word-break: break-word;
}
.role-user { color: #006ce0; }
.role-assistant { color: #067647; }
.role-tool { color: #b42318; }
table { border-collapse: collapse; width: 100%; }
td, th {
  border-bottom: 1px solid var(--aws-soft-line);
  padding: 0.45rem 0.7rem;
  text-align: left;
  vertical-align: top;
}
th { background: #fbfcfe; color: #475467; }
code {
  background: #eef2f6;
  border-radius: 3px;
  color: #1d2939;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.9em;
  padding: 1px 4px;
}
.kv { color: #667085; font-size: 0.85rem; }
.tool-use {
  background: #f0fdf4;
  border: 1px solid #dcfae6;
  border-radius: 4px;
  margin: 0.35rem 0;
  padding: 0.45rem 0.65rem;
}
.tool-name { color: #067647; font-weight: bold; }
details { margin: 0.25rem 0; }
details > summary {
  background: #eff6ff;
  border-radius: 4px;
  color: #1849a9;
  cursor: pointer;
  font-size: 0.85rem;
  list-style: none;
  padding: 0.28rem 0.45rem;
  user-select: none;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before { color: #667085; content: "▶ "; display: inline-block; width: 1em; }
details[open] > summary::before { content: "▼ "; }
details > summary .preview {
  color: #667085;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin-left: 0.5em;
}
.bulkbar {
  background: #fff;
  border: 1px solid var(--aws-line);
  border-radius: 6px;
  box-shadow: var(--shadow-sm);
  font-size: 0.85rem;
  margin-bottom: 0.85rem;
  padding: 0.45rem;
  position: sticky;
  top: 78px;
  z-index: 10;
}
.bulkbar button {
  margin-right: 0.4rem;
  min-height: 29px;
  padding: 0.2rem 0.6rem;
}
.focus-page {
  background:
    radial-gradient(circle at 50% -18%, rgba(255, 153, 0, 0.16), transparent 32%),
    radial-gradient(circle at 82% 26%, rgba(9, 114, 211, 0.12), transparent 28%),
    linear-gradient(180deg, #04070a 0%, #020404 100%);
  color: #eef2f6;
  min-height: 100vh;
  overflow-x: hidden;
  padding: 0 28px 44px;
}
.focus-nav {
  align-items: center;
  display: flex;
  height: 70px;
  justify-content: space-between;
  margin: 0 auto;
  max-width: 1900px;
}
.focus-logo {
  color: #f7f7f7;
  font-size: 20px;
  font-weight: 900;
  letter-spacing: 0.34em;
}
.focus-logo span { color: var(--aws-orange); letter-spacing: 0; margin: 0 3px; }
.focus-links { align-items: center; display: flex; gap: 28px; }
.focus-links a {
  border-radius: 6px;
  color: #f7f7f7;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 19px;
  font-weight: 800;
  padding: 15px 22px;
  text-decoration: none;
}
.focus-links a.active { background: #161a20; box-shadow: inset 0 -2px 0 var(--aws-orange); }
.focus-hero {
  margin: 8px auto 34px;
  max-width: 1900px;
  text-align: center;
}
.focus-hero h1 {
  color: #f7f7f7;
  font-size: clamp(30px, 3vw, 54px);
  font-weight: 850;
  letter-spacing: 0;
  margin-top: 18px;
}
.focus-subtitle {
  color: #98a2b3;
  font-size: 15px;
  margin: 13px auto 20px;
  max-width: 900px;
}
.focus-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  justify-content: center;
}
.focus-chip {
  align-items: center;
  background: rgba(15, 23, 42, 0.78);
  border: 1px solid #293241;
  border-radius: 999px;
  color: #d0d5dd;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  gap: 8px;
  min-height: 42px;
  padding: 9px 18px;
}
.focus-stage {
  background: #12161d;
  border: 1px solid #222936;
  border-radius: 12px;
  box-shadow: 0 22px 80px rgba(0, 0, 0, 0.46);
  display: grid;
  grid-template-columns: 284px minmax(0, 1fr) minmax(430px, 0.92fr);
  margin: 0 auto;
  max-width: 1580px;
  min-height: 720px;
  overflow: hidden;
}
.focus-queue {
  background: #10141b;
  border-right: 1px solid #242b38;
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 28px 22px 20px;
}
.focus-queue-title {
  align-items: center;
  color: #f7f7f7;
  display: flex;
  font-size: 22px;
  font-weight: 850;
  gap: 10px;
  margin-bottom: 40px;
}
.focus-live { color: #35d07f; font-size: 13px; font-weight: 850; letter-spacing: 0.04em; }
.focus-tabs {
  background: #19202b;
  border: 1px solid #283246;
  border-radius: 10px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  margin-bottom: 16px;
  padding: 3px;
}
.focus-tabs span {
  border-radius: 7px;
  color: #667085;
  font-weight: 750;
  padding: 9px 10px;
  text-align: center;
}
.focus-tabs span.active { color: #f7f7f7; background: #222a3a; }
.queue-list {
  display: grid;
  gap: 12px;
  max-height: 500px;
  overflow-y: auto;
  padding-right: 5px;
}
.queue-item {
  background: #171d28;
  border: 1px solid #1f2734;
  border-radius: 7px;
  color: #f2f4f7;
  display: flex;
  justify-content: space-between;
  min-height: 60px;
  padding: 16px 18px;
}
.queue-item.active {
  background: #263044;
  border-color: #7d89bd;
  box-shadow: inset 3px 0 0 var(--aws-orange);
}
.queue-mode {
  align-items: center;
  background: #1c2432;
  border-radius: 999px;
  display: grid;
  gap: 4px;
  grid-template-columns: 1fr 1fr;
  margin-top: auto;
  padding: 5px;
}
.queue-mode span {
  border-radius: 999px;
  color: #98a2b3;
  padding: 9px 12px;
  text-align: center;
}
.queue-mode .active { background: #6573a8; color: #fff; }
.focus-work {
  background: #161c27;
  border-right: 1px solid #242b38;
  min-width: 0;
  padding: 26px 32px;
}
.focus-run-title {
  color: #f7f7f7;
  font-size: clamp(24px, 2vw, 38px);
  font-weight: 850;
  line-height: 1.15;
  margin: 0 0 18px;
}
.run-progress {
  align-items: start;
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 0 8px 28px;
  position: relative;
}
.run-progress::before {
  background: #596170;
  content: "";
  height: 3px;
  left: 12%;
  position: absolute;
  right: 12%;
  top: 14px;
}
.run-step {
  color: #8c94a3;
  font-weight: 850;
  position: relative;
  text-align: center;
  z-index: 1;
}
.run-step .node {
  align-items: center;
  background: #161c27;
  border: 3px solid #596170;
  border-radius: 999px;
  display: inline-flex;
  height: 28px;
  justify-content: center;
  margin-bottom: 7px;
  width: 28px;
}
.run-step.done .node { border-color: #8b96a8; color: #8b96a8; }
.run-step.active { color: #fff; }
.run-step.active .node {
  background: var(--aws-orange);
  border-color: var(--aws-orange);
  color: #12161d;
}
.task-list {
  display: grid;
  gap: 13px;
  font-size: 16px;
}
.task-row {
  align-items: start;
  color: #aeb6c3;
  display: grid;
  gap: 14px;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  min-height: 28px;
}
.task-row strong { color: #f2f4f7; font-weight: 650; }
.task-index { color: #f7f7f7; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.task-status {
  border-radius: 5px;
  font-size: 12px;
  font-weight: 850;
  padding: 4px 9px;
}
.task-status.done { background: rgba(46, 160, 67, 0.15); color: #35d07f; }
.task-status.run { background: rgba(9, 114, 211, 0.16); color: #72b7ff; }
.subtasks {
  color: #8d96a5;
  display: grid;
  gap: 8px;
  margin-top: 12px;
}
.subtasks span {
  align-items: start;
  display: grid;
  gap: 10px;
  grid-template-columns: 18px minmax(0, 1fr);
}
.subtasks .ok { color: #27c46f; }
.subtasks .spin { color: var(--aws-orange); }
.focus-live-panel {
  background: #171e2b;
  min-width: 0;
  padding: 28px 30px;
  position: relative;
}
.live-head {
  align-items: center;
  color: #f7f7f7;
  display: flex;
  font-size: 18px;
  font-weight: 850;
  justify-content: space-between;
  margin-bottom: 18px;
}
.live-stream {
  display: grid;
  gap: 22px;
  max-height: 485px;
  overflow-y: auto;
  padding-right: 10px;
}
.live-event {
  color: #d0d5dd;
  display: grid;
  gap: 9px;
  grid-template-columns: 58px minmax(0, 1fr);
}
.live-time { color: #8d96a5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.live-text { font-size: 15px; }
.tool-pill {
  background: #202837;
  border-radius: 8px;
  color: #dbe4f0;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  gap: 9px;
  margin-top: 7px;
  padding: 9px 12px;
}
.compute-float {
  background: #030506;
  border: 1px solid #181c24;
  border-radius: 10px;
  bottom: 28px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.48);
  padding: 26px 22px 18px;
  position: absolute;
  right: 28px;
  width: 270px;
}
.compute-rings {
  display: grid;
  gap: 14px;
  grid-template-columns: 1fr 1fr;
}
.ring {
  align-items: center;
  border: 10px solid #263047;
  border-top-color: #9eb7ff;
  border-radius: 999px;
  color: #fff;
  display: flex;
  font-size: 25px;
  font-weight: 900;
  height: 84px;
  justify-content: center;
  margin: 0 auto 8px;
  width: 84px;
}
.ring-label { color: #f2f4f7; font-weight: 800; text-align: center; }
.gpu-grid {
  display: grid;
  gap: 6px;
  grid-template-columns: repeat(8, 1fr);
  margin-top: 18px;
}
.gpu-grid span {
  background: #86a5ff;
  height: 8px;
  width: 8px;
}
.gpu-grid span.idle { background: #172033; }
.trace-link {
  color: #ffb84d;
  font-weight: 800;
}
.trace-shell {
  margin: 0 auto;
  max-width: 1480px;
  padding: 10px 0 52px;
}
.trace-hero {
  align-items: flex-start;
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 1fr) auto;
  margin: 8px 0 18px;
}
.trace-kicker {
  color: #ffb84d;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  font-weight: 850;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.trace-hero h1 {
  color: #f8fafc;
  font-size: clamp(28px, 2.7vw, 44px);
  margin: 5px 0 8px;
}
.trace-hero p {
  color: #aeb6c3;
  font-size: 15px;
  margin: 0;
  max-width: 900px;
}
.trace-actions {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}
.trace-actions .button {
  background: #111821;
  border-color: #344054;
  color: #eef2f6;
}
.trace-actions .button:hover {
  background: #1a2230;
  color: #fff;
}
.trace-actions .button.primary {
  background: linear-gradient(180deg, var(--aws-orange), var(--aws-orange-2));
  border-color: #e36d00;
  color: #fff;
}
.trace-tabs {
  align-items: center;
  border-bottom: 1px solid #2b3442;
  display: flex;
  gap: 10px;
  margin-bottom: 22px;
  overflow-x: auto;
}
.trace-tabs a {
  border-bottom: 3px solid transparent;
  color: #98a2b3;
  font-weight: 850;
  padding: 12px 4px 13px;
  text-decoration: none;
  white-space: nowrap;
}
.trace-tabs a.active {
  border-bottom-color: var(--aws-orange);
  color: #fff;
}
.trace-layout {
  align-items: start;
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 1fr) 330px;
}
.trace-stack { display: grid; gap: 16px; min-width: 0; }
.trace-panel {
  background: #111821;
  border: 1px solid #2e3847;
  border-radius: 10px;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
  color: #d9e2ee;
  min-width: 0;
  overflow: hidden;
}
.trace-panel-header {
  align-items: center;
  border-bottom: 1px solid #2a3341;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  min-height: 52px;
  padding: 14px 16px;
}
.trace-panel-title {
  color: #f8fafc;
  font-size: 15px;
  font-weight: 900;
}
.trace-panel-body { padding: 16px; }
.trace-stat-strip {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.trace-stat {
  background: #0b1016;
  border: 1px solid #28313d;
  border-radius: 8px;
  min-height: 88px;
  padding: 13px 14px;
}
.trace-stat span {
  color: #8d96a5;
  display: block;
  font-size: 12px;
  font-weight: 750;
}
.trace-stat strong {
  color: #fff;
  display: block;
  font-size: 27px;
  line-height: 1;
  margin-top: 10px;
}
.trace-summary-list, .trace-rail-list, .trace-agent-list, .trace-step-list {
  display: grid;
  gap: 10px;
}
.trace-summary-item {
  align-items: flex-start;
  background: #0c121a;
  border: 1px solid #26303c;
  border-radius: 8px;
  display: grid;
  gap: 11px;
  grid-template-columns: 30px minmax(0, 1fr);
  padding: 13px;
}
.trace-summary-icon {
  align-items: center;
  background: #1f2937;
  border-radius: 999px;
  color: #ffb84d;
  display: inline-flex;
  font-weight: 900;
  height: 28px;
  justify-content: center;
  width: 28px;
}
.trace-summary-item strong, .trace-agent-main strong, .trace-step-title {
  color: #f8fafc;
}
.trace-summary-item p, .trace-step p {
  color: #aeb6c3;
  margin: 4px 0 0;
}
.trace-agent-row {
  align-items: center;
  background: #0b1016;
  border: 1px solid #26303c;
  border-radius: 8px;
  display: grid;
  gap: 12px;
  grid-template-columns: 64px minmax(0, 1fr) 96px 120px auto;
  min-height: 58px;
  padding: 10px 12px;
}
.trace-agent-id {
  color: #fff;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 850;
}
.trace-agent-main span, .trace-step-meta, .trace-muted {
  color: #8d96a5;
  font-size: 12px;
}
.trace-progress-line {
  background: #26303c;
  border-radius: 999px;
  height: 7px;
  overflow: hidden;
}
.trace-progress-line span {
  background: linear-gradient(90deg, var(--aws-orange), #ffbf66);
  display: block;
  height: 100%;
}
.trace-rail-card {
  background: #0b1016;
  border: 1px solid #26303c;
  border-radius: 8px;
  color: #c7d0dd;
  padding: 14px;
}
.trace-rail-title {
  color: #8d96a5;
  font-size: 11px;
  font-weight: 850;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
  text-transform: uppercase;
}
.trace-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.trace-chip {
  background: #1a2230;
  border: 1px solid #344054;
  border-radius: 999px;
  color: #d9e2ee;
  display: inline-flex;
  font-size: 12px;
  font-weight: 750;
  padding: 5px 9px;
}
.trace-chip.orange {
  background: rgba(255, 153, 0, 0.13);
  border-color: rgba(255, 153, 0, 0.35);
  color: #ffbf66;
}
.trace-step {
  background: #0b1016;
  border: 1px solid #26303c;
  border-radius: 9px;
  display: grid;
  gap: 13px;
  grid-template-columns: 42px minmax(0, 1fr) auto;
  padding: 14px;
}
.trace-step-num {
  align-items: center;
  background: #1d2733;
  border: 1px solid #394557;
  border-radius: 999px;
  color: #ffb84d;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 900;
  height: 36px;
  justify-content: center;
  width: 36px;
}
.trace-detail > summary,
.trace-event > summary {
  background: #151e2a;
  border: 1px solid #2e3847;
  border-radius: 7px;
  color: #d9e2ee;
  font-weight: 750;
  margin-top: 10px;
  padding: 8px 10px;
}
.trace-detail > summary::before,
.trace-event > summary::before {
  content: "";
  display: none;
}
.trace-detail pre,
.trace-event pre {
  background: #05080c;
  border: 1px solid #202938;
  color: #d9e2ee;
  margin-top: 10px;
}
.trace-call-row {
  align-items: center;
  background: #0b1016;
  border: 1px solid #26303c;
  border-radius: 8px;
  display: grid;
  gap: 12px;
  grid-template-columns: 64px minmax(0, 1fr) 80px auto;
  min-height: 56px;
  padding: 10px 12px;
}
.trace-tool-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.trace-event-list { display: grid; gap: 8px; }
.trace-event {
  background: #0b1016;
  border: 1px solid #26303c;
  border-radius: 8px;
  padding: 10px 12px;
}
.trace-event > summary {
  align-items: center;
  background: transparent;
  border: 0;
  display: grid;
  gap: 10px;
  grid-template-columns: 88px minmax(0, 1fr) auto;
  margin: 0;
  padding: 0;
}
.trace-event-kind {
  color: #ffb84d;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
}
.trace-event-preview {
  color: #d9e2ee;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.qp-stage {
  align-items: stretch;
  background: #10151d;
  border: 1px solid #28313d;
  border-radius: 12px;
  box-shadow: 0 22px 80px rgba(0, 0, 0, 0.5);
  display: grid;
  grid-template-columns: 330px minmax(0, 1fr) 390px;
  margin: 0 auto;
  max-width: 1560px;
  min-height: 730px;
  overflow: hidden;
}
.qp-ledger {
  background: #0e131a;
  border-right: 1px solid #28313d;
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 22px 18px;
}
.qp-ledger-head {
  align-items: center;
  color: #f8fafc;
  display: flex;
  justify-content: space-between;
  margin-bottom: 14px;
}
.qp-ledger-title { font-size: 18px; font-weight: 900; }
.qp-ledger-project {
  color: #98a2b3;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
}
.qp-plan-row {
  align-items: center;
  border-bottom: 1px solid #242c38;
  color: #c7d0dd;
  display: grid;
  font-size: 12px;
  gap: 8px;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  min-height: 44px;
  padding: 8px 4px;
  text-decoration: none;
  cursor: pointer;
}
.qp-plan-row:hover { background: #15202f; }
.qp-plan-row.active {
  background: #1b2635;
  border-left: 3px solid var(--aws-orange);
  margin-left: -4px;
  padding-left: 9px;
}
.qp-plan-row.signed { color: #768193; }
.qp-plan-row.signed .qp-row-title { text-decoration: line-through; }
.qp-num {
  color: #8792a3;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  text-align: right;
}
.qp-row-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.qp-type {
  background: #1b2430;
  border-radius: 3px;
  color: #98a2b3;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 9px;
  padding: 2px 5px;
  text-transform: uppercase;
}
.qp-ledger-note {
  color: #98a2b3;
  font-size: 12px;
  line-height: 1.55;
  margin-top: auto;
  padding-top: 16px;
}
.qp-workspace {
  background: #151b25;
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 24px 30px;
}
.qp-workspace-head {
  align-items: center;
  display: flex;
  justify-content: space-between;
  margin-bottom: 14px;
}
.qp-bottom-stats {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: auto;
  padding-top: 16px;
}
.qp-workspace .panel {
  background: #0b1016;
  border: 1px solid #303947;
  border-radius: 7px;
  box-shadow: none;
  color: #e6edf3;
  padding: 12px;
}
.qp-workspace .panel-title {
  color: #98a2b3;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
  text-transform: uppercase;
}
.qp-workspace .panel strong {
  color: #ffb84d;
  display: block;
  font-size: 22px;
  line-height: 1;
}
.qp-card {
  background: #111821;
  border: 1px solid #303947;
  border-radius: 10px;
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
  color: #e6edf3;
  overflow: hidden;
}
.qp-card.eval { border-color: #4d6333; }
.qp-card.plan { border-color: #6a4a16; }
.qp-card.reopen { border-color: #8a4b1e; }
.qp-card.dispute { border-color: #7b2d28; }
.qp-card-header {
  align-items: center;
  background: linear-gradient(90deg, #1d2733, #111821);
  border-bottom: 1px solid #303947;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  min-height: 58px;
  padding: 12px 18px;
}
.qp-card.plan .qp-card-header { background: linear-gradient(90deg, #2f210d, #111821); }
.qp-card.eval .qp-card-header { background: linear-gradient(90deg, #142513, #111821); }
.qp-card.reopen .qp-card-header { background: linear-gradient(90deg, #37210e, #111821); }
.qp-card.dispute .qp-card-header { background: linear-gradient(90deg, #321818, #111821); }
.qp-card-left, .qp-card-right {
  align-items: center;
  display: flex;
  gap: 10px;
  min-width: 0;
}
.qp-card-id {
  background: #070b10;
  border-radius: 3px;
  color: #98a2b3;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  padding: 3px 7px;
}
.qp-card-title {
  color: #f8fafc;
  font-size: 16px;
  font-weight: 850;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.qp-state {
  border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.02em;
  padding: 3px 8px;
  text-transform: uppercase;
  white-space: nowrap;
}
.qp-state.pending { background: #33250d; color: #ffb84d; }
.qp-state.signed { background: #0c2816; color: #3fb950; }
.qp-state.reopened { background: #3a210b; color: #ff9900; }
.qp-score {
  align-items: center;
  background: #070b10;
  border-radius: 999px;
  color: #98a2b3;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  gap: 6px;
  padding: 4px 10px;
}
.qp-score b { color: #ffb84d; font-size: 14px; }
.qp-score.good b { color: #3fb950; }
.qp-card-body { padding: 16px 20px; }
.qp-section {
  background: #0b1016;
  border-left: 2px solid #303947;
  border-radius: 6px;
  margin: 12px 0;
  padding: 10px 14px;
}
.qp-section.highlight { border-left-color: var(--aws-orange); }
.qp-section.warn { border-left-color: #ffb84d; }
.qp-section.red { border-left-color: #f85149; background: #1c1112; }
.qp-section-title {
  color: #98a2b3;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.08em;
  margin-bottom: 7px;
  text-transform: uppercase;
}
.qp-section-body {
  color: #e6edf3;
  font-size: 13px;
  line-height: 1.55;
}
.qp-metric-row {
  display: grid;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  gap: 8px;
  grid-template-columns: 1.35fr 0.8fr 0.8fr 0.7fr;
  padding: 4px 0;
}
.qp-metric-row span:not(:first-child) { text-align: right; }
.qp-delta-up { color: #3fb950; font-weight: 850; }
.qp-delta-down { color: #f85149; font-weight: 850; }
.qp-prereview {
  align-items: center;
  background: #1c1510;
  border-bottom: 1px solid #303947;
  color: #ffb84d;
  display: flex;
  font-size: 12px;
  gap: 8px;
  padding: 9px 18px;
}
.qp-signers {
  align-items: center;
  background: #0b1016;
  border-top: 1px solid #303947;
  display: flex;
  gap: 10px;
  padding: 11px 14px;
}
.qp-signer {
  background: #202938;
  border-radius: 999px;
  color: #98a2b3;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  padding: 5px 10px;
}
.qp-signer.signed { background: #0c2816; color: #3fb950; }
.qp-signer.waiting { border: 1px dashed #667085; }

/* Dual-sign chips: reviewer verdict (left) + human signoff (right).
   Shown on each ledger row and inside the current card header. The old
   ``qp-state`` pill stays adjacent so no regression for callers that
   grep the DOM for state text. */
.qp-dual-sign {
  align-items: center;
  display: inline-flex;
  gap: 6px;
  margin-left: 8px;
  white-space: nowrap;
}
.qp-chip {
  align-items: center;
  background: #202938;
  border: 1px solid #2b3648;
  border-radius: 999px;
  color: #98a2b3;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10.5px;
  font-weight: 700;
  gap: 4px;
  letter-spacing: 0.02em;
  line-height: 1;
  padding: 4px 9px;
  text-transform: uppercase;
}
.qp-chip-label { color: #667085; font-weight: 700; }
.qp-chip.rev-ready { background: #0c2028; border-color: #1c4463; color: #58a6ff; }
.qp-chip.rev-evidence { background: #0c2028; border-color: #1c4463; color: #98c1ff; }
.qp-chip.rev-insufficient { background: #33250d; border-color: #76521b; color: #ffb84d; }
.qp-chip.rev-reject { background: #3a1417; border-color: #7a2d34; color: #f48c9a; }
.qp-chip.rev-none { border-style: dashed; color: #667085; }
.qp-chip.hum-signed { background: #0c2816; border-color: #245d34; color: #3fb950; }
.qp-chip.hum-signed-auto { background: #132019; border-color: #1f3d2a; color: #7ac987; }
.qp-chip.hum-waiting { border-style: dashed; color: #98a2b3; }
.qp-connector { background: #303947; flex: 1; height: 1px; }
.qp-actions {
  background: #0b1016;
  border-top: 1px solid #303947;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 11px 14px;
}
.qp-btn {
  background: #202938;
  border: 1px solid #303947;
  border-radius: 5px;
  color: #e6edf3;
  cursor: pointer;
  font-size: 12px;
  font-weight: 750;
  min-height: 32px;
  padding: 6px 12px;
}
.qp-btn.primary { background: #3a2406; border-color: var(--aws-orange); color: #ffb84d; }
.qp-btn.destructive { background: #2d1618; border-color: #f85149; color: #ff8d86; }
.qp-side {
  background: #0e131a;
  border-left: 1px solid #28313d;
  display: grid;
  gap: 16px;
  align-content: start;
  min-width: 0;
  padding: 22px 18px;
}
.qp-context-head {
  background: #0b1016;
  border: 1px solid #303947;
  border-radius: 10px;
  color: #e6edf3;
  padding: 13px 14px;
}
.qp-context-kicker {
  color: #ffb84d;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.1em;
  margin-bottom: 6px;
  text-transform: uppercase;
}
.qp-context-head h3 {
  color: #f8fafc;
  font-size: 17px;
  line-height: 1.2;
  margin: 0;
}
.qp-context-head p {
  color: #98a2b3;
  font-size: 12px;
  line-height: 1.45;
  margin: 8px 0 0;
}
.qp-context-step {
  align-items: center;
  display: flex;
  gap: 8px;
}
.qp-step-num {
  align-items: center;
  background: #33250d;
  border: 1px solid #76521b;
  border-radius: 999px;
  color: #ffb84d;
  display: inline-flex;
  flex: 0 0 auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  font-weight: 900;
  height: 22px;
  justify-content: center;
  width: 22px;
}
.qp-mini .qp-card-title { white-space: normal; }
.qp-mini .qp-card-header { min-height: 48px; padding: 10px 14px; }
.qp-mini .qp-card-title { font-size: 14px; }
.qp-mini .qp-card-body { padding: 12px 14px; }
.qp-mini .qp-section { margin: 8px 0; padding: 8px 10px; }
.qp-pill {
  background: #1b2430;
  border-radius: 999px;
  color: #98a2b3;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  margin: 3px 4px 3px 0;
  padding: 3px 8px;
}
.qp-pill.ok { background: #0c2816; color: #3fb950; }
.qp-pill.warn { background: #33250d; color: #ffb84d; }

.qp-evidence-group {
  border: 1px solid #232c38;
  border-radius: 6px;
  margin: 8px 0;
  padding: 8px 10px;
}
.qp-evidence-kind {
  color: #c7d0dd;
  display: flex;
  align-items: center;
  font-size: 12px;
  font-weight: 600;
  gap: 6px;
  margin-bottom: 6px;
}
.qp-evidence-item {
  align-items: center;
  background: #0f1824;
  border: 1px solid #1d2735;
  border-radius: 5px;
  color: #c7d0dd;
  display: grid;
  font-size: 12px;
  gap: 6px;
  grid-template-columns: minmax(0, 1fr) auto auto;
  margin: 4px 0;
  padding: 6px 8px;
  text-decoration: none;
}
.qp-evidence-item:hover {
  background: #15202f;
  border-color: #2e3a4d;
}
.qp-evidence-title { color: #e6edf3; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.qp-evidence-meta { color: #768193; font-size: 11px; }
.qp-evidence-id { color: #768193; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
.qp-evidence-empty { color: #768193; font-size: 12px; padding: 4px 0; }
.qp-evidence-extra { color: #768193; font-size: 11px; padding: 4px 0; }

.record-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: 1fr 1fr;
  padding: 18px;
}
.record-pane {
  background: #0f1824;
  border: 1px solid #1d2735;
  border-radius: 6px;
  padding: 12px 14px;
}
.record-pane.full { grid-column: 1 / -1; }
.record-pane h3 {
  color: #e6edf3;
  font-size: 13px;
  letter-spacing: 0.04em;
  margin: 0 0 8px 0;
  text-transform: uppercase;
}
.record-links { list-style: none; margin: 0; padding: 0; }
.record-links li { margin: 3px 0; }
.record-body {
  background: #080c12;
  border: 1px solid #1d2735;
  border-radius: 4px;
  color: #c7d0dd;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  margin: 0;
  max-height: 560px;
  overflow: auto;
  padding: 10px 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.run-rail {
  background: #0e131a;
  border-left: 1px solid #28313d;
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
  padding: 22px 18px;
}
.run-rail-head {
  background: linear-gradient(180deg, #141b25, #0b1016);
  border: 1px solid #303947;
  border-radius: 10px;
  color: #e6edf3;
  padding: 14px;
}
.run-rail-kicker {
  color: #ffb84d;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.1em;
  margin-bottom: 6px;
  text-transform: uppercase;
}
.run-rail-head h3 {
  color: #f8fafc;
  font-size: 18px;
  line-height: 1.2;
  margin: 0;
}
.run-rail-head p {
  color: #98a2b3;
  font-size: 12px;
  line-height: 1.45;
  margin: 8px 0 0;
}
.run-panel {
  background: #0b1016;
  border: 1px solid #303947;
  border-radius: 10px;
  color: #e6edf3;
  padding: 13px 14px;
}
.run-panel.shortcuts { margin-top: auto; }
.run-panel.live {
  background: linear-gradient(180deg, #101925, #0b1016);
  border-color: #4a3411;
}
.run-panel-title {
  color: #98a2b3;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.09em;
  margin-bottom: 10px;
  text-transform: uppercase;
}
.run-stat-grid {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
.run-stat {
  background: #121923;
  border: 1px solid #242e3a;
  border-radius: 7px;
  padding: 9px;
}
.run-stat strong {
  color: #ffb84d;
  display: block;
  font-size: 20px;
  line-height: 1;
}
.run-stat span {
  color: #98a2b3;
  display: block;
  font-size: 11px;
  margin-top: 5px;
}
.run-progress-line {
  background: #202938;
  border-radius: 999px;
  height: 8px;
  overflow: hidden;
}
.run-progress-line span {
  background: linear-gradient(90deg, #ff9900, #ffb84d);
  display: block;
  height: 100%;
}
.live-agent-card {
  display: grid;
  gap: 10px;
}
.live-agent-top {
  align-items: center;
  display: flex;
  gap: 8px;
  justify-content: space-between;
}
.live-agent-name {
  color: #f8fafc;
  font-size: 15px;
  font-weight: 900;
}
.live-dot {
  animation: livePulse 1.6s ease-in-out infinite;
  background: #35d07f;
  border-radius: 999px;
  box-shadow: 0 0 0 0 rgba(53, 208, 127, 0.5);
  display: inline-block;
  height: 8px;
  margin-right: 6px;
  width: 8px;
}
@keyframes livePulse {
  0% { box-shadow: 0 0 0 0 rgba(53, 208, 127, 0.5); }
  70% { box-shadow: 0 0 0 7px rgba(53, 208, 127, 0); }
  100% { box-shadow: 0 0 0 0 rgba(53, 208, 127, 0); }
}
.live-activity-title {
  color: #ffb84d;
  font-size: 13px;
  font-weight: 850;
}
.live-activity-body {
  color: #c7d0dd;
  font-size: 12px;
  line-height: 1.45;
}
.live-meta-grid {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.live-meta {
  background: #121923;
  border: 1px solid #242e3a;
  border-radius: 7px;
  padding: 8px;
}
.live-meta span {
  color: #8d96a5;
  display: block;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.live-meta strong {
  color: #f8fafc;
  display: block;
  font-size: 12px;
  margin-top: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.live-feed {
  display: grid;
  gap: 10px;
}
.live-feed-item {
  border-left: 2px solid #344054;
  padding-left: 10px;
}
.live-feed-item.active { border-left-color: #ff9900; }
.live-feed-title {
  color: #f8fafc;
  font-size: 12px;
  font-weight: 850;
}
.live-feed-body {
  color: #98a2b3;
  font-size: 11px;
  line-height: 1.4;
  margin-top: 2px;
}
.run-readable-list {
  display: grid;
  gap: 10px;
}
.run-readable-item {
  display: grid;
  gap: 9px;
  grid-template-columns: 24px minmax(0, 1fr);
}
.run-readable-dot {
  align-items: center;
  background: #172033;
  border: 1px solid #303947;
  border-radius: 999px;
  color: #98a2b3;
  display: flex;
  font-size: 12px;
  font-weight: 900;
  height: 24px;
  justify-content: center;
  width: 24px;
}
.run-readable-item.done .run-readable-dot {
  background: #0c2816;
  border-color: #245d34;
  color: #3fb950;
}
.run-readable-item.active .run-readable-dot {
  background: #33250d;
  border-color: #76521b;
  color: #ffb84d;
}
.run-readable-item.blocked .run-readable-dot {
  background: #2d1618;
  border-color: #7b2d28;
  color: #ff8d86;
}
.run-readable-title {
  color: #f8fafc;
  font-size: 13px;
  font-weight: 850;
}
.run-readable-body {
  color: #98a2b3;
  font-size: 12px;
  line-height: 1.45;
  margin-top: 2px;
}
.run-note {
  background: #16120a;
  border: 1px solid #4a3411;
  border-radius: 8px;
  color: #ffdb9b;
  font-size: 12px;
  line-height: 1.5;
  padding: 10px 12px;
}
.run-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}
.run-chip { 
  background: #121923;
  border: 1px solid #242e3a;
  border-radius: 999px;
  color: #c7d0dd;
  font-size: 11px;
  font-weight: 750;
  padding: 5px 9px;
}
.run-chip.warn { background: #33250d; border-color: #76521b; color: #ffb84d; }
.run-chip.ok { background: #0c2816; border-color: #245d34; color: #3fb950; }
.lb-shell {
  margin: 0 auto;
  max-width: 1480px;
  padding-bottom: 36px;
}
.lb-header {
  align-items: flex-end;
  display: flex;
  gap: 18px;
  justify-content: space-between;
  margin: 18px 0 22px;
}
.lb-title h1 {
  color: #f8fafc;
  font-size: clamp(30px, 3vw, 48px);
  margin: 0;
}
.lb-title p {
  color: #98a2b3;
  font-size: 14px;
  line-height: 1.5;
  margin: 8px 0 0;
  max-width: 840px;
}
.lb-card {
  background: #10151d;
  border: 1px solid #28313d;
  border-radius: 12px;
  box-shadow: 0 20px 70px rgba(0, 0, 0, 0.42);
  color: #e6edf3;
  overflow: hidden;
}
.lb-toolbar {
  align-items: center;
  background: #0b1016;
  border-bottom: 1px solid #28313d;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: space-between;
  padding: 13px 16px;
}
.lb-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.lb-tab {
  background: #151d28;
  border: 1px solid #293342;
  border-radius: 999px;
  color: #c7d0dd;
  font-size: 12px;
  font-weight: 800;
  padding: 6px 10px;
}
.lb-tab.active { background: #33250d; border-color: #76521b; color: #ffb84d; }
.lb-table { border-collapse: collapse; width: 100%; }
.lb-table th,
.lb-table td {
  border-bottom: 1px solid #242c38;
  padding: 12px 14px;
  text-align: left;
  vertical-align: middle;
}
.lb-table th {
  background: #111821;
  color: #98a2b3;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
.lb-table td { color: #d8dee9; font-size: 13px; }
.lb-rank {
  color: #ffb84d;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 15px;
  font-weight: 900;
}
.lb-run-name {
  color: #f8fafc;
  display: block;
  font-weight: 900;
  margin-bottom: 3px;
}
.lb-run-sub {
  color: #98a2b3;
  display: block;
  font-size: 12px;
}
.lb-score {
  color: #f8fafc;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 16px;
  font-weight: 900;
}
.lb-delta-up { color: #3fb950; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 900; }
.lb-delta-down { color: #f85149; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 900; }
.lb-status {
  border-radius: 999px;
  display: inline-flex;
  font-size: 11px;
  font-weight: 900;
  padding: 5px 9px;
  white-space: nowrap;
}
.lb-status.best { background: #0c2816; color: #3fb950; }
.lb-status.review { background: #33250d; color: #ffb84d; }
.lb-status.blocked { background: #2d1618; color: #ff8d86; }
.lb-status.archived { background: #172033; color: #98a2b3; }
.lb-detail-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 1fr) 360px;
}
.recipe-panel {
  background: #10151d;
  border: 1px solid #28313d;
  border-radius: 12px;
  color: #e6edf3;
  overflow: hidden;
}
.recipe-head {
  background: linear-gradient(90deg, #1b2635, #10151d);
  border-bottom: 1px solid #28313d;
  padding: 15px 18px;
}
.recipe-head h2 {
  color: #f8fafc;
  font-size: 22px;
  margin: 0;
}
.recipe-head p {
  color: #98a2b3;
  font-size: 13px;
  margin: 6px 0 0;
}
.recipe-body { padding: 16px 18px; }
.recipe-section {
  background: #0b1016;
  border-left: 2px solid #303947;
  border-radius: 7px;
  margin-bottom: 13px;
  padding: 11px 13px;
}
.recipe-section.orange { border-left-color: #ff9900; }
.recipe-section.green { border-left-color: #3fb950; }
.recipe-section.red { border-left-color: #f85149; }
.recipe-title {
  color: #98a2b3;
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
  text-transform: uppercase;
}
.recipe-kv {
  display: grid;
  gap: 8px 14px;
  grid-template-columns: 140px minmax(0, 1fr);
  font-size: 13px;
}
.recipe-kv span:nth-child(odd) { color: #98a2b3; }
.recipe-kv span:nth-child(even) { color: #e6edf3; }
.recipe-list {
  color: #e6edf3;
  display: grid;
  font-size: 13px;
  gap: 7px;
  line-height: 1.45;
  margin: 0;
  padding: 0;
}
.recipe-list li {
  color: #e6edf3;
  list-style: none;
}
.recipe-list li::before { color: #ffb84d; content: "→ "; font-weight: 900; }
.breakdown-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.breakdown-cell {
  background: #121923;
  border: 1px solid #242e3a;
  border-radius: 8px;
  padding: 10px;
}
.breakdown-cell strong {
  color: #f8fafc;
  display: block;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 17px;
}
.breakdown-cell span {
  color: #98a2b3;
  display: block;
  font-size: 11px;
  margin-top: 5px;
}

/* SpaceX-style telemetry skin: visual-only overrides, content unchanged. */
body {
  background: var(--spacex-black);
}
.focus-page {
  background:
    radial-gradient(circle at 12% 18%, rgba(255, 255, 255, 0.88) 0 1px, transparent 1.6px),
    radial-gradient(circle at 28% 68%, rgba(180, 212, 245, 0.72) 0 1px, transparent 1.7px),
    radial-gradient(circle at 47% 31%, rgba(255, 255, 255, 0.56) 0 1px, transparent 1.5px),
    radial-gradient(circle at 72% 12%, rgba(214, 232, 250, 0.76) 0 1px, transparent 1.7px),
    radial-gradient(circle at 86% 58%, rgba(255, 255, 255, 0.5) 0 1px, transparent 1.5px),
    linear-gradient(180deg, rgba(2, 6, 11, 0.97), rgba(7, 13, 21, 0.96) 52%, rgba(2, 5, 9, 0.98));
  background-size: 520px 420px, 680px 520px, 760px 540px, 620px 460px, 840px 620px, 100% 100%;
  color: var(--spacex-ice);
}
.focus-nav {
  border-bottom: 1px solid rgba(224, 236, 248, 0.12);
}
.focus-logo {
  color: var(--spacex-ice);
  letter-spacing: 0.28em;
}
.focus-logo span {
  color: var(--spacex-blue);
}
.focus-links a {
  border: 1px solid transparent;
  color: rgba(248, 251, 255, 0.82);
}
.focus-links a.active {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(210, 228, 246, 0.2);
  box-shadow: inset 0 -2px 0 var(--spacex-ice);
  color: #fff;
}
.focus-subtitle,
.trace-hero p,
.lb-title p,
.qp-context-head p,
.trace-muted {
  color: #aab7c6;
}
.focus-chip,
.lb-tab,
.run-chip,
.trace-chip,
.qp-pill {
  background: rgba(10, 18, 28, 0.72);
  border: 1px solid rgba(210, 228, 246, 0.2);
  color: #d8e4f0;
}
.qp-stage,
.lb-card,
.recipe-panel,
.trace-panel,
.qp-card {
  background: linear-gradient(180deg, rgba(14, 24, 36, 0.86), rgba(5, 10, 17, 0.94));
  border-color: rgba(210, 228, 246, 0.22);
  box-shadow: 0 24px 90px rgba(0, 0, 0, 0.55);
}
.qp-ledger,
.qp-workspace,
.qp-side,
.trace-rail-card,
.trace-stat,
.trace-summary-item,
.trace-agent-row,
.trace-step,
.trace-call-row,
.trace-event,
.recipe-section,
.breakdown-cell {
  background: rgba(4, 10, 17, 0.76);
  border-color: rgba(210, 228, 246, 0.18);
}
.qp-card-header,
.recipe-head,
.lb-toolbar {
  background: linear-gradient(90deg, rgba(230, 241, 252, 0.1), rgba(5, 10, 17, 0.2));
  border-color: rgba(210, 228, 246, 0.18);
}
.trace-kicker,
.qp-context-kicker,
.qp-section-title,
.recipe-title,
.lb-rank,
.qp-workspace .panel strong,
.qp-score b,
.qp-step-num,
.trace-summary-icon,
.trace-step-num,
.trace-event-kind,
.recipe-list li::before {
  color: var(--spacex-ice) !important;
}
.trace-link:hover {
  color: var(--spacex-blue) !important;
}
.button.primary,
button.primary,
.trace-actions .button.primary,
.qp-btn.primary {
  background: linear-gradient(180deg, rgba(248, 251, 255, 0.98), rgba(127, 200, 255, 0.86));
  border-color: rgba(248, 251, 255, 0.72);
  color: #050b12;
}
.status-pill.running,
.status-pill.pending_human,
.status-pill.reopened,
.lb-tab.active,
.lb-status.review,
.qp-state.pending,
.qp-state.reopened,
.run-chip.warn,
.qp-pill.warn {
  background: rgba(127, 200, 255, 0.12);
  border-color: rgba(127, 200, 255, 0.34);
  color: #dff2ff;
}
.qp-plan-row.active {
  background: rgba(127, 200, 255, 0.1);
  border-left-color: var(--spacex-ice);
}
.qp-section.highlight,
.recipe-section.orange {
  border-left-color: var(--spacex-ice);
}
.progress-bar {
  background: rgba(210, 228, 246, 0.18);
  border: 1px solid rgba(210, 228, 246, 0.2);
  border-radius: 2px;
  height: 8px;
  overflow: visible;
}
.progress-fill {
  background: linear-gradient(90deg, #ffffff, var(--spacex-blue));
  box-shadow: 0 0 14px rgba(127, 200, 255, 0.55);
}
.run-progress {
  background:
    linear-gradient(180deg, rgba(248, 251, 255, 0.06), rgba(3, 6, 10, 0.1));
  border: 1px solid rgba(210, 228, 246, 0.18);
  border-radius: 10px;
  min-height: 96px;
  padding: 24px 14px 14px;
}
.run-progress::before {
  background: transparent;
  border-bottom: 2px solid rgba(248, 251, 255, 0.72);
  border-radius: 0 0 50% 50% / 0 0 100% 100%;
  height: 68px;
  left: 7%;
  right: 7%;
  top: 0;
}
.run-step .node {
  background: #06101a;
  border-color: rgba(248, 251, 255, 0.64);
  box-shadow: 0 0 0 3px rgba(127, 200, 255, 0.08);
}
.run-step.active .node {
  background: var(--spacex-ice);
  border-color: var(--spacex-ice);
  box-shadow: 0 0 20px rgba(127, 200, 255, 0.75);
  color: #02060a;
}
.launch-page {
  background:
    linear-gradient(180deg, rgba(0, 0, 0, 0.72) 0%, rgba(0, 0, 0, 0.24) 38%, rgba(0, 0, 0, 0.86) 78%, #020305 100%),
    linear-gradient(90deg, rgba(1, 10, 18, 0.72), rgba(0, 0, 0, 0.14) 50%, rgba(36, 13, 5, 0.58)),
    url("https://images-assets.nasa.gov/image/NHQ202409280011/NHQ202409280011~orig.jpg") center 44% / cover no-repeat,
    #010204;
  color: #f8fbff;
  min-height: 100vh;
  overflow: hidden;
  padding: 26px 40px 34px;
  position: relative;
}
.launch-page::before {
  background:
    linear-gradient(180deg, transparent 0 64%, rgba(0, 0, 0, 0.62) 100%);
  content: "";
  inset: 0;
  opacity: 1;
  position: absolute;
}
.launch-page::after {
  background:
    radial-gradient(circle at 8% 24%, rgba(232, 245, 255, 0.9) 0 1px, transparent 1.6px),
    radial-gradient(circle at 84% 16%, rgba(255, 175, 82, 0.9) 0 1px, transparent 1.5px),
    radial-gradient(circle at 70% 68%, rgba(232, 245, 255, 0.48) 0 1px, transparent 1.4px);
  background-size: 460px 360px, 620px 420px, 760px 520px;
  content: "";
  inset: 0;
  opacity: 0.58;
  pointer-events: none;
  position: absolute;
}
.launch-nav,
.launch-scene,
.launch-activity {
  position: relative;
  z-index: 1;
}
.launch-nav {
  align-items: flex-start;
  display: flex;
  justify-content: space-between;
  gap: 18px;
}
.launch-brand {
  color: #fff;
  font-size: clamp(20px, 2vw, 31px);
  font-weight: 850;
  letter-spacing: 0.36em;
  line-height: 1;
}
.launch-brand span {
  color: #ff8b22;
}
.launch-subbrand {
  color: rgba(248, 251, 255, 0.72);
  font-size: 12px;
  font-weight: 750;
  letter-spacing: 0.34em;
  margin-top: 9px;
  text-transform: uppercase;
}
.launch-status {
  align-items: center;
  background: rgba(2, 5, 9, 0.72);
  border: 1px solid rgba(248, 251, 255, 0.28);
  border-radius: 4px;
  color: #f8fbff;
  display: inline-flex;
  font-size: 12px;
  font-weight: 850;
  gap: 10px;
  letter-spacing: 0.08em;
  min-height: 36px;
  padding: 8px 14px;
  text-transform: uppercase;
}
.launch-status-dot {
  background: #30d158;
  border-radius: 999px;
  box-shadow: 0 0 14px rgba(48, 209, 88, 0.72);
  height: 8px;
  width: 8px;
}
.launch-scene {
  align-items: center;
  display: grid;
  min-height: calc(100vh - 245px);
  padding: 24px 0 12px;
}
.launch-title {
  align-self: end;
  justify-self: center;
  margin-bottom: 2px;
  text-align: center;
}
.launch-title h1 {
  color: #fff;
  font-size: clamp(38px, 4vw, 66px);
  font-weight: 850;
  letter-spacing: 0.34em;
  margin: 0;
}
.launch-title p {
  color: rgba(248, 251, 255, 0.78);
  font-size: clamp(13px, 1.2vw, 18px);
  font-weight: 750;
  letter-spacing: 0.42em;
  margin: 12px 0 18px;
  text-transform: uppercase;
}
.launch-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
}
.launch-link {
  background: rgba(248, 251, 255, 0.08);
  border: 1px solid rgba(248, 251, 255, 0.34);
  border-radius: 4px;
  color: #fff;
  font-size: 12px;
  font-weight: 850;
  letter-spacing: 0.12em;
  min-height: 38px;
  padding: 10px 15px;
  text-decoration: none;
  text-transform: uppercase;
}
.launch-link.primary {
  background: #f8fbff;
  color: #03060a;
}
.launch-activity {
  display: grid;
  gap: 0;
  grid-template-columns: repeat(7, minmax(118px, 1fr));
  margin: 0 auto;
  max-width: 1480px;
  overflow-x: auto;
  padding: 0 8px 8px;
  position: relative;
}
.launch-activity[hidden] {
  display: none;
}
.launch-activity::before {
  background:
    radial-gradient(ellipse 84% 144px at 50% 92px,
      transparent 97.1%,
      rgba(248, 251, 255, 0.68) 97.5%,
      rgba(248, 251, 255, 0.68) 98.2%,
      transparent 98.6%);
  content: "";
  height: 82px;
  left: 2.5%;
  pointer-events: none;
  position: absolute;
  right: 2.5%;
  top: 0;
}
.launch-activity-item {
  color: rgba(248, 251, 255, 0.68);
  min-height: 126px;
  min-width: 118px;
  position: relative;
  text-align: center;
  text-decoration: none;
  z-index: 1;
}
.launch-activity-item:hover {
  color: #fff;
  text-decoration: none;
}
.launch-activity-item.active {
  color: #fff;
}
.launch-activity-item.active .launch-activity-node {
  border-color: #ff8b22;
  box-shadow: 0 0 0 3px rgba(255, 139, 34, 0.18), 0 0 24px rgba(255, 139, 34, 0.72);
  color: #ff8b22;
}
.launch-activity-item.done .launch-activity-node {
  border-color: rgba(248, 251, 255, 0.82);
  color: #fff;
}
.launch-activity-node {
  align-items: center;
  background: rgba(1, 3, 7, 0.82);
  border: 2px solid rgba(248, 251, 255, 0.45);
  border-radius: 999px;
  color: rgba(248, 251, 255, 0.76);
  display: inline-flex;
  height: 22px;
  justify-content: center;
  margin-bottom: 9px;
  width: 22px;
}
.launch-activity-node::after {
  background: currentColor;
  border-radius: 999px;
  content: "";
  height: 4px;
  width: 4px;
}
.launch-activity-label {
  color: #f8fbff;
  display: block;
  font-size: 10px;
  font-weight: 850;
  letter-spacing: 0.08em;
  line-height: 1.2;
  margin: 0 auto;
  max-width: 126px;
  overflow: hidden;
  text-overflow: ellipsis;
  text-transform: uppercase;
  white-space: nowrap;
}
.launch-activity-item.active .launch-activity-label { color: #ffb84d; }
.launch-activity-item:nth-child(1),
.launch-activity-item:nth-child(7) { transform: translateY(46px); }
.launch-activity-item:nth-child(2),
.launch-activity-item:nth-child(6) { transform: translateY(30px); }
.launch-activity-item:nth-child(3),
.launch-activity-item:nth-child(5) { transform: translateY(16px); }
.launch-activity-item:nth-child(4) { transform: translateY(8px); }
@media (max-width: 1240px) {
  .main-grid { grid-template-columns: 1fr; }
  .right-rail { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .focus-stage { grid-template-columns: 250px minmax(0, 1fr); }
  .focus-live-panel { grid-column: 1 / -1; min-height: 560px; }
  .qp-stage { grid-template-columns: 300px minmax(0, 1fr); }
  .qp-side, .run-rail { grid-column: 1 / -1; grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .run-rail { display: grid; }
  .run-panel.shortcuts { margin-top: 0; }
  .lb-detail-grid { grid-template-columns: 1fr; }
  .trace-layout { grid-template-columns: 1fr; }
}
@media (max-width: 900px) {
  .topbar { position: static; }
  .brand { min-width: 0; width: 100%; }
  .topbar { flex-wrap: wrap; }
  .topnav { order: 3; width: 100%; }
  .top-actions { margin-left: auto; }
  .app-shell { grid-template-columns: 1fr; }
  .sidebar { border-right: 0; border-bottom: 1px solid var(--aws-line); }
  .workspace { padding: 22px 18px 34px; }
  .page-head { display: grid; }
  .quick-actions { justify-content: flex-start; }
  .checkpoint-card { grid-template-columns: 1fr; }
  .card-grid, .artifact-grid, .right-rail { grid-template-columns: 1fr; }
  .timeline-row { grid-template-columns: 72px minmax(0, 1fr); }
  .timeline-row .button { grid-column: 2; width: fit-content; }
  .focus-page { padding: 0 16px 28px; }
  .focus-nav { height: auto; padding: 18px 0; }
  .focus-links { gap: 8px; }
  .focus-links a { font-size: 14px; padding: 10px 12px; }
  .focus-stage { grid-template-columns: 1fr; }
  .focus-queue, .focus-work { border-right: 0; border-bottom: 1px solid #242b38; }
  .queue-list { max-height: none; }
  .compute-float { position: static; margin-top: 28px; width: 100%; }
  .qp-stage { grid-template-columns: 1fr; }
  .qp-ledger, .qp-side, .run-rail { border: 0; border-bottom: 1px solid #28313d; }
  .qp-side, .run-rail { grid-template-columns: 1fr; }
  .qp-workspace-head { align-items: flex-start; flex-direction: column; gap: 10px; }
  .qp-bottom-stats { grid-template-columns: 1fr; }
  .lb-header { align-items: flex-start; flex-direction: column; }
  .breakdown-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .trace-hero { grid-template-columns: 1fr; }
  .trace-actions { justify-content: flex-start; }
  .trace-stat-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .trace-agent-row { grid-template-columns: 52px minmax(0, 1fr); }
  .trace-agent-row > .trace-muted,
  .trace-agent-row > .status-pill,
  .trace-agent-row > .trace-link { grid-column: 2; width: fit-content; }
  .trace-step { grid-template-columns: 38px minmax(0, 1fr); }
  .trace-step > .trace-link { grid-column: 2; width: fit-content; }
  .trace-call-row { grid-template-columns: 52px minmax(0, 1fr); }
  .trace-call-row > .trace-muted,
  .trace-call-row > .trace-link { grid-column: 2; width: fit-content; }
  .launch-page { overflow-y: auto; padding: 18px 16px 26px; }
  .launch-nav { align-items: flex-start; flex-direction: column; }
  .launch-scene { min-height: 560px; }
  .launch-core { width: min(58vw, 260px); }
  .launch-side-labels { display: none; }
  .launch-stream { width: 64vw; }
  .launch-title h1 { letter-spacing: 0.18em; }
  .launch-title p { letter-spacing: 0.22em; }
  .launch-activity { grid-template-columns: repeat(7, minmax(132px, 1fr)); }
}
@media (max-width: 560px) {
  body { font-size: 13px; }
  .brand-title { font-size: 18px; }
  .top-actions { display: none; }
  .summary-grid { grid-template-columns: 1fr; }
  .metric-row { grid-template-columns: minmax(112px, 1fr) 68px 68px 60px; }
  .data-table { min-width: 720px; }
  .table-scroll { overflow-x: auto; }
  .focus-logo { font-size: 15px; }
  .focus-nav { align-items: flex-start; flex-direction: column; gap: 12px; }
  .focus-hero h1 { font-size: 28px; }
  .run-progress { grid-template-columns: repeat(2, 1fr); }
  .run-progress::before { display: none; }
  .task-row { grid-template-columns: 24px minmax(0, 1fr); }
  .task-status { grid-column: 2; width: fit-content; }
  .live-event { grid-template-columns: 1fr; }
  .qp-card-header, .qp-card-left, .qp-card-right, .qp-signers, .qp-actions {
    align-items: flex-start;
    flex-direction: column;
  }
  .qp-metric-row { grid-template-columns: 1fr 0.75fr 0.75fr 0.75fr; }
  .lb-table { min-width: 980px; }
  .recipe-kv { grid-template-columns: 1fr; }
  .breakdown-grid { grid-template-columns: 1fr; }
  .trace-stat-strip { grid-template-columns: 1fr; }
  .trace-event > summary { grid-template-columns: 1fr; }
  .launch-brand { font-size: 18px; letter-spacing: 0.18em; }
  .launch-subbrand { letter-spacing: 0.18em; }
  .launch-scene { min-height: 500px; }
  .launch-title h1 { font-size: 31px; letter-spacing: 0.14em; }
  .launch-actions { align-items: stretch; flex-direction: column; }
}

/* Chat widget (cockpit leftmost column, above ledger) */
.chat-box {
  background: rgba(17, 25, 40, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.chat-title {
  color: #f8fafc;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.chat-form textarea {
  width: 100%;
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(148, 163, 184, 0.22);
  color: #e2e8f0;
  border-radius: 10px;
  padding: 10px;
  font-family: inherit;
  font-size: 13px;
  resize: vertical;
}
.chat-form-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 8px;
}
.chat-form select {
  background: rgba(15, 23, 42, 0.6);
  color: #e2e8f0;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 8px;
  padding: 6px 8px;
  font-size: 12px;
}
.chat-thread {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 340px;
  overflow-y: auto;
}
.chat-thread.empty { color: #94a3b8; font-size: 12px; font-style: italic; }
.chat-item {
  background: rgba(15, 23, 42, 0.5);
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 10px;
  padding: 10px 12px;
}
.chat-meta {
  display: flex;
  justify-content: space-between;
  color: #94a3b8;
  font-size: 11px;
  margin-bottom: 6px;
}
.chat-actor { font-weight: 600; color: #c4b5fd; }
.chat-urgency { font-style: italic; }
.chat-text { color: #e2e8f0; font-size: 13px; line-height: 1.45; }
.chat-resp {
  margin-top: 8px;
  padding: 8px 10px;
  background: rgba(79, 70, 229, 0.12);
  border-left: 3px solid rgba(139, 92, 246, 0.8);
  border-radius: 6px;
  font-size: 12px;
}
.chat-resp.waiting {
  margin-top: 8px;
  padding: 6px 10px;
  color: #94a3b8;
  font-style: italic;
  font-size: 12px;
}
.chat-resp-label { color: #c4b5fd; font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.08em; }
.chat-resp-text { color: #e2e8f0; margin-top: 4px; }
.chat-resp-meta { color: #94a3b8; margin-top: 4px; font-size: 11px; }

/* Inline sign form on ledger rows + inside the current card */
.qp-sign-form {
  grid-column: 1 / -1;
  display: flex;
  gap: 6px;
  margin-top: 6px;
  padding: 6px 0 2px;
}
.qp-sign-form .qp-sign-note {
  flex: 1;
  background: rgba(15, 23, 42, 0.6);
  color: #e2e8f0;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
}
"""

BULK_SCRIPT = """
<script>
function toggleAll(open) {
  document.querySelectorAll('details').forEach(d => { d.open = open; });
}
</script>
"""

HEADER = (
    "<!doctype html><html><head><meta charset=utf-8>"
    "<meta name=viewport content='width=device-width, initial-scale=1'>"
    f"<style>{STYLE}</style>"
)


# ── Runs-index helpers ───────────────────────────────────────────────

# A run is considered "live" if its driver.log was touched recently OR
# its last trace file mtime is recent. Both signals are cheap and don't
# require process probing.
_LIVE_WINDOW_SECS = 120.0


def _run_summary(name: str) -> dict:
    """One-row summary of a marathon run — used by the runs index and the
    per-run cockpit header. Safe to call pre-launch: if no trace or
    memory files exist yet, returns the scaffolded zero-state."""
    run_dir = RUNS_ROOT / name
    trace_dir = run_dir / "trace"
    memory_path = run_dir / "memory" / "records.jsonl"
    driver_log = run_dir / "driver.log"

    # Trace counts
    n_cycles = 0
    newest_trace_mtime = 0.0
    if trace_dir.is_dir():
        cyc_dirs = [p for p in trace_dir.glob("cycle_*") if p.is_dir()]
        n_cycles = len(cyc_dirs)
        for c in cyc_dirs:
            for p in c.glob("agent_*.jsonl"):
                try:
                    newest_trace_mtime = max(newest_trace_mtime,
                                             p.stat().st_mtime)
                except OSError:
                    pass

    # Record counts (lightweight; don't parse JSON)
    n_records = 0
    memory_mtime = 0.0
    if memory_path.is_file():
        try:
            memory_mtime = memory_path.stat().st_mtime
            with memory_path.open("rb") as f:
                n_records = sum(1 for _ in f)
        except OSError:
            pass

    # Driver log
    driver_mtime = 0.0
    if driver_log.is_file():
        try:
            driver_mtime = driver_log.stat().st_mtime
        except OSError:
            pass

    # Liveness
    last_activity = max(newest_trace_mtime, memory_mtime, driver_mtime)
    is_live = (time.time() - last_activity) < _LIVE_WINDOW_SECS if last_activity else False
    if n_records == 0 and n_cycles == 0 and last_activity == 0.0:
        state = "empty"
    elif is_live:
        state = "live"
    else:
        state = "finished"

    started_at = 0.0
    if driver_log.is_file():
        # Driver log ctime is usually when the run was launched; fall
        # back to the run dir's mtime if that's missing.
        try:
            started_at = driver_log.stat().st_ctime
        except OSError:
            pass
    if not started_at and run_dir.is_dir():
        try:
            started_at = run_dir.stat().st_mtime
        except OSError:
            pass

    return {
        "name": name,
        "state": state,
        "n_cycles": n_cycles,
        "n_records": n_records,
        "last_activity": last_activity,
        "started_at": started_at,
        "trace_dir": str(trace_dir),
        "memory_path": str(memory_path),
    }


def _fmt_relative_time(ts: float) -> str:
    if not ts:
        return "—"
    delta = time.time() - ts
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


def _fmt_abs_time(ts: float) -> str:
    if not ts:
        return "—"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def render_runs_index() -> str:
    runs = list_runs()
    summaries = [_run_summary(n) for n in runs]
    # Newest first.
    summaries.sort(key=lambda s: s["last_activity"] or s["started_at"],
                   reverse=True)

    cards: list[str] = []
    for s in summaries:
        state = s["state"]
        badge_cls = {"live": "ok", "finished": "muted",
                     "empty": "warn"}.get(state, "muted")
        badge_label = {"live": "LIVE",
                       "finished": "finished",
                       "empty": "waiting for first record"}.get(state, state)
        last_seen = _fmt_relative_time(s["last_activity"])
        started = _fmt_abs_time(s["started_at"])
        cards.append(
            "<li class=run-card>"
            f"<a class=run-card-main href='/runs/{_h(s['name'])}'>"
            f"<div class=run-card-head>"
            f"<h3>{_h(s['name'])}</h3>"
            f"<span class='run-badge {badge_cls}'>{_h(badge_label)}</span>"
            f"</div>"
            f"<div class=run-card-stats>"
            f"<span><b>{s['n_cycles']}</b> cycles</span>"
            f"<span><b>{s['n_records']}</b> records</span>"
            f"<span>last activity: {_h(last_seen)}</span>"
            f"<span>started: {_h(started)}</span>"
            f"</div>"
            f"</a></li>"
        )
    if not cards:
        body = (
            "<div class=run-empty>"
            f"<p>No marathon runs found under <code>{_h(str(RUNS_ROOT))}</code>.</p>"
            "<p>Launch one with <code>examples/nemo_mas_reasoning_example/drive_nemo_mas.py</code>, "
            "specifying <code>--work-dir runs/&lt;your-run-name&gt;</code>. "
            "The viewer picks it up automatically.</p>"
            "</div>"
        )
    else:
        body = "<ul class=runs-list>" + "".join(cards) + "</ul>"

    extra_css = """
    .runs-wrap { max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }
    .runs-header { display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px; }
    .runs-header h1 { font-size: 28px; }
    .runs-header .meta { color:#64748b; font-size:13px; }
    .runs-list { list-style:none; margin:0; padding:0; display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }
    .run-card { background: #fff; border: 1px solid var(--aws-line); border-radius: 8px; box-shadow: var(--shadow-sm); overflow: hidden; }
    .run-card-main { display:block; padding: 16px 18px; color:inherit; text-decoration:none; }
    .run-card-main:hover { background: #f8fafc; text-decoration:none; }
    .run-card-head { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:10px; }
    .run-card-head h3 { margin:0; font-size:15px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color:#111827; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .run-badge { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size:11px; font-weight:700; letter-spacing:0.02em; }
    .run-badge.ok { background:#ecfdf3; color:#067647; border:1px solid #abefc6; }
    .run-badge.muted { background:#f1f5f9; color:#475467; border:1px solid #e5e7eb; }
    .run-badge.warn { background:#fff7ed; color:#c2410c; border:1px solid #fed7aa; }
    .run-card-stats { display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:#475467; }
    .run-card-stats b { color:#111827; }
    .run-empty { background:#fff; border:1px dashed var(--aws-line); border-radius:8px; padding:28px; color:#475467; }
    """

    return (
        HEADER
        + f"<style>{extra_css}</style>"
        + "<title>Marathon runs — A-Evolve-MAS-Train</title></head><body>"
        + "<div class=runs-wrap>"
        + "<header class=runs-header>"
        + "<div><h1>A-Evolve-MAS-Train</h1>"
        + f"<div class=meta>Runs root: <code>{_h(str(RUNS_ROOT))}</code></div></div>"
        # Re-pick at request time so the button tracks live runs as they
        # appear (the CLI-snapshot ``DEFAULT_RUN`` only moves on restart).
        + (lambda tgt: (f"<a class='button' href='/runs/{_h(tgt)}'>"
                         f"Open {'live' if _is_run_live(tgt) else 'latest'} run →</a>")
                        if tgt else "")(_pick_default_run())
        + "</header>"
        + body
        + "</div></body></html>"
    )


# ── helpers ──────────────────────────────────────────────────────────

def _cycle_dirs() -> list[Path]:
    trace_dir = trace_dir_for()
    if not trace_dir.is_dir():
        return []
    return sorted([p for p in trace_dir.glob("cycle_*") if p.is_dir()])


def _agent_files(cycle_dir: Path) -> list[Path]:
    def key(p: Path):
        m = re.match(r"agent_(\d+)\.jsonl", p.name)
        return int(m.group(1)) if m else 999
    return sorted(cycle_dir.glob("agent_*.jsonl"), key=key)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        return []
    return rows


def _h(s) -> str:
    return html.escape(str(s))


def _preview(text: str, limit: int = 120) -> str:
    """First line, truncated — shown next to the summary when collapsed."""
    s = str(text).strip().splitlines()[0] if str(text).strip() else ""
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


def _collapsible(
    text: str,
    *,
    label: str,
    threshold_chars: int = 600,
    threshold_lines: int = 14,
    open_by_default: bool = False,
) -> str:
    """Render text as <pre>, wrapping in <details> when it's long.

    Short blocks render inline (no fold). Long blocks collapse with a
    summary line showing the label + a one-line preview. The full text
    is always present in the DOM — nothing is trimmed.
    """
    text = "" if text is None else str(text)
    pre = f"<pre>{_h(text)}</pre>"
    nlines = text.count("\n") + 1
    if len(text) <= threshold_chars and nlines <= threshold_lines:
        return pre
    preview = _preview(text)
    meta = (f"<span class=kv>[{len(text):,} chars · {nlines:,} lines]</span> "
            f"<span class=preview>{_h(preview)}</span>")
    state = " open" if open_by_default else ""
    return (f"<details{state}><summary>{_h(label)} {meta}</summary>"
            f"{pre}</details>")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _tool_calls(rows: list[dict]) -> list[dict]:
    """Extract the tool-call timeline from one agent's trace rows.

    Walks `event=turn` rows, collects each toolUse (request) and matches
    the next user message's toolResult by ``toolUseId``. Returns a list
    of dicts: {turn, name, input, status, result_excerpt}.

    For spawn calls, pulls the spawned role/task out of the input so the
    cycle-level call-graph view can wire parent → child.
    """
    # First pass: collect toolResults indexed by toolUseId.
    results: dict[str, dict] = {}
    for r in rows:
        if r.get("event") != "message":
            continue
        content = r.get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "toolResult" in block:
                tr = block["toolResult"]
                uid = tr.get("toolUseId") or tr.get("tool_use_id")
                if uid:
                    results[uid] = tr

    calls: list[dict] = []
    for r in rows:
        if r.get("event") != "turn":
            continue
        turn = r.get("turn", "?")
        assistant = (r.get("assistant") or {})
        content = assistant.get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or "toolUse" not in block:
                continue
            tu = block["toolUse"]
            uid = tu.get("toolUseId") or tu.get("id")
            name = tu.get("name", "?")
            tin = tu.get("input") or {}
            tr = results.get(uid, {})
            status = tr.get("status", "pending" if not tr else "ok")
            # Extract a compact excerpt of the result for the timeline.
            res_content = tr.get("content") or []
            excerpt = ""
            if isinstance(res_content, list):
                for rb in res_content:
                    if isinstance(rb, dict) and "text" in rb:
                        excerpt = str(rb["text"])
                        break
                    if isinstance(rb, dict) and "json" in rb:
                        excerpt = json.dumps(rb["json"], default=str)
                        break
            elif isinstance(res_content, str):
                excerpt = res_content

            calls.append({
                "turn": turn,
                "tool_use_id": uid,
                "name": name,
                "input": tin,
                "status": status,
                "result_excerpt": excerpt,
            })
    return calls


_ROLE_PATTERNS: list[tuple[str, str]] = [
    (r"you are the orchestrator",                 "orchestrator"),
    (r"you are (a |an |the )?reviewer",           "reviewer"),
    (r"you are (a |an |the )?data[\s-]?worker",   "data_worker"),
    (r"you are (a |an |the )?planner",            "planner"),
    (r"you are (a |an |the )?trainer",            "trainer"),
]


def _detect_role(system_excerpt: str, tool_names: list[str] | None) -> str:
    """Return the role string by matching the ``You are the X`` opener.

    Substring matching against arbitrary text is unreliable — every
    worker prompt mentions other roles by name (the reviewer's prompt
    says "the Orchestrator decides …"). We anchor the match to the
    "You are …" self-introduction instead, and fall back to tool
    shape if that fails.
    """
    if system_excerpt:
        s = system_excerpt.lower()
        for pat, role in _ROLE_PATTERNS:
            if re.search(pat, s):
                return role
    # Fallback: tool shape uniquely identifies each role.
    names = set(tool_names or [])
    if "spawn_and_run_subagent" in names:
        return "orchestrator"
    if "launch_training" in names:
        return "trainer"
    if "call_teacher_model" in names or "mix_sources" in names:
        return "data_worker"
    if "checkpoint_review_suggest" in names or "run_eval" in names:
        return "reviewer"
    if "diff_yaml" in names or "render_recipe_diff" in names:
        return "planner"
    return "?"


def _agent_summary_from_rows(path: Path, rows: list[dict]) -> dict:
    turns = sum(1 for r in rows if r.get("event") == "turn")
    done = next((r for r in reversed(rows) if r.get("event") == "done"), None)
    start = next((r for r in rows if r.get("event") == "start"), None)
    role = _detect_role(
        (start or {}).get("system_excerpt", "") or "",
        (start or {}).get("tool_names") or [],
    )
    size = path.stat().st_size if path.exists() else 0
    mtime = path.stat().st_mtime if path.exists() else 0
    return {
        "role": role,
        "turns": turns,
        "size": size,
        "mtime": mtime,
        "input_tokens": (done or {}).get("input_tokens"),
        "output_tokens": (done or {}).get("output_tokens"),
        "finished": done is not None,
    }


def _agent_summary(path: Path) -> dict:
    return _agent_summary_from_rows(path, _load_jsonl(path))


# Legacy name retained for callers; the authoritative slot declarations live
# in agent_evolve.model.algorithms.nemo_mas.checkpoints. Derived per-request
# by _derive_checkpoints() from records.jsonl + the current mode.
QUALITY_CHECKPOINTS: list[dict] = []

BASELINE_CATEGORIES = [
    ("Data", 5, "schema, distribution, leakage, 20+ examples, verifier yield"),
    ("Model", 4, "init loss, overfit batch, forward shape, parameter count"),
    ("Training", 4, "seed log, gradient sanity, loss trend, resources"),
    ("Evaluation", 4, "frozen holdout, 10+ samples, slice metrics, robustness"),
    ("Artifacts", 2, "versioning and Model Card"),
]

# Legacy name retained for callers; real rows come from _derive_eval_runs()
# which walks cv_result → training_run → recipe_proposal → eval_report via
# the refs DAG in records.jsonl.
EVAL_RUNS: list[dict] = []


# ── records.jsonl loader + derivation ─────────────────────────────────

_records_lock = threading.Lock()
# {memory_path_str: {"mtime": float, "rows": list[dict]}} — one entry per run.
_records_cache: dict[str, dict] = {}


def _load_records() -> list[dict]:
    """Return every record from the active run's ``records.jsonl``.

    Cache is keyed by path so viewing two runs concurrently doesn't thrash
    each other's cached rows. Entry is refreshed only when the file mtime
    changes, so the cockpit surfaces new state within one live-status
    poll (6 seconds).
    """
    path = memory_path_for()
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        with _records_lock:
            _records_cache[key] = {"mtime": -1.0, "rows": []}
            return []
    with _records_lock:
        entry = _records_cache.get(key)
        if entry and entry["mtime"] == mtime:
            return list(entry["rows"])
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        rows = []
    with _records_lock:
        _records_cache[key] = {"mtime": mtime, "rows": list(rows)}
    return rows


def _records_by_kind(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r.get("kind") == kind]


def _record_by_id(records: list[dict], rec_id: str) -> dict | None:
    for r in records:
        if r.get("id") == rec_id:
            return r
    return None


def _resolve_ref_chain(records: list[dict], start_id: str,
                       target_kind: str, max_hops: int = 6) -> dict | None:
    """Walk ``refs`` upstream from ``start_id`` until we hit ``target_kind``.

    The refs DAG is sparse — cv_result → training_run → {recipe_proposal,
    dataset_snapshot} — so a BFS stays fast. Returns the first matching
    record, or None.
    """
    seen: set[str] = set()
    queue = [start_id]
    hops = 0
    while queue and hops < max_hops:
        next_queue: list[str] = []
        for rid in queue:
            if rid in seen:
                continue
            seen.add(rid)
            rec = _record_by_id(records, rid)
            if rec is None:
                continue
            if rec.get("kind") == target_kind:
                return rec
            next_queue.extend(rec.get("refs") or [])
        queue = next_queue
        hops += 1
    return None


def _reverse_find_by_ref(records: list[dict], target_id: str,
                         kind: str) -> dict | None:
    """Return the most-recent record of ``kind`` whose refs include ``target_id``.

    Used to go from training_run → eval_report (the eval points back at the
    training run, not the other way round).
    """
    matches = [r for r in records
               if r.get("kind") == kind
               and target_id in (r.get("refs") or [])]
    if not matches:
        return None
    matches.sort(key=lambda r: r.get("ts", ""))
    return matches[-1]


_FENCED_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_fenced_json(body: str, key: str) -> dict | None:
    """Find the first fenced ```json``` block containing top-level ``key``.

    Records are expected to end with a metrics / recipe JSON block per the
    worker prompt contracts; this parser is forgiving of additional blocks.
    """
    if not body:
        return None
    for match in _FENCED_JSON_RE.finditer(body):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None


def _parse_findings(body: str) -> list[str]:
    """Pull bullet lines (``- `` or ``* ``) out of a record body."""
    if not body:
        return []
    lines = []
    for raw in body.splitlines():
        stripped = raw.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            lines.append(stripped[2:].strip())
    return lines


def _parse_score_note(body: str) -> str:
    if not body:
        return ""
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            return ""
        return stripped
    return ""


def _parse_artifacts(body: str) -> list[str]:
    if not body:
        return []
    return [ln.strip() for ln in body.splitlines()
            if ln.strip().startswith("artifact://")]


def _parse_ts_to_epoch(ts: str) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _derive_eval_runs(records: list[dict]) -> list[dict]:
    """Turn evaluated runs into leaderboard rows.

    Sources, strongest first:
      1. ``cv_result`` records (N seeds × M splits, the gold promotion gate),
      2. ``eval_report`` records whose refs point at a ``training_run``
         (one-seed eval — sufficient for a reasonable validation signal).

    Each eval'd training_run contributes at most one row. The row shape
    matches what ``render_leaderboard`` / ``render_run_detail`` already
    consume; ``status`` is ``best`` (top kaggle metric), ``promoted``
    (cv_result-backed), ``eval`` (eval_report only), or ``blocked``
    (cv_result refs a failed_attempt).
    """
    cv_results = _records_by_kind(records, "cv_result")
    eval_reports = _records_by_kind(records, "eval_report")
    if not cv_results and not eval_reports:
        return []

    # Build (training_run_id → best-signal record) so one run doesn't show
    # up twice. A cv_result always wins over an eval_report for the same
    # training_run.
    anchor_for_run: dict[str, dict] = {}
    for er in eval_reports:
        for rid in er.get("refs") or []:
            target = _record_by_id(records, rid)
            if target and target.get("kind") == "training_run":
                anchor_for_run[rid] = {"kind": "eval_report", "record": er}
                break
    for cv in cv_results:
        for rid in cv.get("refs") or []:
            target = _record_by_id(records, rid)
            if target and target.get("kind") == "training_run":
                anchor_for_run[rid] = {"kind": "cv_result", "record": cv}
                break

    if not anchor_for_run:
        return []

    # Row timeline: sort by anchor record ts, newest first for display;
    # asc order for stage indexing.
    anchors_asc = sorted(
        anchor_for_run.items(),
        key=lambda kv: kv[1]["record"].get("ts", ""),
    )
    cycle_order: list[str] = []
    seen_cycles: set[str] = set()
    for _, anchor in anchors_asc:
        cid = anchor["record"].get("cycle_id", "")
        if cid not in seen_cycles:
            seen_cycles.add(cid)
            cycle_order.append(cid)

    # Best = highest kaggle metric, preferring stable cv_result.
    best_id: str | None = None
    best_metric: float = float("-inf")
    best_prefers_cv = False
    for _, anchor in anchors_asc:
        rec = anchor["record"]
        body = rec.get("body") or ""
        metrics = (_parse_fenced_json(body, "metrics") or {}).get("metrics") or {}
        kaggle = metrics.get("kaggle")
        if not isinstance(kaggle, (int, float)):
            continue
        is_cv = anchor["kind"] == "cv_result"
        stable = (_parse_fenced_json(body, "stable") or {}).get("stable")
        # CV with stable=True beats a raw eval_report at the same score;
        # otherwise pure numeric max wins.
        if is_cv and not stable:
            continue
        if (float(kaggle) > best_metric) or (
            float(kaggle) == best_metric and is_cv and not best_prefers_cv
        ):
            best_metric = float(kaggle)
            best_id = rec.get("id")
            best_prefers_cv = is_cv

    rows: list[dict] = []
    for run_id, anchor in sorted(anchor_for_run.items(),
                                  key=lambda kv: kv[1]["record"].get("ts", ""),
                                  reverse=True):
        rec = anchor["record"]
        is_cv = anchor["kind"] == "cv_result"
        body = rec.get("body") or ""
        metrics = (_parse_fenced_json(body, "metrics") or {}).get("metrics") or {}

        training_run = _record_by_id(records, run_id)
        recipe_proposal = (
            _resolve_ref_chain(records, run_id, "recipe_proposal")
            if training_run else None
        )
        eval_report = (
            rec if not is_cv else
            _reverse_find_by_ref(records, run_id, "eval_report")
        )

        recipe_json: dict = {}
        if training_run:
            recipe_json = (_parse_fenced_json(training_run.get("body") or "",
                                              "recipe") or {}).get("recipe") or {}
        if not recipe_json and recipe_proposal:
            recipe_json = (_parse_fenced_json(recipe_proposal.get("body") or "",
                                              "recipe") or {}).get("recipe") or {}

        score_note = ""
        findings: list[str] = []
        if eval_report:
            score_note = _parse_score_note(eval_report.get("body") or "")
            findings = _parse_findings(eval_report.get("body") or "")

        artifacts: list[str] = []
        for src in filter(None, [rec, training_run, recipe_proposal, eval_report]):
            artifacts.extend(_parse_artifacts(src.get("body") or ""))
        seen_art: set[str] = set()
        artifacts = [a for a in artifacts if not (a in seen_art or seen_art.add(a))]

        cycle_id = rec.get("cycle_id", "")
        stage_idx = (cycle_order.index(cycle_id) + 1
                     if cycle_id in seen_cycles else len(cycle_order))

        has_failed = any(
            (_record_by_id(records, rid) or {}).get("kind") == "failed_attempt"
            for rid in rec.get("refs") or []
        )

        if rec.get("id") == best_id:
            status = "best"
        elif has_failed:
            status = "blocked"
        elif is_cv:
            status = "promoted"   # cv_result: cross-validated
        else:
            status = "eval"       # eval_report only

        rows.append({
            "id": rec.get("id", ""),
            "name": rec.get("title", "") or f"{anchor['kind']} {rec.get('id', '')}",
            "cycle": cycle_id,
            "stage": (f"Round {stage_idx} {'cv' if is_cv else 'eval'}"),
            "status": status,
            "decision": status,
            "kaggle": _safe_float(metrics.get("kaggle")),
            "local": _safe_float(metrics.get("local")),
            "hard": _safe_float(metrics.get("hard")),
            "delta": metrics.get("delta") or "—",
            "score_note": score_note or rec.get("title", ""),
            "recipe": recipe_proposal.get("title", "") if recipe_proposal else "—",
            "base_model": recipe_json.get("base_model", "") or "—",
            "data_mix": recipe_json.get("data_mix", "") or "—",
            "training": recipe_json.get("training", "") or "—",
            "quality_gate": recipe_json.get("quality_gate", "") or "—",
            "findings": findings,
            "breakdown": dict(metrics.get("breakdown") or {}),
            "artifacts": artifacts,
            "source_kind": anchor["kind"],
        })
    return rows


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _slots_for_active_run() -> list[dict]:
    """Load the active run's ``checkpoints.yaml``.

    Uses ``cycle_workspace_path`` from the algorithm module as the single
    source of truth for the fork layout — keeps producer (driver) and
    consumer (viewer) in sync. Missing file (pre-launch or a benchmark
    that opted out of gates) ⇒ empty list.
    """
    run = active_run()
    if run is None:
        return []
    cycles_root = RUNS_ROOT / run / "cycles"
    if not cycles_root.is_dir():
        return []
    cycle_dirs = sorted(p for p in cycles_root.iterdir() if p.is_dir())
    if not cycle_dirs:
        return []
    latest_id = cycle_dirs[-1].name
    workspace = cycle_workspace_path(RUNS_ROOT / run, latest_id)
    if (workspace / "checkpoints.yaml").is_file():
        return load_slot_decls(workspace)
    return []


def _derive_checkpoints(records: list[dict], mode: str | None = None) -> list[dict]:
    """Fold ``checkpoint_event`` + ``checkpoint_review`` into one entry per slot.

    Emits the same dict shape the old hardcoded ``QUALITY_CHECKPOINTS``
    array used, plus fields (``can_sign``, ``evidence_counts``,
    ``last_event_ts``, ``last_review_verdict``, ``last_review_reason``)
    that the new Sign button + reviewer badge logic consume.
    """
    mode = mode or _mode_for_active_run()
    slots = _slots_for_active_run()
    folded = fold_checkpoints(records, mode, slots=slots)
    return [
        {
            "id": s.id,
            "short": s.short,
            "title": s.title,
            "type": s.type,
            "template": s.template,
            "signers": s.signers,
            "last_review_verdict": s.last_review_verdict,
            "last_review_reason": s.last_review_reason,
            "last_review_cycle": s.last_review_cycle,
            "state": s.state,
            "required": s.required,
            "depends_on": list(s.depends_on),
            "requires_evidence": list(s.requires_evidence),
            "evidence_counts": dict(s.evidence_counts),
            "last_event_ts": s.last_event_ts,
            "last_event_actor": s.last_event_actor,
            "can_sign": s.can_sign,
        }
        for s in folded
    ]


def _derive_chat_thread(records: list[dict], limit: int = 5) -> list[dict]:
    """Return the last ``limit`` directive/response pairs, newest first."""
    directives = sorted(
        _records_by_kind(records, "human_directive"),
        key=lambda r: r.get("ts", ""),
        reverse=True,
    )[:limit]
    responses = _records_by_kind(records, "directive_response")

    thread: list[dict] = []
    for d in directives:
        dbody_obj = {}
        try:
            dbody_obj = json.loads(d.get("body") or "{}")
        except json.JSONDecodeError:
            pass
        directive_text = dbody_obj.get("text") if isinstance(dbody_obj, dict) else None
        if not directive_text:
            directive_text = d.get("body") or ""
        urgency = ""
        for t in d.get("tags") or []:
            if t.startswith("urgency:"):
                urgency = t[len("urgency:"):]
                break

        matching = [
            r for r in responses
            if d["id"] in (r.get("refs") or [])
            or any(t == f"reply_to:{d['id']}" for t in (r.get("tags") or []))
        ]
        matching.sort(key=lambda r: r.get("ts", ""))
        response_entry = None
        if matching:
            last = matching[-1]
            try:
                rbody = json.loads(last.get("body") or "{}")
            except json.JSONDecodeError:
                rbody = {}
            response_entry = {
                "id": last.get("id", ""),
                "ts": last.get("ts", ""),
                "summary": rbody.get("summary") or last.get("title", ""),
                "action": rbody.get("action", ""),
                "spawned_role": rbody.get("spawned_role") or "",
            }
        thread.append({
            "id": d.get("id", ""),
            "ts": d.get("ts", ""),
            "actor": d.get("author", ""),
            "text": directive_text,
            "urgency": urgency,
            "response": response_entry,
        })
    return thread


# ── records.jsonl append (POST endpoints) ─────────────────────────────


_append_lock = threading.Lock()


def _new_record_id() -> str:
    return f"rec_{secrets.token_hex(6)}"


def _iso_now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _latest_cycle_id_from_records(records: list[dict]) -> str:
    cycles = sorted({r.get("cycle_id", "") for r in records} - {""})
    if cycles:
        return cycles[-1]
    latest = _latest_cycle_id()
    return latest or "0001"


def _append_record(record: dict) -> None:
    """Append one ``MemoryRecord``-shaped dict to the records.jsonl file.

    Writes through the same lock we use for cache reads so a reader never
    sees a partial line. Invalidates the cache so the new row is visible
    on the very next request instead of the next mtime check.
    """
    path = memory_path_for()
    if not str(path):
        raise RuntimeError("no active run; cannot append record")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _append_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    # Invalidate this path's cache entry so the new row is visible on
    # the next request without waiting for an mtime poll.
    with _records_lock:
        _records_cache.pop(str(path), None)


def _format_ts(ts: float | int | None) -> str:
    if not ts:
        return "n/a"
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _cycle_id(cycle_dir: Path) -> str:
    return cycle_dir.name.replace("cycle_", "")


def _role_display(role: str) -> str:
    return {
        "orchestrator": "MAS Orchestrator",
        "reviewer":     "Reviewer",
        "data_worker":  "Data Worker",
        "trainer":      "Trainer",
        "planner":      "Planner",
        "?":            "Unclassified",
    }.get(role, role.replace("_", " ").title())


def _role_class(role: str) -> str:
    return role if role in {"orchestrator", "reviewer", "data_worker", "trainer", "planner"} else "unknown"


def _state_label(state: str) -> str:
    return {
        "signed": "Signed",
        "reopened": "Reopened",
        "pending_human": "Pending human",
        "pending_evidence": "Pending evidence",
        "pending_pre_review": "Pre-review",
        "draft": "Draft",
        "final": "Final",
        "rejected": "Rejected",
    }.get(state, state.replace("_", " ").title())


def _quality_state_counts() -> Counter:
    return Counter(cp["state"] for cp in _derive_checkpoints(_load_records()))


def _checkpoint_progress() -> int:
    slots = _derive_checkpoints(_load_records())
    finished = sum(1 for cp in slots if cp["state"] in {"signed", "reopened"})
    return round(100 * finished / max(len(slots), 1))


def _latest_cycle_id() -> str | None:
    cycles = _cycle_dirs()
    if not cycles:
        return None
    return _cycle_id(cycles[-1])


def _cycle_summaries(*, include_roles: bool = False) -> list[dict]:
    summaries = []
    for c in _cycle_dirs():
        agents = _agent_files(c)
        total = sum(p.stat().st_size for p in agents)
        mtime = max((p.stat().st_mtime for p in agents), default=0)
        role_counts: Counter = Counter()
        finished = 0
        in_tokens = 0
        out_tokens = 0
        if include_roles:
            for p in agents:
                s = _agent_summary(p)
                role_counts[s["role"]] += 1
                finished += int(bool(s["finished"]))
                in_tokens += s["input_tokens"] or 0
                out_tokens += s["output_tokens"] or 0
        summaries.append({
            "id": _cycle_id(c),
            "name": c.name,
            "agents": len(agents),
            "finished": finished,
            "total": total,
            "mtime": mtime,
            "roles": role_counts,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        })
    return summaries


def _trace_totals() -> dict:
    cycles = _cycle_summaries(include_roles=True)
    roles: Counter = Counter()
    for c in cycles:
        roles.update(c["roles"])
    return {
        "cycles": len(cycles),
        "agents": sum(c["agents"] for c in cycles),
        "bytes": sum(c["total"] for c in cycles),
        "last_activity": max((c["mtime"] for c in cycles), default=0),
        "roles": roles,
        "input_tokens": sum(c["input_tokens"] for c in cycles),
        "output_tokens": sum(c["output_tokens"] for c in cycles),
    }


def _eval_runs_ranked() -> list[dict]:
    runs = _derive_eval_runs(_load_records())
    return sorted(runs, key=lambda r: r["kaggle"], reverse=True)


def _eval_run(run_id: str) -> dict | None:
    for r in _derive_eval_runs(_load_records()):
        if r["id"] == run_id:
            return r
    return None


def _leaderboard_status(run: dict) -> tuple[str, str]:
    return {
        "best": ("best", "Best candidate"),
        "promoted": ("review", "CV promoted"),
        "eval": ("review", "Eval only"),
        "review": ("review", "Review"),
        "blocked": ("blocked", "Blocked"),
        "archived": ("archived", "Archived"),
    }.get(run["status"], ("archived", run["status"].title()))


def _run_for_cycle(cycle: str) -> dict | None:
    for r in _derive_eval_runs(_load_records()):
        if r["cycle"] == cycle:
            return r
    return None


def _fmt_int(n) -> str:
    return f"{int(n):,}" if isinstance(n, (int, float)) else "—"


def _content_plaintext(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, default=str)
    pieces: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            pieces.append(json.dumps(block, default=str))
        elif "text" in block:
            pieces.append(str(block["text"]))
        elif "toolUse" in block:
            tu = block["toolUse"] or {}
            pieces.append(f"calls {tu.get('name', '?')}")
        elif "toolResult" in block:
            pieces.append("tool result returned")
    return " · ".join(p for p in pieces if p)


def _tool_activity_summary(name: str, tin: dict) -> tuple[str, str]:
    tin = tin or {}
    if name == "launch_training":
        ckpt = tin.get("ckpt_out", "checkpoint")
        data = tin.get("data_path", "training data")
        recipe = tin.get("recipe_path", "recipe")
        return (
            "Launching training",
            f"Running {recipe} on {data}; checkpoint target {ckpt}.",
        )
    if name == "read_training_log":
        return ("Reading training log", f"Checking log at {tin.get('path', 'latest run log')}.")
    if name == "read_checkpoint_metric":
        return ("Checking checkpoint metric", f"Reading metric from {tin.get('ckpt_path', 'checkpoint')}.")
    if name == "read_file":
        return ("Verifying config file", f"Reading {tin.get('path', 'a config file')}.")
    if name == "mem_search":
        return ("Searching run memory", f"Looking up: {_preview(tin.get('query', ''), 90)}")
    if name == "mem_get":
        return ("Loading evidence record", f"Opening record {tin.get('id', 'unknown')}.")
    if name == "mem_write":
        kind = tin.get("kind", "record")
        return ("Writing run memory", f"Recording {kind} evidence for the plan ledger.")
    if name == "skill_load":
        return ("Loading role skill", f"Activating {tin.get('name', 'role skill')}.")
    if name == "spawn_and_run_subagent":
        role = tin.get("role", "agent")
        return (
            f"Delegating to {_role_display(role)}",
            _preview(" ".join(str(tin.get("task", "")).split()), 150),
        )
    if name == "call_existing_agent":
        return ("Resuming agent", f"Continuing agent_{tin.get('agent_id', '?')}.")
    if name:
        return (f"Calling {name}", _preview(json.dumps(tin, default=str), 150))
    return ("Thinking", "No tool call in the latest event.")


def _event_live_summary(ev: dict) -> dict:
    kind = ev.get("event", "?")
    base = {
        "title": "Waiting for activity",
        "body": "No recent event found.",
        "tool": "",
        "turn": ev.get("turn"),
        "state": kind,
        "ts": ev.get("ts", 0),
    }
    if kind == "start":
        tools = ev.get("tool_names") or []
        base.update({
            "title": "Agent started",
            "body": f"Model {ev.get('model_id', 'unknown')} loaded with {len(tools)} tools.",
        })
        return base
    if kind == "message":
        base.update({
            "title": "Received task",
            "body": _preview(" ".join(_content_plaintext(ev.get("content")).split()), 180),
        })
        return base
    if kind == "done":
        base.update({
            "title": "Agent completed",
            "body": (
                f"{ev.get('total_turns', '?')} turns; "
                f"{ev.get('input_tokens', '?')} input / {ev.get('output_tokens', '?')} output tokens."
            ),
            "state": "done",
        })
        return base
    if kind == "turn":
        content = ((ev.get("assistant") or {}).get("content") or [])
        text_bits = []
        latest_tool = None
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    text_bits.append(str(block["text"]))
                elif "toolUse" in block:
                    latest_tool = block["toolUse"] or {}
        if latest_tool:
            tool_name = latest_tool.get("name", "?")
            title, body = _tool_activity_summary(tool_name, latest_tool.get("input") or {})
            lead = _preview(" ".join(" ".join(text_bits).split()), 110)
            if lead and tool_name not in {"launch_training", "read_training_log", "read_checkpoint_metric"}:
                body = f"{lead} {body}"
            base.update({
                "title": title,
                "body": body,
                "tool": tool_name,
                "state": ev.get("stop_reason", "tool_use"),
            })
            return base
        body = _preview(" ".join(" ".join(text_bits).split()), 180)
        base.update({
            "title": "Writing response",
            "body": body or "Assistant turn without a visible tool call.",
            "state": ev.get("stop_reason", "turn"),
        })
        return base
    base.update({
        "title": kind.replace("_", " ").title(),
        "body": _preview(json.dumps(ev, default=str), 180),
    })
    return base


def _launch_role_label(role: str) -> str:
    label = _role_display(role)
    if label == "MAS Orchestrator":
        return "Orchestrator"
    if label == "Unclassified":
        return "Agent"
    return label


def _launch_action_label(live: dict) -> str:
    tool = live.get("tool") or ""
    if tool:
        return {
            "launch_training": "Launch training",
            "read_training_log": "Read training log",
            "read_checkpoint_metric": "Check metric",
            "read_file": "Read config",
            "list_dir": "Inspect files",
            "mem_search": "Search memory",
            "mem_recent": "Read memory",
            "mem_get": "Read memory",
            "mem_write": "Write memory",
            "skill_load": "Load skill",
            "skill_index": "Find skill",
            "spawn_and_run_subagent": "Delegate",
            "call_existing_agent": "Resume agent",
            "run_eval": "Run eval",
            "checkpoint_review_suggest": "Review gate",
            "call_teacher_model": "Call teacher",
            "mix_sources": "Mix data",
            "diff_yaml": "Diff recipe",
            "render_recipe_diff": "Render recipe",
        }.get(tool, tool.replace("_", " ").title())
    state = live.get("state") or ""
    title = live.get("title") or "Working"
    if state == "done":
        return "Completed"
    return {
        "Agent started": "Started",
        "Agent completed": "Completed",
        "Received task": "Received task",
        "Writing response": "Responding",
    }.get(title, title)


def _recent_agent_activity(limit: int = 7) -> list[dict]:
    run = active_run()
    run_pref = f"/runs/{run}" if run else ""
    items: list[dict] = []
    for cycle_dir in reversed(_cycle_dirs()):
        cycle = _cycle_id(cycle_dir)
        cycle_items: list[dict] = []
        for p in _agent_files(cycle_dir):
            m = re.match(r"agent_(\d+)\.jsonl", p.name)
            if not m:
                continue
            aid = int(m.group(1))
            rows = _load_jsonl(p)
            if not rows:
                continue
            summary = _agent_summary_from_rows(p, rows)
            last_ts = max((r.get("ts", 0) for r in rows), default=0)
            for ev in rows:
                if ev.get("event") not in {"start", "turn", "done"}:
                    continue
                ts = ev.get("ts") or summary["mtime"]
                live = _event_live_summary(ev)
                is_current = bool(ts == last_ts and not summary["finished"])
                if ev.get("event") == "done":
                    state = "done"
                elif is_current:
                    state = "active"
                else:
                    state = "idle"
                cycle_items.append({
                    "cycle": cycle,
                    "agent_id": aid,
                    "agent_label": f"ag_{aid:02d}",
                    "agent_kind": "agent" if aid == 0 else "subagent",
                    "role": _launch_role_label(summary["role"]),
                    "action": _launch_action_label(live),
                    "title": live["title"],
                    "body": live["body"],
                    "tool": live["tool"],
                    "turn": live["turn"] if live["turn"] is not None else "",
                    "updated": _fmt_relative_time(ts),
                    "ts": ts,
                    "state": state,
                    "href": f"{run_pref}/cycle/{cycle}/{aid}",
                })
        items.extend(cycle_items)
        if len(items) >= limit:
            break
    items.sort(key=lambda item: (item["ts"], item["cycle"], item["agent_id"]), reverse=True)
    return items[:limit]


def _live_snapshot() -> dict:
    run = active_run()
    run_pref = f"/runs/{run}" if run else ""
    latest = _latest_cycle_id()
    if not latest:
        return {
            "cycle": "n/a",
            "cycle_label": "no trace",
            "agent_id": "n/a",
            "agent_label": "agent n/a",
            "role": "No agent",
            "status": "idle",
            "activity_title": "No trace yet",
            "activity_body": "The trace directory does not contain cycles.",
            "tool": "",
            "turn": "",
            "updated": "n/a",
            "progress": 0,
            "agents_done": 0,
            "agents_total": 0,
            "cycle_url": "#",
            "agent_url": "#",
            "calls_url": "#",
            "feed": [],
            "launch_activity": [],
        }

    cycle_dir = trace_dir_for() / f"cycle_{latest}"
    records = _cycle_agent_records(cycle_dir)
    if not records:
        return {
            "cycle": latest,
            "cycle_label": f"cycle_{latest}",
            "agent_id": "n/a",
            "agent_label": "agent n/a",
            "role": "No agent",
            "status": "idle",
            "activity_title": "Cycle has no agent files",
            "activity_body": "Waiting for the first agent trace.",
            "tool": "",
            "turn": "",
            "updated": "n/a",
            "progress": 0,
            "agents_done": 0,
            "agents_total": 0,
            "cycle_url": f"{run_pref}/cycle/{latest}",
            "agent_url": f"{run_pref}/cycle/{latest}",
            "calls_url": f"{run_pref}/cycle/{latest}/calls",
            "feed": [],
            "launch_activity": _recent_agent_activity(limit=7),
        }

    enriched = []
    for r in records:
        last_ev = max(r["rows"], key=lambda ev: ev.get("ts", 0), default={})
        live = _event_live_summary(last_ev)
        enriched.append({**r, "last_event": last_ev, "live": live})

    unfinished = [r for r in enriched if not r["summary"]["finished"]]
    active = max(
        unfinished or enriched,
        key=lambda r: (r["last_event"].get("ts", 0), r["summary"]["mtime"], r["aid"]),
    )
    done = sum(1 for r in enriched if r["summary"]["finished"])
    progress = round(100 * done / max(len(enriched), 1))
    active_live = active["live"]
    status = "running" if not active["summary"]["finished"] else "completed"
    feed = []
    for r in sorted(
        enriched,
        key=lambda item: (item["last_event"].get("ts", 0), item["summary"]["mtime"]),
        reverse=True,
    )[:5]:
        live = r["live"]
        feed.append({
            "agent_id": r["aid"],
            "agent_label": f"ag_{r['aid']:02d}",
            "role": _role_display(r["summary"]["role"]),
            "title": live["title"],
            "body": live["body"],
            "state": "done" if r["summary"]["finished"] else "active",
            "href": f"{run_pref}/cycle/{latest}/{r['aid']}",
        })

    return {
        "cycle": latest,
        "cycle_label": f"cycle_{latest}",
        "agent_id": active["aid"],
        "agent_label": f"ag_{active['aid']:02d}",
        "role": _role_display(active["summary"]["role"]),
        "status": status,
        "activity_title": active_live["title"],
        "activity_body": active_live["body"],
        "tool": active_live["tool"],
        "turn": active_live["turn"] if active_live["turn"] is not None else "",
        "updated": _format_ts(active_live["ts"] or active["summary"]["mtime"]),
        "progress": progress,
        "agents_done": done,
        "agents_total": len(enriched),
        "cycle_url": f"{run_pref}/cycle/{latest}",
        "agent_url": f"{run_pref}/cycle/{latest}/{active['aid']}",
        "calls_url": f"{run_pref}/cycle/{latest}/calls",
        "feed": feed,
        "launch_activity": _recent_agent_activity(limit=7),
    }


def _trace_detail(text: str, label: str, *, limit: int = 320) -> str:
    if not text:
        return ""
    preview = _preview(" ".join(str(text).split()), limit)
    return (
        f"<details class=trace-detail><summary>{_h(label)}"
        f"<span class=preview> · {_h(preview)}</span></summary>"
        f"<pre>{_h(text)}</pre></details>"
    )


def _focus_nav(active: str = "train", trace_cycle: str | None = None) -> str:
    latest = trace_cycle or _latest_cycle_id() or "0001"
    items = [
        ("train", "/train", "TRAIN"),
        ("leaderboard", "/leaderboard", "LEADERBOARD"),
        ("trace", f"/cycle/{latest}", "TRACE"),
        ("about", ABOUT_URL, "ABOUT"),
    ]
    links = "".join(
        f"<a class='{('active' if key == active else '')}' href='{href}'>{label}</a>"
        for key, href, label in items
    )
    return (
        "<nav class=focus-nav>"
        "<div class=focus-logo>A-EVOLVE<span>·</span>MAS<span>·</span>TRAIN</div>"
        f"<div class=focus-links>{links}</div></nav>"
    )


def _trace_tabs(cycle: str, active: str) -> str:
    tabs = [
        ("overview", f"/cycle/{cycle}", "Overview"),
        ("sequence", f"/cycle/{cycle}/sequence", "Agent handoffs"),
        ("calls", f"/cycle/{cycle}/calls", "Tool activity"),
    ]
    return (
        "<nav class=trace-tabs>"
        + "".join(
            f"<a class='{('active' if key == active else '')}' href='{href}'>{label}</a>"
            for key, href, label in tabs
        )
        + "</nav>"
    )


def _trace_page(title: str, body: str, *, cycle: str, active: str = "overview") -> str:
    return (
        HEADER
        + f"<title>{_h(title)}</title></head><body>"
        + "<div class=focus-page>"
        + _focus_nav("trace", cycle)
        + "<main class=trace-shell>"
        + body
        + "</main></div></body></html>"
    )


def _trace_hero(cycle: str, title: str, subtitle: str, *, active: str) -> str:
    return (
        "<section class=trace-hero>"
        "<div><div class=trace-kicker>MAS TRACE · cycle_"
        + _h(cycle)
        + "</div>"
        + f"<h1>{_h(title)}</h1>"
        + f"<p>{_h(subtitle)}</p></div></section>"
        + _trace_tabs(cycle, active)
    )


def _cycle_agent_records(cycle_dir: Path) -> list[dict]:
    records: list[dict] = []
    for p in _agent_files(cycle_dir):
        m = re.match(r"agent_(\d+)\.jsonl", p.name)
        if not m:
            continue
        rows = _load_jsonl(p)
        summary = _agent_summary_from_rows(p, rows)
        records.append({
            "aid": int(m.group(1)),
            "path": p,
            "rows": rows,
            "summary": summary,
            "calls": _tool_calls(rows),
        })
    return records


def _final_assistant_text(rows: list[dict]) -> str:
    for r in reversed(rows):
        if r.get("event") != "turn" or r.get("stop_reason") != "end_turn":
            continue
        content = ((r.get("assistant") or {}).get("content") or [])
        for blk in content:
            if isinstance(blk, dict) and "text" in blk:
                return str(blk["text"])
    return ""


def _trace_agent_status(finished: bool) -> str:
    if finished:
        return "<span class='status-pill done'>Done</span>"
    return "<span class='status-pill running'>Running</span>"


def _dual_sign_chips(cp: dict) -> str:
    """Render the reviewer-verdict + human-signoff chip pair for a slot.

    Uses the folded ``last_review_verdict``, ``state``, and
    ``last_event_actor`` fields that ``_derive_checkpoints`` already
    produces. The old single ``qp-state`` pill is kept adjacent by the
    caller; these chips are extra visual info, not a replacement.
    """
    verdict = (cp.get("last_review_verdict") or "").strip()
    verdict_map = {
        "ready_to_sign": ("rev-ready", "ready_to_sign"),
        "evidence_attached": ("rev-evidence", "evidence"),
        "insufficient": ("rev-insufficient", "insufficient"),
        "reject": ("rev-reject", "reject"),
    }
    if verdict in verdict_map:
        cls, label = verdict_map[verdict]
    else:
        cls, label = ("rev-none", "no verdict")
    reviewer_chip = (
        f"<span class='qp-chip {cls}' title='Latest reviewer verdict on this slot'>"
        f"<span class=qp-chip-label>rev</span>{_h(label)}</span>"
    )

    state = cp.get("state") or ""
    actor = (cp.get("last_event_actor") or "").strip().lower()
    if state == "signed":
        if "auto" in actor or actor.startswith("orchestrator"):
            human_cls, human_label = ("hum-signed-auto", "auto-signed")
            title = f"Auto-signed by {actor or 'orchestrator_auto'}"
        else:
            human_cls, human_label = ("hum-signed", "signed")
            title = f"Signed by {actor or 'human'}"
    elif state == "reopened":
        human_cls, human_label = ("hum-waiting", "reopened")
        title = "Slot was signed but evidence changed; re-sign required"
    else:
        human_cls, human_label = ("hum-waiting", "awaiting")
        title = "Awaiting human signature"
    human_chip = (
        f"<span class='qp-chip {human_cls}' title='{_h(title)}'>"
        f"<span class=qp-chip-label>human</span>{_h(human_label)}</span>"
    )

    return f"<span class=qp-dual-sign>{reviewer_chip}{human_chip}</span>"


def _role_count_chips(roles: Counter) -> str:
    if not roles:
        return "<span class=trace-chip>no roles detected</span>"
    return "".join(
        f"<span class=trace-chip>{_h(_role_display(role))} · {count}</span>"
        for role, count in roles.most_common()
    )


def _live_feed_html(feed: list[dict]) -> str:
    if not feed:
        return "<div class=run-readable-body>No live agent activity yet.</div>"
    return "".join(
        f"<a class='live-feed-item {('active' if item.get('state') == 'active' else '')}' "
        f"href='{_h(item.get('href', '#'))}'>"
        f"<div class=live-feed-title>{_h(item.get('agent_label', 'ag_??'))} · "
        f"{_h(item.get('role', 'Agent'))}</div>"
        f"<div class=live-feed-body>{_h(item.get('title', 'Working'))}: "
        f"{_h(_preview(item.get('body', ''), 100))}</div></a>"
        for item in feed
    )


def _live_script() -> str:
    run = _ACTIVE_RUN.get()
    prefix_literal = f'"/runs/{run}"' if run else '""'
    # Use plain concat instead of an f-string so the JS `{}` tokens don't
    # need to be escaped; only the RUN_PREFIX line gets templated.
    return ("""
<script>
(() => {
  const RUN_PREFIX = """ + prefix_literal + """;
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
  const setText = (id, value) => { const el = $(id); if (el) el.textContent = value ?? ""; };
  const setHref = (id, value) => { const el = $(id); if (el) el.href = value || "#"; };
  async function refreshLive() {
    try {
      const res = await fetch(RUN_PREFIX + "/live-status.json?t=" + Date.now(), {cache: "no-store"});
      if (!res.ok) return;
      const live = await res.json();
      setText("live-cycle", live.cycle_label);
      setText("live-agent", live.agent_label + " · " + live.role);
      setText("live-status", live.status === "running" ? "LIVE" : "DONE");
      setText("live-title", live.activity_title);
      setText("live-body", live.activity_body);
      setText("live-turn", live.turn === "" ? "n/a" : "turn " + live.turn);
      setText("live-tool", live.tool || "no tool");
      setText("live-updated", live.updated);
      setText("live-done", live.agents_done + " / " + live.agents_total);
      setText("live-done-progress", live.agents_done + " / " + live.agents_total);
      setText("live-progress-label", live.progress + "%");
      const fill = $("live-progress-fill");
      if (fill) fill.style.width = live.progress + "%";
      setHref("live-cycle-link", live.cycle_url);
      setHref("live-agent-link", live.agent_url);
      setHref("live-calls-link", live.calls_url);
      const feed = $("live-feed");
      if (feed) {
        feed.innerHTML = (live.feed || []).map((item) => (
          `<a class="live-feed-item ${item.state === "active" ? "active" : ""}" href="${esc(item.href || "#")}">` +
          `<div class="live-feed-title">${esc(item.agent_label)} · ${esc(item.role)}</div>` +
          `<div class="live-feed-body">${esc(item.title)}: ${esc(item.body).slice(0, 120)}</div></a>`
        )).join("");
      }
      const launchActivity = $("launch-activity");
      if (launchActivity) {
        const items = live.launch_activity || [];
        launchActivity.hidden = items.length === 0;
        launchActivity.innerHTML = items.map((item) => (
          `<a class="launch-activity-item ${esc(item.state || "idle")}" href="${esc(item.href || "#")}">` +
          `<span class="launch-activity-node"></span>` +
          `<span class="launch-activity-label">${esc(item.role || "Agent")}: ${esc(item.action || item.title || "Working")}</span>` +
          `</a>`
        )).join("");
      }
    } catch (_) {}
  }
  setInterval(refreshLive, 6000);
  setTimeout(refreshLive, 400);
})();
</script>
""")


def _render_run_rail(live: dict, totals: dict, progress: int, signed_count: int,
                     reopened_count: int, human_gate_count: int) -> str:
    return (
        "<aside class=run-rail>"
        + "<div class=run-rail-head>"
        + "<div class=run-rail-kicker>Live Run Pulse</div>"
        + "<h3><span class=live-dot></span><span id=live-cycle>"
        + _h(live["cycle_label"])
        + "</span> · <span id=live-status>"
        + ("LIVE" if live["status"] == "running" else "DONE")
        + "</span></h3>"
        + "<p>A compact live progress view: current cycle, active agent, latest action, and clickable trace links. It reads the latest JSONL trace but only shows human-readable summaries here.</p>"
        + "</div>"
        + "<div class='run-panel live'>"
        + "<div class=run-panel-title>Current agent</div>"
        + "<div class=live-agent-card>"
        + "<div class=live-agent-top><div class=live-agent-name id=live-agent>"
        + _h(f"{live['agent_label']} · {live['role']}")
        + "</div><a class=trace-link id=live-agent-link href='"
        + _h(live["agent_url"])
        + "'>Open</a></div>"
        + "<div class=live-activity-title id=live-title>"
        + _h(live["activity_title"])
        + "</div>"
        + "<div class=live-activity-body id=live-body>"
        + _h(live["activity_body"])
        + "</div>"
        + "<div class=live-meta-grid>"
        + "<div class=live-meta><span>Turn</span><strong id=live-turn>"
        + (_h(f"turn {live['turn']}") if live["turn"] != "" else "n/a")
        + "</strong></div>"
        + "<div class=live-meta><span>Tool</span><strong id=live-tool>"
        + _h(live["tool"] or "no tool")
        + "</strong></div>"
        + "<div class=live-meta><span>Updated</span><strong id=live-updated>"
        + _h(live["updated"])
        + "</strong></div>"
        + "<div class=live-meta><span>Agents done</span><strong id=live-done>"
        + _h(f"{live['agents_done']} / {live['agents_total']}")
        + "</strong></div>"
        + "</div></div></div>"
        + "<div class=run-panel>"
        + "<div class=run-panel-title>Cycle progress</div>"
        + "<div class=run-progress-line><span id=live-progress-fill style='width:"
        + _h(live["progress"])
        + "%'></span></div>"
        + "<div style='align-items:center;display:flex;justify-content:space-between;margin-top:9px;color:#98a2b3;font-size:12px;'>"
        + "<span id=live-done-progress>"
        + _h(f"{live['agents_done']} / {live['agents_total']}")
        + "</span><strong id=live-progress-label style='color:#ffb84d;'>"
        + _h(f"{live['progress']}%")
        + "</strong></div>"
        + "</div>"
        + "<div class=run-panel>"
        + "<div class=run-panel-title>Recent agent activity</div>"
        + "<div class=live-feed id=live-feed>"
        + _live_feed_html(live["feed"])
        + "</div>"
        + "</div>"
        + "<div class=run-panel>"
        + "<div class=run-panel-title>Quality gate</div>"
        + f"<div class=run-readable-body>{signed_count} signed · {reopened_count} reopened · {human_gate_count} human gate<br>"
        + f"Plan ledger progress: <b style='color:#ffb84d;'>{progress}%</b></div>"
        + "</div>"
        + "<div class='run-panel shortcuts'>"
        + "<div class=run-panel-title>Trace shortcuts</div>"
        + "<div class=run-readable-body>"
        + "<a class=trace-link id=live-cycle-link href='"
        + _h(live["cycle_url"])
        + "'>Open current cycle</a><br>"
        + "<a class=trace-link id=live-calls-link href='"
        + _h(live["calls_url"])
        + "'>Open tool activity</a><br>"
        + f"{totals['cycles']} cycles · {totals['agents']} agent traces"
        + "</div></div>"
        + "</aside>"
    )


def _top_nav(active: str) -> str:
    items = [
        ("runs", "/train", "Research Runs"),
        ("papers", "#", "Papers"),
        ("experiments", "#checkpoint-table", "Experiments"),
        ("evals", "#eval-card", "Evaluations"),
        ("literature", "#", "Literature"),
        ("settings", "#", "Settings"),
    ]
    top_active = active if active in {key for key, _, _ in items} else "runs"
    return "".join(
        f"<a class='{('active' if key == top_active else '')}' href='{href}'>{label}</a>"
        for key, href, label in items
    )


def _sidebar(active: str) -> str:
    counts = _quality_state_counts()
    latest = _latest_cycle_id()
    trace_href = f"/cycle/{latest}" if latest else "#trace-cycles"
    nav_items = [
        ("runs", "/train", "▣", "Research Runs"),
        ("plan", "#checkpoint-table", "◇", "Quality Plan"),
        ("trace", trace_href, "⌁", "MAS Trace"),
        ("cards", "#eval-card", "▤", "Checkpoint Cards"),
        ("evidence", "#evidence", "□", "Evidence Bundles"),
        ("audit", "#audit", "∥", "Audit Log"),
    ]
    nav = "".join(
        f"<a class='{('active' if key == active else '')}' href='{href}'>"
        f"<span class=side-icon>{icon}</span>{label}</a>"
        for key, href, icon, label in nav_items
    )
    return f"""
    <aside class=sidebar>
      <section class=side-section>
        <div class=side-title>Workspace</div>
        <nav class=side-nav>{nav}</nav>
      </section>
      <section class=side-section>
        <div class=side-title>Checkpoint State</div>
        <div class=filter-row><span><span class='dot green'></span> Signed</span><span class=badge>{counts.get('signed', 0)}</span></div>
        <div class=filter-row><span><span class='dot orange'></span> Human gate</span><span class=badge>{counts.get('pending_human', 0)}</span></div>
        <div class=filter-row><span><span class='dot blue'></span> Evidence</span><span class=badge>{counts.get('pending_evidence', 0)}</span></div>
        <div class=filter-row><span><span class='dot red'></span> Reopened</span><span class=badge>{counts.get('reopened', 0)}</span></div>
        <div class=filter-row><span><span class='dot gray'></span> Draft</span><span class=badge>{counts.get('draft', 0)}</span></div>
      </section>
      <section class=side-section>
        <div class=side-title>Signers</div>
        <div class=mini-row><span>Alice · owner</span><span class=badge>5</span></div>
        <div class=mini-row><span>Bob · reviewer</span><span class=badge>1</span></div>
        <div class=mini-row><span>Card Agent</span><span class=badge>0</span></div>
      </section>
      <div class='collapse-link meta'>« Collapse</div>
    </aside>
    """


def _page_shell(title: str, main: str, *, active: str = "runs") -> str:
    return (
        HEADER
        + f"<title>{_h(title)}</title></head><body>"
        + "<header class=topbar>"
        + "<div class=brand><div class=brand-mark></div><div>"
        + "<div class=brand-title>A-Evolve-MAS-Train<span class=beta>beta</span></div>"
        + "<div class=brand-subtitle>Evolvable Model Auto Training System</div>"
        + "</div></div>"
        + f"<nav class=topnav>{_top_nav(active)}</nav>"
        + "<div class=top-actions><span>Env: Internal Beta ▾</span><span>?</span><span>♧</span><span class=avatar>QP</span></div>"
        + "</header>"
        + "<div class=app-shell>"
        + _sidebar(active)
        + f"<main class=workspace>{main}</main>"
        + "</div></body></html>"
    )


def _render_launch_activity(items: list[dict]) -> str:
    if not items:
        return "<section class=launch-activity id=launch-activity aria-live=polite hidden></section>"
    cards = []
    for item in items:
        state = item.get("state") if item.get("state") in {"active", "done", "idle"} else "idle"
        cards.append(
            f"<a class='launch-activity-item {state}' href='{_h(item.get('href', '#'))}'>"
            "<span class=launch-activity-node></span>"
            f"<span class=launch-activity-label>{_h(item.get('role', 'Agent'))}: "
            f"{_h(item.get('action') or item.get('title', 'Working'))}</span>"
            "</a>"
        )
    return (
        "<section class=launch-activity id=launch-activity aria-live=polite>"
        + "".join(cards)
        + "</section>"
    )


def render_pre_launch() -> str:
    """Shown when the active run has no cycles and no memory records yet.

    Rather than paint the full cockpit with empty slots (confusing —
    looks like things are broken), show one card: what path the viewer
    is watching, what state each signal is in, and the command the
    user probably needs to run. Polls every 4s and auto-refreshes
    once the driver touches any of the watched paths.
    """
    run = active_run() or "(unscoped)"
    trace_dir = trace_dir_for()
    memory_path = memory_path_for()
    driver_log = driver_log_path_for()

    def _state_row(label: str, path: Path, expected_cmd: str) -> str:
        exists = path.is_file() if str(path).endswith(".jsonl") or str(path).endswith(".log") else path.is_dir()
        badge = ("<span class='run-badge ok'>ready</span>" if exists
                 else "<span class='run-badge warn'>waiting</span>")
        return (
            "<tr>"
            f"<td>{_h(label)}</td>"
            f"<td style='font-family:ui-monospace,Menlo,monospace;font-size:12px;'>{_h(str(path))}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )

    extra_css = """
    .pre-launch-wrap { max-width: 880px; margin: 56px auto; padding: 0 24px; }
    .pre-launch-card { background:#fff; border: 1px solid var(--aws-line); border-radius: 10px; box-shadow: var(--shadow-sm); overflow: hidden; }
    .pre-launch-head { padding: 24px 28px; border-bottom: 1px solid var(--aws-soft-line); display:flex; justify-content:space-between; align-items:center; gap:16px; }
    .pre-launch-head h1 { margin:0; font-size: 22px; }
    .pre-launch-head .meta { color:#667085; font-size:13px; margin-top: 4px; }
    .pre-launch-body { padding: 20px 28px 24px; }
    .pre-launch-body table { width: 100%; border-collapse: collapse; margin: 12px 0 20px; }
    .pre-launch-body td { padding: 10px 12px; border-bottom: 1px solid var(--aws-soft-line); font-size:13px; vertical-align: top; }
    .pre-launch-body code.block { display:block; background:#0f172a; color:#e5e7eb; padding: 14px 16px; border-radius: 6px; font-size: 12px; white-space: pre-wrap; word-break: break-all; }
    .pre-launch-hint { color: #475467; font-size: 13px; margin-top: 14px; line-height: 1.55; }
    """

    launch_cmd = (
        f"cd /fsx/zzsamshi/a-evolve\n"
        f"PYTHONPATH=/fsx/zzsamshi/a-evolve .venv/bin/python \\\n"
        f"  examples/nemo_mas_reasoning_example/drive_nemo_mas.py \\\n"
        f"  --mode real --backend k8s --cycles 50 \\\n"
        f"  --work-dir {RUNS_ROOT / run} \\\n"
        f"  --trace-dir {RUNS_ROOT / run / 'trace'}"
    )

    return (
        HEADER
        + f"<style>{extra_css}</style>"
        + f"<title>{_h(run)} · waiting to launch</title></head><body>"
        + "<meta http-equiv=refresh content='10'>"
        + "<div class=pre-launch-wrap>"
        + "<div class=pre-launch-card>"
        + "<div class=pre-launch-head>"
        + f"<div><h1>{_h(run)}</h1>"
        + "<div class=meta>Waiting for the driver to write the first cycle or memory record.</div></div>"
        + "<a class='button' href='/runs'>← All runs</a>"
        + "</div>"
        + "<div class=pre-launch-body>"
        + "<table>"
        + _state_row("trace/", trace_dir, "driver writes cycle_NNNN/ subdirs here")
        + _state_row("memory/records.jsonl", memory_path, "driver appends typed records here")
        + _state_row("driver.log", driver_log, "driver stdout+stderr")
        + "</table>"
        + "<div class=pre-launch-hint>Launch command (run it on the host, not here):</div>"
        + f"<code class=block>{_h(launch_cmd)}</code>"
        + "<div class=pre-launch-hint>This page auto-reloads every 10s. Once the driver writes its first record "
        + "the full cockpit appears automatically — no manual refresh needed.</div>"
        + "</div></div></div>"
        + "</body></html>"
    )


def render_entry() -> str:
    # Pre-launch short-circuit: the run exists on disk but the driver
    # hasn't written any trace or memory yet. Showing the full Model
    # Forge UI at this point looks like things are broken — instead
    # bounce to the clean "waiting" card (same as /train pre-launch).
    if not _cycle_dirs() and not _load_records():
        return render_pre_launch()
    live = _live_snapshot()
    activity = live.get("launch_activity", [])
    status_label = "Agent activity live" if activity and live["status"] == "running" else "System nominal"
    enter_run = _pick_default_run() or active_run()
    enter_href = f"/runs/{enter_run}/train" if enter_run else "/train"
    return (
        HEADER
        + "<title>A-Evolve Model Forge</title></head><body>"
        + "<main class=launch-page>"
        + "<nav class=launch-nav>"
        + "<div><div class=launch-brand>A-EVOLVE<span>|FORGE</span></div>"
        + "<div class=launch-subbrand>model training and evolution system</div></div>"
        + f"<div class=launch-status><span class=launch-status-dot></span>{_h(status_label)}</div>"
        + "</nav>"
        + "<section class=launch-scene>"
        + "<div class=launch-title>"
        + "<div class=launch-actions>"
        + f"<a class='launch-link primary' href='{_h(enter_href)}'>ENTER</a>"
        + "<a class=launch-link href='/runs'>Past runs</a>"
        + "</div>"
        + "</div>"
        + "</section>"
        + _render_launch_activity(activity)
        + "</main>"
        + _live_script()
        + "</body></html>"
    )


# ── rendering ────────────────────────────────────────────────────────

def render_index(selected_slot_id: str | None = None) -> str:
    totals = _trace_totals()
    cycles = _cycle_summaries(include_roles=True)
    records_preview = _load_records()
    # Pre-launch short-circuit: driver hasn't written the first trace or
    # the first record yet. Show a clean "waiting" card instead of the
    # full cockpit (which would render as a row of empty panels pretending
    # we have checkpoints / eval runs / a chat thread).
    if not cycles and not records_preview:
        return render_pre_launch()
    latest = cycles[-1] if cycles else None
    latest_id = latest["id"] if latest else "0000"
    records = records_preview
    run_mode = _mode_for_active_run()
    slots = _derive_checkpoints(records, run_mode)
    eval_runs = _derive_eval_runs(records)
    thread = _derive_chat_thread(records, limit=5)
    progress = _checkpoint_progress()
    live = _live_snapshot()
    last_cycle_href = f"/cycle/{latest_id}" if latest else "#"
    signed_count = sum(1 for cp in slots if cp["state"] == "signed")
    reopened_count = sum(1 for cp in slots if cp["state"] == "reopened")
    human_gate_count = sum(1 for cp in slots if cp["state"] == "pending_human")
    sequence_href = f"{last_cycle_href}/sequence" if latest else "#"

    # Active slot = URL-pinned one (if it exists), else first pending required,
    # else last. Row click carries ?slot=<id> so the card inspects any slot.
    pinned_slot = next(
        (cp for cp in slots if cp["id"] == selected_slot_id),
        None,
    ) if selected_slot_id else None
    active_slot = pinned_slot or next(
        (cp for cp in slots if cp["state"] not in {"signed", "reopened"}),
        slots[-1] if slots else None,
    )

    ledger_rows = []
    for i, cp in enumerate(slots):
        is_active = active_slot and cp["id"] == active_slot["id"]
        active_cls = " active" if is_active else ""
        signed_cls = " signed" if cp["state"] == "signed" else ""
        state_class = {
            "signed": "signed",
            "reopened": "reopened",
            "pending_human": "pending",
            "pending_evidence": "pending",
            "pending": "pending",
        }.get(cp["state"], "pending")
        sign_btn = ""
        if cp["can_sign"]:
            # Sibling of the row anchor (not nested — <form> inside <a> is
            # illegal HTML and browsers reparent it, breaking the submit).
            sign_btn = (
                f"<form class=qp-sign-form method=post action='/checkpoint/{_h(cp['id'])}/sign'>"
                f"<input type=hidden name=actor value='human:owner'>"
                f"<input class=qp-sign-note type=text name=note placeholder='Signoff note (optional)'>"
                f"<button class='qp-btn primary' type=submit>Sign</button>"
                f"</form>"
            )
        row_href = f"/train?slot={_h(cp['id'])}"
        dual_chips = _dual_sign_chips(cp)
        ledger_rows.append(
            f"<div class=qp-plan-entry>"
            f"<a class='qp-plan-row{active_cls}{signed_cls}' href='{row_href}'>"
            f"<span class=qp-num>{i:02d}</span>"
            f"<span class=qp-row-title>{_h(cp['title'])}</span>"
            f"<span class=qp-type>{_h(cp['type'])}</span>"
            f"<span class='qp-state {state_class}'>{_h(_state_label(cp['state']))}</span>"
            f"{dual_chips}"
            f"</a>"
            f"{sign_btn}"
            f"</div>"
        )

    # Current card = the active slot's state + evidence + the best linked run.
    linked_run = next(
        (r for r in eval_runs if active_slot and r["cycle"]
         and r["cycle"] == _latest_cycle_id_from_records(records)),
        eval_runs[0] if eval_runs else None,
    )
    current_card = _render_current_card(active_slot, linked_run, records)

    chat_html = _render_chat_widget(thread)

    mode_chip = (
        "<span class=focus-chip>Auto-sign mode</span>"
        if run_mode == CHECKPOINT_MODE_AUTO
        else "<span class=focus-chip>Manual sign mode</span>"
    )

    return (
        HEADER
        + "<title>A-Evolve-MAS-Train</title></head><body>"
        + "<div class=focus-page>"
        + "<nav class=focus-nav>"
        + "<div class=focus-logo>A-EVOLVE<span>·</span>MAS<span>·</span>TRAIN</div>"
        + "<div class=focus-links>"
        + "<a class=active href='/train'>TRAIN</a><a href='/leaderboard'>LEADERBOARD</a><a href='/cycle/"
        + _h(latest_id)
        + f"'>TRACE</a><a href='{ABOUT_URL}'>ABOUT</a>"
        + "</div></nav>"
        + "<section class=focus-hero>"
        + "<h1>A-Evolve-MAS-Train: A Evolvable Model Auto Training System</h1>"
        + "<div class=focus-subtitle>A contract-driven MAS training cockpit where every expensive training decision closes through a signed Quality Plan Card.</div>"
        + "<div class=focus-chips>"
        + f"<span class=focus-chip>Signature ledger · {signed_count} / {len(slots)} signed</span>"
        + f"<span class=focus-chip>Reopened · {reopened_count}</span>"
        + (f"<span class=focus-chip>Current gate · {_h(active_slot['id'])} {_h(_state_label(active_slot['state']))}</span>"
           if active_slot else "<span class=focus-chip>No active gate</span>")
        + mode_chip
        + "</div></section>"
        + "<section class=qp-stage id=checkpoint-table>"
        + "<aside class=qp-ledger>"
        + chat_html
        + "<div class=qp-ledger-head>"
        + "<div><div class=qp-ledger-title>Quality Plan Ledger</div>"
        + f"<div class=qp-ledger-project>mode: {_h(run_mode)} · {len(records)} records</div></div>"
        + f"<span class='qp-state pending'>{progress}%</span>"
        + "</div>"
        + "".join(ledger_rows)
        + "<div class=qp-ledger-note>This is a ledger of signatures, not a telemetry dashboard. Slot state folds <code>checkpoint_event</code> records from the memory store; click Sign on a <em>pending_human</em> row to append a signoff event.</div>"
        + "</aside>"
        + "<main class=qp-workspace>"
        + "<div class=qp-workspace-head>"
        + "<div><h2 style='color:#f8fafc;margin:0;font-size:28px;'>Current Quality Plan Card</h2>"
        + "<div style='color:#98a2b3;font-size:13px;margin-top:4px;'>Human action surface for the next gate. Evidence is the backend's records.jsonl.</div></div>"
        + f"<a class=trace-link href='{sequence_href}'>Open agent sequence</a>"
        + "</div>"
        + current_card
        + "<div class=qp-bottom-stats>"
        + f"<div class=panel><div class=panel-title>Trace cycles</div><strong>{totals['cycles']}</strong><div class=meta>{totals['agents']} agent traces</div></div>"
        + f"<div class=panel><div class=panel-title>Memory records</div><strong>{len(records)}</strong><div class=meta>across {len({r.get('cycle_id','') for r in records if r.get('cycle_id')})} cycles</div></div>"
        + f"<div class=panel><div class=panel-title>Checkpoint mode</div><strong>{_h(run_mode)}</strong><div class=meta>from meta.json</div></div>"
        + "</div>"
        + "</main>"
        + _render_run_rail(live, totals, progress, signed_count,
                           reopened_count, human_gate_count)
        + "</section></div>"
        + _live_script()
        + "</body></html>"
    )


def _render_chat_widget(thread: list[dict]) -> str:
    thread_items = []
    for entry in thread:
        resp = entry.get("response")
        if resp:
            resp_html = (
                "<div class=chat-resp>"
                f"<span class=chat-resp-label>Orchestrator · {_h(resp.get('action',''))}</span>"
                f"<div class=chat-resp-text>{_h(resp.get('summary') or '(no summary)')}</div>"
                + (f"<div class=chat-resp-meta>spawned {_h(resp['spawned_role'])}</div>"
                   if resp.get("spawned_role") else "")
                + "</div>"
            )
        else:
            resp_html = "<div class='chat-resp waiting'>Orchestrator has not replied yet.</div>"
        thread_items.append(
            "<li class=chat-item>"
            f"<div class=chat-meta><span class=chat-actor>{_h(entry.get('actor','human'))}</span>"
            f"<span class=chat-urgency>{_h(entry.get('urgency') or '—')}</span></div>"
            f"<div class=chat-text>{_h(entry.get('text') or '')}</div>"
            f"{resp_html}"
            "</li>"
        )
    thread_html = (
        "<ol class=chat-thread>" + "".join(thread_items) + "</ol>"
        if thread_items
        else "<div class='chat-thread empty'>No directives yet — send the orchestrator a hint.</div>"
    )
    return (
        "<section class=chat-box>"
        "<div class=chat-title>Talk to the orchestrator</div>"
        "<form class=chat-form method=post action='/directive'>"
        "<textarea name=text required placeholder='Share an idea, research hint, or redirect…' rows=3></textarea>"
        "<div class=chat-form-row>"
        "<select name=urgency>"
        "<option value=when_convenient>when convenient</option>"
        "<option value=next_cycle selected>next cycle</option>"
        "<option value=immediate>immediate</option>"
        "</select>"
        "<input type=hidden name=actor value='human:owner'>"
        "<button class='qp-btn primary' type=submit>Send</button>"
        "</div></form>"
        + thread_html
        + "</section>"
    )


def _render_slot_evidence_list(slot: dict, records: list[dict]) -> str:
    """Group slot-tagged records by required kind, render one row per record
    with a link to /record/<id>. Falls back to a plain count pill for any
    kind that has zero matching records.
    """
    requires = list(slot.get("requires_evidence") or ())
    sid = slot["id"]
    # bucket records by kind, slot-tagged only
    buckets: dict[str, list[dict]] = {k: [] for k in requires}
    for rec in records:
        kind = rec.get("kind")
        if kind not in buckets:
            continue
        tags = rec.get("tags") or []
        if f"checkpoint:{sid}" not in tags:
            continue
        buckets[kind].append(rec)
    for k in buckets:
        buckets[k].sort(key=lambda r: r.get("ts", ""), reverse=True)

    if not requires:
        return "<div class=qp-evidence-empty>no required evidence kinds declared</div>"

    blocks = []
    for kind in requires:
        rows = buckets[kind]
        if not rows:
            blocks.append(
                f"<div class=qp-evidence-group>"
                f"<div class=qp-evidence-kind><code>{_h(kind)}</code>"
                f" <span class='qp-pill warn'>0</span></div>"
                f"<div class=qp-evidence-empty>no records with tag "
                f"<code>checkpoint:{_h(sid)}</code> yet</div>"
                f"</div>"
            )
            continue
        items = []
        for r in rows[:8]:
            rid = r.get("id", "")
            title = r.get("title") or rid
            cycle = r.get("cycle_id") or ""
            author = r.get("author") or ""
            items.append(
                f"<a class=qp-evidence-item href='/record/{_h(rid)}'>"
                f"<span class=qp-evidence-title>{_h(title)}</span>"
                f"<span class=qp-evidence-meta>{_h(cycle)} · {_h(author)}</span>"
                f"<code class=qp-evidence-id>{_h(rid)}</code>"
                f"</a>"
            )
        extra = (f"<div class=qp-evidence-extra>+{len(rows) - 8} more</div>"
                 if len(rows) > 8 else "")
        blocks.append(
            f"<div class=qp-evidence-group>"
            f"<div class=qp-evidence-kind><code>{_h(kind)}</code>"
            f" <span class='qp-pill ok'>{len(rows)}</span></div>"
            + "".join(items) + extra +
            f"</div>"
        )
    return "".join(blocks)


def _render_current_card(slot: dict | None, run: dict | None,
                         records: list[dict]) -> str:
    """Render the 'Current Quality Plan Card' for the active slot.

    Shape mirrors the old hardcoded eval card but fills from derived data.
    Empty evidence renders explicit 'not yet' pills so the human sees what
    the agent still needs to produce.
    """
    if slot is None:
        return (
            "<div class='qp-card eval'><div class=qp-card-header>"
            "<div class=qp-card-left><span class=qp-card-id>—</span>"
            "<span class=qp-card-title>No active checkpoint</span></div>"
            "</div><div class=qp-card-body>"
            "<div class=qp-section><div class=qp-section-body>"
            "Nothing to sign; the records store has no slot declarations or "
            "every slot is signed. Launch a cycle to start producing evidence."
            "</div></div></div></div>"
        )

    state_cls = "pending" if slot["state"].startswith("pending") else slot["state"]
    evidence_html = _render_slot_evidence_list(slot, records)

    if run:
        metrics_html = (
            "<div class=qp-metric-row><span>Public LB</span>"
            f"<span>{run['kaggle']:.3f}</span><span>{_h(run['delta'])}</span></div>"
            "<div class=qp-metric-row><span>Local</span>"
            f"<span>{run['local']:.3f}</span><span>&nbsp;</span></div>"
            "<div class=qp-metric-row><span>Hard</span>"
            f"<span>{run['hard']:.3f}</span><span>&nbsp;</span></div>"
        )
        interp = run.get("score_note") or "—"
        recipe_line = f"{run['recipe']} · gate {run['quality_gate']}"
    else:
        metrics_html = (
            "<div class=qp-metric-row><span>Public LB</span><span>—</span><span>—</span></div>"
            "<div class=qp-metric-row><span>Local</span><span>—</span><span>—</span></div>"
            "<div class=qp-metric-row><span>Hard</span><span>—</span><span>—</span></div>"
        )
        interp = "No cv_result linked to this cycle yet. Evidence is still pending."
        recipe_line = "(no recipe attached)"

    deps = ", ".join(slot["depends_on"]) or "(none)"
    artifacts_html = ""
    if run and run.get("artifacts"):
        artifacts_html = "<br>".join(_h(a) for a in run["artifacts"][:8])
    else:
        artifacts_html = "artifact://… (attach via mem_write bodies)"

    sign_form = ""
    if slot["can_sign"]:
        sign_form = (
            f"<form class=qp-sign-form method=post action='/checkpoint/{_h(slot['id'])}/sign'>"
            "<input type=hidden name=actor value='human:owner'>"
            "<input class=qp-sign-note type=text name=note placeholder='Signoff note'>"
            "<button class='qp-btn primary' type=submit>Sign</button>"
            "</form>"
        )

    return (
        "<div class='qp-card eval'>"
        "<div class=qp-card-header>"
        f"<div class=qp-card-left><span class=qp-card-id>{_h(slot['id'])}</span>"
        f"<span class=qp-card-title>{_h(slot['title'])}</span></div>"
        f"<div class=qp-card-right><span class='qp-state {state_cls}'>"
        f"{_h(_state_label(slot['state']))} · {_h(slot['signers'])}</span>"
        f"{_dual_sign_chips(slot)}</div>"
        "</div>"
        f"<div class=qp-prereview>Depends on: {_h(deps)}. Requires evidence: {_h(', '.join(slot['requires_evidence']) or '(none)')}</div>"
        "<div class=qp-card-body>"
        "<div class='qp-section highlight'><div class=qp-section-title>Linked run</div>"
        f"<div class=qp-section-body>{metrics_html}</div></div>"
        "<div class='qp-section warn'><div class=qp-section-title>Interpretation</div>"
        f"<div class=qp-section-body>{_h(interp)}</div></div>"
        "<div class=qp-section><div class=qp-section-title>Recipe</div>"
        f"<div class=qp-section-body>{_h(recipe_line)}</div></div>"
        "<div class=qp-section><div class=qp-section-title>Required evidence slots</div>"
        f"<div class=qp-section-body>{evidence_html}</div></div>"
        "<div class=qp-section><div class=qp-section-title>Artifacts</div>"
        f"<div class=qp-section-body style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;'>{artifacts_html}</div></div>"
        "</div>"
        f"{sign_form}"
        "</div>"
    )


def _pretty_body(body: str) -> str:
    """Render a record body: treat as JSON if it parses, else as plain text.
    Keeps a fenced-JSON tail intact — they're short, so dump the whole thing.
    """
    if not body:
        return "<em>(empty body)</em>"
    stripped = body.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            return f"<pre class=record-body>{_h(json.dumps(parsed, indent=2, ensure_ascii=False))}</pre>"
        except json.JSONDecodeError:
            pass
    return f"<pre class=record-body>{_h(body)}</pre>"


def render_record(rec_id: str) -> str:
    """Human view of a single memory record. Refs + tags become links back
    into the viewer: refs → /record/<id>, checkpoint tag → /train?slot=<id>.
    """
    records = _load_records()
    rec = _record_by_id(records, rec_id)
    if rec is None:
        return (
            HEADER + "<title>Record not found</title></head><body>"
            "<div class=focus-page><main class=lb-shell>"
            f"<h1>Unknown record: <code>{_h(rec_id)}</code></h1>"
            "<p>This id is not present in the current run's records.jsonl.</p>"
            "<a class=trace-link href='/train'>Back to Quality Plan</a>"
            "</main></div></body></html>"
        )

    refs_html = ""
    for ref in rec.get("refs") or []:
        target = _record_by_id(records, ref)
        label = (target.get("title") or target.get("kind") or ref) if target else ref
        refs_html += (
            f"<li><a class=trace-link href='/record/{_h(ref)}'>"
            f"<code>{_h(ref)}</code> · {_h(label)}</a></li>"
        )
    refs_html = refs_html or "<li><em>(no refs)</em></li>"

    tag_html = ""
    for tag in rec.get("tags") or []:
        if tag.startswith("checkpoint:"):
            sid = tag.split(":", 1)[1]
            tag_html += (
                f"<a class='qp-pill ok' href='/train?slot={_h(sid)}'>"
                f"{_h(tag)}</a> "
            )
        else:
            tag_html += f"<span class='qp-pill'>{_h(tag)}</span> "
    tag_html = tag_html or "<em>(no tags)</em>"

    # Which records cite this one?
    referenced_by = [
        r for r in records if rec_id in (r.get("refs") or [])
    ]
    ref_by_html = ""
    for r in referenced_by[:20]:
        rid = r.get("id", "")
        ref_by_html += (
            f"<li><a class=trace-link href='/record/{_h(rid)}'>"
            f"<code>{_h(rid)}</code> · {_h(r.get('kind') or '')} · "
            f"{_h(r.get('title') or '')}</a></li>"
        )
    ref_by_html = ref_by_html or "<li><em>(not referenced by any record yet)</em></li>"

    cycle = rec.get("cycle_id", "")
    cycle_link = (f"<a class=trace-link href='/cycle/{_h(cycle)}'>cycle {_h(cycle)}</a>"
                  if cycle else "—")

    return (
        HEADER + "<title>Record · " + _h(rec_id) + "</title></head><body>"
        "<div class=focus-page>"
        + "<nav class=focus-nav><div class=focus-logo>A-EVOLVE<span>·</span>MAS<span>·</span>TRAIN</div>"
        + "<div class=focus-links>"
        + "<a href='/train'>TRAIN</a><a href='/leaderboard'>LEADERBOARD</a>"
        + f"<a href='/cycle/{_h(_latest_cycle_id() or '0001')}'>TRACE</a>"
        + f"<a href='{ABOUT_URL}'>ABOUT</a></div></nav>"
        + "<main class=lb-shell>"
        + f"<section class=lb-header><div class=lb-title>"
        + f"<h1>{_h(rec.get('title') or rec_id)}</h1>"
        + f"<p><code>{_h(rec_id)}</code> · <b>{_h(rec.get('kind') or '')}</b>"
        + f" · author <code>{_h(rec.get('author') or '')}</code>"
        + f" · {cycle_link} · {_h(rec.get('ts') or '')}</p>"
        + "</div></section>"
        + "<section class=lb-card>"
        + "<div class=record-grid>"
        + "<div class=record-pane><h3>Tags</h3>"
        + f"<div>{tag_html}</div></div>"
        + "<div class=record-pane><h3>Refs (upstream)</h3>"
        + f"<ul class=record-links>{refs_html}</ul></div>"
        + "<div class=record-pane><h3>Referenced by (downstream)</h3>"
        + f"<ul class=record-links>{ref_by_html}</ul></div>"
        + "<div class='record-pane full'><h3>Body</h3>"
        + _pretty_body(rec.get("body") or "")
        + "</div></div>"
        + "<a class=trace-link href='/train'>Back to Quality Plan</a>"
        + "</section></main></div></body></html>"
    )


def render_leaderboard() -> str:
    runs = _eval_runs_ranked()
    best = runs[0] if runs else None
    rows = []
    for rank, run in enumerate(runs, 1):
        status_cls, status_label = _leaderboard_status(run)
        delta_cls = "lb-delta-up" if str(run["delta"]).startswith("+") else "lb-delta-down"
        rows.append(
            f"<tr><td><span class=lb-rank>#{rank}</span></td>"
            f"<td><a class=lb-run-name href='/run/{_h(run['id'])}'>{_h(run['name'])}</a>"
            f"<span class=lb-run-sub>{_h(run['stage'])} · cycle_{_h(run['cycle'])}</span></td>"
            f"<td><span class=lb-score>{run['kaggle']:.3f}</span></td>"
            f"<td>{run['local']:.3f}</td>"
            f"<td>{run['hard']:.3f}</td>"
            f"<td><span class='{delta_cls}'>{_h(run['delta'])}</span></td>"
            f"<td>{_h(run['recipe'])}</td>"
            f"<td><span class='lb-status {status_cls}'>{_h(status_label)}</span></td>"
            f"<td><a class=trace-link href='/run/{_h(run['id'])}'>Open</a></td></tr>"
        )

    return (
        HEADER
        + "<title>A-Evolve-MAS-Train · Leaderboard</title></head><body>"
        + "<div class=focus-page>"
        + "<nav class=focus-nav><div class=focus-logo>A-EVOLVE<span>·</span>MAS<span>·</span>TRAIN</div>"
        + "<div class=focus-links><a href='/train'>TRAIN</a><a class=active href='/leaderboard'>LEADERBOARD</a>"
        + f"<a href='/cycle/{_h(_latest_cycle_id() or '0001')}'>TRACE</a><a href='{ABOUT_URL}'>ABOUT</a></div></nav>"
        + "<main class=lb-shell>"
        + "<section class=lb-header><div class=lb-title>"
        + "<h1>Kaggle Training Leaderboard</h1>"
        + "<p>All evaluated runs, ranked by public LB proxy. Rows are human-readable summaries; click a run to inspect recipe, data mix, training setup, eval breakdown, and Quality Gate status.</p>"
        + "</div>"
        + (f"<div class=run-note><b>Current best:</b> {_h(best['name'])}<br>"
           f"Public LB {best['kaggle']:.3f} · hard split {best['hard']:.3f}</div>" if best else "")
        + "</section>"
        + "<section class=lb-card>"
        + "<div class=lb-toolbar><div class=lb-tabs>"
        + "<span class='lb-tab active'>All evals</span><span class=lb-tab>Submission candidates</span>"
        + "<span class=lb-tab>Blocked</span><span class=lb-tab>Archived</span>"
        + "</div><a class=trace-link href='/train'>Back to Quality Plan</a></div>"
        + "<div class=table-scroll><table class=lb-table>"
        + "<tr><th>Rank</th><th>Run</th><th>Public LB</th><th>Local</th><th>Hard</th>"
        + "<th>Delta</th><th>Recipe</th><th>Status</th><th></th></tr>"
        + "".join(rows)
        + "</table></div></section>"
        + "</main></div></body></html>"
    )


def render_run_detail(run_id: str) -> str:
    run = _eval_run(run_id)
    if not run:
        return (
            HEADER
            + "<title>Run not found</title></head><body><div class=focus-page>"
            + "<main class=lb-shell><section class=lb-header><div class=lb-title>"
            + f"<h1>Unknown run: {_h(run_id)}</h1><p>No evaluated recipe has this id.</p>"
            + "</div><a class=trace-link href='/leaderboard'>Back to leaderboard</a></section>"
            + "</main></div></body></html>"
        )

    ranked = _eval_runs_ranked()
    rank = ranked.index(run) + 1
    status_cls, status_label = _leaderboard_status(run)
    breakdown = "".join(
        f"<div class=breakdown-cell><strong>{score:.3f}</strong><span>{_h(name)}</span></div>"
        for name, score in run["breakdown"].items()
    )
    findings = "".join(f"<li>{_h(item)}</li>" for item in run["findings"])
    artifacts = "".join(f"<li>{_h(item)}</li>" for item in run["artifacts"])
    cycle_href = f"/cycle/{_h(run['cycle'])}"

    return (
        HEADER
        + f"<title>{_h(run['name'])}</title></head><body>"
        + "<div class=focus-page>"
        + "<nav class=focus-nav><div class=focus-logo>A-EVOLVE<span>·</span>MAS<span>·</span>TRAIN</div>"
        + "<div class=focus-links><a href='/train'>TRAIN</a><a class=active href='/leaderboard'>LEADERBOARD</a>"
        + f"<a href='{cycle_href}'>TRACE</a><a href='{ABOUT_URL}'>ABOUT</a></div></nav>"
        + "<main class=lb-shell>"
        + "<section class=lb-header><div class=lb-title>"
        + f"<h1>{_h(run['name'])}</h1>"
        + f"<p>{_h(run['score_note'])} This page translates the evaluated run into recipe-level decisions rather than raw logs.</p>"
        + "</div>"
        + f"<a class=trace-link href='/leaderboard'>Back to leaderboard</a>"
        + "</section>"
        + "<section class=lb-detail-grid>"
        + "<article class=recipe-panel><div class=recipe-head>"
        + f"<h2>Recipe Card · rank #{rank}</h2>"
        + f"<p>{_h(run['stage'])} · cycle_{_h(run['cycle'])} · "
        + f"<span class='lb-status {status_cls}'>{_h(status_label)}</span></p>"
        + "</div><div class=recipe-body>"
        + "<div class='recipe-section orange'><div class=recipe-title>Leaderboard metrics</div>"
        + f"<div class=breakdown-grid><div class=breakdown-cell><strong>{run['kaggle']:.3f}</strong><span>public LB</span></div>"
        + f"<div class=breakdown-cell><strong>{run['local']:.3f}</strong><span>local holdout</span></div>"
        + f"<div class=breakdown-cell><strong>{run['hard']:.3f}</strong><span>hard split</span></div>"
        + f"<div class=breakdown-cell><strong>{_h(run['delta'])}</strong><span>vs baseline</span></div></div></div>"
        + "<div class='recipe-section'><div class=recipe-title>Recipe</div><div class=recipe-kv>"
        + f"<span>Summary</span><span>{_h(run['recipe'])}</span>"
        + f"<span>Base model</span><span>{_h(run['base_model'])}</span>"
        + f"<span>Data mix</span><span>{_h(run['data_mix'])}</span>"
        + f"<span>Training</span><span>{_h(run['training'])}</span>"
        + "</div></div>"
        + "<div class='recipe-section green'><div class=recipe-title>Eval breakdown</div>"
        + f"<div class=breakdown-grid>{breakdown}</div></div>"
        + "<div class='recipe-section'><div class=recipe-title>MAS findings</div>"
        + f"<ul class=recipe-list>{findings}</ul></div>"
        + "</div></article>"
        + "<aside class=recipe-panel><div class=recipe-head><h2>Run Context</h2>"
        + "<p>Human-readable details for deciding whether to submit, ablate, or archive.</p></div>"
        + "<div class=recipe-body>"
        + "<div class='recipe-section orange'><div class=recipe-title>Quality Gate</div>"
        + f"<div class=recipe-section-body>{_h(run['quality_gate'])}</div></div>"
        + "<div class=recipe-section><div class=recipe-title>Decision</div>"
        + f"<div class=recipe-section-body>{_h(run['decision'])}</div></div>"
        + "<div class=recipe-section><div class=recipe-title>Artifacts</div>"
        + f"<ul class=recipe-list>{artifacts}</ul></div>"
        + "<div class=recipe-section><div class=recipe-title>Trace views</div>"
        + f"<ul class=recipe-list><li><a class=trace-link href='{cycle_href}'>agent roster</a></li>"
        + f"<li><a class=trace-link href='{cycle_href}/sequence'>agent sequence</a></li>"
        + f"<li><a class=trace-link href='{cycle_href}/calls'>tool-call timeline</a></li></ul></div>"
        + "</div></aside></section>"
        + "</main></div></body></html>"
    )


def render_cycle(cycle: str) -> str:
    cycle_dir = trace_dir_for() / f"cycle_{cycle}"
    if not cycle_dir.is_dir():
        body = (
            _trace_hero(
                cycle,
                "Trace cycle not found",
                "This cycle id is not present in the local MAS trace directory.",
                active="overview",
            )
            + "<section class=trace-panel><div class=trace-panel-body>"
            + f"Unknown cycle: {_h(cycle)}</div></section>"
        )
        return _trace_page("Unknown cycle", body, cycle=cycle, active="overview")

    records = _cycle_agent_records(cycle_dir)
    agents = len(records)
    finished = sum(1 for r in records if r["summary"]["finished"])
    progress = round(100 * finished / max(agents, 1))
    turns = sum(r["summary"]["turns"] for r in records)
    call_count = sum(len(r["calls"]) for r in records)
    in_tokens = sum(r["summary"]["input_tokens"] or 0 for r in records)
    out_tokens = sum(r["summary"]["output_tokens"] or 0 for r in records)
    roles = Counter(r["summary"]["role"] for r in records)
    mtime = max((r["summary"]["mtime"] for r in records), default=0)
    run = _run_for_cycle(cycle)
    run_status = _leaderboard_status(run)[1] if run else "Trace only"

    agent_rows = []
    for r in records:
        s = r["summary"]
        tokens = (
            "tokens n/a" if s["input_tokens"] is None
            else f"{s['input_tokens']:,} in · {s['output_tokens']:,} out"
        )
        agent_rows.append(
            f"<div class=trace-agent-row>"
            f"<div class=trace-agent-id>ag_{r['aid']:02d}</div>"
            f"<div class=trace-agent-main>"
            f"<strong>{_h(_role_display(s['role']))}</strong>"
            f"<span>{s['turns']} turns · {tokens}</span></div>"
            f"<div class=trace-muted>{len(r['calls'])} tool calls</div>"
            f"{_trace_agent_status(s['finished'])}"
            f"<a class=trace-link href='/cycle/{cycle}/{r['aid']}'>Inspect</a>"
            f"</div>"
        )

    if run:
        summary_items = [
            (
                "#1",
                "Linked leaderboard run",
                f"{run['name']} is attached to this cycle. Public LB {run['kaggle']:.3f}, "
                f"local {run['local']:.3f}, hard split {run['hard']:.3f}.",
            ),
            (
                "#2",
                "Recipe and gate context",
                f"{run['recipe']}. Quality gate: {run['quality_gate']}.",
            ),
            (
                "#3",
                "What the trace is for",
                "Use this page to see who acted and where to drill in. Full prompts, "
                "tool inputs, and JSONL stay one click deeper.",
            ),
        ]
        subtitle = (
            f"{run['stage']} for {run['name']}. A compact view of the MAS work "
            "behind the training decision."
        )
    else:
        summary_items = [
            (
                "#1",
                "Cycle-level MAS execution",
                "This cycle is not tied to an evaluated Kaggle run, so the page shows "
                "execution health and drill-down points only.",
            ),
            (
                "#2",
                "Human-readable by default",
                "Agent details and raw JSONL are available, but the top-level trace "
                "stays focused on progress and ownership.",
            ),
            (
                "#3",
                "Where to inspect next",
                "Open Agent handoffs for role-to-role flow, or Tool activity for the "
                "calls that changed files, data, training, or evaluation state.",
            ),
        ]
        subtitle = "A compact, human-readable MAS trace overview for this training cycle."

    summary_html = "".join(
        "<div class=trace-summary-item>"
        f"<span class=trace-summary-icon>{_h(icon)}</span>"
        f"<div><strong>{_h(title)}</strong><p>{_h(text)}</p></div></div>"
        for icon, title, text in summary_items
    )

    linked_run = (
        f"<div class=trace-rail-card><div class=trace-rail-title>Leaderboard run</div>"
        f"<strong style='color:#f8fafc;'>{_h(run['name'])}</strong>"
        f"<div class=trace-muted style='margin-top:6px;'>{_h(run['score_note'])}</div>"
        f"<div style='margin-top:10px;'><a class=trace-link href='/run/{_h(run['id'])}'>View recipe and evals</a></div>"
        f"</div>"
        if run else
        "<div class=trace-rail-card><div class=trace-rail-title>Leaderboard run</div>"
        "<strong style='color:#f8fafc;'>No evaluated run attached</strong>"
        "<div class=trace-muted style='margin-top:6px;'>This cycle is available for trace inspection only.</div></div>"
    )

    body = (
        _trace_hero(cycle, "Trace Overview", subtitle, active="overview")
        + "<section class=trace-layout>"
        + "<main class=trace-stack>"
        + "<div class=trace-stat-strip>"
        + f"<div class=trace-stat><span>Agents</span><strong>{agents}</strong></div>"
        + f"<div class=trace-stat><span>Finished</span><strong>{finished}/{agents}</strong></div>"
        + f"<div class=trace-stat><span>Turns</span><strong>{_fmt_int(turns)}</strong></div>"
        + f"<div class=trace-stat><span>Tool calls</span><strong>{_fmt_int(call_count)}</strong></div>"
        + "</div>"
        + "<section class=trace-panel>"
        + "<div class=trace-panel-header><div class=trace-panel-title>Readable Cycle Summary</div>"
        + f"<span class='status-pill running'>{_h(run_status)}</span></div>"
        + f"<div class=trace-panel-body><div class=trace-summary-list>{summary_html}</div></div>"
        + "</section>"
        + "<section class=trace-panel>"
        + "<div class=trace-panel-header><div class=trace-panel-title>Agents</div>"
        + "<div class=trace-muted>compact roster · details are one click deeper</div></div>"
        + f"<div class=trace-panel-body><div class=trace-agent-list>{''.join(agent_rows)}</div></div>"
        + "</section>"
        + "</main>"
        + "<aside class=trace-stack>"
        + "<div class=trace-rail-card><div class=trace-rail-title>Cycle progress</div>"
        + f"<div class=trace-progress-line><span style='width:{progress}%'></span></div>"
        + f"<div style='display:flex;justify-content:space-between;margin-top:9px;'><span>{finished} done</span><strong style='color:#ffb84d;'>{progress}%</strong></div>"
        + f"<div class=trace-muted style='margin-top:8px;'>Updated {_h(_format_ts(mtime))}</div></div>"
        + linked_run
        + "<div class=trace-rail-card><div class=trace-rail-title>Roles</div>"
        + f"<div class=trace-chip-row>{_role_count_chips(roles)}</div></div>"
        + "<div class=trace-rail-card><div class=trace-rail-title>Token footprint</div>"
        + f"<div class=trace-chip-row><span class='trace-chip orange'>{_fmt_int(in_tokens)} input</span>"
        + f"<span class='trace-chip orange'>{_fmt_int(out_tokens)} output</span></div></div>"
        + "<div class=trace-rail-card><div class=trace-rail-title>Quick inspect</div>"
        + f"<div class=trace-rail-list><a class=trace-link href='/cycle/{cycle}/sequence'>Agent handoffs</a>"
        + f"<a class=trace-link href='/cycle/{cycle}/calls'>Tool activity</a>"
        + "<a class=trace-link href='/train'>Back to Train cockpit</a></div></div>"
        + "</aside></section>"
    )
    return _trace_page(f"cycle_{cycle} · trace overview", body, cycle=cycle, active="overview")


def render_sequence(cycle: str) -> str:
    """Chronological cross-agent call sequence for one cycle.

    In this workspace only the orchestrator (agent 0) can spawn workers
    (no nested spawns), so the cross-agent call graph is flat: a time-
    ordered list of `orchestrator → worker_N` handoffs. Each row pairs
    the orchestrator's spawn call with the worker's final return
    summary, so you can read the conversation A → B → C top-to-bottom.
    """
    cycle_dir = trace_dir_for() / f"cycle_{cycle}"
    if not cycle_dir.is_dir():
        body = (
            _trace_hero(
                cycle,
                "Agent handoffs not found",
                "This cycle id is not present in the local MAS trace directory.",
                active="sequence",
            )
            + "<section class=trace-panel><div class=trace-panel-body>"
            + f"Unknown cycle: {_h(cycle)}</div></section>"
        )
        return _trace_page("Unknown cycle", body, cycle=cycle, active="sequence")

    records = _cycle_agent_records(cycle_dir)
    by_aid = {r["aid"]: r for r in records}
    orch = by_aid.get(0)
    if not orch:
        body = (
            _trace_hero(
                cycle,
                "Agent handoffs",
                "The orchestrator trace is missing, so handoffs cannot be reconstructed.",
                active="sequence",
            )
            + "<section class=trace-panel><div class=trace-panel-body>"
            + "Orchestrator trace missing.</div></section>"
        )
        return _trace_page(f"cycle_{cycle} sequence", body, cycle=cycle, active="sequence")

    # Walk orchestrator calls: match spawn_and_run_subagent with workers in order.
    steps: list[dict] = []
    worker_counter = 0
    for call in orch["calls"]:
        if call["name"] == "spawn_and_run_subagent":
            worker_counter += 1
            tin = call["input"] or {}
            target_aid = worker_counter
            w = by_aid.get(target_aid, {})
            steps.append({
                "kind": "spawn",
                "turn": call["turn"],
                "target_aid": target_aid,
                "role": tin.get("role", "?"),
                "suggested_skills": tin.get("suggested_skills", []) or [],
                "budget_tokens": tin.get("budget_tokens"),
                "task": tin.get("task", "") or "",
                "result_text": call["result_excerpt"],
                "worker_summary": w.get("summary", {}) if w else {},
                "worker_final_text": _final_assistant_text(w.get("rows", [])) if w else "",
                "worker_exists": bool(w),
            })
        elif call["name"] == "call_existing_agent":
            tin = call["input"] or {}
            try:
                target_aid = int(str(tin.get("agent_id", "0")))
            except (TypeError, ValueError):
                target_aid = 0
            steps.append({
                "kind": "resume",
                "turn": call["turn"],
                "target_aid": target_aid,
                "role": (by_aid.get(target_aid, {})
                         .get("summary", {}).get("role", "?")),
                "suggested_skills": [],
                "budget_tokens": None,
                "task": tin.get("task", "") or "",
                "result_text": call["result_excerpt"],
                "worker_summary": by_aid.get(target_aid, {}).get("summary", {}),
                "worker_final_text": "",
                "worker_exists": target_aid in by_aid,
            })

    orch_summary = orch["summary"]
    step_cards = []
    if not steps:
        step_cards.append("<div class=trace-muted>No cross-agent calls in this cycle.</div>")
    else:
        for i, s in enumerate(steps, 1):
            role = s["role"] or s["worker_summary"].get("role", "?")
            target = s["target_aid"]
            ws = s["worker_summary"] or {}
            tokens = (
                "tokens n/a" if not ws.get("input_tokens")
                else f"{ws.get('input_tokens'):,} in · {ws.get('output_tokens'):,} out"
            )
            detail = _trace_detail(s["worker_final_text"] or s["result_text"], "Returned summary")
            step_cards.append(
                f"<div class=trace-step>"
                f"<div class=trace-step-num>{i:02d}</div>"
                f"<div><div class=trace-step-title>{_h(_role_display(role))}</div>"
                f"<div class=trace-step-meta>orchestrator turn {_h(s['turn'])} · "
                f"agent_{target} · {ws.get('turns', 0)} turns · {tokens}</div>"
                f"<p>{_h(_preview(' '.join(str(s['task']).split()), 240))}</p>"
                f"{detail}</div>"
                f"<a class=trace-link href='/cycle/{cycle}/{target}'>Inspect</a>"
                f"</div>"
            )

    final_orch = _final_assistant_text(orch["rows"])
    if final_orch:
        step_cards.append(
            "<div class=trace-step><div class=trace-step-num>✓</div>"
            "<div><div class=trace-step-title>Orchestrator closes cycle</div>"
            "<div class=trace-step-meta>final readable handoff summary</div>"
            f"<p>{_h(_preview(' '.join(final_orch.split()), 260))}</p>"
            f"{_trace_detail(final_orch, 'Final orchestrator message')}</div>"
            f"<a class=trace-link href='/cycle/{cycle}/0'>Inspect</a></div>"
        )

    body = (
        _trace_hero(
            cycle,
            "Agent Handoffs",
            "A compact reading path for role-to-role MAS work. Task briefings are previews; detailed prompts stay behind each agent link.",
            active="sequence",
        )
        + "<section class=trace-layout>"
        + "<main class=trace-stack>"
        + "<div class=trace-stat-strip>"
        + f"<div class=trace-stat><span>Handoffs</span><strong>{len(steps)}</strong></div>"
        + f"<div class=trace-stat><span>Workers</span><strong>{max(len(by_aid) - 1, 0)}</strong></div>"
        + f"<div class=trace-stat><span>Orchestrator turns</span><strong>{orch_summary['turns']}</strong></div>"
        + f"<div class=trace-stat><span>Orchestrator calls</span><strong>{len(orch['calls'])}</strong></div>"
        + "</div>"
        + "<section class=trace-panel><div class=trace-panel-header>"
        + "<div class=trace-panel-title>Readable Handoff Timeline</div>"
        + "<div class=trace-muted>spawn/resume events only</div></div>"
        + f"<div class=trace-panel-body><div class=trace-step-list>{''.join(step_cards)}</div></div></section>"
        + "</main>"
        + "<aside class=trace-stack>"
        + "<div class=trace-rail-card><div class=trace-rail-title>How to read it</div>"
        + "Each row is one orchestrator handoff. Open an agent only when you need the full prompt, tool transcript, or raw JSONL.</div>"
        + "<div class=trace-rail-card><div class=trace-rail-title>Shortcuts</div>"
        + f"<div class=trace-rail-list><a class=trace-link href='/cycle/{cycle}'>Overview</a>"
        + f"<a class=trace-link href='/cycle/{cycle}/calls'>Tool activity</a>"
        + "<a class=trace-link href='/train'>Back to Train cockpit</a></div></div>"
        + "</aside></section>"
    )
    return _trace_page(
        f"cycle_{cycle} · agent handoffs",
        body,
        cycle=cycle,
        active="sequence",
    )


def render_calls(cycle: str) -> str:
    cycle_dir = trace_dir_for() / f"cycle_{cycle}"
    if not cycle_dir.is_dir():
        body = (
            _trace_hero(
                cycle,
                "Tool activity not found",
                "This cycle id is not present in the local MAS trace directory.",
                active="calls",
            )
            + "<section class=trace-panel><div class=trace-panel-body>"
            + f"Unknown cycle: {_h(cycle)}</div></section>"
        )
        return _trace_page("Unknown cycle", body, cycle=cycle, active="calls")

    agents = _cycle_agent_records(cycle_dir)

    # Build spawn edges: orchestrator (aid 0) spawns workers via
    # spawn_and_run_subagent. We tag each spawn with its index so we can
    # map worker aid 1..N to the orchestrator's Nth spawn call.
    spawns: list[dict] = []
    orch = next((a for a in agents if a["aid"] == 0), None)
    if orch:
        i = 0
        for call in orch["calls"]:
            if call["name"] in ("spawn_and_run_subagent", "call_existing_agent"):
                i += 1
                tin = call["input"] or {}
                target_aid = i  # workers are aid 1,2,3,... in spawn order
                if call["name"] == "call_existing_agent":
                    try:
                        target_aid = int(str(tin.get("agent_id", i)))
                    except (TypeError, ValueError):
                        pass
                spawns.append({
                    "from": 0,
                    "to": target_aid,
                    "turn": call["turn"],
                    "role": tin.get("role", "?"),
                    "task": tin.get("task", "") or "",
                    "kind": call["name"],
                })

    tool_counter: Counter = Counter()
    call_rows = []
    for a in agents:
        aid = a["aid"]
        s = a["summary"]
        calls = a["calls"]
        top = Counter(c["name"] for c in calls)
        tool_counter.update(top)
        chips = "".join(
            f"<span class=trace-chip>{_h(name)} · {count}</span>"
            for name, count in top.most_common(3)
        ) or "<span class=trace-chip>no calls</span>"
        call_rows.append(
            f"<div class=trace-call-row id='agent-{aid}'>"
            f"<div class=trace-agent-id>ag_{aid:02d}</div>"
            f"<div><strong style='color:#f8fafc;'>{_h(_role_display(s['role']))}</strong>"
            f"<div class=trace-tool-cloud style='margin-top:7px;'>{chips}</div></div>"
            f"<div class=trace-muted>{len(calls)} calls</div>"
            f"<a class=trace-link href='/cycle/{cycle}/{aid}'>Inspect</a>"
            f"</div>"
        )

    top_tools = "".join(
        f"<span class='trace-chip orange'>{_h(name)} · {count}</span>"
        for name, count in tool_counter.most_common(8)
    ) or "<span class=trace-chip>no tool calls</span>"
    spawn_rows = "".join(
        "<div class=trace-summary-item>"
        f"<span class=trace-summary-icon>{i:02d}</span>"
        f"<div><strong>agent_{s['from']} → agent_{s['to']} · {_h(s['kind'])}</strong>"
        f"<p>{_h(_role_display(s['role']))} · turn {_h(s['turn'])} · "
        f"{_h(_preview(' '.join(str(s['task']).split()), 180))}</p></div></div>"
        for i, s in enumerate(spawns, 1)
    ) or "<div class=trace-muted>No orchestrator handoffs detected.</div>"
    total_calls = sum(len(a["calls"]) for a in agents)
    failed = sum(
        1 for a in agents for c in a["calls"]
        if c["status"] not in {"ok", "success"}
    )

    body = (
        _trace_hero(
            cycle,
            "Tool Activity",
            "A high-signal view of what the agents actually invoked. Full inputs and outputs are kept inside each agent detail page.",
            active="calls",
        )
        + "<section class=trace-layout>"
        + "<main class=trace-stack>"
        + "<div class=trace-stat-strip>"
        + f"<div class=trace-stat><span>Total calls</span><strong>{_fmt_int(total_calls)}</strong></div>"
        + f"<div class=trace-stat><span>Agents with tools</span><strong>{sum(1 for a in agents if a['calls'])}</strong></div>"
        + f"<div class=trace-stat><span>Handoffs</span><strong>{len(spawns)}</strong></div>"
        + f"<div class=trace-stat><span>Non-ok results</span><strong>{failed}</strong></div>"
        + "</div>"
        + "<section class=trace-panel><div class=trace-panel-header>"
        + "<div class=trace-panel-title>Most Used Tools</div>"
        + "<div class=trace-muted>counts across this cycle</div></div>"
        + f"<div class=trace-panel-body><div class=trace-tool-cloud>{top_tools}</div></div></section>"
        + "<section class=trace-panel><div class=trace-panel-header>"
        + "<div class=trace-panel-title>Agent Tool Activity</div>"
        + "<div class=trace-muted>top tools per agent</div></div>"
        + f"<div class=trace-panel-body><div class=trace-agent-list>{''.join(call_rows)}</div></div></section>"
        + "</main>"
        + "<aside class=trace-stack>"
        + "<section class=trace-panel><div class=trace-panel-header>"
        + "<div class=trace-panel-title>Orchestrator Handoffs</div></div>"
        + f"<div class=trace-panel-body><div class=trace-summary-list>{spawn_rows}</div></div></section>"
        + "<div class=trace-rail-card><div class=trace-rail-title>Shortcuts</div>"
        + f"<div class=trace-rail-list><a class=trace-link href='/cycle/{cycle}'>Overview</a>"
        + f"<a class=trace-link href='/cycle/{cycle}/sequence'>Agent handoffs</a>"
        + "<a class=trace-link href='/train'>Back to Train cockpit</a></div></div>"
        + "</aside></section>"
    )
    return _trace_page(f"cycle_{cycle} · tool activity", body, cycle=cycle, active="calls")


def render_agent(cycle: str, aid: str) -> str:
    cycle_dir = trace_dir_for() / f"cycle_{cycle}"
    path = cycle_dir / f"agent_{aid}.jsonl"
    if not path.is_file():
        body = (
            _trace_hero(
                cycle,
                "Agent trace not found",
                "The selected agent JSONL file does not exist for this cycle.",
                active="overview",
            )
            + "<section class=trace-panel><div class=trace-panel-body>"
            + f"Not found: {_h(path)}</div></section>"
        )
        return _trace_page("Agent trace not found", body, cycle=cycle, active="overview")
    rows = _load_jsonl(path)
    summary = _agent_summary_from_rows(path, rows)
    calls = _tool_calls(rows)
    event_rows = []
    for idx, ev in enumerate(rows, 1):
        kind = ev.get("event", "?")
        meta = []
        if "turn" in ev:
            meta.append(f"turn {_h(ev['turn'])}")
        if "stop_reason" in ev:
            meta.append(f"stop={_h(ev['stop_reason'])}")
        preview = ""
        detail = ""
        if kind == "start":
            tn = ev.get("tool_names", []) or []
            preview = f"model={ev.get('model_id', '?')} · {len(tn)} tools available"
            meta.append(f"role={_role_display(summary['role'])}")
            detail = (
                f"<div class=trace-muted>agent_id={_h(ev.get('agent_id'))} · "
                f"model={_h(ev.get('model_id'))}</div>"
            )
            tn = ev.get("tool_names", []) or []
            if tn:
                detail += (
                    "<div class=trace-tool-cloud style='margin-top:10px;'>"
                    + "".join(f"<span class=trace-chip>{_h(t)}</span>" for t in tn)
                    + "</div>"
                )
            sys_ex = ev.get("system_excerpt", "")
            if sys_ex:
                detail += _trace_detail(sys_ex, "System prompt")
        elif kind == "message":
            content = ev.get("content")
            preview = _preview(" ".join(_content_plaintext(content).split()), 220)
            detail = _render_content(content)
        elif kind == "turn":
            u = ev.get("usage") or {}
            if u:
                meta.append(
                    f"in={u.get('inputTokens','?')} out={u.get('outputTokens','?')}"
                )
            assistant = ev.get("assistant")
            if assistant:
                content = assistant.get("content")
                preview = _preview(" ".join(_content_plaintext(content).split()), 220)
                detail = _render_content(content)
        elif kind == "done":
            preview = (
                f"total_turns={ev.get('total_turns')} · "
                f"in={ev.get('input_tokens')} · out={ev.get('output_tokens')}"
            )
        else:
            preview = _preview(json.dumps(ev, default=str), 220)
            detail = _collapsible(json.dumps(ev, indent=2, default=str), label="event JSON")

        event_rows.append(
            "<details class=trace-event>"
            f"<summary><span class=trace-event-kind>{_h(kind)}</span>"
            f"<span class=trace-event-preview>{_h(preview or '(no preview)')}</span>"
            f"<span class=trace-muted>{' · '.join(meta)}</span></summary>"
            f"{detail}</details>"
        )

    tokens = (
        "tokens n/a" if summary["input_tokens"] is None
        else f"{summary['input_tokens']:,} in · {summary['output_tokens']:,} out"
    )
    body = (
        _trace_hero(
            cycle,
            f"Agent ag_{int(aid):02d}",
            f"{_role_display(summary['role'])} · {len(rows)} events · {tokens}. Open only the rows you need.",
            active="overview",
        )
        + "<section class=trace-layout>"
        + "<main class=trace-stack>"
        + "<div class=trace-stat-strip>"
        + f"<div class=trace-stat><span>Events</span><strong>{len(rows)}</strong></div>"
        + f"<div class=trace-stat><span>Turns</span><strong>{summary['turns']}</strong></div>"
        + f"<div class=trace-stat><span>Tool calls</span><strong>{len(calls)}</strong></div>"
        + f"<div class=trace-stat><span>Size</span><strong>{_h(_fmt_bytes(summary['size']))}</strong></div>"
        + "</div>"
        + "<section class=trace-panel><div class=trace-panel-header>"
        + "<div class=trace-panel-title>Readable Event Stream</div>"
        + "<div class=trace-muted>collapsed by default</div></div>"
        + f"<div class=trace-panel-body><div class=trace-event-list>{''.join(event_rows)}</div></div>"
        + "</section></main>"
        + "<aside class=trace-stack>"
        + "<div class=trace-rail-card><div class=trace-rail-title>Agent context</div>"
        + f"<strong style='color:#f8fafc;'>{_h(_role_display(summary['role']))}</strong>"
        + f"<div class=trace-muted style='margin-top:7px;'>{tokens}</div>"
        + f"<div style='margin-top:10px;'>{_trace_agent_status(summary['finished'])}</div></div>"
        + "<div class=trace-rail-card><div class=trace-rail-title>Shortcuts</div>"
        + f"<div class=trace-rail-list><a class=trace-link href='/cycle/{cycle}'>Overview</a>"
        + f"<a class=trace-link href='/cycle/{cycle}/sequence'>Agent handoffs</a>"
        + f"<a class=trace-link href='/cycle/{cycle}/calls'>Tool activity</a>"
        + f"<a class=trace-link href='/raw/{cycle}/{aid}'>Raw JSONL</a>"
        + "<a class=trace-link href='/train'>Back to Train cockpit</a></div></div>"
        + "</aside></section>"
    )
    return _trace_page(f"cycle_{cycle}/agent_{aid}", body, cycle=cycle, active="overview")


def _render_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _collapsible(content, label="text")
    if not isinstance(content, list):
        return _collapsible(
            json.dumps(content, indent=2, default=str), label="content",
        )
    chunks = []
    for block in content:
        if not isinstance(block, dict):
            chunks.append(_collapsible(
                json.dumps(block, indent=2, default=str), label="block",
            ))
            continue
        if "text" in block:
            chunks.append(_collapsible(block["text"], label="text"))
        elif "toolUse" in block:
            tu = block["toolUse"]
            name = tu.get("name", "?")
            tin = tu.get("input", {})
            chunks.append(
                f"<div class=tool-use><span class=tool-name>→ {_h(name)}</span>"
                + _collapsible(
                    json.dumps(tin, indent=2, default=str),
                    label=f"tool_use input ({_h(name)})",
                )
                + "</div>"
            )
        elif "toolResult" in block:
            tr = block["toolResult"]
            chunks.append(
                f"<div class=tool-use><span class=tool-name>"
                f"← tool_result (use_id={_h(tr.get('toolUseId'))})</span>"
                + _collapsible(
                    json.dumps(tr.get("content"), indent=2, default=str),
                    label="tool_result content",
                )
                + "</div>"
            )
        else:
            chunks.append(_collapsible(
                json.dumps(block, indent=2, default=str), label="block",
            ))
    return "".join(chunks)


# ── HTTP handler ─────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] "
                         f"{self.address_string()} {fmt % args}\n")

    def _send_html(self, body: str, status: int = 200) -> None:
        # Rewrite every ``href='/...'`` / ``action='/...'`` so links stay
        # inside the active run's URL scope. No-op when the request
        # isn't scoped to a specific run (the runs index, 404 pages).
        body = _apply_run_scope(body)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _send_raw(self, path: Path) -> None:
        try:
            data = path.read_bytes()
        except OSError as e:
            self._send_html(f"<pre>error: {_h(e)}</pre>", status=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _dispatch_get(self, path: str, query: dict | None = None) -> bool:
        """Route a path (already stripped of any ``/runs/<name>`` prefix and
        with the active-run ContextVar set). ``query`` is the parsed query
        string (flat dict, first value wins). Returns True if handled."""
        query = query or {}
        if path == "/":
            # Per-run landing page: the Model Forge hero + live agent
            # activity. Cockpit details live under /train.
            self._send_html(render_entry()); return True
        if path == "/train":
            self._send_html(render_index(
                selected_slot_id=query.get("slot") or None
            )); return True
        if path == "/live-status.json":
            self._send_json(_live_snapshot()); return True
        if path == "/leaderboard":
            self._send_html(render_leaderboard()); return True
        m = re.match(r"^/run/([A-Za-z0-9_-]+)$", path)
        if m:
            self._send_html(render_run_detail(m.group(1))); return True
        m = re.match(r"^/record/([A-Za-z0-9_.:-]+)$", path)
        if m:
            self._send_html(render_record(m.group(1))); return True
        m = re.match(r"^/cycle/(\d+)$", path)
        if m:
            self._send_html(render_cycle(m.group(1))); return True
        m = re.match(r"^/cycle/(\d+)/calls$", path)
        if m:
            self._send_html(render_calls(m.group(1))); return True
        m = re.match(r"^/cycle/(\d+)/sequence$", path)
        if m:
            self._send_html(render_sequence(m.group(1))); return True
        m = re.match(r"^/cycle/(\d+)/(\d+)$", path)
        if m:
            self._send_html(render_agent(m.group(1), m.group(2))); return True
        m = re.match(r"^/raw/(\d+)/(\d+)$", path)
        if m:
            p = trace_dir_for() / f"cycle_{m.group(1)}" / f"agent_{m.group(2)}.jsonl"
            self._send_raw(p); return True
        return False

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = {k: (v[0] if v else "") for k, v in parse_qs(parsed.query).items()}
        try:
            # Root ``/`` lands on the *default* run's MODEL FORGE entry —
            # whichever is live, or the most recently-active one otherwise.
            # Past-runs browser lives at ``/runs``. Legacy single-run
            # (--trace-dir) mode jumps straight into the pinned run.
            if path == "/":
                if LEGACY_PINNED and DEFAULT_RUN:
                    self._redirect(f"/runs/{DEFAULT_RUN}/train"); return
                target = _pick_default_run()
                if target:
                    self._redirect(f"/runs/{target}"); return
                # No runs on disk yet — fall back to the list (empty-state
                # copy is better than a 404).
                self._send_html(render_runs_index()); return
            if path in ("/runs", "/runs/"):
                self._send_html(render_runs_index()); return

            # /runs/<name>/<rest> — scope the request to that marathon.
            m = re.match(r"^/runs/([A-Za-z0-9_.-]+)(/.*)?$", path)
            if m:
                run_name = m.group(1)
                if run_name not in list_runs():
                    self._send_html(
                        f"<p>Unknown run: <code>{_h(run_name)}</code>. "
                        f"<a href='/runs'>See all runs.</a></p>",
                        status=404,
                    )
                    return
                sub = m.group(2) or "/"
                token = _ACTIVE_RUN.set(run_name)
                try:
                    if self._dispatch_get(sub, qs):
                        return
                finally:
                    _ACTIVE_RUN.reset(token)
                self._send_html("<p>404</p>", status=404)
                return

            # Legacy path without a /runs/ prefix — scope to DEFAULT_RUN.
            token = _ACTIVE_RUN.set(DEFAULT_RUN) if DEFAULT_RUN else None
            try:
                if self._dispatch_get(path, qs):
                    return
            finally:
                if token is not None:
                    _ACTIVE_RUN.reset(token)
            self._send_html("<p>404</p>", status=404)
        except Exception as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            self._send_html(f"<pre>{_h(tb)}</pre>", status=500)

    # ── POST: signoff + human directives ──────────────────────────

    def _read_form(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > _MAX_POST_BYTES:
            return {}
        body = self.rfile.read(length)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return {}
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        if ctype == "application/json":
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                return {}
        parsed = parse_qs(text, keep_blank_values=True)
        return {k: (v[0] if v else "") for k, v in parsed.items()}

    def _redirect(self, target: str) -> None:
        self.send_response(303)
        self.send_header("Location", target)
        self.end_headers()

    def _dispatch_post(self, path: str, run_prefix: str) -> bool:
        """POST routing scoped to the active run. ``run_prefix`` is the
        URL piece to redirect back to after a successful POST so the user
        stays on the run they came from."""
        m = re.match(r"^/checkpoint/([A-Za-z0-9_]+)/sign$", path)
        if m:
            form = self._read_form()
            resp = _sign_checkpoint(
                slot_id=m.group(1),
                actor=(form.get("actor") or "human:owner").strip(),
                note=(form.get("note") or "").strip(),
            )
            if resp.get("ok"):
                self._redirect(f"{run_prefix}/train" if run_prefix else "/train"); return True
            self._send_json(resp, status=409); return True
        if path == "/directive":
            form = self._read_form()
            resp = _append_directive(
                text=(form.get("text") or "").strip(),
                urgency=(form.get("urgency") or "next_cycle").strip(),
                actor=(form.get("actor") or "human:owner").strip(),
            )
            if resp.get("ok"):
                self._redirect(f"{run_prefix}/train" if run_prefix else "/train"); return True
            self._send_json(resp, status=400); return True
        return False

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            m = re.match(r"^/runs/([A-Za-z0-9_.-]+)(/.*)?$", path)
            if m:
                run_name = m.group(1)
                if run_name not in list_runs():
                    self._send_html(
                        f"<p>Unknown run: <code>{_h(run_name)}</code>.</p>",
                        status=404,
                    )
                    return
                sub = m.group(2) or "/"
                token = _ACTIVE_RUN.set(run_name)
                try:
                    if self._dispatch_post(sub, f"/runs/{run_name}"):
                        return
                finally:
                    _ACTIVE_RUN.reset(token)
                self._send_html("<p>404</p>", status=404)
                return

            token = _ACTIVE_RUN.set(DEFAULT_RUN) if DEFAULT_RUN else None
            try:
                if self._dispatch_post(path, ""):
                    return
            finally:
                if token is not None:
                    _ACTIVE_RUN.reset(token)
            self._send_html("<p>404</p>", status=404)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            self._send_html(f"<pre>{_h(tb)}</pre>", status=500)


def _sign_checkpoint(*, slot_id: str, actor: str, note: str) -> dict:
    run_mode = _mode_for_active_run()
    if run_mode != CHECKPOINT_MODE_MANUAL:
        return {"ok": False, "error": f"sign refused: mode={run_mode}"}
    slots = _slots_for_active_run()
    slot_decl = next((s for s in slots if s["id"] == slot_id), None)
    if slot_decl is None:
        return {"ok": False, "error": f"unknown slot_id {slot_id!r}"}
    if not actor.startswith("human:"):
        actor = f"human:{actor or 'owner'}"

    records = _load_records()
    folded = {s.id: s for s in fold_checkpoints(records, run_mode, slots=slots)}
    slot = folded.get(slot_id)
    if slot is None:
        return {"ok": False, "error": f"no fold state for {slot_id!r}"}
    if slot.state in ("signed", "reopened"):
        return {"ok": False, "error": f"slot {slot_id!r} already {slot.state}"}

    refs = evidence_refs_for_slot(slot, records)
    missing = [k for k in slot_decl["requires_evidence"]
               if all(_record_by_id(records, r) is not None
                      and _record_by_id(records, r).get("kind") != k
                      for r in refs)]
    if missing or len(refs) < len(slot_decl["requires_evidence"]):
        return {
            "ok": False,
            "error": (
                f"cannot sign {slot_id!r}: evidence kinds "
                f"{slot_decl['requires_evidence']} not all present in memory "
                f"(resolved refs: {refs})"
            ),
        }

    body = json.dumps({
        "checkpoint_id": slot_id,
        "event": "signoff",
        "actor": actor,
        "note": note,
    })
    rec = {
        "id": _new_record_id(),
        "cycle_id": _latest_cycle_id_from_records(records),
        "author": actor,
        "kind": "checkpoint_event",
        "title": f"signoff {slot_id}",
        "body": body,
        "tags": [f"checkpoint:{slot_id}", "event:signoff", f"actor:{actor}"],
        "refs": refs,
        "ts": _iso_now(),
    }
    _append_record(rec)
    return {"ok": True, "id": rec["id"], "slot_id": slot_id}


def _append_directive(*, text: str, urgency: str, actor: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "directive text is empty"}
    if len(text) > _MAX_POST_BYTES:
        return {"ok": False, "error": "directive too long"}
    if urgency not in {"immediate", "next_cycle", "when_convenient"}:
        urgency = "next_cycle"
    if not actor.startswith("human:"):
        actor = f"human:{actor or 'owner'}"
    records = _load_records()
    rec = {
        "id": _new_record_id(),
        "cycle_id": _latest_cycle_id_from_records(records),
        "author": actor,
        "kind": "human_directive",
        "title": text[:80] or "human directive",
        "body": json.dumps({"text": text, "urgency": urgency}),
        "tags": ["channel:chat", f"urgency:{urgency}"],
        "refs": [],
        "ts": _iso_now(),
    }
    _append_record(rec)
    return {"ok": True, "id": rec["id"]}


def main() -> int:
    global RUNS_ROOT, DEFAULT_RUN, LEGACY_PINNED, CHECKPOINT_MODE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Multi-run mode: point at the directory that contains all marathons.
    ap.add_argument("--runs-root",
                    default="/fsx/zzsamshi/a-evolve/runs",
                    help="Parent directory holding one subdir per marathon "
                         "run (each with trace/ + memory/).")
    # Legacy single-run mode (backward compat): point at one run's
    # trace/ dir and that run gets pinned as DEFAULT_RUN so legacy URLs
    # (/train, /cycle/…) keep working.
    ap.add_argument("--trace-dir", default=None,
                    help="Pin a single run's trace/ dir. Derives its "
                         "run name + runs-root from the path.")
    ap.add_argument("--port", type=int, default=7890)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    if args.trace_dir:
        trace_dir = Path(args.trace_dir).resolve()
        if not trace_dir.is_dir():
            print(f"trace dir not found: {trace_dir}", file=sys.stderr)
            return 2
        # <runs-root>/<run_name>/trace/
        run_dir = trace_dir.parent
        RUNS_ROOT = run_dir.parent
        DEFAULT_RUN = run_dir.name
        LEGACY_PINNED = True
    else:
        RUNS_ROOT = Path(args.runs_root).resolve()
        if not RUNS_ROOT.is_dir():
            print(f"runs root not found: {RUNS_ROOT}", file=sys.stderr)
            return 2
        DEFAULT_RUN = _pick_default_run()
        LEGACY_PINNED = False

    env_mode = os.environ.get("NEMO_MAS_CHECKPOINT_MODE", CHECKPOINT_MODE_MANUAL)
    CHECKPOINT_MODE = (env_mode if env_mode in (CHECKPOINT_MODE_AUTO,
                                                CHECKPOINT_MODE_MANUAL)
                       else CHECKPOINT_MODE_MANUAL)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving runs under {RUNS_ROOT} "
          f"(default run: {DEFAULT_RUN or '(none — pre-launch)'}) "
          f"at http://{args.host}:{args.port}/ (mode={CHECKPOINT_MODE})",
          file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
