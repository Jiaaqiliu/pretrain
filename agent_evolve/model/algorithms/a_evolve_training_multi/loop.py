"""``run_cycle`` — the entire orchestration loop.

The platform does three things and nothing else:

1. Allocate a monotonic cycle id (``0001``, ``0002``, ...).
2. Make a private subdirectory for each role.
3. Drop a ``_done`` sentinel when the cycle finishes.

Everything else — strategy, plans, reports, schemas, communication
between roles — happens through files written into role directories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .role import Role


def run_cycle(workspace: Path, roles: Sequence[Role]) -> Path:
    """Run one evolution cycle.

    Args:
        workspace: Root of the evolution workspace. The function will
            create ``<workspace>/cycles/`` on first call.
        roles: Role instances to run, in order. Duplicates are allowed
            (e.g. running ``data`` twice in one cycle); each invocation
            shares the same role directory by name, so the second call
            sees the first call's output.

    Returns:
        The path of the newly created cycle directory.
    """
    cycles_root = workspace / "cycles"
    cycles_root.mkdir(parents=True, exist_ok=True)

    cycle_id = next_cycle_id(cycles_root)
    cycle_dir = cycles_root / cycle_id
    cycle_dir.mkdir()

    for role in roles:
        my_dir = cycle_dir / role.name
        my_dir.mkdir(exist_ok=True)
        role.execute(my_dir, cycle_dir)

    (cycle_dir / "_done").touch()
    return cycle_dir


def next_cycle_id(cycles_root: Path) -> str:
    """Return the next monotonic 4-digit cycle id (``0001`` / ``0002`` / ...)."""
    if not cycles_root.exists():
        return "0001"
    existing = [
        int(p.name)
        for p in cycles_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    return f"{(max(existing) + 1) if existing else 1:04d}"
