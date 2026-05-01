"""The ``Role`` Protocol — the only contract every role must satisfy.

A role has a name and an ``execute(my_dir, cycle_dir)`` method. What
gets written into ``my_dir`` is entirely the role's choice (file format,
filenames, schemas — none of it is fixed by the platform).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Role(Protocol):
    """Anything with a stable ``name`` and an ``execute`` method.

    The platform calls ``execute`` once per cycle, in the order given to
    :func:`run_cycle`. Roles communicate by reading sibling subdirectories
    of ``cycle_dir`` (e.g. ``cycle_dir / "data"``) — there is no typed
    pipe. If a role needs cross-cycle context, it walks
    ``cycle_dir.parent`` (which is ``<workspace>/cycles/``).
    """

    name: str

    def execute(self, my_dir: Path, cycle_dir: Path) -> None:
        """Run this role for one cycle.

        Args:
            my_dir: This role's private directory (already created).
                Write whatever you want here in whatever format you want.
            cycle_dir: The cycle root. ``cycle_dir / "<other_role>"``
                holds upstream output you may want to read.
        """
        ...
