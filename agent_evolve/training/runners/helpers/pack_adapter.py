"""Package an SFT checkpoint into an adapter :class:`CheckpointRef`."""

from __future__ import annotations

import json
from pathlib import Path

from ...types import CheckpointRef


def pack_adapter(
    source: CheckpointRef,
    *,
    target_root: Path,
    adapter_name: str,
    metadata: dict | None = None,
) -> CheckpointRef:
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    adapter_dir = target_root / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "name": adapter_name,
        "source": source.path,
        "source_kind": source.kind,
        "metadata": metadata or {},
    }
    (adapter_dir / "adapter.json").write_text(json.dumps(info, indent=2))

    return CheckpointRef(
        name=adapter_name,
        path=str(adapter_dir),
        kind="adapter",
        metadata=info["metadata"],
    )
