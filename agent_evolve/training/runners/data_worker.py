"""Dataset rendering.

In ``smoke`` mode we load one of:
  1. ``data/*.jsonl`` files referenced by ``data/sources.yaml``, or
  2. ``eval/local_holdout_small.jsonl`` as a tiny fallback, or
  3. a deterministic synthetic batch.

The caller is expected to batch the returned iterable into Datums of the
appropriate size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from ...backends.tinkerlite.base import Datum, ModelInput


def render_datums(
    workspace: Any,
    split: str = "train",
    *,
    max_items: int = 32,
    smoke: bool = True,
) -> Iterator[Datum]:
    root = Path(workspace.root)
    sources_path = root / "data" / "sources.yaml"
    rendered = 0

    for path in _discover_jsonl_paths(sources_path, root, split):
        for item in _read_jsonl(path, limit=max_items - rendered):
            yield _item_to_datum(item)
            rendered += 1
            if rendered >= max_items:
                return

    if rendered == 0:
        fallback = root / "eval" / "local_holdout_small.jsonl"
        if fallback.exists():
            for item in _read_jsonl(fallback, limit=max_items - rendered):
                yield _item_to_datum(item)
                rendered += 1
                if rendered >= max_items:
                    return

    if rendered == 0 and smoke:
        for i in range(max_items):
            yield Datum(model_input=ModelInput.from_ints([0, 1, 2, i + 1]))


def _discover_jsonl_paths(sources_path: Path, root: Path, split: str) -> Iterable[Path]:
    if not sources_path.exists():
        return []
    try:
        with open(sources_path) as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return []
    paths: list[Path] = []
    for entry in raw.get("sources", []) or []:
        if isinstance(entry, dict):
            entry_split = entry.get("split") or "train"
            path_rel = entry.get("path")
            if path_rel and entry_split == split:
                path = (root / path_rel).resolve()
                if path.exists():
                    paths.append(path)
    return paths


def _read_jsonl(path: Path, *, limit: int) -> Iterable[dict]:
    if limit <= 0:
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _item_to_datum(item: dict) -> Datum:
    # We don't tokenize here — workers that run a real model (PR7+) must
    # override tokenization. Smoke path treats integers directly.
    tokens = item.get("input_ids") or [0, 1, 2, 3]
    target_ids = item.get("target_ids")
    weights = item.get("weights")
    loss_fn_inputs: dict[str, Any] = {}
    if target_ids is not None:
        loss_fn_inputs["target_tokens"] = list(target_ids)
    if weights is not None:
        loss_fn_inputs["weights"] = list(weights)
    return Datum(model_input=ModelInput.from_ints(list(tokens)), loss_fn_inputs=loss_fn_inputs)
