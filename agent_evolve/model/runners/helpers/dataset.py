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

from ....backends.tinkerlite.base import Datum, ModelInput


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
    # Smoke-path-only datum builder. Real SFT must go through
    # :func:`render_hf_dataset` below — never this function.
    tokens = item.get("input_ids") or [0, 1, 2, 3]
    target_ids = item.get("target_ids")
    weights = item.get("weights")
    loss_fn_inputs: dict[str, Any] = {}
    if target_ids is not None:
        loss_fn_inputs["target_tokens"] = list(target_ids)
    if weights is not None:
        loss_fn_inputs["weights"] = list(weights)
    return Datum(model_input=ModelInput.from_ints(list(tokens)), loss_fn_inputs=loss_fn_inputs)


# ── Real-SFT tokenization (prompt-masked completion-only loss) ──────────

def render_hf_dataset(
    workspace: Any,
    tokenizer: Any,
    *,
    split: str = "train",
    max_len: int = 2560,
    max_items: int | None = None,
):
    """Tokenize ``data/sources.yaml`` entries into an HF ``Dataset``.

    Expected row shape (matches ``../nemotron-auto-research/data/sft/*.jsonl``):
      ``{"prompt_rendered": "<full prompt template>", "completion": "<answer>", ...}``

    Produces ``{"input_ids", "attention_mask", "labels"}`` where labels are
    ``-100`` on the prompt segment so loss only fires on the completion.
    When the combined length exceeds ``max_len``, we truncate from the LEFT
    of the prompt so the completion stays intact.
    """
    from datasets import Dataset  # deferred — only needed in the real path

    root = Path(workspace.root)
    sources_path = root / "data" / "sources.yaml"
    paths = list(_discover_jsonl_paths(sources_path, root, split))
    if not paths:
        raise FileNotFoundError(
            f"No training sources found for split={split!r} in {sources_path}"
        )

    out = {"input_ids": [], "attention_mask": [], "labels": []}
    total = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prompt_rendered" not in row or "completion" not in row:
                    raise ValueError(
                        f"Row in {path} missing prompt_rendered/completion keys; "
                        f"got {list(row.keys())}"
                    )
                prompt_ids = tokenizer.encode(
                    row["prompt_rendered"], add_special_tokens=False
                )
                completion_ids = tokenizer.encode(
                    row["completion"], add_special_tokens=False
                )
                input_ids = prompt_ids + completion_ids
                labels = [-100] * len(prompt_ids) + completion_ids
                if len(input_ids) > max_len:
                    over = len(input_ids) - max_len
                    input_ids = input_ids[over:]
                    labels = labels[over:]
                attn = [1] * len(input_ids)
                out["input_ids"].append(input_ids)
                out["attention_mask"].append(attn)
                out["labels"].append(labels)
                total += 1
                if max_items is not None and total >= max_items:
                    return Dataset.from_dict(out)

    if total == 0:
        raise ValueError(
            f"Tokenized 0 rows from {paths}; training data is empty."
        )
    return Dataset.from_dict(out)


# ── Chat-formatted row loader (Unsloth SFT) ─────────────────────────────

def load_chat_rows(
    workspace: Any,
    *,
    split: str = "train",
    max_items: int | None = None,
) -> list[dict]:
    """Load raw ``{"messages": [...]}`` rows from ``data/sources.yaml``.

    Unlike :func:`render_hf_dataset` (which pre-tokenizes with a prompt /
    completion mask), this returns the untransformed rows so the
    Unsloth/SFTTrainer stage can apply its own chat template. Fails loud
    if no sources are found for ``split``.
    """
    root = Path(workspace.root)
    sources_path = root / "data" / "sources.yaml"
    paths = list(_discover_jsonl_paths(sources_path, root, split))
    if not paths:
        raise FileNotFoundError(
            f"No training sources found for split={split!r} in {sources_path}"
        )
    rows: list[dict] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(row)
                if max_items is not None and len(rows) >= max_items:
                    return rows
    return rows


class PadToLongest:
    """Dynamic-padding collator preserving ``-100`` labels. Torch-backed."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        import torch  # deferred — real path only

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            L = len(b["input_ids"])
            pad = max_len - L
            input_ids.append(b["input_ids"] + [self.pad_token_id] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
