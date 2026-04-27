"""TrainingWorkspace — filesystem contract for evolvable training state."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .schema import validate_training_workspace
from .types import (
    PatchOperation,
    TrainingSearchNode,
    TrainingWorkspaceNotFound,
    TrainingWorkspaceValidationError,
    WorkspaceFingerprint,
    WorkspaceMutation,
)


_DEFAULT_PROTECTED = [
    "model/base.yaml",
    "eval/local_splits.yaml",
    "eval/private_holdout.jsonl",
    "data/raw",
]

_DEFAULT_ARTIFACT = ["memory", "checkpoints", "evolution"]


class TrainingWorkspace:
    """Filesystem-backed handle to a training workspace tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._manifest_cache: dict[str, Any] | None = None

    # ── Loading / validation ─────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> "TrainingWorkspace":
        root = Path(path)
        if not root.exists():
            raise TrainingWorkspaceNotFound(f"No training workspace at {root}")
        ws = cls(root)
        ws.validate()
        return ws

    def validate(self) -> None:
        errors = validate_training_workspace(self.root)
        if errors:
            raise TrainingWorkspaceValidationError(str(self.root), errors)

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest_cache is None:
            with open(self.root / "manifest.yaml") as f:
                self._manifest_cache = yaml.safe_load(f) or {}
        return self._manifest_cache

    @property
    def name(self) -> str:
        return str(self.manifest.get("name", self.root.name))

    @property
    def evolvable_layers(self) -> list[str]:
        return list(self.manifest.get("evolvable_layers", []))

    @property
    def protected_layers(self) -> list[str]:
        return list(self.manifest.get("protected_layers", _DEFAULT_PROTECTED))

    @property
    def artifact_layers(self) -> list[str]:
        return list(self.manifest.get("artifact_layers", _DEFAULT_ARTIFACT))

    # ── IO helpers ───────────────────────────────────────────────────

    def read_yaml(self, relpath: str) -> dict:
        with open(self.root / relpath) as f:
            return yaml.safe_load(f) or {}

    def write_yaml(self, relpath: str, value: dict) -> None:
        self._assert_not_protected(relpath)
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            yaml.safe_dump(value, f, default_flow_style=False, sort_keys=False)
        self._manifest_cache = None

    def append_jsonl(self, relpath: str, row: dict) -> None:
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a") as f:
            f.write(json.dumps(row) + "\n")

    # ── Fingerprint ──────────────────────────────────────────────────

    def fingerprint(self) -> WorkspaceFingerprint:
        manifest_path = self.root / "manifest.yaml"
        manifest_hash = _hash_file(manifest_path) if manifest_path.exists() else ""

        evolvable_paths: list[Path] = []
        for rel in self.evolvable_layers:
            evolvable_paths.extend(self._expand_layer(rel))
        evolvable_hash = _hash_tree(evolvable_paths)

        protected_paths: list[Path] = []
        for rel in self.protected_layers:
            protected_paths.extend(self._expand_layer(rel))
        protected_hash = _hash_tree(protected_paths)

        git_sha = _git_sha(self.root)
        return WorkspaceFingerprint(
            workspace_name=self.name,
            manifest_hash=manifest_hash,
            evolvable_hash=evolvable_hash,
            protected_hash=protected_hash,
            git_sha=git_sha,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Fork / incumbent materialization ─────────────────────────────

    def fork(
        self,
        node_id: str,
        mutation: WorkspaceMutation,
        work_dir: str | Path,
    ) -> "TrainingWorkspace":
        """Copy this workspace into a candidate directory and apply ``mutation``."""
        work_dir = Path(work_dir)
        dest = work_dir / "nodes" / node_id / "workspace"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.root, dest, dirs_exist_ok=False)

        candidate = TrainingWorkspace(dest)
        candidate._apply_patch(mutation)
        candidate.validate()
        return candidate

    def materialize_incumbent(
        self,
        node: TrainingSearchNode,
        target_dir: str | Path | None = None,
    ) -> Path:
        """Copy the candidate workspace belonging to ``node`` into ``target_dir``.

        If the node exposes a workspace path via ``artifacts['workspace_path']``,
        copy that tree. Otherwise fall back to writing an incumbent pointer file.
        """
        target = Path(target_dir) if target_dir else (self.root / "evolution" / "incumbent")
        target.mkdir(parents=True, exist_ok=True)
        src = node.artifacts.get("workspace_path") if node.artifacts else None
        if src and Path(src).is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
        pointer = self.root / "evolution" / "incumbent.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        with open(pointer, "w") as f:
            json.dump(
                {
                    "node_id": node.node_id,
                    "metric": node.metric,
                    "reward": node.reward,
                    "checkpoint": node.checkpoint.path if node.checkpoint else None,
                    "materialized_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        return target

    # ── Private helpers ──────────────────────────────────────────────

    def _expand_layer(self, rel: str) -> list[Path]:
        path = self.root / rel
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(p for p in path.rglob("*") if p.is_file())
        return []

    def _assert_not_protected(self, relpath: str) -> None:
        norm = Path(relpath).as_posix()
        for prot in self.protected_layers:
            prot_norm = Path(prot).as_posix()
            if norm == prot_norm or norm.startswith(prot_norm + "/"):
                raise TrainingWorkspaceValidationError(
                    str(self.root),
                    [f"Cannot mutate protected layer: {relpath}"],
                )

    def _apply_patch(self, mutation: WorkspaceMutation) -> None:
        for op in mutation.patch.operations:
            self._apply_op(op)

    def _apply_op(self, op: PatchOperation) -> None:
        self._assert_not_protected(op.path)
        target = self.root / op.path
        if op.key_path is None:
            # File-level operation.
            if op.op == "remove":
                if target.exists():
                    target.unlink()
                return
            if op.op in ("replace", "add"):
                target.parent.mkdir(parents=True, exist_ok=True)
                if op.path.endswith((".yaml", ".yml")):
                    with open(target, "w") as f:
                        yaml.safe_dump(op.value, f, default_flow_style=False, sort_keys=False)
                else:
                    with open(target, "w") as f:
                        f.write(str(op.value) if op.value is not None else "")
                return
        # Nested YAML key update.
        data: dict[str, Any] = {}
        if target.exists():
            with open(target) as f:
                data = yaml.safe_load(f) or {}
        cursor: Any = data
        *parents, last = op.key_path
        for key in parents:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        if op.op == "remove":
            cursor.pop(last, None)
        else:
            cursor[last] = op.value
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tree(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        rel = p.as_posix().encode()
        h.update(rel)
        h.update(b"\x00")
        h.update(_hash_file(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def _git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None
