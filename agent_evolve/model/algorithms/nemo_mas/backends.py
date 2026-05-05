"""Backend handlers for the nemo_mas worker tools.

Two tiers:

  * **Local handlers** — pure Python, no GPU / Bedrock required. Cover:
    sample_jsonl, count_by_field, length_distribution, format_validate,
    minhash_dedup, mix_sources, write_jsonl, filter_by_gold,
    apply_format_filter, compute_stability, plot_loss_curve,
    compute_data_gap_table, diff_yaml, render_recipe_diff,
    read_training_log, read_checkpoint_metric.

  * **Compute-bound bridge** — wraps an actual ``TrainingJobRunner``
    backend (e.g. ``SingleNodeTinkerLiteBackend(mock=True)``) +
    benchmark + workspace, and exposes ``run_eval``, ``run_short_training``,
    ``launch_training``, ``rerun_recipe_with_seeds``,
    ``load_checkpoint_for_inference``, ``batch_generate``,
    ``call_teacher_model``. If the backend is in mock mode these still
    work (they just return mock outputs).

Recommended composition::

    from agent_evolve.model.algorithms.nemo_mas import NemoMASAlgorithm
    from agent_evolve.model.algorithms.nemo_mas.backends import (
        local_handlers, BackendBridge, demo_compute_handlers,
    )

    bridge = BackendBridge(workspace_root, benchmark, backend)
    algo = NemoMASAlgorithm(backend_registry={
        **local_handlers(workspace_root),
        **bridge.as_registry(),
    })

For purely offline / demo work::

    algo = NemoMASAlgorithm(backend_registry={
        **local_handlers(workspace_root),
        **demo_compute_handlers(),
    })

Both styles produce structured JSON outputs that the LLM workers can
parse. None of these handlers raise — failures come back as
``{"ok": false, "reason": "..."}`` so the agent can adapt.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


# ── Generic helpers ─────────────────────────────────────────────────


def _ok(**kw) -> str:
    return json.dumps({"ok": True, **kw})


def _err(reason: str, **kw) -> str:
    return json.dumps({"ok": False, "reason": reason, **kw})


def _safe_path(workspace_root: Path, path: str) -> Path | None:
    p = Path(path)
    if p.is_absolute():
        full = p.resolve()
    else:
        full = (workspace_root / path.lstrip("/")).resolve()
    ws_resolved = workspace_root.resolve()
    if full == ws_resolved:
        return full
    if ws_resolved not in full.parents:
        return None
    return full


def _approx_token_len(text: str, tokenizer: str = "approx") -> int:
    """Heuristic: ~chars/4. Fast, no model dependency. Underestimates
    for code-heavy text; overestimates for some Asian scripts. Good
    enough for length-distribution histograms on English reasoning
    traces."""
    return max(1, len(text) // 4)


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _percentile(data: list[float], q: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


# ── Tier 1: Local handlers ─────────────────────────────────────────


def local_handlers(
    workspace_root: Path | str | Callable[[], Path | str],
) -> dict[str, Callable[..., str]]:
    """Build the full local-tool handler dict for a given workspace root.

    ``workspace_root`` may be a fixed path or a zero-arg callable that
    returns the current cycle's forked workspace path. Callable form lets
    ``NemoMASAlgorithm`` redirect writes into ``work_dir/cycles/<NNNN>/
    .fork_target/nodes/workspace/workspace`` each cycle without rebuilding
    the registry — otherwise the seed workspace gets mutated in-place.

    All returned handlers accept keyword args matching the tool spec in
    ``tools.py::_BACKEND_TOOL_CATALOGUE`` and return JSON-serialized
    strings (per the BedrockAgent tool contract).
    """
    if callable(workspace_root):
        _resolve_ws = lambda: Path(workspace_root())  # noqa: E731
    else:
        _fixed_ws = Path(workspace_root)
        _resolve_ws = lambda: _fixed_ws  # noqa: E731

    # ── Analyst: data inspection ────────────────────────────────

    def sample_jsonl(*, path: str, n: int = 50, seed: int = 0) -> str:
        ws = _resolve_ws()
        full = _safe_path(ws, path)
        if not full or not full.exists():
            return _err(f"path not found or escaped sandbox: {path}")
        try:
            rows = _read_jsonl(full)
        except OSError as e:
            return _err(str(e))
        if not rows:
            return _ok(rows=[], n=0, summary={})
        # Reproducible sample.
        import random
        rng = random.Random(seed)
        sample = rng.sample(rows, min(n, len(rows)))
        # Summary: per-field type histogram + length stats for str fields.
        summary: dict[str, Any] = {}
        for k in sample[0]:
            types = Counter()
            lengths = []
            for r in sample:
                v = r.get(k)
                types[type(v).__name__] += 1
                if isinstance(v, str):
                    lengths.append(len(v))
            summary[k] = {
                "types": dict(types),
                "str_len_p50": _percentile(lengths, 0.5) if lengths else None,
                "str_len_p95": _percentile(lengths, 0.95) if lengths else None,
            }
        return _ok(rows=sample, n=len(sample), total=len(rows), summary=summary)

    def count_by_field(*, path: str, field: str) -> str:
        ws = _resolve_ws()
        full = _safe_path(ws, path)
        if not full or not full.exists():
            return _err(f"path not found: {path}")
        rows = _read_jsonl(full)
        c: Counter = Counter()
        for r in rows:
            v = r.get(field)
            if v is None:
                c["<missing>"] += 1
            else:
                c[str(v)] += 1
        return _ok(field=field, total=len(rows), counts=dict(c))

    def length_distribution(*, path: str, field: str,
                            tokenizer: str = "approx") -> str:
        ws = _resolve_ws()
        full = _safe_path(ws, path)
        if not full or not full.exists():
            return _err(f"path not found: {path}")
        rows = _read_jsonl(full)
        lens = []
        for r in rows:
            v = r.get(field)
            if isinstance(v, str):
                lens.append(_approx_token_len(v, tokenizer))
        if not lens:
            return _ok(n=0, note="no string values found in field")
        return _ok(
            n=len(lens), tokenizer=tokenizer, field=field,
            p50=_percentile(lens, 0.5),
            p95=_percentile(lens, 0.95),
            p99=_percentile(lens, 0.99),
            max=max(lens),
            mean=sum(lens) / len(lens),
        )

    # ── Analyst: profiling helpers ──────────────────────────────

    def plot_loss_curve(*, training_run_ids: list[str]) -> str:
        # Without matplotlib here, return a markdown table of last-N losses
        # so the LLM can read the trajectory directly.
        return _ok(
            note="No PNG generated in local mode. The LLM should request the "
                 "training_run records via mem_get and inspect the loss "
                 "trajectory in their bodies directly.",
            ids=training_run_ids,
        )

    def compute_data_gap_table(*, eval_report_id: str) -> str:
        return _ok(
            note="Local mode: no per-row eval JSONL available. Read the "
                 "eval_report record via mem_get and use its 'Cross-tab "
                 "(category × bucket)' section directly.",
            eval_report_id=eval_report_id,
        )

    # ── DataEngineer: rejection / dedup / mix / write ──────────

    _BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")

    def filter_by_gold(*, generations: list, golds: list) -> str:
        if len(generations) != len(golds):
            return _err("generations and golds length mismatch")
        kept, rejected = [], []
        for gen, gold in zip(generations, golds):
            text = gen.get("completion") if isinstance(gen, dict) else str(gen)
            box_match = _BOX_RE.search(text or "")
            if not box_match:
                rejected.append({"reason": "no_box", "gen": text[:120]})
                continue
            extracted = box_match.group(1).strip()
            gold_str = str(gold).strip()
            if extracted == gold_str:
                kept.append({"completion": text, "extracted": extracted, "gold": gold_str})
                continue
            # numeric tolerance
            try:
                e = float(extracted.replace(",", ""))
                g = float(gold_str.replace(",", ""))
                if g != 0 and abs(e - g) / abs(g) < 1e-2:
                    kept.append({"completion": text, "extracted": extracted, "gold": gold_str})
                    continue
                if g == 0 and abs(e) < 1e-9:
                    kept.append({"completion": text, "extracted": extracted, "gold": gold_str})
                    continue
            except (ValueError, TypeError):
                pass
            rejected.append({"reason": "wrong", "extracted": extracted, "gold": gold_str})
        return _ok(kept=kept, n_kept=len(kept), n_rejected=len(rejected),
                   yield_=round(len(kept) / max(len(generations), 1), 4))

    def minhash_dedup(*, input_path: str, key_field: str,
                      threshold: float = 0.85) -> str:
        ws = _resolve_ws()
        full = _safe_path(ws, input_path)
        if not full or not full.exists():
            return _err(f"path not found: {input_path}")
        rows = _read_jsonl(full)
        # Simple shingle-based Jaccard (no datasketch dep). For ~10k rows
        # this is O(N) with a hash set; we approximate with content
        # fingerprints rather than true minhash to stay dep-free.
        seen: dict[str, int] = {}
        kept_idxs: list[int] = []
        dropped: list[dict] = []
        for i, r in enumerate(rows):
            key = str(r.get(key_field, ""))
            shingles = _shingles(key, n=4)
            fp = _fingerprint(shingles)
            if fp in seen:
                dropped.append({"i": i, "near_neighbor_i": seen[fp]})
                continue
            seen[fp] = i
            kept_idxs.append(i)
        out_path = full.with_suffix(".dedup.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for i in kept_idxs:
                f.write(json.dumps(rows[i]) + "\n")
        return _ok(input=str(full), output=str(out_path),
                   total=len(rows), kept=len(kept_idxs),
                   dropped=len(dropped),
                   note="Fingerprint-based dedup (not full MinHash); "
                        "threshold arg recorded but not used.")

    def apply_format_filter(*, input_path: str) -> str:
        ws = _resolve_ws()
        full = _safe_path(ws, input_path)
        if not full or not full.exists():
            return _err(f"path not found: {input_path}")
        rows = _read_jsonl(full)
        kept, drops = [], Counter()
        for r in rows:
            comp = r.get("completion", "")
            if not isinstance(comp, str):
                drops["non_str_completion"] += 1
                continue
            if "[verify]: PASS" not in comp:
                drops["no_verify_pass"] += 1
                continue
            if not _BOX_RE.search(comp):
                drops["no_box"] += 1
                continue
            if _approx_token_len(comp) > 7600:
                drops["overlong"] += 1
                continue
            kept.append(r)
        out_path = full.with_suffix(".filtered.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        return _ok(input=str(full), output=str(out_path),
                   total=len(rows), kept=len(kept), drops=dict(drops))

    def format_validate(*, path: str) -> str:
        ws = _resolve_ws()
        full = _safe_path(ws, path)
        if not full or not full.exists():
            return _err(f"path not found: {path}")
        rows = _read_jsonl(full)
        required = {"prompt_rendered", "completion", "category", "source"}
        problems: dict[str, int] = Counter()
        sample_problems: list[dict] = []
        for i, r in enumerate(rows):
            missing = required - set(r.keys())
            if missing:
                problems["missing_fields"] += 1
                if len(sample_problems) < 5:
                    sample_problems.append({"i": i, "missing": sorted(missing)})
                continue
            comp = r.get("completion", "")
            if not _BOX_RE.search(comp or ""):
                problems["no_box"] += 1
            if "[verify]: PASS" not in (comp or ""):
                problems["no_verify_pass"] += 1
            if _approx_token_len(comp or "") > 7600:
                problems["overlong"] += 1
        fail_rate = sum(problems.values()) / max(len(rows), 1)
        return _ok(total=len(rows), problems=dict(problems),
                   fail_rate=round(fail_rate, 4),
                   verdict="pass" if fail_rate <= 0.01 else "fail",
                   sample_problems=sample_problems)

    def mix_sources(*, sources: list[str], weights: list[float],
                    curriculum_yaml: str | None = None) -> str:
        ws = _resolve_ws()
        if len(sources) != len(weights):
            return _err("sources and weights length mismatch")
        all_rows: list[dict] = []
        per_source: dict[str, int] = {}
        for src, w in zip(sources, weights):
            full = _safe_path(ws, src)
            if not full or not full.exists():
                return _err(f"source path not found: {src}")
            rows = _read_jsonl(full)
            # Weight as a target proportion: keep round(w * len(rows)) rows.
            # Simplified: for "weight" semantics we just take a prefix.
            n = max(1, int(round(w * len(rows))))
            picked = rows[:n]
            all_rows.extend(picked)
            per_source[src] = len(picked)
        # Output path
        out = ws / "data" / "final" / "train.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
        return _ok(output=str(out), total=len(all_rows),
                   per_source=per_source, sha256_short=digest,
                   curriculum_yaml=curriculum_yaml)

    def write_jsonl(*, path: str, rows: list[dict]) -> str:
        ws = _resolve_ws()
        full = _safe_path(ws, path)
        if not full:
            return _err(f"path escaped sandbox: {path}")
        full.parent.mkdir(parents=True, exist_ok=True)
        with full.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return _ok(output=str(full), n=len(rows))

    # ── Theorist helpers ────────────────────────────────────────

    def diff_yaml(*, a: str, b: str) -> str:
        ws = _resolve_ws()
        # Treat a, b as either inline YAML or paths.
        def _resolve(s: str) -> str:
            p = _safe_path(ws, s)
            if p and p.exists() and p.is_file():
                return p.read_text(encoding="utf-8")
            return s
        a_text = _resolve(a)
        b_text = _resolve(b)
        diff = "\n".join(difflib.unified_diff(
            a_text.splitlines(), b_text.splitlines(),
            fromfile="a", tofile="b", lineterm="",
        ))
        return _ok(diff=diff, a_lines=len(a_text.splitlines()),
                   b_lines=len(b_text.splitlines()))

    def render_recipe_diff(*, proposal_body: str) -> str:
        # The proposal body is expected to contain a YAML block — we just
        # return it pre-formatted as a fenced block. Theorist's job is to
        # author the diff; we package it.
        return _ok(rendered=f"```yaml\n{proposal_body.strip()}\n```")

    # ── Engineer: training log + checkpoint metric reads ────────
    #
    # Runner scaffolding tools were removed: training always goes through
    # agent_evolve/model/runners/stages/*.py via the StageRegistry, not
    # through workspace-scaffolded scripts. The Engineer role reads logs
    # and metrics produced by those platform runners.

    def read_training_log(*, job_id: str) -> str:
        ws = _resolve_ws()
        log = ws / "logs" / f"{job_id}.log"
        if not log.exists():
            return _err(f"no log at {log}")
        text = log.read_text(encoding="utf-8")
        return _ok(job_id=job_id, log_tail="\n".join(text.splitlines()[-200:]))

    def read_checkpoint_metric(*, ckpt_path: str) -> str:
        ws = _resolve_ws()
        ckpt = _safe_path(ws, ckpt_path) or Path(ckpt_path)
        candidate = ckpt / "metric.json" if ckpt.is_dir() else ckpt.parent / "metric.json"
        if not candidate.exists():
            return _err(f"no metric.json at {candidate}")
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return _err(f"invalid metric.json: {e}")
        return _ok(path=str(candidate), metric=data)

    def compute_stability(*, training_run_ids: list[str]) -> str:
        # Looks for metric.json beside each ckpt — but since we don't have
        # those record bodies here, we accept that the LLM provides
        # numerics in the call. A more capable impl would look records up
        # via memory; that would need a memory handle, which we'd rather
        # keep out of backends.py to avoid coupling.
        return _ok(
            note="compute_stability is a record-derived metric. The "
                 "Engineer skill instructs the LLM to mem_get each "
                 "training_run id, extract the primary metric from the "
                 "body, and include the mean / std in the cv_result body.",
            ids=training_run_ids,
        )

    # ── Submission packaging (trainer) ──────────────────────────
    #
    # The Kaggle host loads our adapter onto the frozen Nemotron-3-Nano-30B
    # base and scores it in vLLM. Our job: package the LoRA adapter dir
    # into ``submission.zip`` with ``adapter_config.json`` at the archive
    # root, sanity-check rank <= 32, and report what shipped. The reviewer
    # later cites the resulting record when posting a cp_submission_ready
    # verdict.

    def pack_submission(*, ckpt_path: str, out_zip: str) -> str:
        import zipfile
        ws = _resolve_ws()
        src = _safe_path(ws, ckpt_path) or Path(ckpt_path)
        if not src.is_dir():
            return _err(f"ckpt_path is not a directory: {src}")
        adapter_cfg = src / "adapter_config.json"
        if not adapter_cfg.is_file():
            return _err(
                f"no adapter_config.json under {src} — is this a LoRA adapter "
                "dir? (Full-weight checkpoints are not a valid Kaggle submission.)"
            )
        try:
            cfg = json.loads(adapter_cfg.read_text())
        except (OSError, json.JSONDecodeError) as e:
            return _err(f"adapter_config.json unreadable: {e}")
        rank = cfg.get("r") or cfg.get("rank")
        if rank is not None and int(rank) > 32:
            return _err(
                f"adapter rank={rank} > 32; Kaggle rejects rank > 32. "
                "Re-train with lora_rank <= 32."
            )

        # Resolve output zip path. Must live under workspace root (or an
        # absolute path on FSx) so the file is reachable from the driver
        # after the cycle closes.
        out = Path(out_zip)
        if not out.is_absolute():
            out = ws / out
        out.parent.mkdir(parents=True, exist_ok=True)

        # Write a flat zip with the adapter files at the archive root —
        # the host unzips straight into a LoRA dir.
        try:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(src.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(src))
        except OSError as e:
            return _err(f"zip write failed: {e}")

        size = out.stat().st_size
        return _ok(
            zip_path=str(out),
            size_bytes=size,
            adapter_rank=rank,
            target_modules=cfg.get("target_modules"),
            base_model_name_or_path=cfg.get("base_model_name_or_path"),
            peft_type=cfg.get("peft_type"),
            message=(
                f"packaged {src} → {out} ({size:,} bytes). Trainer should "
                "write a `submission_artifact` record with this zip_path + "
                "refs to the training_run that produced the checkpoint."
            ),
        )

    # ── Kaggle API (reviewer) ───────────────────────────────────
    #
    # Uses the `kaggle` CLI (credentials expected at ~/.kaggle/kaggle.json).
    # We keep these out of the k8s pods — they're cheap HTTPS calls and
    # the driver host already has AWS + Kaggle creds. Submission is
    # gated by the Quality Plan (reviewer only calls this after verdict
    # ready_to_sign on cp_submission_ready) and by the per-run budget
    # enforced in the reviewer's task brief.

    _KAGGLE_COMPETITION = "nvidia-nemotron-model-reasoning-challenge"
    # Hard budget: Kaggle gives us 5 submits/day and we pick 2 finals; a
    # single marathon run shouldn't burn more than 1. If the
    # NemoMASAlgorithm sets a higher per-run limit, this handler-level
    # fallback still refuses once the shared records.jsonl has already
    # captured that many kaggle_submission_result records.
    _KAGGLE_MAX_PER_RUN = int(os.environ.get("NEMO_MAS_KAGGLE_MAX_PER_RUN", "1"))

    def _count_prior_submits() -> int:
        # Memory lives at <work_dir>/memory/records.jsonl (the shared
        # cross-cycle store the driver seeds). We don't have a direct
        # handle, but the driver threads it via env var NEMO_MAS_MEMORY_PATH.
        mem_path = os.environ.get("NEMO_MAS_MEMORY_PATH")
        if not mem_path or not Path(mem_path).is_file():
            return 0
        n = 0
        try:
            with Path(mem_path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if json.loads(line).get("kind") == "kaggle_submission_result":
                            n += 1
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return n

    def kaggle_submit(*, zip_path: str, message: str) -> str:
        import shutil
        import subprocess
        if not shutil.which("kaggle"):
            return _err("`kaggle` CLI not on PATH; `pip install kaggle`")
        prior = _count_prior_submits()
        if prior >= _KAGGLE_MAX_PER_RUN:
            return _err(
                f"kaggle_submit refused: {prior} submit(s) already made this "
                f"run, budget is {_KAGGLE_MAX_PER_RUN}. Post verdict="
                "ready_to_sign without submitting; the human can push via "
                "the CLI manually."
            )
        ws = _resolve_ws()
        zp = Path(zip_path)
        if not zp.is_absolute():
            zp = ws / zp
        if not zp.is_file():
            return _err(f"zip not found: {zp}")
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", _KAGGLE_COMPETITION,
            "-f", str(zp),
            "-m", message or "auto submit from nemo_mas",
        ]
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return _err(f"kaggle submit failed: {e}")
        if out.returncode != 0:
            return _err(
                f"kaggle submit returncode={out.returncode}; "
                f"stderr={out.stderr.strip()}"
            )
        # The CLI prints "Successfully submitted to ..." but doesn't
        # expose the submission id. We pull it via a follow-up list call
        # so the reviewer can track score status later.
        list_out = subprocess.run(
            ["kaggle", "competitions", "submissions",
             "-c", _KAGGLE_COMPETITION, "-v"],
            capture_output=True, text=True, timeout=120,
        )
        submission_id = ""
        latest_status = ""
        if list_out.returncode == 0:
            # CSV: fileName,date,description,status,publicScore,privateScore
            lines = list_out.stdout.strip().splitlines()
            for row in lines[1:2]:   # most recent is first
                parts = row.split(",")
                if len(parts) >= 4:
                    submission_id = parts[1].strip()   # date acts as handle
                    latest_status = parts[3].strip()
                break
        return _ok(
            submission_id=submission_id,
            status=latest_status or "pending",
            competition=_KAGGLE_COMPETITION,
            stdout=out.stdout.strip()[-500:],
            message=(
                "Submission pushed to Kaggle. Public-LB score arrives after "
                "host-side inference (~30-60 min). Call `kaggle_fetch_score` "
                "later to update the kaggle_submission_result record."
            ),
        )

    def kaggle_fetch_score(*, submission_id: str = "") -> str:
        import shutil
        import subprocess
        if not shutil.which("kaggle"):
            return _err("`kaggle` CLI not on PATH")
        cmd = ["kaggle", "competitions", "submissions",
               "-c", _KAGGLE_COMPETITION, "-v"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, OSError) as e:
            return _err(f"kaggle list failed: {e}")
        if out.returncode != 0:
            return _err(
                f"kaggle list returncode={out.returncode}; "
                f"stderr={out.stderr.strip()}"
            )
        lines = out.stdout.strip().splitlines()
        rows = []
        for row in lines[1:]:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) < 6:
                continue
            rows.append({
                "fileName":    parts[0],
                "date":        parts[1],
                "description": parts[2],
                "status":      parts[3],
                "publicScore": parts[4],
                "privateScore": parts[5],
            })
        if not rows:
            return _err("no submissions found for this competition")
        # If submission_id (really the date stamp from kaggle_submit) is
        # given, return that row; otherwise the most recent.
        target = next((r for r in rows if r["date"] == submission_id), rows[0])
        return _ok(
            submission_id=target["date"],
            status=target["status"],
            public_score=target["publicScore"],
            private_score=target["privateScore"],
            all_submissions=rows[:5],
        )

    return {
        "sample_jsonl":              sample_jsonl,
        "count_by_field":            count_by_field,
        "length_distribution":       length_distribution,
        "plot_loss_curve":           plot_loss_curve,
        "compute_data_gap_table":    compute_data_gap_table,
        "filter_by_gold":            filter_by_gold,
        "minhash_dedup":             minhash_dedup,
        "apply_format_filter":       apply_format_filter,
        "format_validate":           format_validate,
        "mix_sources":               mix_sources,
        "write_jsonl":               write_jsonl,
        "diff_yaml":                 diff_yaml,
        "render_recipe_diff":        render_recipe_diff,
        "read_training_log":         read_training_log,
        "read_checkpoint_metric":    read_checkpoint_metric,
        "compute_stability":         compute_stability,
        "pack_submission":           pack_submission,
        "kaggle_submit":             kaggle_submit,
        "kaggle_fetch_score":        kaggle_fetch_score,
    }


# ── Tier 2: BackendBridge ───────────────────────────────────────────


class BackendBridge:
    """Wrap a TrainingJobRunner backend + benchmark + workspace path
    behind the compute-bound tool registry.

    Designed to be permissive: if the caller's backend is in mock mode
    (``SingleNodeTinkerLiteBackend(mock=True)``) the handlers still
    return plausible structured outputs. If a real backend is provided
    they delegate to it.

    The bridge does not import any model or framework — all heavy work
    happens inside the backend / benchmark.
    """

    def __init__(
        self,
        *,
        workspace_root: Path | str | Callable[[], Path | str],
        benchmark: Any,
        backend: Any,
    ) -> None:
        # Accept a callable so the bridge follows the current cycle's
        # forked workspace rather than pinning to the seed. See
        # ``local_handlers`` for the same shape.
        if callable(workspace_root):
            self._resolve_ws_root = lambda: Path(workspace_root())
        else:
            fixed = Path(workspace_root)
            self._resolve_ws_root = lambda: fixed
        self.benchmark = benchmark
        self.backend = backend
        self._infer_handles: dict[str, Any] = {}

    @property
    def workspace_root(self) -> Path:
        return self._resolve_ws_root()

    @property
    def workspace(self):
        # The platform backend expects a ``TrainingWorkspace`` object with a
        # ``.root`` attribute, not a raw ``Path``. Passing a ``Path`` makes
        # ``workspace.root`` resolve to ``"/"`` (the filesystem root), which
        # then fails as "pipeline missing: /train/pipeline.yaml" before any
        # stage runs. Re-wrap each access so per-cycle forks work.
        from agent_evolve.model.workspace import TrainingWorkspace
        return TrainingWorkspace(self._resolve_ws_root())

    def as_registry(self) -> dict[str, Callable[..., str]]:
        return {
            "run_eval":                    self.run_eval,
            "run_short_training":          self.run_short_training,
            "launch_training":             self.launch_training,
            "rerun_recipe_with_seeds":     self.rerun_recipe_with_seeds,
            "load_checkpoint_for_inference": self.load_checkpoint_for_inference,
            "batch_generate":              self.batch_generate,
            "call_teacher_model":          self.call_teacher_model,
        }

    # ── Eval / training delegation ───────────────────────────────

    def run_eval(self, *, ckpt_path: str, split: str,
                 limit: int | None = None) -> str:
        try:
            from agent_evolve.model.types import CheckpointRef
        except ImportError as e:
            return _err(f"types import failed: {e}")
        ckpt = CheckpointRef(name=Path(ckpt_path).name, path=ckpt_path,
                             kind="adapter")
        try:
            out_path = self.benchmark.evaluate(
                self.workspace, ckpt, self.backend, split,
            )
        except Exception as e:                # noqa: BLE001
            return _err(f"benchmark.evaluate failed: {e}")
        return _ok(eval_output_path=str(out_path), split=split,
                   ckpt_path=ckpt_path, limit=limit,
                   note="Per-row JSONL is at eval_output_path. "
                        "Use sample_jsonl on it to inspect failures.")

    def run_short_training(self, *, recipe_diff: str,
                           max_steps: int = 200,
                           log_every: int = 10) -> str:
        # We don't have a "short training" hook on TrainingJobRunner
        # directly. The simplest correct behavior is to refuse and ask
        # Engineer to use launch_training with a step cap.
        return _err(
            "run_short_training is not directly delegable to "
            "TrainingJobRunner. Use launch_training with a small "
            "max_steps and monitor=true.",
            recipe_diff_preview=recipe_diff[:200], max_steps=max_steps,
            log_every=log_every,
        )

    def launch_training(self, *, recipe_path: str,
                        data_path: str, ckpt_out: str,
                        max_steps: int | None = None,
                        monitor: bool = True) -> str:
        # Training always runs through the platform's StageRegistry — the
        # backend's run_trial dispatches to model/runners/stages/*.py.
        # We never shell out to a workspace-local script; there's no
        # runner_path argument.
        try:
            from agent_evolve.model.types import (
                CheckpointRef, TrainingSearchNode, TrialBudget,
            )
        except ImportError as e:
            return _err(f"types import failed: {e}")
        node = TrainingSearchNode(
            node_id=f"node-nemomas-{uuid.uuid4().hex[:8]}",
            parent_id="",
            branch_id=0,
            mutation_plan=f"manual launch recipe={recipe_path} data={data_path}",
            workspace_patch=None,
        )
        budget = TrialBudget(seconds=None, steps=max_steps, tokens=None)
        try:
            result = self.backend.run_trial(
                self.workspace, node, budget, self.benchmark,
            )
        except Exception as e:                # noqa: BLE001
            return _err(f"backend.run_trial failed: {e}")
        ckpt = result.checkpoint
        ckpt_path = ckpt.path if isinstance(ckpt, CheckpointRef) else None
        metric_name = result.eval_metrics.primary_metric_name if result.eval_metrics else None
        metric_value = result.eval_metrics.primary_metric_value if result.eval_metrics else None
        return _ok(
            job_id=node.node_id, status=str(getattr(result, "status", "")),
            ckpt_path=ckpt_path, metric_name=metric_name,
            metric_value=metric_value,
            cost=getattr(result, "cost", {}),
            note="If monitor=true and the backend killed mid-run, status "
                 "will be train_failed; write a failed_attempt rather "
                 "than a training_run.",
        )

    def rerun_recipe_with_seeds(self, *, recipe_path: str, data_path: str,
                                seeds: list[int],
                                splits: list[str] | None = None) -> str:
        # Sequential reruns. Each spawns a TrialBudget with the seed
        # threaded into the workspace patch via mutation_plan. The real
        # backend may or may not honor this — we surface the constraint
        # so callers know.
        ids = []
        for s in seeds:
            r = self.launch_training(
                recipe_path=recipe_path,
                data_path=data_path, ckpt_out=f"cv/seed{s}/",
                max_steps=None, monitor=True,
            )
            ids.append(r)
        return _ok(rerun_results=ids, seeds=seeds, splits=splits or [],
                   note="Each result is the JSON returned by launch_training.")

    # ── Inference delegation (vLLM / Bedrock / etc.) ─────────────

    def load_checkpoint_for_inference(self, *, ckpt_path: str) -> str:
        try:
            from agent_evolve.model.types import CheckpointRef
        except ImportError as e:
            return _err(f"types import failed: {e}")
        ckpt = CheckpointRef(name=Path(ckpt_path).name, path=ckpt_path,
                             kind="adapter")
        try:
            client = self.backend.create_sampling_client(self.workspace, ckpt)
        except Exception as e:                # noqa: BLE001
            return _err(f"create_sampling_client failed: {e}")
        handle = f"infer-{uuid.uuid4().hex[:8]}"
        self._infer_handles[handle] = client
        return _ok(handle=handle, ckpt_path=ckpt_path)

    def batch_generate(self, *, handle: str, prompts: list[str],
                       sampling_config: dict | None = None) -> str:
        client = self._infer_handles.get(handle)
        if client is None:
            return _err(f"unknown inference handle: {handle}")
        cfg = dict(sampling_config or {})
        # Default to Kaggle eval contract (deterministic).
        cfg.setdefault("temperature", 0.0)
        cfg.setdefault("top_p", 1.0)
        cfg.setdefault("max_tokens", 7680)
        try:
            outs = client.generate(prompts, **cfg)
        except TypeError:
            # Older clients may take a single dict.
            try:
                outs = client.generate(prompts, sampling_config=cfg)
            except Exception as e:           # noqa: BLE001
                return _err(f"client.generate failed: {e}")
        except Exception as e:                # noqa: BLE001
            return _err(f"client.generate failed: {e}")
        return _ok(generations=outs, n=len(outs),
                   sampling_config=cfg, handle=handle)

    def call_teacher_model(self, *, model: str, prompts: list[str],
                           max_tokens: int = 8000,
                           temperature: float = 0.7,
                           system_prompt: str | None = None) -> str:
        # Teacher distill is delegated to whatever LLM client the user
        # has wired in. Keep this as a structured "not wired" stub by
        # default — let users override via a custom registry. (We don't
        # auto-import boto3 here because that would force a dep on
        # backends.py.)
        return _err(
            "call_teacher_model is not implemented at the bridge level. "
            "Wire in a teacher-model handler explicitly via the "
            "backend_registry kwarg of NemoMASAlgorithm.",
            model=model, n_prompts=len(prompts),
            max_tokens=max_tokens, temperature=temperature,
        )


# ── Tier 2 fallback: pure-demo handlers ─────────────────────────────


def demo_compute_handlers() -> dict[str, Callable[..., str]]:
    """Compute-bound handlers that return plausible mock outputs.

    Use this when no backend is available (tests, dry-run drivers).
    All outputs are structured so the LLM workers can write valid
    records from them. Numbers are obviously fake — don't ship them.
    """
    def run_eval(*, ckpt_path: str, split: str, limit: int | None = None) -> str:
        return _ok(eval_output_path="(demo)/eval_rows.jsonl", split=split,
                   ckpt_path=ckpt_path,
                   primary_metric=0.50,  # sentinel; obviously fake
                   per_category={"bit_manipulation": 0.5, "cryptarithm": 0.4},
                   error_buckets={"format_error": 5, "wrong_rule": 30},
                   note="DEMO MODE — numbers are placeholders.")

    def launch_training(*, recipe_path: str,
                        data_path: str, ckpt_out: str,
                        max_steps: int | None = None,
                        monitor: bool = True) -> str:
        return _ok(job_id=f"demo-{uuid.uuid4().hex[:6]}",
                   status="success", ckpt_path=ckpt_out,
                   metric_name="kaggle_nemo_boxed_em",
                   metric_value=0.50,
                   cost={"seconds": 60.0},
                   note="DEMO MODE — no actual training happened.")

    def rerun_recipe_with_seeds(*, recipe_path: str, data_path: str,
                                seeds: list[int],
                                splits: list[str] | None = None) -> str:
        return _ok(rerun_results=[
            {"seed": s, "metric_value": 0.50 + 0.001 * (s % 7)}
            for s in seeds
        ], note="DEMO MODE.")

    def call_teacher_model(*, model: str, prompts: list[str],
                           max_tokens: int = 8000,
                           temperature: float = 0.7,
                           system_prompt: str | None = None) -> str:
        return _ok(generations=[
            {"prompt": p, "completion": "[verify]: PASS\n\\boxed{42}"}
            for p in prompts
        ], cost_usd=0.001 * len(prompts), model=model,
           note="DEMO MODE — no real teacher call.")

    def load_checkpoint_for_inference(*, ckpt_path: str) -> str:
        return _ok(handle=f"demo-{uuid.uuid4().hex[:6]}", ckpt_path=ckpt_path,
                   note="DEMO MODE.")

    def batch_generate(*, handle: str, prompts: list[str],
                       sampling_config: dict | None = None) -> str:
        return _ok(generations=[
            {"prompt": p, "completion": f"[verify]: PASS\n\\boxed{{demo-{i}}}"}
            for i, p in enumerate(prompts)
        ], handle=handle, sampling_config=sampling_config or {},
           note="DEMO MODE.")

    def run_short_training(*, recipe_diff: str, max_steps: int = 200,
                           log_every: int = 10) -> str:
        # A loss curve that decreases monotonically (the "passes profile" case).
        losses = [round(2.0 * math.exp(-i / 80), 4)
                  for i in range(0, max_steps, log_every)]
        return _ok(loss_curve=losses, max_steps=max_steps, log_every=log_every,
                   verdict="usable",
                   note="DEMO MODE — synthetic loss curve.")

    def pack_submission(*, ckpt_path: str, out_zip: str) -> str:
        return _ok(zip_path=out_zip, size_bytes=123456, adapter_rank=32,
                   target_modules=["q_proj", "k_proj"], peft_type="LORA",
                   note="DEMO MODE — no real zip written.")

    def kaggle_submit(*, zip_path: str, message: str) -> str:
        return _ok(submission_id="2026-05-04 00:00:00",
                   status="pending", competition="demo",
                   note="DEMO MODE — nothing sent to Kaggle.")

    def kaggle_fetch_score(*, submission_id: str = "") -> str:
        return _ok(submission_id=submission_id or "demo",
                   status="complete", public_score="0.500",
                   private_score="", note="DEMO MODE.")

    return {
        "run_eval":                    run_eval,
        "run_short_training":          run_short_training,
        "launch_training":             launch_training,
        "rerun_recipe_with_seeds":     rerun_recipe_with_seeds,
        "load_checkpoint_for_inference": load_checkpoint_for_inference,
        "batch_generate":              batch_generate,
        "call_teacher_model":          call_teacher_model,
        "pack_submission":             pack_submission,
        "kaggle_submit":               kaggle_submit,
        "kaggle_fetch_score":          kaggle_fetch_score,
    }


# ── Helpers for minhash_dedup ───────────────────────────────────────


def _shingles(text: str, *, n: int = 4) -> set[str]:
    text = text.lower()
    return {text[i:i + n] for i in range(0, max(len(text) - n + 1, 0))}


def _fingerprint(shingles: set[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(shingles)[:128]:   # cap for stability
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
