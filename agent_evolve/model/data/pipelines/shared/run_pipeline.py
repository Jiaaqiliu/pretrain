"""Domain-agnostic pipeline driver.

Reads ``pipeline.yaml`` + ``prompt_templates.yaml``, walks the five
generation stages in order, halts on threshold violations, and prints a
single run log. The pipeline is generation-only — mixing the curated
output into the training set is done separately by the data_worker's
``dw-curate-mix`` skill, which consumes the Stage 5 ``curated/<hash>/``
JSONL.

Usage:
    python -m agent_evolve.model.data.pipelines.shared.run_pipeline \\
        --config   agent_evolve/model/data/pipelines/bits/pipeline.yaml \\
        --templates agent_evolve/model/data/pipelines/bits/prompt_templates.yaml \\
        [--from-stage 4]   # resume after a halt
        [--to-stage 2]     # stop early (e.g. for smoke testing)
        [--dry-run]        # validate config + endpoint reachability only
        [--limit N]        # cap source rows to N (smoke testing)

Each stage's outputs land at the path declared in ``pipeline.yaml``;
re-running with ``--from-stage`` reads them from disk rather than
recomputing. The driver does NOT touch the nemo_mas ledger — that's the
data_worker's job once curate succeeds.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from .stages import (
    curate,
    opus_supervise,
    self_distill,
    teacher_distill,
    witness_search,
)


_STAGE_ORDER = [
    ("stage_1_witness_search", witness_search),
    ("stage_2_teacher_distill", teacher_distill),
    ("stage_3_self_distill",    self_distill),
    ("stage_4_opus_supervision", opus_supervise),
    ("stage_5_curate",          curate),
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _make_logger():
    def log(msg: str) -> None:
        ts = datetime.datetime.utcnow().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
    return log


def _probe_endpoint(base_url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models",
                                     timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Substitute ``${name}`` (the pipeline name) into stage out_paths."""
    name = cfg["name"]
    def _sub(v):
        if isinstance(v, str): return v.replace("${name}", name)
        if isinstance(v, dict): return {k: _sub(x) for k, x in v.items()}
        if isinstance(v, list): return [_sub(x) for x in v]
        return v
    return _sub(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--templates", required=True, type=Path)
    ap.add_argument("--from-stage", type=int, default=1, help="1-5")
    ap.add_argument("--to-stage",   type=int, default=5, help="1-5")
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--limit",      type=int, default=None,
                    help="cap source rows in stage 1 (smoke testing)")
    args = ap.parse_args(argv)

    log = _make_logger()
    cfg = _resolve_paths(_load_yaml(args.config))
    templates_doc = _load_yaml(args.templates)
    prompt_templates = templates_doc.get("templates", {})
    prompt_blocks    = templates_doc.get("blocks", {})

    domain = cfg["domain"]
    log(f"pipeline={cfg['name']}  domain={domain}  "
        f"from_stage={args.from_stage} to_stage={args.to_stage}")

    # Endpoint reachability
    for stage_key, _ in _STAGE_ORDER:
        sc = cfg.get(stage_key, {})
        ep = sc.get("endpoint", {}).get("base_url")
        if ep and sc.get("enabled", True):
            ok = _probe_endpoint(ep)
            log(f"  endpoint check: {ep} → {'ok' if ok else 'UNREACHABLE'}")
            if not ok and not args.dry_run:
                log(f"  WARNING: {stage_key} endpoint unreachable; will fail at run-time")

    if args.dry_run:
        log("dry-run: config validated, exiting")
        return 0

    kaggle_csv = Path(cfg["source"]["kaggle_csv"])
    if not kaggle_csv.exists():
        log(f"FATAL: kaggle_csv not found: {kaggle_csv}")
        return 2

    # Pipe state: files each stage produces / consumes
    s1_out: Path | None = None
    s2_out: Path | None = None
    s3_out: Path | None = None

    for idx, (stage_key, stage_mod) in enumerate(_STAGE_ORDER, start=1):
        if idx < args.from_stage or idx > args.to_stage:
            log(f"-- skip {stage_key} (range)")
            # Re-resolve persisted paths so later stages can pick them up
            sc = cfg.get(stage_key, {})
            if stage_key == "stage_1_witness_search" and "out_path" in sc:
                s1_out = Path(sc["out_path"])
            if stage_key == "stage_2_teacher_distill" and "out_path" in sc:
                s2_out = Path(sc["out_path"])
            if stage_key == "stage_3_self_distill" and "out_path" in sc:
                s3_out = Path(sc["out_path"])
            continue

        log(f"== {stage_key}")
        sc = cfg.get(stage_key, {})
        try:
            if stage_key == "stage_1_witness_search":
                # apply --limit as a temporary cap by symlinking a head'd CSV
                if args.limit:
                    capped = Path(f"/tmp/{cfg['name']}_capped_kaggle.csv")
                    with kaggle_csv.open() as src, capped.open("w") as dst:
                        header = src.readline()
                        dst.write(header)
                        # We can't filter to domain here cheaply; cap on
                        # raw read order. Stage 1 will still domain-filter.
                        for i, line in enumerate(src):
                            dst.write(line)
                            if i >= args.limit * 20:  # ~20× headroom for filter
                                break
                    csv_for_stage = capped
                else:
                    csv_for_stage = kaggle_csv
                witness_search.run(sc, domain, csv_for_stage, log)
                s1_out = Path(sc["out_path"])

            elif stage_key == "stage_2_teacher_distill":
                if s1_out is None: s1_out = Path(sc["out_path"]).parent / "stage1.jsonl"
                teacher_distill.run(sc, s1_out, prompt_templates, log)
                if sc.get("enabled", True):
                    s2_out = Path(sc["out_path"])

            elif stage_key == "stage_3_self_distill":
                self_distill.run(sc, s1_out, prompt_templates, log)
                if sc.get("enabled", False):
                    s3_out = Path(sc["out_path"])

            elif stage_key == "stage_4_opus_supervision":
                # Audit BOTH teacher (stage 2) and self-distill (stage 3)
                # outputs. On --from-stage 4 resumes, recover paths from
                # the YAML so we don't depend on earlier stage state.
                if s2_out is None and "out_path" in cfg.get("stage_2_teacher_distill", {}):
                    s2_out = Path(cfg["stage_2_teacher_distill"]["out_path"])
                if s3_out is None and cfg.get("stage_3_self_distill", {}).get("enabled", False):
                    s3_path = cfg["stage_3_self_distill"].get("out_path")
                    if s3_path:
                        s3_out = Path(s3_path)
                opus_supervise.run(sc, {"teacher": s2_out, "self": s3_out},
                                   prompt_blocks, log)

            elif stage_key == "stage_5_curate":
                inputs = [p for p in (s2_out, s3_out) if p is not None]
                # Pass the Stage 4 audit path so curate can intersect on
                # `require_audit_pass`. On --from-stage 5 resumes, the
                # earlier branch is skipped, so re-resolve from cfg.
                s4_out = None
                s4_cfg = cfg.get("stage_4_opus_supervision", {})
                if s4_cfg.get("enabled", True) and "out_path" in s4_cfg:
                    s4_out = Path(s4_cfg["out_path"])
                curate.run(sc, inputs, log, audit_path=s4_out)

        except Exception as exc:
            log(f"FATAL in {stage_key}: {exc!r}")
            return 1

    log("pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
