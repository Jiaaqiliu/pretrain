"""Stage 1 — per-row witness/hint extraction.

Loads the domain-specific hint provider, walks the source CSV's rows for
this domain, and writes one JSONL row per input::

    {"row_id": ..., "prompt": ..., "kaggle_answer": ...,
     "hint": {"rule_summary": ..., "per_component": [...],
              "applies_cleanly": ..., "extras": {...}},
     "status": "ok" | "no_fit" | "parse_failed"}

``no_fit`` rows are dropped from the downstream stages by default
(``on_no_fit: drop``). The output is read by Stage 2 (teacher distill) and
optionally Stage 3 (self distill).
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


# Domain → keyword used by the verifier's infer_domain heuristic.
_DOMAIN_HEURISTIC = {
    "bits":      lambda p: "8-bit binary" in p or "determine the output for:" in p,
    "cipher":    lambda p: "decrypt the following text" in p,
    "equations": lambda p: ("transformation rules is applied to equations" in p
                            or "determine the result for:" in p),
    "gravity":   lambda p: "falling distance" in p
                            or ("For t =" in p and "distance" in p),
    "numerals":  lambda p: "write the number" in p and "Wonderland" in p,
    "units":     lambda p: "convert the following measurement" in p,
}


def _domain_filter(prompt: str, domain: str) -> bool:
    fn = _DOMAIN_HEURISTIC.get(domain)
    if fn is None:
        raise ValueError(f"unknown domain {domain!r}")
    return fn(prompt)


def _iter_kaggle_rows(csv_path: Path, domain: str) -> Iterable[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with csv_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if _domain_filter(r["prompt"], domain):
                yield r


def run(stage_cfg: dict[str, Any], domain: str, kaggle_csv: Path,
        log) -> dict[str, int]:
    provider_path = stage_cfg["hint_provider"]
    on_no_fit = stage_cfg.get("on_no_fit", "drop")
    out_path = Path(stage_cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    provider = importlib.import_module(provider_path)
    if not hasattr(provider, "compute_hint"):
        raise RuntimeError(f"{provider_path} missing compute_hint")

    counts = {"ok": 0, "no_fit": 0, "parse_failed": 0, "scanned": 0}
    bar = tqdm(
        unit="rows", smoothing=0.05,
        mininterval=2.0, maxinterval=30.0,
        desc="stage_1", file=sys.stderr,
        dynamic_ncols=True,
    )
    log_every_n = 200
    with out_path.open("w") as out:
        for r in _iter_kaggle_rows(kaggle_csv, domain):
            counts["scanned"] += 1
            row = {"row_id": r["id"], "prompt": r["prompt"],
                   "kaggle_answer": r["answer"]}
            try:
                hint = provider.compute_hint(r["prompt"], r["answer"])
            except Exception as exc:
                row["hint"] = None
                row["status"] = "parse_failed"
                row["error"] = repr(exc)
                counts["parse_failed"] += 1
            else:
                if hint is None:
                    row["hint"] = None
                    row["status"] = "no_fit"
                    counts["no_fit"] += 1
                    if on_no_fit == "drop":
                        out.write(json.dumps(row) + "\n")
                        bar.update(1)
                        bar.set_postfix(
                            ok=counts["ok"], no_fit=counts["no_fit"],
                            refresh=False,
                        )
                        continue
                else:
                    row["hint"] = asdict(hint)
                    row["status"] = "ok"
                    counts["ok"] += 1
            out.write(json.dumps(row) + "\n")
            bar.update(1)
            bar.set_postfix(
                ok=counts["ok"], no_fit=counts["no_fit"],
                refresh=False,
            )
            if counts["scanned"] % log_every_n == 0:
                log(f"    progress {counts['scanned']} scanned "
                    f"ok={counts['ok']} no_fit={counts['no_fit']} "
                    f"parse_failed={counts['parse_failed']}")
    bar.close()

    log(f"  scanned={counts['scanned']}  ok={counts['ok']}  "
        f"no_fit={counts['no_fit']}  parse_failed={counts['parse_failed']}")
    log(f"  wrote {out_path}")
    return counts
