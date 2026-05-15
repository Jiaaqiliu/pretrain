#!/usr/bin/env python3
"""Pack a trained LoRA adapter into a Kaggle-compatible submission.zip.

Kaggle's scoring notebook loads the LoRA via vLLM's ``LoRARequest`` — it
expects a flat zip whose root contains ``adapter_config.json`` +
``adapter_model.safetensors`` + tokenizer files. No on-cluster transform
is applied, so the submission must already be in the vLLM-compatible
form (expanded expert keys, Mamba already merged into in_proj). Our
``train_unsloth.py`` already produces that form, so packaging is mostly
a zip step with two optional tweaks:

  --rename-lm-head   rewrite lm_head keys from
                     ``base_model.model.lm_head.*`` →
                     ``base_model.model.backbone.lm_head.*``
                     (matches the Nemotron-H module path under .backbone)
  --drop-lm-head     strip lm_head LoRA entirely (matches the known-
                     working 0.74 LB W4/step_200 shape)

Default: keep lm_head as-is. Our step_250 submission (2026-05-12) used
the default and is pending Kaggle score; if it lands ≥0.74 that
confirms the non-renamed, lm-head-included form is valid.

Usage:
    python -m agent_evolve.backends.nemo_reasoner.pack_for_kaggle \
        --ckpt /path/to/step_250 \
        --out  /path/to/submission.zip

    # Strip lm_head (replicate W4/step_200 shape exactly):
    python -m agent_evolve.backends.nemo_reasoner.pack_for_kaggle \
        --ckpt /path/to/step_250 --out /path/to/sub.zip --drop-lm-head

    # Rename lm_head to backbone.lm_head (if the non-renamed form errors
    # out at Kaggle's vLLM load):
    python -m agent_evolve.backends.nemo_reasoner.pack_for_kaggle \
        --ckpt /path/to/step_250 --out /path/to/sub.zip --rename-lm-head
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "README.md",
]


def _detect_pretransformed(cfg: dict, keys: list[str]) -> list[str]:
    """Return signals that ``ckpt`` was already mutated for local vLLM eval.

    A raw unsloth-saved checkpoint has:
      * ``base_model_name_or_path`` ending in ``-unsloth``
      * ``.base_layer.*`` shadow tensors (unsloth's PEFT wrap leaves them)
      * lm_head keys under ``base_model.model.lm_head.*``

    Local eval staging (see ``model/runners/stages/eval.py``
    ``_stage_vllm_compatible_adapter``) rewrites all three. That staged
    form has been validated against local vLLM but has NEVER been scored
    on the Kaggle LB — all successful Kaggle submissions used the raw
    form. Packing a pre-transformed adapter silently is how a packed
    submission ends up at a different byte size than the reference
    (step_250 → 0.78 → 1.37 GB) and risks either a host-side load error
    or a silent scoring divergence.
    """
    signals = []
    base = str(cfg.get("base_model_name_or_path", ""))
    if base and not base.endswith("-unsloth"):
        signals.append(f"base_model_name_or_path={base!r} (raw form points to *-unsloth)")
    if not any(".base_layer." in k for k in keys):
        signals.append("no .base_layer.* tensors (raw unsloth saves always include them)")
    if any(k.startswith("base_model.model.backbone.lm_head.") for k in keys):
        signals.append("lm_head keys already under .backbone.lm_head.* (raw form uses .lm_head.*)")
    return signals


def pack(
    ckpt: Path,
    out_zip: Path,
    *,
    drop_lm_head: bool,
    rename_lm_head: bool,
    allow_pretransformed: bool = False,
) -> dict:
    if not ckpt.is_dir():
        raise SystemExit(f"error: --ckpt is not a directory: {ckpt}")
    if drop_lm_head and rename_lm_head:
        raise SystemExit("error: --drop-lm-head and --rename-lm-head are mutually exclusive")

    src_st = ckpt / "adapter_model.safetensors"
    src_cfg = ckpt / "adapter_config.json"
    if not src_st.is_file() or not src_cfg.is_file():
        raise SystemExit(f"error: missing adapter files under {ckpt}")

    cfg = json.loads(src_cfg.read_text())
    rank = cfg.get("r") or cfg.get("rank")
    if rank is not None and int(rank) > 32:
        raise SystemExit(f"error: adapter rank={rank} > 32 (Kaggle rejects rank > 32)")

    # Load tensors (copy into memory — adapters are <5GB).
    tensors: dict[str, object] = {}
    with safe_open(src_st, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    keys = list(tensors.keys())
    signals = _detect_pretransformed(cfg, keys)
    if signals:
        lines = "\n  - ".join(signals)
        raw_hint = ""
        if ckpt.name.endswith("_vllm"):
            sibling = ckpt.with_name(ckpt.name[:-len("_vllm")])
            if sibling.is_dir():
                raw_hint = f"\n\nRaw sibling found at: {sibling}\nRe-run with --ckpt {sibling}"
        if not allow_pretransformed:
            raise SystemExit(
                "error: checkpoint looks pre-transformed for local vLLM eval, not "
                "raw training output. Only the raw form has ever scored on the "
                "Kaggle LB (step_200 → 0.74, step_250 → 0.78).\n\n"
                f"Detected:\n  - {lines}"
                f"{raw_hint}\n\n"
                "If you have evidence the transformed form is Kaggle-compatible, "
                "pass --allow-pretransformed to override."
            )
        print(
            "[pack] WARNING: packing pre-transformed adapter (override via "
            "--allow-pretransformed):",
            flush=True,
        )
        for s in signals:
            print(f"[pack]   - {s}", flush=True)

    n_before = len(tensors)
    n_lm = sum(1 for k in tensors if ".lm_head." in k)

    if drop_lm_head:
        tensors = {k: v for k, v in tensors.items() if ".lm_head." not in k}
        cfg["target_modules"] = [t for t in cfg.get("target_modules", []) if t != "lm_head"]
        action = f"dropped {n_lm} lm_head keys"
    elif rename_lm_head:
        old = "base_model.model.lm_head."
        new = "base_model.model.backbone.lm_head."
        renamed = {}
        for k, v in tensors.items():
            renamed[k.replace(old, new) if k.startswith(old) else k] = v
        tensors = renamed
        action = f"renamed {n_lm} lm_head keys ({old}* → {new}*)"
    else:
        action = f"kept {n_lm} lm_head keys as-is"

    out_zip = out_zip.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        save_file(tensors, stage / "adapter_model.safetensors")
        (stage / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        for name in TOKENIZER_FILES:
            src = ckpt / name
            if src.is_file():
                shutil.copy(src, stage / name)

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(stage.iterdir()):
                if p.is_file():
                    zf.write(p, p.name)

    size = out_zip.stat().st_size
    return {
        "zip_path": str(out_zip),
        "size_bytes": size,
        "action": action,
        "n_keys_before": n_before,
        "n_keys_after": len(tensors),
        "target_modules": cfg.get("target_modules"),
        "adapter_rank": rank,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, type=Path, help="LoRA checkpoint dir (contains adapter_config.json)")
    ap.add_argument("--out", required=True, type=Path, help="Output submission.zip path")
    ap.add_argument("--drop-lm-head", action="store_true", help="Strip lm_head LoRA keys + remove from target_modules (matches W4/step_200 shape)")
    ap.add_argument("--rename-lm-head", action="store_true", help="Rewrite lm_head keys to backbone.lm_head.* (if Kaggle rejects the default naming)")
    ap.add_argument("--allow-pretransformed", action="store_true", help="Override the safety check that refuses pre-transformed (local vLLM eval) adapter dirs. Only use if you have LB evidence the transformed form scores.")
    args = ap.parse_args()

    result = pack(
        args.ckpt, args.out,
        drop_lm_head=args.drop_lm_head,
        rename_lm_head=args.rename_lm_head,
        allow_pretransformed=args.allow_pretransformed,
    )

    print(f"[pack] {result['action']}")
    print(f"[pack] keys: {result['n_keys_before']} → {result['n_keys_after']}")
    print(f"[pack] target_modules: {sorted(result['target_modules'] or [])}")
    print(f"[pack] rank: {result['adapter_rank']}")
    print(f"[pack] wrote {result['zip_path']} ({result['size_bytes']:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
