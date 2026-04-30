"""Generate tiny-scale `.ddp_config.json` files for k8s smoke jobs.

Writes each kind (sft / gspo) to a dedicated directory on FSx so pods can
read it. Deliberately uses the same ``common_cfg`` builders as production
so smoke exercises the identical cfg schema.

Usage:
    python make_smoke_cfg.py <kind> <out_root>
where kind in {sft, gspo}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/fsx/zzsamshi/a-evolve")
sys.path.insert(0, str(REPO))

from agent_evolve.backends.tinkerlite.common_cfg import build_gspo_cfg, build_sft_cfg  # noqa: E402


class _WS:
    def __init__(self, root: Path):
        self.root = str(root)


def _load_yaml(p: Path) -> dict:
    import yaml
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    kind, out_root = sys.argv[1], Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)

    seed = REPO / "seed_workspaces" / "nemotron_reasoner"
    # We don't fork the workspace for smoke — pods read the seed directly and
    # write outputs to a smoke-specific dir under out_root.
    ws = _WS(seed)

    base = _load_yaml(seed / "model" / "base.yaml")
    adapter = _load_yaml(seed / "model" / "adapter.yaml")
    opt = _load_yaml(seed / "train" / "optimizer.yaml")
    batch = _load_yaml(seed / "train" / "batching.yaml")

    outdir = out_root / f"{kind}_adapter"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_path = outdir / ".ddp_config.json"
    result_path = outdir / ".ddp_result.json"

    if kind == "sft":
        stage = {
            "name": "sft_smoke", "epochs": 1, "max_steps": 1,
            "loss": "cross_entropy", "seed": 42,
        }
        # Tiny batching — we just need to prove the worker runs end-to-end.
        batch = {**batch, "per_device_bs": 1, "grad_accum": 1,
                 "max_seq_len": 512, "log_every": 1}
        cfg = build_sft_cfg(
            ws, stage,
            base_cfg=base, adapter_cfg=adapter,
            optimizer_cfg=opt, batching_cfg=batch,
            outdir=outdir, result_path=result_path,
            budget_seconds=600,
        )
    elif kind == "gspo":
        # Need a tiny synthetic rollouts.jsonl for the worker to consume.
        rollouts_path = outdir / "rollouts.jsonl"
        if not rollouts_path.is_file():
            # Fake a handful of rollout records in the shape the worker expects.
            # See train_worker_ddp._run_gspo for fields.
            import random
            random.seed(0)
            rows = []
            for i in range(4):
                prompt_ids = [random.randint(1, 1000) for _ in range(32)]
                completion_tokens = [random.randint(1, 1000) for _ in range(16)]
                logprobs_old = [random.uniform(-5.0, -0.01) for _ in completion_tokens]
                rows.append({
                    "pid": f"smoke_{i}",
                    "prompt_ids": prompt_ids,
                    "completion_tokens": completion_tokens,
                    "logprobs_old": logprobs_old,
                    "advantage": 0.1 * (i - 1.5),
                })
            rollouts_path.write_text("\n".join(json.dumps(r) for r in rows))

        stage = {"name": "rl_gspo_smoke", "seed": 11}
        gspo = {
            "lr": 1e-5, "epochs": 1, "grad_accum": 1,
            "eps_low": 3e-4, "eps_high": 4e-4,
            "dapo_token_level": False, "max_steps": 1, "log_every": 1, "seed": 11,
        }
        cfg = build_gspo_cfg(
            ws, stage,
            base_cfg=base, adapter_cfg=adapter,
            optimizer_cfg=opt,
            rollouts_path=rollouts_path,
            start_adapter_path="",  # will let worker init fresh
            gspo_cfg=gspo,
            outdir=outdir, result_path=result_path,
        )
    else:
        print(f"unknown kind: {kind}", file=sys.stderr)
        return 2

    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {cfg_path}")
    print(f"result will land at {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
