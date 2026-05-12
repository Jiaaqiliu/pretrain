# Legacy scripts

Frozen snapshots of the scripts used during initial Unsloth bring-up. None
are live; future runs use `../k8s/entries/train_unsloth.py` via `../k8s/submit.sh`.

Kept here for reference when someone needs to see "how did we get here"
or reproduce an earlier result.

## What each script was

| file | purpose | status |
|---|---|---|
| `unsloth_smoke.py` | 1-GPU Unsloth load + LoRA wrap + 1 forward, on host EC2 | passed |
| `unsloth_smoke2.py` | + huikang patches (lm_head, fp32 LoRA, Mamba fast path) + forward+backward | passed |
| `unsloth_ddp_smoke.py` | 8-GPU DDP accelerate smoke — 2 optimizer steps on tiny corpus | passed on host, failed on k8s (driver 570) |
| `unsloth_w4_1gpu.py` | Early 1-GPU attempt using SFTTrainer (pre-huikang recipe) | superseded by next |
| `unsloth_huikang_1gpu.py` | **The proven-working local-EC2 training** — full custom loop, CCE, MoE tying, huikang literal | superseded by `k8s/entries/train_unsloth.py` |
| `unsloth_huikang_entry.py` | First k8s pod entry — had Trainer patches to work around `_old_compute_loss=None` issue | dead; driver upgrade made it unnecessary |
| `unsloth_huikang_1gpu_entry.py` | Revised k8s 1-GPU entry | dead; driver upgrade + canonical path superseded |
| `unsloth_smoke_entry.py` | Entry for the 8-shard-on-1-node smoke that proved parallel Unsloth works | dead; smoke complete, pattern absorbed |

## Why we went this direction

1. Driver 570 on k8s nodes couldn't load Mamba-SSM's Triton SASS binaries, but host (driver 580) could. Forced local development until the cluster nodegroup was upgraded.
2. Unsloth's SFTTrainer path hit a `_old_compute_loss=None` gotcha under certain import orders. Fixed by switching to huikang's custom training loop (no Trainer at all) in `unsloth_huikang_1gpu.py`.
3. Final canonical form is a single parameterized entry in `../k8s/entries/train_unsloth.py` that the `submit.sh` CLI invokes.
