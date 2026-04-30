"""Real ``TrainingClient`` implementations backed by HuggingFace + PEFT.

Mirrors the verified recipes from ``../nemotron-auto-research``:
  * SFT cross-entropy matches ``scripts/train_sft_lora.py``.
  * GSPO sequence-level + DAPO token-level objectives match
    ``scripts/gspo_update.py``.

The client is *eager*: the base model and LoRA adapter are loaded in the
constructor, and gradients live on-device until ``save_weights_for_sampler``
is called. Callers are expected to:

  1. ``client = backend.create_training_client(workspace, checkpoint)``
  2. loop over ``forward_backward(...) + optim_step(...)``
  3. ``ckpt = client.save_weights_for_sampler(name)``
  4. drop the client (``del client``) so CUDA memory can be reclaimed before
     the next stage (e.g. vLLM eval) boots.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any

from ....training.types import CheckpointRef
from ..base import (
    AdamParams,
    Datum,
    ForwardBackwardResult,
    OptimStepResult,
    TrainingClient,
)

logger = logging.getLogger(__name__)


class HFTrainingClient(TrainingClient):
    """Eager HF + PEFT training client.

    Supported ``loss_fn`` values:
      * ``"cross_entropy"`` — standard LM loss. Each ``Datum`` supplies
        ``model_input.tokens`` (input_ids) and ``loss_fn_inputs = {
        "attention_mask": List[int], "labels": List[int] (-100 masks)}``.
        ``loss_config`` optionally sets ``"accumulate": bool`` (default True).
      * ``"gspo"`` — sequence-level clipped objective. Each ``Datum`` supplies
        ``model_input.tokens = prompt_ids + completion_ids`` and
        ``loss_fn_inputs = {"logprobs_old": List[float],
        "advantage": float, "prompt_len": int}``.
        ``loss_config`` reads ``eps_low``, ``eps_high`` (both default to the
        GSPO Qwen3-30B values 3e-4 / 4e-4).
      * ``"dapo_token_level"`` — same Datum shape as gspo; token-level clip
        per GSPO paper DAPO variant.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        model_path: str,
        start_adapter_path: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        lr: float = 5e-5,
        device_map: str | dict | None = None,
    ) -> None:
        """HF + PEFT training client.

        ``device_map`` controls how the base model is placed across GPUs:
          * ``None`` (default): single-GPU, the entire model lands on
            ``cuda:0``. Matches the verified reference recipe.
          * ``"auto"``: HF/accelerate shards the model across all visible
            GPUs. Forward + backward flow through the shards. Uses every
            visible GPU, not just ``cuda:0``.
          * ``dict``: explicit layer→device mapping (advanced).

        When ``device_map`` is set, we skip the post-hoc ``.to("cuda:0")``
        since ``from_pretrained`` already placed the weights. ``cuda_input``
        in forward/backward paths is resolved via the first model parameter.
        """
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._root = Path(workspace_root)
        self._model_path = model_path
        self._start_adapter_path = start_adapter_path
        self._torch = torch
        self._step = 0
        self._lr = lr

        logger.info("[hf-client] loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        logger.info(
            "[hf-client] loading base model in bf16 (device_map=%r)", device_map
        )
        base = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if device_map is None:
            base = base.to("cuda:0")
        base.config.use_cache = False

        if start_adapter_path is not None and Path(start_adapter_path).is_dir():
            logger.info("[hf-client] loading start adapter from %s", start_adapter_path)
            self.model = PeftModel.from_pretrained(
                base, start_adapter_path, is_trainable=True
            )
        else:
            logger.info("[hf-client] attaching fresh LoRA adapter rank=%d", lora_rank)
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(
                    target_modules
                    or ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]
                ),
            )
            self.model = get_peft_model(base, lora_cfg)

        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.train()
        try:
            self.model.print_trainable_parameters()
        except Exception:
            pass

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._trainable = trainable
        # Fused AdamW matches the GSPO recipe; falls back to regular AdamW on
        # CPU (unit-test path).
        try:
            self.optimizer = torch.optim.AdamW(trainable, lr=lr, fused=True)
        except (RuntimeError, TypeError):
            self.optimizer = torch.optim.AdamW(trainable, lr=lr)
        self.optimizer.zero_grad()

    # ── TrainingClient protocol ─────────────────────────────────────────

    def forward_backward(
        self,
        batch: list[Datum],
        loss_fn: str,
        loss_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardResult:
        loss_config = loss_config or {}
        if loss_fn == "cross_entropy":
            return self._forward_backward_ce(batch, loss_config)
        if loss_fn in ("gspo", "dapo_token_level"):
            token_level = loss_fn == "dapo_token_level" or bool(
                loss_config.get("dapo_token_level", False)
            )
            return self._forward_backward_gspo(batch, loss_config, token_level=token_level)
        raise ValueError(f"Unknown loss_fn={loss_fn!r}")

    def optim_step(self, params: AdamParams) -> OptimStepResult:
        if params.learning_rate != self._lr:
            for pg in self.optimizer.param_groups:
                pg["lr"] = params.learning_rate
            self._lr = params.learning_rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._step += 1
        return OptimStepResult(step=self._step, learning_rate=self._lr)

    def save_state(self, name: str) -> CheckpointRef:
        # Full optimizer+model state isn't needed for our current flow; the
        # adapter-only snapshot via save_weights_for_sampler is sufficient.
        raise NotImplementedError(
            "HFTrainingClient.save_state is not implemented; "
            "use save_weights_for_sampler(name) to snapshot the adapter."
        )

    def save_weights_for_sampler(self, name: str) -> CheckpointRef:
        outdir = self._root / "checkpoints" / "adapters" / name
        outdir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(outdir))
        self.tokenizer.save_pretrained(str(outdir))
        if not (outdir / "adapter_config.json").is_file():
            raise RuntimeError(
                f"save_pretrained did not emit adapter_config.json under {outdir}"
            )
        return CheckpointRef(
            name=name,
            path=str(outdir),
            kind="adapter",
            metadata={"lr": self._lr, "step": self._step},
        )

    # ── Loss implementations ────────────────────────────────────────────

    def _forward_backward_ce(
        self, batch: list[Datum], loss_config: dict[str, Any]
    ) -> ForwardBackwardResult:
        torch = self._torch
        pad_id = int(self.tokenizer.pad_token_id)

        rows = []
        for d in batch:
            tokens = list(d.model_input.tokens)
            attn = d.loss_fn_inputs.get("attention_mask") or [1] * len(tokens)
            labels = d.loss_fn_inputs.get("labels")
            if labels is None:
                raise ValueError(
                    "cross_entropy loss requires Datum.loss_fn_inputs['labels']"
                )
            rows.append((tokens, list(attn), list(labels)))

        max_len = max(len(t) for t, _, _ in rows)
        input_ids, attn_mask, labels_arr = [], [], []
        for t, a, lab in rows:
            pad = max_len - len(t)
            input_ids.append(t + [pad_id] * pad)
            attn_mask.append(a + [0] * pad)
            labels_arr.append(lab + [-100] * pad)

        device = next(self.model.parameters()).device
        x = torch.tensor(input_ids, dtype=torch.long, device=device)
        am = torch.tensor(attn_mask, dtype=torch.long, device=device)
        lab = torch.tensor(labels_arr, dtype=torch.long, device=device)

        out = self.model(input_ids=x, attention_mask=am, labels=lab)
        loss = out.loss
        grad_accum = int(loss_config.get("grad_accum", 1))
        (loss / grad_accum).backward()
        return ForwardBackwardResult(
            loss=float(loss.detach()),
            extras={"batch_size": float(len(batch)), "loss_fn": 0.0},  # extras: floats only
        )

    def _forward_backward_gspo(
        self,
        batch: list[Datum],
        loss_config: dict[str, Any],
        *,
        token_level: bool,
    ) -> ForwardBackwardResult:
        """Sequence- or token-level clipped importance-sampling objective.

        Ported from ``gspo_update.py``. We process each Datum individually
        (no padding) — one rollout per datum is the natural granularity.
        The caller drives grad-accum by varying how many forward_backward
        calls occur before each optim_step.
        """
        torch = self._torch
        eps_low = float(loss_config.get("eps_low", 3e-4))
        eps_high = float(loss_config.get("eps_high", 4e-4))
        grad_accum = int(loss_config.get("grad_accum", 1))
        device = next(self.model.parameters()).device

        total_loss = 0.0
        total_s = 0.0
        total_clipped = 0.0
        for d in batch:
            full_ids = list(d.model_input.tokens)
            lp_old_list = d.loss_fn_inputs.get("logprobs_old")
            if lp_old_list is None:
                raise ValueError("gspo loss requires loss_fn_inputs['logprobs_old']")
            advantage = float(d.loss_fn_inputs.get("advantage", 0.0))
            prompt_len = int(d.loss_fn_inputs.get("prompt_len", 0))
            if prompt_len <= 0 or prompt_len >= len(full_ids):
                raise ValueError(
                    f"gspo prompt_len={prompt_len} is inconsistent with full length {len(full_ids)}"
                )

            x = torch.tensor([full_ids], dtype=torch.long, device=device)
            out = self.model(x)
            logits = out.logits  # [1, T, V]
            pred_logits = logits[0, prompt_len - 1 : -1, :]  # [C, V]
            compl_ids = torch.tensor(
                full_ids[prompt_len:], dtype=torch.long, device=device
            )
            if pred_logits.shape[0] != compl_ids.shape[0]:
                raise RuntimeError(
                    f"gspo logits/completion shape mismatch: "
                    f"{pred_logits.shape[0]} vs {compl_ids.shape[0]}"
                )
            lp_new = torch.nn.functional.log_softmax(
                pred_logits.float(), dim=-1
            ).gather(-1, compl_ids.unsqueeze(-1)).squeeze(-1)  # [C]
            lp_old = torch.tensor(lp_old_list, dtype=torch.float32, device=device)
            if lp_new.shape != lp_old.shape:
                raise RuntimeError(
                    f"gspo lp_new/lp_old shape mismatch: {lp_new.shape} vs {lp_old.shape}"
                )

            if token_level:
                log_ratio_tok = lp_new - lp_old
                s_tok = torch.exp(log_ratio_tok)
                s_tok_clipped = torch.clamp(
                    s_tok, min=1.0 - eps_low, max=1.0 + eps_high
                )
                obj = torch.min(s_tok * advantage, s_tok_clipped * advantage).mean()
                s_scalar = float(s_tok.mean())
                clipped_flag = float((s_tok != s_tok_clipped).any())
            else:
                log_ratio = (lp_new - lp_old).mean()
                s = torch.exp(log_ratio)
                s_clipped = torch.clamp(s, min=1.0 - eps_low, max=1.0 + eps_high)
                obj = torch.min(s * advantage, s_clipped * advantage)
                s_scalar = float(s)
                clipped_flag = float(s != s_clipped)

            loss = -(obj) / grad_accum
            loss.backward()
            total_loss += float(loss) * grad_accum
            total_s += s_scalar
            total_clipped += clipped_flag

        n = max(1, len(batch))
        return ForwardBackwardResult(
            loss=total_loss / n,
            extras={
                "mean_s": total_s / n,
                "clip_frac": total_clipped / n,
                "batch_size": float(n),
            },
        )

    # ── Teardown ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release GPU memory. Callers should call this before booting vLLM."""
        torch = self._torch
        try:
            del self.optimizer
        except Exception:
            pass
        try:
            del self.model
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover — best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


# Convenience: construct from workspace YAML files.

def build_hf_client_from_workspace(
    workspace: Any,
    checkpoint: CheckpointRef | None = None,
) -> HFTrainingClient:
    import yaml

    root = Path(workspace.root)

    def _load(p: Path) -> dict:
        if not p.exists():
            return {}
        try:
            with open(p) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    base_cfg = _load(root / "model" / "base.yaml")
    adapter_cfg = _load(root / "model" / "adapter.yaml")
    optimizer_cfg = _load(root / "train" / "optimizer.yaml")

    model_path = base_cfg.get("path") or os.environ.get("AE_BASE_MODEL_PATH")
    if not model_path:
        raise RuntimeError(
            "HFTrainingClient requires model/base.yaml::path or AE_BASE_MODEL_PATH"
        )

    start_adapter_path = checkpoint.path if checkpoint is not None else None

    return HFTrainingClient(
        workspace_root=root,
        model_path=str(model_path),
        start_adapter_path=start_adapter_path,
        lora_rank=int(adapter_cfg.get("rank", 16)),
        lora_alpha=int(adapter_cfg.get("alpha", 32)),
        lora_dropout=float(adapter_cfg.get("dropout", 0.05)),
        target_modules=list(
            adapter_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
            )
        ),
        lr=float(optimizer_cfg.get("lr", 5e-5)),
        # Resolution order for device_map:
        #   1. AE_HF_DEVICE_MAP env var
        #   2. adapter.yaml::device_map
        #   3. None (single-GPU, verified-recipe default)
        device_map=(
            os.environ.get("AE_HF_DEVICE_MAP")
            or adapter_cfg.get("device_map")
            or None
        ),
    )


__all__ = ["HFTrainingClient", "build_hf_client_from_workspace"]
