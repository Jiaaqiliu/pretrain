"""Real ``SamplingClient`` implementation backed by vLLM + LoRA.

Used by the GSPO rollout phase to sample G completions per prompt and
capture per-token logprobs under the *old* policy. Matches the engine
kwargs used by ``eval_worker._run_vllm_eval`` and the sampling kwargs in
``../nemotron-auto-research/scripts/gspo_rollout.py``.

Notes:
  * ``sample(prompts, params)`` expects the caller to have tokenized +
    chat-templated the prompts already. The tokens carried by ``Prompt``
    are NOT used by vLLM's generate (which re-tokenizes strings) — the
    wrapper stashes the matching prompt string via ``set_prompt_strings``.
  * Per-token logprobs are returned on ``Sample`` via a non-protocol
    attribute ``_logprobs_per_token: list[float]``. Callers that need
    rollout logprobs access this attribute explicitly.
  * ``compute_logprobs`` is intentionally a stub: GSPO's rollout phase
    already emits lp_old in the same call as generation, so we don't need
    a separate logprob-scoring path today. Raising NotImplementedError
    makes mis-use loud; flip to a real implementation when needed.
"""

from __future__ import annotations

import logging
from typing import Any

from ..base import (
    Logprobs,
    Prompt,
    Sample,
    SampleResponse,
    SamplingClient,
    SamplingParams,
    TokenSequence,
)

logger = logging.getLogger(__name__)


class VLLMSamplingClient(SamplingClient):
    def __init__(
        self,
        *,
        model_path: str,
        adapter_path: str,
        adapter_name: str = "candidate",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        max_lora_rank: int = 32,
        max_num_seqs: int = 128,
        gpu_memory_utilization: float = 0.85,
        seed: int = 0,
        enable_prefix_caching: bool = True,
    ) -> None:
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        engine_kwargs = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            max_model_len=max_model_len,
            max_lora_rank=max_lora_rank,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=True,
            dtype="auto",
            trust_remote_code=True,
        )
        logger.info(
            "[vllm-sampling] booting engine: model=%s tp=%d max_lora_rank=%d",
            model_path,
            tensor_parallel_size,
            max_lora_rank,
        )
        self.llm = LLM(**engine_kwargs)
        self._lora_request = LoRARequest(adapter_name, 1, adapter_path)
        self._model_path = model_path
        # prompt_text lookup keyed by id(prompt) — populated by callers to
        # avoid re-tokenizing. If missing we fall back to Prompt.tokens.
        self._prompt_text: dict[int, str] = {}

    # ── Helpers ─────────────────────────────────────────────────────────

    def set_prompt_strings(self, prompts: list[Prompt], texts: list[str]) -> None:
        assert len(prompts) == len(texts)
        for p, t in zip(prompts, texts):
            self._prompt_text[id(p)] = t

    # ── SamplingClient protocol ─────────────────────────────────────────

    def sample(
        self, prompts: list[Prompt], params: SamplingParams
    ) -> list[SampleResponse]:
        from vllm import SamplingParams as VSamplingParams

        vparams = VSamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            n=max(1, params.n),
            logprobs=0,
        )
        prompt_strs: list[str] = []
        for p in prompts:
            text = self._prompt_text.get(id(p))
            if text is None:
                raise ValueError(
                    "VLLMSamplingClient.sample requires prompt text; call "
                    "set_prompt_strings(prompts, texts) before sample()."
                )
            prompt_strs.append(text)

        outputs = self.llm.generate(
            prompt_strs, sampling_params=vparams, lora_request=self._lora_request
        )

        responses: list[SampleResponse] = []
        for out in outputs:
            samples: list[Sample] = []
            for comp in out.outputs:
                token_ids = list(comp.token_ids)
                lp_per_token: list[float] = []
                lp_sum = 0.0
                if comp.logprobs is not None:
                    for tid, lp_dict in zip(token_ids, comp.logprobs):
                        lp = float(lp_dict[tid].logprob)
                        lp_per_token.append(lp)
                        lp_sum += lp
                mean_lp = (lp_sum / len(token_ids)) if token_ids else 0.0
                sample = Sample(
                    tokens=token_ids,
                    text=comp.text,
                    logprob=mean_lp,
                )
                # Non-protocol attribute — callers opt into it explicitly.
                sample._logprobs_per_token = lp_per_token  # type: ignore[attr-defined]
                samples.append(sample)
            responses.append(SampleResponse(samples=samples))
        return responses

    def compute_logprobs(self, sequences: list[TokenSequence]) -> list[Logprobs]:
        raise NotImplementedError(
            "VLLMSamplingClient.compute_logprobs is not implemented — "
            "GSPO rollouts already emit logprobs_old during sample(). "
            "Use HFTrainingClient to compute on-policy logprobs during update."
        )

    # ── Teardown ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Best-effort teardown; vLLM engines hold CUDA state that the OS
        reclaims on subprocess exit. In-process teardown is brittle — if you
        need guaranteed reclamation, run this client inside a subprocess.
        """
        import gc

        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ["VLLMSamplingClient"]
