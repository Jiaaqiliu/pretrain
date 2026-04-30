"""Concrete TrainingClient / SamplingClient implementations.

Shared between all TinkerLite backends (``single_node`` and ``elastic``).
Each submodule houses one client family:

- ``mock``  — in-memory fakes for unit tests and the smoke path.
- ``hf``    — real HuggingFace + PEFT-backed training client.
- ``vllm``  — real vLLM-backed sampling client for rollouts + eval.
"""
