# Trainer image for K8sTinkerLiteBackend.
#
# Deliberately thin: NO application code is COPY'd in. The a-evolve
# repo lives on an FSx PersistentVolume and is mounted into every pod at
# /fsx/zzsamshi/a-evolve — iteration is "save file -> kubectl create Job"
# with no rebuild.
#
# Versions are pinned to match the host venv at
# /fsx/zzsamshi/nemotron-auto-research/.venv so the k8s path and local
# subprocess produce byte-identical results:
#   torch         2.10.0
#   transformers  4.57.6
#   peft          0.19.1
#   vllm          0.19.0
#   accelerate    1.13.0
#   datasets      4.8.4
#
# To build + push:
#   cd agent_evolve/backends/tinkerlite/k8s
#   ./build_and_push.sh

FROM nvcr.io/nvidia/pytorch:25.09-py3

# Replace the NGC-bundled torch with the version the host venv uses.
# (NGC 25.09 ships torch ~2.8; we need 2.10 for the worker code path.)
RUN pip install --no-cache-dir --upgrade pip \
 && pip uninstall -y \
      torch torchvision torchaudio torchao \
      flash-attn flash-attn-2 flashinfer flashinfer-python \
      || true \
 && pip install --no-cache-dir \
      "torch==2.10.0" \
      "transformers==4.57.6" \
      "peft==0.19.1" \
      "accelerate==1.13.0" \
      "datasets==4.8.4" \
      "vllm==0.19.0" \
      "kubernetes>=31.0" \
      "pyyaml>=6.0" \
      "numpy<3.0"

# Nemotron-H's modeling_nemotron_h.py unconditionally imports
# `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn` and raises
# ImportError otherwise. The real mamba-ssm wheel needs CUDA-specific
# build flags. The host venv ships a tiny pure-PyTorch shim that covers
# the one symbol the model actually uses — we bake the same shim into
# the image so k8s pods load the model the same way.
COPY mamba_ssm_shim/ /usr/local/lib/python3.12/dist-packages/mamba_ssm/

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_DISABLED=true \
    NCCL_DEBUG=WARN \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/fsx/zzsamshi/a-evolve

WORKDIR /fsx/zzsamshi/a-evolve
