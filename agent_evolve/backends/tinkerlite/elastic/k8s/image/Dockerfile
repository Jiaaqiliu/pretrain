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

# Nemotron-H's modeling_nemotron_h.py relies on mamba_ssm's triton fast
# paths (selective_state_update, causal_conv1d_fn) for any reasonable
# generation speed. Without them, HF transformers silently falls back to
# a pure-Python state-space loop that runs ~100× slower. vLLM-on-this-
# cluster is not an option (triton kernel load fails under driver 570),
# so eval goes through HF and NEEDS these packages to be real.
#
# NGC 25.09 ships cuda-toolkit 13.0 but we pip-installed torch==2.10.0
# built for cu128. mamba-ssm's setup.py refuses to build with mismatched
# toolkits, so we install cuda-toolkit-12-8 side-by-side and point CUDA_HOME
# at it for the duration of this RUN. Both packages must build against
# the torch installed above (2.10.0) — `--no-build-isolation` keeps pip
# from pulling a fresh torch during compile.
# Use the AWS region-local Ubuntu mirror so apt transfers don't timeout
# against the Canonical-global mirror. Region is hardcoded to ap-southeast-3
# (the cluster's region). APT::Acquire::Retries handles transient flakes.
RUN printf 'APT::Acquire::Retries "5";\nAcquire::http::Timeout "60";\nAcquire::https::Timeout "60";\n' \
      > /etc/apt/apt.conf.d/99-retries \
 && sed -i 's|http://archive.ubuntu.com/ubuntu|http://ap-southeast-3.ec2.archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://ap-southeast-3.ec2.archive.ubuntu.com/ubuntu|g' \
      /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null || true \
 && apt-get update \
 && apt-get install -y --no-install-recommends wget gnupg ca-certificates \
 && wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      cuda-nvcc-12-8 \
      cuda-cudart-dev-12-8 \
      cuda-nvrtc-dev-12-8 \
 && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

# Install from git — the PyPI sdist for causal-conv1d <1.6 omits csrc/,
# which ninja needs to find the .cpp source file. Tagged v1.5.0.post8 has
# the full tree. Also need git for pip to clone; apt gets it.
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME_MAMBA=/usr/local/cuda-12.8
RUN PATH="${CUDA_HOME_MAMBA}/bin:${PATH}" \
    CUDA_HOME="${CUDA_HOME_MAMBA}" \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAMBA_FORCE_BUILD=TRUE \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    pip install --no-cache-dir --no-build-isolation \
      "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.0.post8" \
      "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.5"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_DISABLED=true \
    NCCL_DEBUG=WARN \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/fsx/zzsamshi/a-evolve

WORKDIR /fsx/zzsamshi/a-evolve
