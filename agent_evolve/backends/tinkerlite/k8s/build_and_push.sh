#!/usr/bin/env bash
# Build + push the K8sTinkerLiteBackend trainer image to ECR.
# Tag is derived from git SHA + "-dirty" if there are unstaged changes.
# Usage:
#   cd agent_evolve/backends/tinkerlite/k8s
#   ./build_and_push.sh                  # build + push with auto tag
#   ./build_and_push.sh latest           # build + push with explicit tag
#   AE_ECR_REGION=us-west-2 ./build_and_push.sh   # override region

set -euo pipefail

REGION="${AE_ECR_REGION:-ap-southeast-3}"
ACCOUNT="${AE_ECR_ACCOUNT:-801953956576}"
REPO="${AE_ECR_REPO:-zzsamshi/a-evolve}"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

HERE="$(cd "$(dirname "$0")" && pwd)"
AE_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
cd "${AE_ROOT}"

if [[ $# -ge 1 ]]; then
  TAG="$1"
else
  SHA="$(git rev-parse --short HEAD)"
  if ! git diff --quiet || ! git diff --cached --quiet; then
    TAG="${SHA}-dirty-$(date +%Y%m%d%H%M%S)"
  else
    TAG="${SHA}"
  fi
fi

IMAGE="${REGISTRY}/${REPO}:${TAG}"
echo "[build] building ${IMAGE}"
echo "[build] context=${AE_ROOT}"
echo "[build] dockerfile=${HERE}/Dockerfile"

aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

# Use the repo root as context but don't COPY anything in — the Dockerfile
# doesn't reference local files. We set context to the Dockerfile's dir to
# keep the build upload tiny.
docker build -t "${IMAGE}" -f "${HERE}/Dockerfile" "${HERE}"

echo "[build] pushing ${IMAGE}"
docker push "${IMAGE}"

# Also tag latest so consumers that don't pin can still grab the newest.
LATEST="${REGISTRY}/${REPO}:latest"
docker tag "${IMAGE}" "${LATEST}"
docker push "${LATEST}"

echo
echo "pushed: ${IMAGE}"
echo "pushed: ${LATEST}"
echo
echo "use in K8sTinkerLiteBackend via image=... kwarg, or:"
echo "  export AE_K8S_IMAGE=${IMAGE}"
