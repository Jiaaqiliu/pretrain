#!/usr/bin/env bash
# Build + push the Unsloth trainer image for the nemo_mas_reasoner workspace.
# Thin wrapper: delegates to the platform builder but passes the workspace's
# Dockerfile.unsloth explicitly.
#
# Usage:
#   cd seed_workspaces/nemo_mas_reasoner/k8s/image
#   ./build_and_push.sh unsloth-v6    # tag to push (e.g. unsloth-v6, unsloth-latest)
set -euo pipefail

TAG="${1:-unsloth-latest}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PLATFORM_BUILDER="$(cd "$HERE/../../../../agent_evolve/backends/tinkerlite/elastic/k8s/image" && pwd)/build_and_push.sh"

if [[ ! -x "$PLATFORM_BUILDER" ]]; then
  echo "error: platform builder not found at $PLATFORM_BUILDER" >&2
  exit 2
fi

# Stage our Dockerfile in the platform image dir so the platform builder can
# find it (it looks for $DOCKERFILE relative to its own dir).
cp "$HERE/Dockerfile.unsloth" "$(dirname "$PLATFORM_BUILDER")/Dockerfile.unsloth"
trap 'rm -f "$(dirname "$PLATFORM_BUILDER")/Dockerfile.unsloth"' EXIT

DOCKERFILE=Dockerfile.unsloth "$PLATFORM_BUILDER" "$TAG"
