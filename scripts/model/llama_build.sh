#!/usr/bin/env bash
set -euo pipefail

IMAGE="mccviahat-llama:dev"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Building image: $IMAGE"
cd "$HERE/docker/llama_server"
# Always rebuild to pick up any updates
docker build -t "$IMAGE" .