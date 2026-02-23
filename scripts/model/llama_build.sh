#!/usr/bin/env bash
set -euo pipefail

IMAGE="mccviahat-llama:dev"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image exists: $IMAGE"
  exit 0
fi

echo "Building image: $IMAGE"
cd "$HERE/docker/llama_server"
docker build -t "$IMAGE" .