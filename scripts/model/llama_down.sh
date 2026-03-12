#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/model/llama_down.sh [7b|70b]
MODEL_SIZE="${1:-70b}"

NAME="mccviahat-llama"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null
  echo "Stopped/removed: $NAME ($MODEL_SIZE)"
else
  echo "Not found: $NAME"
fi
