#!/usr/bin/env bash
# scripts/model/reset_server.sh
#
# Restart the llama.cpp container between runs to clear the KV cache.
# Usage: bash scripts/model/reset_server.sh [7b|70b]

set -euo pipefail
MODEL_SIZE="${1:-70b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Stopping llama server ($MODEL_SIZE)..."
bash "$SCRIPT_DIR/llama_down.sh" "$MODEL_SIZE"

echo ">>> Starting llama server ($MODEL_SIZE)..."
bash "$SCRIPT_DIR/llama_up.sh" "$MODEL_SIZE"

echo ">>> Waiting for server to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo ">>> Server is up and healthy (attempt $i)"
    exit 0
  fi
  sleep 2
done

echo "ERROR: Server did not become healthy within 60 seconds"
exit 1