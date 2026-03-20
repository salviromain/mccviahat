#!/usr/bin/env bash
# scripts/model/reset_server.sh
#
# Restart the llama.cpp container between runs to clear the KV cache.
# All timeouts are read from model_config.sh (written by model_fetch.sh).
#
# Usage: bash scripts/model/reset_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/model_config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: model_config.sh not found."
  echo "Run first: bash scripts/model/model_fetch.sh [7b|70b]"
  exit 1
fi

# shellcheck source=model_config.sh
source "$CONFIG_FILE"

echo ">>> Stopping llama server (model=${MODEL_SIZE})..."
bash "$SCRIPT_DIR/llama_down.sh"

echo ">>> Starting llama server (model=${MODEL_SIZE})..."
bash "$SCRIPT_DIR/llama_up.sh"

echo ">>> Waiting for server to be ready (up to ${HEALTH_RETRIES}x${HEALTH_SLEEP}s)..."
for i in $(seq 1 "${HEALTH_RETRIES}"); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo ">>> Server is up and healthy (attempt ${i})"
    exit 0
  fi
  echo "    attempt ${i}/${HEALTH_RETRIES} — not ready yet, sleeping ${HEALTH_SLEEP}s..."
  sleep "${HEALTH_SLEEP}"
done

echo "ERROR: Server did not become healthy within $((HEALTH_RETRIES * HEALTH_SLEEP))s"
exit 1
