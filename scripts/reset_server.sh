#!/usr/bin/env bash
# scripts/reset_server.sh
#
# Restart the llama.cpp container between runs to clear the KV cache.
# This ensures each run starts cold (no cached prompts), so prompt
# processing times are comparable across conditions.
#
# Usage:  bash scripts/reset_server.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ">>> Stopping llama server..."
bash "$SCRIPT_DIR/llama_down.sh"

echo ">>> Starting llama server..."
bash "$SCRIPT_DIR/llama_up.sh"

echo ">>> Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo ">>> Server is up and healthy (attempt $i)"
        exit 0
    fi
    sleep 2
done

echo "ERROR: Server did not become healthy within 60 seconds"
exit 1
