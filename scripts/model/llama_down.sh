#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/model/llama_down.sh
#
# Stops the llama container regardless of which model is running.
# Model size is read from model_config.sh for display purposes only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/model_config.sh"

MODEL_SIZE="unknown"
if [ -f "$CONFIG_FILE" ]; then
  # shellcheck source=model_config.sh
  source "$CONFIG_FILE"
fi

NAME="mccviahat-llama"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null
  echo "Stopped and removed: $NAME  (model=${MODEL_SIZE})"
else
  echo "Not running: $NAME"
fi
