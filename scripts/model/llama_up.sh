#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/model/llama_up.sh
#
# All parameters are read from model_config.sh (written by model_fetch.sh).
# To switch models: bash scripts/model/model_fetch.sh [7b|70b]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/model_config.sh"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: model_config.sh not found."
  echo "Run first: bash scripts/model/model_fetch.sh [7b|70b]"
  exit 1
fi

# shellcheck source=model_config.sh
source "$CONFIG_FILE"

IMAGE="mccviahat-llama:dev"
NAME="mccviahat-llama"
PORT="8000"
N_THREADS=$(nproc)


# If already running, do nothing
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Already running: $NAME"
  docker ps --filter "name=$NAME"
  exit 0
fi

# If an old stopped container exists, remove it
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm "$NAME" >/dev/null
fi

echo "Starting llama server: $NAME  model=${MODEL_SIZE}  ctx=${CTX_SIZE}  cpuset=${CPUSET}"
docker run -d \
  --name "$NAME" \
  --cpuset-cpus "${CPUSET}" \
  -p "${PORT}:8000" \
  -v "${MODEL_DIR}:/models" \
  "$IMAGE" \
  /opt/llama.cpp/build/bin/llama-server \
    --host 0.0.0.0 --port 8000 \
    --model "${MODEL_PATH_IN_CONTAINER}" \
    --ctx-size "${CTX_SIZE}" \
    --threads "${N_THREADS}" \
    --threads-batch "${N_THREADS}" \
    --parallel 1 \
    --override-kv tokenizer.ggml.eos_token_id=int:-1 \
  >/dev/null

echo "Started."
docker ps --filter "name=$NAME"
echo "PID: $(docker inspect -f '{{.State.Pid}}' "$NAME")"
