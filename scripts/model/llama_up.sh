#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/model/llama_up.sh [7b|70b]
MODEL_SIZE="${1:-70b}"

IMAGE="mccviahat-llama:dev"
NAME="mccviahat-llama"
PORT="8000"
MODEL_DIR="${HOME}/model_cache/llama"

case "$MODEL_SIZE" in
  7b)
    MODEL_PATH="/models/llama-2-7b.Q4_K_M.gguf"
    ;;
  70b)
    MODEL_PATH="/models/llama-3.1-70b.Q4_K_M.gguf"
    ;;
  *)
    echo "Usage: $0 [7b|70b]"
    exit 1
    ;;
esac

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

echo "Starting llama server container: $NAME"
docker run -d \
  --name "$NAME" \
  --cpuset-cpus "0-15" \
  -p "${PORT}:8000" \
  -v "${MODEL_DIR}:/models" \
  "$IMAGE" \
  /opt/llama.cpp/build/bin/llama-server \
    --host 0.0.0.0 --port 8000 \
    --model "$MODEL_PATH" \
    --override-kv tokenizer.ggml.eos_token_id=int:-1 \
  >/dev/null

echo "Started."
docker ps --filter "name=$NAME"
echo "PID: $(docker inspect -f '{{.State.Pid}}' "$NAME")"
