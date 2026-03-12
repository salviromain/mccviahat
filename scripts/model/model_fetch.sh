#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/model/model_fetch.sh [7b|70b]
MODEL_SIZE="${1:-70b}"

MODEL_DIR="${HOME}/model_cache/llama"
mkdir -p "$MODEL_DIR"

case "$MODEL_SIZE" in
  7b)
    MODEL_FILE="${MODEL_DIR}/llama-2-7b.Q4_K_M.gguf"
    URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
    ;;
  70b)
    MODEL_FILE="${MODEL_DIR}/llama-3.1-70b.Q4_K_M.gguf"
    URL="https://huggingface.co/bartowski/Llama-3.1-70B-Instruct-GGUF/resolve/main/Llama-3.1-70B-Instruct-Q4_K_M.gguf"
    ;;
  *)
    echo "Usage: $0 [7b|70b]"
    exit 1
    ;;
esac

if [ -f "$MODEL_FILE" ]; then
  echo "Model exists: $MODEL_FILE"
  ls -lh "$MODEL_FILE"
  exit 0
fi

echo "Downloading $MODEL_SIZE model to: $MODEL_FILE"
wget -O "$MODEL_FILE" "$URL"

echo "Downloaded:"
ls -lh "$MODEL_FILE"
