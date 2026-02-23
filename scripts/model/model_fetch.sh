#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${HOME}/model_cache/llama"
MODEL_FILE="${MODEL_DIR}/llama-2-7b.Q4_K_M.gguf"
URL="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  echo "Model exists: $MODEL_FILE"
  ls -lh "$MODEL_FILE"
  exit 0
fi

echo "Downloading model to: $MODEL_FILE"
wget -O "$MODEL_FILE" "$URL"

echo "Downloaded:"
ls -lh "$MODEL_FILE"
