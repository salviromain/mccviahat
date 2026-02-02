#!/usr/bin/env bash
set -euo pipefail

# Run the dockerized LLM server on the node
# Model weights will be cached in ~/model_cache on the host
mkdir -p ~/model_cache

docker run --rm \
  -p 8000:8000 \
  -v ~/model_cache:/cache \
  -e MODEL_ID=gpt2 \
  mccviahat-llm:dev
