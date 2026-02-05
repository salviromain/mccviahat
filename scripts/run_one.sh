#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_one.sh "Say hello in one sentence." [duration] [interval]
PROMPT="${1:-Say hello in one sentence.}"
DURATION="${2:-10}"
INTERVAL="${3:-0.2}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="mccviahat-llama"

# Ensure server is running
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "ERROR: container '$NAME' not running. Start it with: bash scripts/llama_up.sh"
  exit 1
fi

PID="$(docker inspect -f '{{.State.Pid}}' "$NAME")"
if [ -z "$PID" ] || [ "$PID" = "0" ]; then
  echo "ERROR: could not resolve PID for container '$NAME'"
  exit 1
fi

RUN_ID="$(date +%Y-%m-%dT%H-%M-%S)_${NAME}"
OUT_DIR="${HERE}/runs/${RUN_ID}"
mkdir -p "$OUT_DIR"

# Record metadata
{
  echo "run_id: $RUN_ID"
  echo "timestamp: $(date -Is)"
  echo "host: $(hostname)"
  echo "container: $NAME"
  echo "pid: $PID"
  echo "duration: $DURATION"
  echo "interval: $INTERVAL"
  echo "prompt: $PROMPT"
} > "${OUT_DIR}/meta.txt"

# Start sampler in background
CSV="${OUT_DIR}/proc_sample.csv"
python3 "${HERE}/collectors/proc_sampler.py" \
  --pid "$PID" \
  --out "$CSV" \
  --interval "$INTERVAL" \
  --duration "$DURATION" &
SAMP_PID="$!"

# Give sampler a moment to start
sleep 0.2

# Trigger inference (store response)
RESP="${OUT_DIR}/response.json"
curl -s http://localhost:8000/completion \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"${PROMPT//\"/\\\"}\",\"n_predict\":40}" > "$RESP" || true

# Wait for sampler to finish
wait "$SAMP_PID" || true

echo "Wrote:"
echo "  $CSV"
echo "  $RESP"
echo "  ${OUT_DIR}/meta.txt"
