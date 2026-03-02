#!/usr/bin/env bash
set -euo pipefail

NAME="mccviahat-llama"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null
  echo "Stopped/removed: $NAME"
else
  echo "Not found: $NAME"
fi
