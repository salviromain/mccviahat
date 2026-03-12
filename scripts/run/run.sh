#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run/run.sh [clemson|utah]

if [ $# -lt 1 ]; then
  echo "Usage: $0 [clemson|utah]"
  exit 1
fi

MACHINE="$1"

case "$MACHINE" in
  clemson)
    echo ">>> Running on clemson (64 cores)"
    rm -rf ~/mccviahat/runs/*
    python scripts/run/run_prompts_isolated.py \
      --json prompts/20base/mixed_independent.json \
      --llm_cpus 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62 \
      --perf_cpu 1
    ;;
  utah)
    echo ">>> Running on utah (56 cores)"
    rm -rf ~/mccviahat/runs/*
    python scripts/run/run_prompts_isolated.py \
      --json prompts/20base/mixed_independent.json \
      --llm_cpus 0-26,28-54 \
      --perf_cpu 27,55
    ;;
  *)
    echo "Error: Unknown machine '$MACHINE'"
    echo "Usage: $0 [clemson|utah]"
    exit 1
    ;;
esac
