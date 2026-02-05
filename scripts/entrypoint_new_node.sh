#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "1) Bootstrap node (perf sysctls + docker install)"
bash "$HERE/scripts/bootstrap_node.sh"

echo
echo "NOTE: If docker commands fail with permission denied, run:"
echo "  newgrp docker"
echo "or log out/in to refresh group membership."
echo

echo "2) Build llama image (if missing)"
bash "$HERE/scripts/llama_build.sh"

echo "3) Fetch model (if missing)"
bash "$HERE/scripts/model_fetch.sh"

echo "4) Start llama server"
bash "$HERE/scripts/llama_up.sh"

echo
echo "Baseline ready. Next run:"
echo "  bash scripts/run_one.sh \"Say hello in one sentence.\""
