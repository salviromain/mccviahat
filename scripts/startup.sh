#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────
# startup.sh — One-shot CloudLab node bootstrap
#
# Clones the repo, installs dependencies, fetches the model,
# builds the llama.cpp Docker image, and starts the server.
#
# Usage:
#   curl -sL https://raw.githubusercontent.com/salviromain/mccviahat/main/scripts/startup.sh | bash
#   # or
#   bash scripts/startup.sh
# ─────────────────────────────────────────────────────────

echo "══════════════════════════════════════════"
echo "  mccviahat — CloudLab node startup"
echo "══════════════════════════════════════════"

# 1. Clone repo (skip if already inside it)
if [ ! -f "scripts/startup.sh" ]; then
    echo "» Cloning repo..."
    git clone https://github.com/salviromain/mccviahat
    cd mccviahat
else
    echo "» Already inside mccviahat repo, skipping clone."
fi

# 2. Bootstrap node (install perf, docker, etc.)
echo "» Bootstrapping node..."
bash scripts/bootstrap_node.sh

# 3. Ensure current user is in docker group
#    newgrp replaces the current shell, so we use sg instead
#    to run the remaining commands within the docker group.
echo "» Activating docker group..."
if ! groups | grep -qw docker; then
    echo "  (running remaining steps under 'sg docker')"
    sg docker -c "
        set -euo pipefail
        echo '» Fetching model...'
        bash scripts/model_fetch.sh
        echo '» Building llama.cpp Docker image...'
        bash scripts/llama_build.sh
        echo '» Starting server...'
        bash scripts/reset_server.sh
        echo ''
        echo '══════════════════════════════════════════'
        echo '  ✓ Node is ready. Server is running.'
        echo '══════════════════════════════════════════'
    "
else
    echo "» Fetching model..."
    bash scripts/model_fetch.sh
    echo "» Building llama.cpp Docker image..."
    bash scripts/llama_build.sh
    echo "» Starting server..."
    bash scripts/reset_server.sh
    echo ""
    echo "══════════════════════════════════════════"
    echo "  ✓ Node is ready. Server is running."
    echo "══════════════════════════════════════════"
fi
