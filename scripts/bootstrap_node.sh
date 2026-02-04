#!/usr/bin/env bash
set -euo pipefail

# --- perf permissions (required for HAT collection) ---
# Allow user-space access to CPU performance counters
sudo sysctl -w kernel.perf_event_paranoid=1
# Allow kernel symbol access for profiling tools
sudo sysctl -w kernel.kptr_restrict=0

# Persist settings across reboot (best-effort on CloudLab)
if [ -d /etc/sysctl.d ]; then
  sudo tee /etc/sysctl.d/99-mccviahat-perf.conf >/dev/null <<'EOF'
kernel.perf_event_paranoid = 1
kernel.kptr_restrict = 0
EOF
  sudo sysctl --system >/dev/null || true
fi

# --- base system setup ---
sudo apt-get update -y
sudo apt-get install -y git curl ca-certificates

# --- Docker (simple Ubuntu packages) ---
sudo apt-get install -y docker.io docker-compose-plugin

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo
echo "Docker installed."
echo "IMPORTANT: run 'newgrp docker' (or log out/in) before using docker without sudo."
echo "Test: docker run --rm hello-world"
