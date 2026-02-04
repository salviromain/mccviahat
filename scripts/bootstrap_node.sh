
#!/usr/bin/env bash
set -euo pipefail

# --- perf permissions (required for HAT collection) ---
sudo sysctl -w kernel.perf_event_paranoid=1
sudo sysctl -w kernel.kptr_restrict=0

# Persist settings across reboot (best-effort)
if [ -d /etc/sysctl.d ]; then
  sudo tee /etc/sysctl.d/99-mccviahat-perf.conf >/dev/null <<'EOC'
kernel.perf_event_paranoid = 1
kernel.kptr_restrict = 0
EOC
  sudo sysctl --system >/dev/null || true
fi

# --- base system setup ---
sudo apt-get update -y
sudo apt-get install -y git curl ca-certificates

# --- Docker + Compose ---
sudo apt-get install -y docker.io
sudo apt-get install -y docker-compose-v2 || sudo apt-get install -y docker-compose

sudo systemctl enable --now docker

# Ensure docker group exists, then add user
sudo groupadd -f docker
sudo usermod -aG docker "$USER"

echo
echo "Docker installed."
echo "IMPORTANT: run 'newgrp docker' (or log out/in) before using docker without sudo."
echo "Test: docker run --rm hello-world"
echo "Compose: docker compose version (or docker-compose --version)"
