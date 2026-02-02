#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y git curl ca-certificates

# Docker (simple Ubuntu packages)
sudo apt-get install -y docker.io docker-compose-plugin

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo
echo "Docker installed."
echo "IMPORTANT: run 'newgrp docker' (or log out/in) before using docker without sudo."
echo "Test: docker run --rm hello-world"
