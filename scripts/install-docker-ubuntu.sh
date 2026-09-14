#!/usr/bin/env bash
# Docker Engine et plugins officiels : https://docs.docker.com/engine/install/ubuntu/
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Executer ce script avec sudo." >&2
    exit 1
fi
. /etc/os-release
if [[ "$ID" != ubuntu ]]; then
    echo "Ce script est reserve a Ubuntu." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-remove ca-certificates curl
install -d -m 0755 /etc/apt/keyrings
curl -fsSL --retry 3 https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod 0644 /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt-get update
apt-get install -y --no-remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker.service containerd.service
usermod -aG docker pc

docker --version
docker compose version
docker buildx version
docker run --rm hello-world
echo 'DOCKER_INSTALL_OK'
