#!/usr/bin/env bash
# Procedure officielle NVIDIA Container Toolkit pour Ubuntu.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Executer avec sudo.' >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-remove ca-certificates curl gnupg2
install -d -m 0755 /usr/share/keyrings
curl -fsSL --retry 3 https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL --retry 3 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y --no-remove nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
if [[ -f /etc/docker/daemon.json ]]; then
    cp -a /etc/docker/daemon.json "/etc/docker/daemon.json.before-nvidia-$(date +%Y%m%d%H%M%S)"
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
echo 'NVIDIA_DOCKER_OK'
