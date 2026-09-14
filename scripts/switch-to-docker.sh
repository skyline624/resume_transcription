#!/usr/bin/env bash
# A executer apres validation des conteneurs sur le port temporaire 18000.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
compose=(docker compose -f docker-compose.studio-linux.yml)
curl -fsS http://127.0.0.1:18000/health >/dev/null
"${compose[@]}" config --quiet

restore_native() {
    result=$?
    trap - ERR
    echo 'La bascule a echoue : restauration du service natif.' >&2
    "${compose[@]}" down || true
    systemctl --user enable --now resume-transcription-ollama resume-transcription
    exit "$result"
}
trap restore_native ERR

systemctl --user disable --now resume-transcription resume-transcription-ollama
TRANSCRIPTION_PORT=8000 "${compose[@]}" up -d --wait --wait-timeout 600
mkdir -p runtime/verification-docker
curl -fsS http://127.0.0.1:8000/health > runtime/verification-docker/health-production.json
trap - ERR
echo 'DOCKER_CUTOVER_OK'
"${compose[@]}" ps
