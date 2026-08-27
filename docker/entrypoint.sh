#!/usr/bin/env bash
set -euo pipefail

MAIN_PYTHON=/opt/venv-main/bin/python
QWEN_PYTHON=/opt/venv-qwen/bin/python
WORKER_SOCKET="${TTS_WORKER_SOCKET:-/run/qwen-tts/worker.sock}"
worker_pid=""
main_pid=""

cleanup() {
    trap - EXIT
    set +e
    [[ -n "$main_pid" ]] && kill -TERM "$main_pid" 2>/dev/null
    [[ -n "$worker_pid" ]] && kill -TERM "$worker_pid" 2>/dev/null
    [[ -n "$main_pid" ]] && wait "$main_pid" 2>/dev/null
    [[ -n "$worker_pid" ]] && wait "$worker_pid" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 143' TERM INT

echo "[entrypoint] Vérification de l'environnement CUDA…"
"$MAIN_PYTHON" - <<'PY'
import torch
print(f"  torch           : {torch.__version__}")
print(f"  cuda disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu             : {torch.cuda.get_device_name(0)}")
    libre, total = torch.cuda.mem_get_info()
    print(f"  vram            : {libre // 1024**2} Mo libres / {total // 1024**2} Mo")
PY

echo "[entrypoint] Version de ffmpeg :"
ffmpeg -version 2>&1 | head -1 | sed 's/^/  /'

if [[ "${ENABLE_TTS:-true}" == "true" ]]; then
    mkdir -p "$(dirname "$WORKER_SOCKET")"
    rm -f -- "$WORKER_SOCKET"
    echo "[entrypoint] Vérification des checkpoints Qwen…"
    "$QWEN_PYTHON" -m qwen_tts_worker.download
    echo "[entrypoint] Démarrage du worker Qwen privé…"
    "$QWEN_PYTHON" -m qwen_tts_worker.main &
    worker_pid=$!
    ready=false
    for _ in $(seq 1 600); do
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            echo "[entrypoint] Le worker Qwen s'est arrêté avant d'être prêt." >&2
            exit 1
        fi
        if curl --silent --fail --unix-socket "$WORKER_SOCKET" http://localhost/health >/dev/null; then
            ready=true
            break
        fi
        sleep 1
    done
    if [[ "$ready" != "true" ]]; then
        echo "[entrypoint] Timeout au démarrage du worker Qwen." >&2
        exit 1
    fi
fi

echo "[entrypoint] Démarrage de l'API principale…"
"$@" &
main_pid=$!

set +e
if [[ -n "$worker_pid" ]]; then
    wait -n "$main_pid" "$worker_pid"
    status=$?
else
    wait "$main_pid"
    status=$?
fi
set -e
exit "$status"
