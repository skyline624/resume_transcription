#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Vérification de l'environnement CUDA…"
python - <<'PY'
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

echo "[entrypoint] Démarrage…"
exec "$@"
