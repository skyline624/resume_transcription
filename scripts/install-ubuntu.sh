#!/usr/bin/env bash
# Installation native sur Ubuntu 24.04 avec NVIDIA, sans privileges root.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd -P)
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$ROOT/runtime/uv-cache"
export UV_LINK_MODE=hardlink
export TMPDIR="$ROOT/runtime/tmp"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$ROOT/runtime/logs"
command -v uv >/dev/null
command -v ffmpeg >/dev/null
if ! command -v sox >/dev/null; then
    mkdir -p runtime/apt runtime/vendor
    (cd runtime/apt && apt download sox libsox3 libsox-fmt-base libltdl7)
    find runtime/apt -name '*.deb' -exec dpkg-deb -x {} runtime/vendor \;
    export PATH="$ROOT/runtime/vendor/usr/bin:$PATH"
    export LD_LIBRARY_PATH="$ROOT/runtime/vendor/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    sox --version
fi
nvidia-smi
test -x .venv/bin/python || uv venv --python /usr/bin/python3.12 .venv
test -x .venv-qwen/bin/python || uv venv --python /usr/bin/python3.12 .venv-qwen
printf 'torch==2.11.0\ntorchaudio==2.11.0\n' > runtime/constraints.txt
for env in .venv .venv-qwen; do
    uv pip install --python "$env/bin/python" --default-index https://download.pytorch.org/whl/cu128 \
        --index https://pypi.org/simple --index-strategy unsafe-best-match \
        torch==2.11.0+cu128 torchaudio==2.11.0+cu128
done
uv pip install --python .venv/bin/python -c runtime/constraints.txt -e '.[gpu,dev]'
uv pip install --python .venv-qwen/bin/python -c runtime/constraints.txt -r docker/requirements-qwen.txt
uv pip install --python .venv-qwen/bin/python --no-deps -e qwen_worker
.venv/bin/python -c 'import nemo, torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
.venv-qwen/bin/python -c 'import qwen_tts, torch, transformers; assert torch.cuda.is_available(); assert transformers.__version__ == "4.57.3"'
(cd web && npm ci && npm test && npm run build)
echo 'Installation terminee. Configurer .env.studio-linux puis installer le service utilisateur.'
