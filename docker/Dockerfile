# torch 2.11 satisfait NeMo (>=2.6) et pyannote (>=2.8) ; CUDA 12.8 est le
# socle le mieux eprouve pour NeMo, et le driver 610.88 de l'hote est
# retro-compatible.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# ffmpeg pour le decodage audio, libsndfile1 pour soundfile (dependance de
# NeMo), git parce que certaines dependances de NeMo se resolvent par git,
# curl pour le healthcheck de Compose.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# L'image de base expose un Python systeme marque PEP 668, donc pip refuse
# d'installer sans cette levee. Dans un conteneur dedie a cette seule
# application, il n'y a pas d'OS partage a proteger : lever la garde est plus
# simple que de superposer un venv a une image qui n'en a pas besoin.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

COPY docker/pin_torch.py /tmp/pin_torch.py
RUN python /tmp/pin_torch.py /tmp/constraints.txt

COPY pyproject.toml README.md ./

# Seul le fichier de paquet est copie avant l'installation : hatchling exige
# qu'il existe, mais `pip install -e` n'enregistre qu'un chemin — le code est
# lu a l'execution. Copier tout `src/` ici invaliderait le cache de cette
# couche a chaque ligne modifiee, imposant dix minutes de reinstallation de
# NeMo pour un caractere.
COPY src/transcription_server/__init__.py ./src/transcription_server/__init__.py

RUN pip install --no-cache-dir -c /tmp/constraints.txt -e ".[gpu,dev]"

COPY src/ ./src/

# Verification au build : si l'installation de NeMo ou pyannote a remplace le
# torch CUDA par une roue CPU, autant l'apprendre maintenant qu'au premier
# demarrage sur la machine de l'utilisateur.
RUN python -c "import torch; assert torch.version.cuda, 'torch a perdu CUDA pendant pip install'; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"

COPY tests/ ./tests/

EXPOSE 8000

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Docker Desktop peut fournir le contexte depuis un checkout Windows en CRLF.
# Normaliser dans l'image évite un shebang interprété comme `bash\r`.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# Indispensable : l'entrypoint fait `exec "$@"` sous `set -u`, donc un
# `docker run` sans commande echouerait sur une variable non liee.
CMD ["python", "-m", "transcription_server.main"]
