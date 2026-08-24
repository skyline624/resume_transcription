"""Decodage de n'importe quel format audio vers du PCM mono 16 kHz.

On delegue a ffmpeg plutot qu'a une bibliotheque Python : cela couvre mp3,
m4a, ogg, flac, mp4, webm et le reste sans dependance supplementaire, et
c'est le meme binaire dans le conteneur et sur la machine de developpement.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE: int = 16000


class AudioDecodeError(Exception):
    """L'audio n'a pas pu etre decode. Correspond a un HTTP 400."""


def decode_to_pcm(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode un fichier audio en float32 mono, normalise dans [-1, 1]."""
    source = Path(path)
    if not source.exists():
        raise AudioDecodeError(f"Fichier introuvable : {source}")
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("ffmpeg est introuvable dans le PATH.")

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-f", "f32le",       # float32 little-endian brut
        "-acodec", "pcm_f32le",
        "-ac", "1",          # mono
        "-ar", str(sample_rate),
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:  # pragma: no cover - defense en profondeur
        raise AudioDecodeError(f"Echec de l'appel a ffmpeg : {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(f"ffmpeg n'a pas pu décoder le fichier : {detail}")

    pcm = np.frombuffer(result.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise AudioDecodeError("Le fichier ne contient aucun échantillon audio.")

    # ffmpeg rend deja du float normalise, mais un fichier deja sature
    # pourrait depasser les bornes : le flottant n'est pas rabote comme
    # l'entier. La division est conditionnelle a dessein -- normaliser sans
    # condition amplifierait un enregistrement discret jusqu'a la saturation,
    # et rendrait des NaN sur un passage entierement silencieux, ou le pic
    # vaut zero. Les deux cas sont ordinaires sur de l'audio reel.
    peak = float(np.abs(pcm).max())
    if peak > 1.0:
        return (pcm / peak).astype(np.float32, copy=False)

    # np.frombuffer n'a rendu qu'une vue en lecture seule sur les octets de
    # ffmpeg. La copie rend un tableau possede et inscriptible, comme celui de
    # la branche ci-dessus, et libere le buffer de sortie entier -- qu'une
    # seule fenetre decoupee suffirait sinon a maintenir en vie.
    return pcm.copy()


def duration_seconds(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / float(sample_rate)
