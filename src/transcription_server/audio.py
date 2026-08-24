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

# Le stderr de ffmpeg n'est pas borne, meme a -loglevel error : un fichier
# abime en produit des dizaines de lignes. Ce texte finit dans le message
# d'une exception qui remonte jusqu'a une reponse HTTP, on le tronque donc
# ici, quoi que decide la couche API.
_DETAIL_MAX = 500

# Taille d'un echantillon float32, en octets.
_TAILLE_ECHANTILLON = 4


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
        # ffmpeg reconnait le format d'apres le contenu et non d'apres le nom :
        # un fichier de type playlist lui ferait ouvrir d'autres ressources que
        # celle qu'on lui confie. La 8.1.1 les refuse deja par defaut, mais
        # l'image installera ffmpeg depuis les depots de la distribution -- ce
        # defaut-la, nous ne le choisissons pas. Option d'entree : doit
        # imperativement preceder -i.
        "-protocol_whitelist", "file",
        "-i", str(source),
        "-f", "f32le",       # float32 little-endian brut
        "-acodec", "pcm_f32le",
        "-ac", "1",          # mono
        "-ar", str(sample_rate),
        "-",
    ]
    try:
        # -nostdin desarme la lecture du stdin par ffmpeg lui-meme ; DEVNULL
        # ferme la couche en dessous, le descripteur herite du parent, que
        # capture_output ne redirige pas.
        result = subprocess.run(
            command, capture_output=True, check=False, stdin=subprocess.DEVNULL
        )
    except OSError as exc:  # pragma: no cover - defense en profondeur
        raise AudioDecodeError(f"Échec de l'appel à ffmpeg : {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:_DETAIL_MAX]
        raise AudioDecodeError(f"ffmpeg n'a pas pu décoder le fichier : {detail}")

    # np.frombuffer exige un multiple de la taille d'un echantillon et
    # leverait sinon un ValueError -- un 500 la ou toutes les autres sorties
    # de cette fonction donnent un 400.
    if len(result.stdout) % _TAILLE_ECHANTILLON:
        raise AudioDecodeError(
            f"Sortie de ffmpeg tronquée : {len(result.stdout)} octets ne "
            "forment pas un nombre entier d'échantillons."
        )

    pcm = np.frombuffer(result.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise AudioDecodeError("Le fichier ne contient aucun échantillon audio.")

    # Equivalent de np.abs(pcm).max() sans materialiser un second tableau de
    # la taille du PCM, et propageant NaN et infinis a l'identique.
    peak = float(max(-pcm.min(), pcm.max()))

    # Un wav flottant peut porter des NaN ou des infinis -- format d'export
    # courant, et un fichier tronque en contient trivialement -- et ffmpeg les
    # transmet sans broncher, code de retour nul. Les laisser passer romprait
    # le contrat de sortie ; pire, sur un infini le pic vaut inf, donc la
    # division ci-dessous ramenerait tout le signal fini a zero et rendrait un
    # 200 sur une transcription vide, indistinguable d'un vrai silence.
    if not np.isfinite(peak):
        raise AudioDecodeError(
            "Le fichier contient des échantillons non finis (NaN ou infini)."
        )

    # ffmpeg rend deja du float normalise, mais un fichier deja sature
    # pourrait depasser les bornes : le flottant n'est pas rabote comme
    # l'entier. La division est conditionnelle a dessein -- normaliser sans
    # condition amplifierait un enregistrement discret jusqu'a la saturation,
    # et rendrait des NaN sur un passage entierement silencieux, ou le pic
    # vaut zero. Les deux cas sont ordinaires sur de l'audio reel.
    if peak > 1.0:
        return (pcm / peak).astype(np.float32, copy=False)

    # np.frombuffer n'a rendu qu'une vue en lecture seule sur les octets de
    # ffmpeg. La copie rend un tableau possede et inscriptible, comme celui de
    # la branche ci-dessus, et libere le buffer de sortie entier -- qu'une
    # seule fenetre decoupee suffirait sinon a maintenir en vie.
    return pcm.copy()


def duration_seconds(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / float(sample_rate)
