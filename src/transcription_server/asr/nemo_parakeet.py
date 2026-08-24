"""Adaptateur du modele Parakeet via NVIDIA NeMo.

NeMo et torch sont importes paresseusement, a l'interieur de
`load_nemo_engine` : la classe reste donc importable sans l'extra `gpu`, ce qui
permet aux tests de conformite de signature de tourner sur Windows.
"""

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

from transcription_server.audio import SAMPLE_RATE
from transcription_server.domain import Word

logger = logging.getLogger(__name__)


class NemoParakeetEngine:
    """Transcrit avec un modele NeMo ASR, timestamps mot a mot inclus."""

    def __init__(self, model, model_name: str, device: str) -> None:
        self._model = model
        self._name = model_name
        self._device = device

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]:
        """Rend les mots horodates, en temps relatif au debut de `audio`.

        `language` est accepte pour respecter le contrat du Protocol, mais
        Parakeet TDT v3 detecte la langue lui-meme et n'expose pas de parametre
        pour la forcer : la valeur est donc ignoree. L'appelant la recupere
        telle quelle dans la reponse, ce qui est documente comme une limite.
        """
        if audio.size == 0:
            return []

        # NeMo lit un chemin de fichier de maniere fiable quelle que soit sa
        # version ; passer un tableau change de signature d'une release a
        # l'autre. On ecrit donc un wav temporaire, supprime dans tous les cas.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            chemin = Path(handle.name)
        try:
            _write_wav(chemin, audio)
            sorties = self._model.transcribe([str(chemin)], timestamps=True)
        finally:
            chemin.unlink(missing_ok=True)

        if not sorties:
            return []
        return _extract_words(sorties[0])


def _write_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Ecrit un tableau float32 dans [-1, 1] en wav mono 16 bits."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm16.tobytes())


def _extract_words(hypothesis) -> list[Word]:
    """Extrait les mots horodates d'une hypothese NeMo.

    NeMo expose `hypothesis.timestamp["word"]`, une liste de dictionnaires
    portant `word`, `start` et `end` en secondes.
    """
    timestamps = getattr(hypothesis, "timestamp", None)
    if not timestamps or "word" not in timestamps:
        texte = (getattr(hypothesis, "text", "") or "").strip()
        if not texte:
            return []
        logger.warning(
            "NeMo n'a pas rendu de timestamps mot à mot ; le texte est restitué "
            "en un seul bloc, sans horodatage fin."
        )
        return [Word(text=texte, start=0.0, end=0.0)]

    mots: list[Word] = []
    for entree in timestamps["word"]:
        texte = (entree.get("word") or "").strip()
        if not texte:
            continue
        mots.append(
            Word(
                text=texte,
                start=float(entree["start"]),
                end=float(entree["end"]),
            )
        )
    # Le contrat de AsrEngine.transcribe exige des mots chronologiques : ni
    # merge_windows ni group_into_turns ne trient, et une entree desordonnee
    # produit silencieusement des tours dont end < start. NeMo les rend
    # normalement dans l'ordre, mais rien ne le garantit — le tri est bon
    # marche et supprime la dependance a cette hypothese.
    mots.sort(key=lambda mot: (mot.start, mot.end))
    return mots


def load_nemo_engine(
    model_name: str,
    device: str,
    compute_type: str,
) -> NemoParakeetEngine:
    """Charge le modele et le place sur le peripherique demande."""
    import nemo.collections.asr as nemo_asr
    import torch

    logger.info("Chargement du modèle ASR %s…", model_name)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(torch.device(device))
    if device == "cuda" and compute_type == "float16":
        model = model.half()
    model.eval()
    logger.info("Modèle ASR %s chargé sur %s.", model_name, device)
    return NemoParakeetEngine(model=model, model_name=model_name, device=device)
