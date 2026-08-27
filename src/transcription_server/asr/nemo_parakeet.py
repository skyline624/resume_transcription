"""Adaptateur du modele Parakeet via NVIDIA NeMo.

NeMo et torch sont importes paresseusement, a l'interieur de
`load_nemo_engine` : la classe reste donc importable sans l'extra `gpu`, ce qui
permet aux tests de conformite de signature de tourner sur Windows.
"""

import gc
import logging
import re
import tempfile
import wave
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path

import numpy as np

from transcription_server.audio import SAMPLE_RATE
from transcription_server.chunking import offset_words
from transcription_server.domain import Word

logger = logging.getLogger(__name__)

_RETRY_SEGMENT_S = 3.0
_MOTS_ANGLAIS = set(
    "the and because you that with this what your they their it is are was were "
    "for but from have has had will would should could in of to".split()
)
_MOTS_FRANCAIS = set(
    "le la les un une des du de et que qui dans pour avec ce cette ces est sont "
    "tu vous il elle on nous mais pas plus au aux en ça donc parce".split()
)


class NemoParakeetEngine:
    """Transcrit avec un modele NeMo ASR, timestamps mot a mot inclus."""

    def __init__(
        self,
        model,
        model_name: str,
        device: str,
        precision_context: Callable[[], AbstractContextManager] = nullcontext,
        reload_model: Callable[[], object] | None = None,
    ) -> None:
        self._model = model
        self._name = model_name
        self._device = device
        self._resident_device = device
        self._precision_context = precision_context
        self._reload_model = reload_model

    @property
    def name(self) -> str:
        return self._name

    def release_gpu(self) -> None:
        """Detruit le modele reel afin de ne conserver aucune adresse CUDA."""
        if self._resident_device != "cuda":
            return
        import torch

        if self._reload_model is not None:
            synchronize = getattr(getattr(torch, "cuda", None), "synchronize", None)
            if synchronize is not None:
                synchronize()
            model = self._model
            self._model = None
            self._resident_device = "unloaded"
            del model
            gc.collect()
            return

        # Repli pour les adaptateurs construits directement par des clients ou
        # des tests. La fabrique de production fournit toujours reload_model.
        self._model = self._model.to(torch.device("cpu"))
        self._resident_device = "cpu"

    def _ensure_target_device(self) -> None:
        if self._resident_device == "unloaded":
            if self._reload_model is None:
                raise RuntimeError("Le modèle ASR déchargé ne peut pas être recréé.")
            self._model = self._reload_model()
            self._resident_device = self._device
            return
        if self._resident_device == self._device:
            return
        import torch

        self._model = self._model.to(torch.device(self._device))
        self._resident_device = self._device

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]:
        """Rend les mots horodates, en temps relatif au debut de `audio`.

        Parakeet TDT v3 détecte la langue lui-même et ne permet pas de la
        forcer. Quand `fr` est explicitement demandé, la valeur sert toutefois
        de garde-fou : une sortie nettement anglaise est retentée sur des
        sous-segments plus courts.
        """
        if audio.size == 0:
            return []

        self._ensure_target_device()
        mots = self._transcribe_once(audio)
        texte = " ".join(mot.text for mot in mots)
        francais_demande = bool(language and language.lower().split("-", 1)[0] == "fr")
        taille_retry = int(_RETRY_SEGMENT_S * SAMPLE_RATE)
        if francais_demande and _semble_anglais(texte) and audio.size > taille_retry:
            logger.info(
                "Sortie probablement anglaise malgré language=fr ; "
                "nouvelle inférence par segments de %.1f s.",
                _RETRY_SEGMENT_S,
            )
            mots = []
            for debut in range(0, audio.size, taille_retry):
                locaux = self._transcribe_once(audio[debut : debut + taille_retry])
                mots.extend(offset_words(locaux, debut / SAMPLE_RATE))
        return mots

    def _transcribe_once(self, audio: np.ndarray) -> list[Word]:
        """Exécute une inférence NeMo sans logique de seconde passe."""

        # NeMo lit un chemin de fichier de maniere fiable quelle que soit sa
        # version ; passer un tableau change de signature d'une release a
        # l'autre. On ecrit donc un wav temporaire, supprime dans tous les cas.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            chemin = Path(handle.name)
        try:
            _write_wav(chemin, audio)
            with self._precision_context():
                sorties = self._model.transcribe([str(chemin)], timestamps=True)
        finally:
            chemin.unlink(missing_ok=True)

        if not sorties:
            return []
        return _extract_words(sorties[0])


def _semble_anglais(phrase: str) -> bool:
    """Repère une sortie nettement anglaise, sans classer les mots métier."""
    mots = re.findall(r"[a-zà-ÿ']+", phrase.lower())
    anglais = sum(mot in _MOTS_ANGLAIS for mot in mots)
    francais = sum(mot in _MOTS_FRANCAIS for mot in mots)
    return anglais >= 2 and anglais > francais


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

    def build_model():
        logger.info("Chargement du modèle ASR %s…", model_name)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model = model.to(torch.device(device))
        model.eval()
        # Les graphes CUDA de NeMo capturent des adresses de tenseurs. Même
        # après recréation, le mode mobile n'a aucun intérêt à les conserver.
        disable_cuda_graphs = getattr(model, "disable_cuda_graphs", None)
        if disable_cuda_graphs is not None:
            disable_cuda_graphs()
        logger.info("Modèle ASR %s chargé sur %s.", model_name, device)
        return model

    model = build_model()

    @contextmanager
    def precision_context():
        with torch.inference_mode():
            if device == "cuda" and compute_type == "float16":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    yield
            else:
                yield

    return NemoParakeetEngine(
        model=model,
        model_name=model_name,
        device=device,
        precision_context=precision_context,
        reload_model=build_model,
    )
