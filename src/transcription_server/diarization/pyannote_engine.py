"""Adaptateur du pipeline de diarization pyannote.audio.

pyannote et torch sont importes paresseusement, a l'interieur de
`load_pyannote_engine` : la classe reste importable sans l'extra `gpu` ni
token, ce qui permet aux tests de conformite de signature de tourner sur
Windows.
"""

import logging

import numpy as np

from transcription_server.audio import SAMPLE_RATE
from transcription_server.domain import SpeakerSegment

logger = logging.getLogger(__name__)


def _annotation_de(sortie):
    """Rend l'`Annotation` portee par la sortie du pipeline.

    pyannote 4 n'appelle plus directement une `Annotation` : il enveloppe son
    resultat dans un `DiarizeOutput`, dataclass portant `speaker_diarization`,
    `exclusive_speaker_diarization` et `speaker_embeddings`. La branche 3.x
    rendait l'`Annotation` telle quelle.

    On retient `speaker_diarization`, qui autorise la parole simultanee :
    l'attribution des mots se fait par recouvrement maximal, donc les
    chevauchements sont deja traites en aval, et les ecraser ici perdrait
    l'information sans rien simplifier.
    """
    return getattr(sortie, "speaker_diarization", sortie)


class PyannoteEngine:
    """Separe les locuteurs avec un pipeline pyannote deja charge."""

    def __init__(self, pipeline, model_name: str, device: str = "cpu") -> None:
        self._pipeline = pipeline
        self._name = model_name
        self._device = device
        self._resident_device = device

    @property
    def name(self) -> str:
        return self._name

    def release_gpu(self) -> None:
        """Place le pipeline en RAM jusqu'a la prochaine diarization."""
        if self._resident_device != "cuda":
            return
        import torch

        self._pipeline.to(torch.device("cpu"))
        self._resident_device = "cpu"

    def _ensure_target_device(self) -> None:
        if self._resident_device == self._device:
            return
        import torch

        self._pipeline.to(torch.device(self._device))
        self._resident_device = self._device

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        """Rend des segments tries par debut croissant.

        Les etiquettes viennent de pyannote et suivent deja la forme
        `SPEAKER_00`, `SPEAKER_01`, … : aucune renumerotation n'est faite ici.
        """
        import torch

        if audio.size == 0:
            return []

        self._ensure_target_device()
        # pyannote attend un tenseur (canaux, echantillons). `ascontiguousarray`
        # protege d'un tableau non contigu venant d'une tranche de fenetre.
        forme_onde = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

        options: dict = {}
        if num_speakers is not None:
            # `num_speakers` fixe le nombre exact ; le combiner aux bornes est
            # contradictoire, et la route rejette deja cette combinaison en 400.
            options["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                options["min_speakers"] = min_speakers
            if max_speakers is not None:
                options["max_speakers"] = max_speakers

        sortie = self._pipeline(
            {"waveform": forme_onde, "sample_rate": SAMPLE_RATE},
            **options,
        )
        annotation = _annotation_de(sortie)

        segments = [
            SpeakerSegment(
                speaker=str(locuteur),
                start=float(tour.start),
                end=float(tour.end),
            )
            for tour, _, locuteur in annotation.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda segment: (segment.start, segment.end))
        return segments


def load_pyannote_engine(
    model_name: str,
    hf_token: str,
    device: str,
) -> PyannoteEngine:
    """Charge le pipeline et le place sur le peripherique demande."""
    import torch
    from pyannote.audio import Pipeline

    logger.info("Chargement du pipeline de diarization %s…", model_name)
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    except TypeError:
        # pyannote 3.x attendait `use_auth_token`, renomme en `token` en 4.x.
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)

    if pipeline is None:
        # `from_pretrained` rend None au lieu de lever quand l'acces est refuse :
        # sans ce garde-fou, la panne n'apparaitrait qu'a la premiere requete,
        # sous la forme d'un AttributeError incomprehensible.
        raise RuntimeError(
            f"pyannote n'a pas pu charger {model_name}. Vérifiez que le compte "
            "HuggingFace a bien accepté les conditions du modèle et que "
            "HF_TOKEN est un token de type read valide."
        )

    pipeline.to(torch.device(device))
    logger.info("Pipeline de diarization chargé sur %s.", device)
    return PyannoteEngine(pipeline=pipeline, model_name=model_name, device=device)
