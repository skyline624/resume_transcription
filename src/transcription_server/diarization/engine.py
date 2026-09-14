"""Contrat des moteurs de diarization."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from transcription_server.domain import SpeakerSegment


@dataclass(frozen=True)
class SpeakerEmbedding:
    model: str
    values: tuple[float, ...]
    speech_s: float


class DiarizationSegments(list[SpeakerSegment]):
    """Segments et empreintes issus du meme appel, sans etat partage du moteur."""

    def __init__(self, segments, embeddings: dict[str, SpeakerEmbedding]):
        super().__init__(segments)
        self.embeddings = embeddings


@runtime_checkable
class DiarizationEngine(Protocol):
    """Decoupe une waveform en intervalles attribues a des locuteurs."""

    @property
    def name(self) -> str: ...

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        """Rend des segments non vides, tries par debut croissant."""
        ...


class NullDiarizationEngine:
    """Ne separe aucun locuteur.

    Utilise quand ENABLE_DIARIZATION=false : le serveur demarre alors sans
    token HuggingFace, et toutes les transcriptions sortent en un seul flux.
    """

    @property
    def name(self) -> str:
        return "none"

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        return []


class StubDiarizationEngine:
    """Moteur a sortie fixe, pour tester les routes sans GPU."""

    def __init__(
        self,
        segments: list[SpeakerSegment],
        name: str = "stub-diarization",
    ) -> None:
        self._segments = list(segments)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        return list(self._segments)
