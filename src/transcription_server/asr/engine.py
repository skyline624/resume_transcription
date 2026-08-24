"""Contrat des moteurs de transcription."""

from typing import Protocol, runtime_checkable

import numpy as np

from transcription_server.domain import Word


@runtime_checkable
class AsrEngine(Protocol):
    """Transforme une waveform mono 16 kHz en mots horodates."""

    @property
    def name(self) -> str:
        """Identifiant du modele, expose par /health et /v1/models."""
        ...

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]:
        """Rend les mots avec des timestamps relatifs au debut de `audio`."""
        ...


class StubAsrEngine:
    """Moteur a sortie fixe, pour tester les routes sans GPU."""

    def __init__(self, words: list[Word], name: str = "stub-asr") -> None:
        self._words = list(words)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]:
        return list(self._words)
