"""Contrat minimal d'un détecteur d'activité vocale."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from transcription_server.chunking import plan_windows


class VadEngine(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def device(self) -> str: ...

    def plan(self, audio: np.ndarray) -> list[tuple[float, float]]: ...


@dataclass(frozen=True)
class FixedWindowVadEngine:
    """Repli déterministe qui conserve la limite de durée des inférences."""

    max_segment_s: float = 5.0
    overlap_s: float = 1.0

    @property
    def name(self) -> str:
        return "fixed-windows"

    @property
    def device(self) -> str:
        return "cpu"

    def plan(self, audio: np.ndarray) -> list[tuple[float, float]]:
        if len(audio) == 0:
            return []
        return [
            (window.start, window.end)
            for window in plan_windows(
                duration_s=len(audio) / 16000,
                chunk_length_s=self.max_segment_s,
                overlap_s=self.overlap_s,
            )
        ]
