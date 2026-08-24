"""Types du domaine, sans aucune dependance lourde.

Ce module ne doit jamais importer torch, NeMo, pyannote ni numpy :
c'est ce qui rend la logique metier testable hors du conteneur.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """Un mot transcrit, horodate en secondes absolues."""

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SpeakerSegment:
    """Un intervalle attribue a un locuteur par la diarization."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Turn:
    """Un tour de parole : des mots consecutifs d'un meme locuteur."""

    speaker: str | None
    start: float
    end: float
    text: str
    words: tuple[Word, ...]
