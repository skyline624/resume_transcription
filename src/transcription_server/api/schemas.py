"""Modeles de reponse de l'API."""

from typing import Any

from pydantic import BaseModel, Field

from transcription_server.domain import Turn
from transcription_server.pipeline import TranscriptionResult


class WordOut(BaseModel):
    word: str
    start: float
    end: float


class TurnOut(BaseModel):
    speaker: str | None
    start: float
    end: float
    text: str
    words: list[WordOut] = Field(default_factory=list)


class TranscriptionOut(BaseModel):
    text: str
    language: str | None = None
    duration: float
    speakers: list[str] = Field(default_factory=list)
    turns: list[TurnOut] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)


class HealthOut(BaseModel):
    status: str
    device: str
    asr_model: str
    diarization_model: str
    diarization_enabled: bool
    # dict nu vaudrait dict[Any, Any] : une cle non serialisable y passerait la
    # validation pour ne se manifester qu'a la serialisation, en avertissement.
    gpu: dict[str, Any] | None = None


def turn_to_out(turn: Turn, include_words: bool) -> TurnOut:
    return TurnOut(
        speaker=turn.speaker,
        start=round(turn.start, 3),
        end=round(turn.end, 3),
        text=turn.text,
        words=[
            WordOut(word=w.text, start=round(w.start, 3), end=round(w.end, 3))
            for w in turn.words
        ]
        if include_words
        else [],
    )


def result_to_out(
    result: TranscriptionResult, include_words: bool = True
) -> TranscriptionOut:
    return TranscriptionOut(
        text=result.text,
        language=result.language,
        duration=round(result.duration, 3),
        speakers=result.speakers,
        turns=[turn_to_out(t, include_words) for t in result.turns],
        # `pipeline` arrondit deja, mais l'unite du corps JSON -- tous les
        # flottants au millieme -- est un contrat de l'API, pas une propriete
        # empruntee a son fournisseur. On le tient ici, a la frontiere.
        timing={nom: round(valeur, 3) for nom, valeur in result.timing.items()},
    )
