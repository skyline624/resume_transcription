"""Modeles de reponse de l'API."""

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
    gpu: dict | None = None


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
        timing=result.timing,
    )
