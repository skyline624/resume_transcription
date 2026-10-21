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
    speaker_label: str | None = None
    start: float
    end: float
    text: str
    words: list[WordOut] = Field(default_factory=list)


class SpeakerMatchOut(BaseModel):
    speaker_id: str | None = None
    name: str | None = None
    score: float | None = None
    enrollable: bool = False


class SpeakerEmbeddingOut(BaseModel):
    model: str
    #: Normalise ; nul quand le registre refuserait l'empreinte (parole trop
    #: courte, vecteur invalide). Au millionieme : au millieme, l'erreur
    #: d'arrondi deplacerait les similarites d'environ 0,005.
    vector: list[float] | None = None
    speech_s: float


class TranscriptionOut(BaseModel):
    text: str
    language: str | None = None
    duration: float
    speakers: list[str] = Field(default_factory=list)
    turns: list[TurnOut] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)
    #: Nombre de canaux transcrits separement. 1 en mode `mix`.
    #: Expose pour que l'appelant sache ce que le serveur a fait de son
    #: fichier, plutot que de devoir le deduire.
    channels_used: int = 1
    transcription_id: str | None = None
    speaker_matches: dict[str, SpeakerMatchOut] = Field(default_factory=dict)
    #: Seulement avec `return_embeddings` : absent du corps sinon.
    speaker_embeddings: dict[str, SpeakerEmbeddingOut] | None = None


class HealthOut(BaseModel):
    status: str
    device: str
    asr_model: str
    diarization_model: str
    diarization_enabled: bool
    speaker_recognition_enabled: bool = False
    vad_model: str
    vad_device: str | None
    vad_enabled: bool
    summary_model: str
    summary_enabled: bool
    asr_state: str
    diarization_state: str
    # dict nu vaudrait dict[Any, Any] : une cle non serialisable y passerait la
    # validation pour ne se manifester qu'a la serialisation, en avertissement.
    gpu: dict[str, Any] | None = None
    tts: "TtsHealthOut"


class TtsHealthOut(BaseModel):
    enabled: bool
    worker: bool
    state: str
    downloaded_models: list[str] = Field(default_factory=list)
    loaded_model: str | None = None
    precision: str | None = None
    device: str | None = None
    attention: str | None = None
    features: list[str] = Field(default_factory=list)
    last_error: str | None = None
    pid: int | None = None
    vram_allocated_mib: float | None = None


def turn_to_out(turn: Turn, include_words: bool) -> TurnOut:
    return TurnOut(
        speaker=turn.speaker,
        speaker_label=turn.speaker_label or turn.speaker,
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
        channels_used=result.channels_used,
        transcription_id=result.transcription_id,
        speaker_matches=result.speaker_matches,
        speaker_embeddings=None if result.speaker_embeddings is None else {
            label: SpeakerEmbeddingOut(
                model=entry["model"],
                vector=None if entry["vector"] is None
                else [round(value, 6) for value in entry["vector"]],
                speech_s=entry["speech_s"],
            )
            for label, entry in result.speaker_embeddings.items()
        },
    )


def transcription_body(out: TranscriptionOut) -> dict:
    """Le corps JSON ; sans `speaker_embeddings` quand elles n'ont pas ete demandees."""
    exclude = {"speaker_embeddings"} if out.speaker_embeddings is None else set()
    return out.model_dump(mode="json", exclude=exclude)
