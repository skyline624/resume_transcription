"""Apprentissage explicite et gestion des personnes reconnues par leur voix."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from transcription_server.speakers import SpeakerRegistry, clean_name
from transcription_server.state import AppState, get_state

router = APIRouter(prefix="/speakers", tags=["Speaker recognition"])


class SpeakerName(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value):
        return clean_name(value)


class IdentifySpeaker(SpeakerName):
    transcription_id: UUID
    speaker: str = Field(pattern=r"^SPEAKER_\d+$", max_length=64)


class PersonOut(BaseModel):
    id: str
    name: str
    samples: int


def _registry(state: Annotated[AppState, Depends(get_state)]) -> SpeakerRegistry:
    if state.speaker_registry is None:
        raise HTTPException(503, "La reconnaissance des personnes est désactivée.")
    return state.speaker_registry


Registry = Annotated[SpeakerRegistry, Depends(_registry)]


def _call(operation, *args):
    try:
        return operation(*args)
    except KeyError as exc:
        raise HTTPException(404, exc.args[0]) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/identify", response_model=PersonOut)
def identify(body: IdentifySpeaker, registry: Registry):
    """Associe un nom au label d'une transcription récente (24 h par défaut).

    Le même nom ajoute une référence à la personne existante (maximum cinq).
    Renvoyer le même extrait avec un autre nom corrige son attribution.
    Utiliser le label original SPEAKER_XX, même si un nom avait été reconnu.
    """
    return _call(registry.identify, str(body.transcription_id), body.speaker, body.name)


@router.get("", response_model=list[PersonOut])
def list_speakers(registry: Registry):
    return registry.list()


@router.patch("/{speaker_id}", response_model=PersonOut)
def rename(speaker_id: UUID, body: SpeakerName, registry: Registry):
    return _call(registry.rename, str(speaker_id), body.name)


@router.delete("/{speaker_id}", status_code=204)
def delete(speaker_id: UUID, registry: Registry):
    _call(registry.delete, str(speaker_id))
    return Response(status_code=204)
