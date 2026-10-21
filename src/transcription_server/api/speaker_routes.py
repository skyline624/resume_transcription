"""Apprentissage explicite et gestion des personnes reconnues par leur voix."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from transcription_server.speakers import SpeakerRegistry, clean_name, embedding_model_id
from transcription_server.state import AppState, get_state

router = APIRouter(prefix="/speakers", tags=["Speaker recognition"])

Label = Annotated[str, Field(pattern=r"^SPEAKER_\d+$", max_length=64)]
Source = Annotated[str, Field(min_length=1, max_length=200)]
Vector = Annotated[list[float], Field(min_length=1, max_length=4096)]


class SpeakerName(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value):
        return clean_name(value)


class IdentifySpeaker(SpeakerName):
    transcription_id: UUID
    speaker: Label


class PersonOut(BaseModel):
    id: str
    name: str
    samples: int
    #: Voix automatique (« Voix N ») creee pour un inconnu, pas encore nommee.
    auto: bool = False
    #: Date de la reference la plus recente (ISO 8601, UTC).
    last_seen: str | None = None


class ReferenceIn(BaseModel):
    """Une empreinte gardee par le client, sous sa propre cle (source, speaker)."""

    source: Source
    speaker: Label
    model: str = Field(min_length=1, max_length=200)
    vector: Vector
    speech_s: float | None = None
    name: str | None = Field(default=None, min_length=1, max_length=100)
    person_id: UUID | None = None
    auto: bool = False

    @field_validator("name")
    @classmethod
    def valid_name(cls, value):
        return None if value is None else clean_name(value)

    @model_validator(mode="after")
    def one_target(self):
        if sum((self.name is not None, self.person_id is not None, self.auto)) != 1:
            raise ValueError("Indiquez une seule cible : name, person_id ou auto.")
        return self


class MergeIn(BaseModel):
    into: UUID


class ImportReference(BaseModel):
    source: Source
    speaker: Label
    vector: Vector


class ImportPerson(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=100)
    auto: bool = False
    auto_number: int | None = Field(default=None, ge=1)
    references: list[ImportReference] = Field(default_factory=list, max_length=1000)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value):
        return clean_name(value)


class ImportIn(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    next_auto_number: int | None = Field(default=None, ge=1)
    persons: list[ImportPerson] = Field(max_length=1000)


class ImportOut(BaseModel):
    persons: dict[str, PersonOut]


class RegistryInfo(BaseModel):
    model: str
    threshold: float
    margin: float
    min_speech_s: float
    max_references: int


def _registry(state: Annotated[AppState, Depends(get_state)]) -> SpeakerRegistry:
    if state.speaker_registry is None:
        raise HTTPException(503, "La reconnaissance des personnes est désactivée.")
    return state.speaker_registry


Registry = Annotated[SpeakerRegistry, Depends(_registry)]


def _call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
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


# Les chemins fixes passent avant /{speaker_id}, qui accepterait leur nom
# avant de le refuser comme UUID.

@router.get("/info", response_model=RegistryInfo)
def info(registry: Registry, state: Annotated[AppState, Depends(get_state)]):
    """Étiquette des empreintes du moteur actif et règles de reconnaissance."""
    return {"model": embedding_model_id(state.settings.diarization_model), **registry.info()}


@router.get("/aliases", response_model=dict[str, str])
def aliases(registry: Registry):
    """Personnes fusionnées : ancien id -> id de la personne qui les a reçues."""
    return registry.aliases()


@router.post("/references", response_model=PersonOut)
def add_reference(body: ReferenceIn, registry: Registry):
    """Rattache une empreinte gardée par le client à une personne, sans limite de durée.

    La clé (source, speaker) est celle du client : la renvoyer vers une autre
    personne déplace la référence. `auto` crée une « Voix N » pour un inconnu.
    """
    return _call(registry.add_reference, body.source, body.speaker, body.model, body.vector,
                 speech_s=body.speech_s, name=body.name,
                 person_id=None if body.person_id is None else str(body.person_id),
                 auto=body.auto)


@router.delete("/references", status_code=204)
def remove_reference(registry: Registry, source: Annotated[Source, Query()],
                     speaker: Annotated[Label, Query()]):
    """Détache une empreinte ; une personne restée sans référence disparaît."""
    _call(registry.remove_reference, source, speaker)
    return Response(status_code=204)


@router.post("/import", response_model=ImportOut)
def import_people(body: ImportIn, registry: Registry):
    """Importe une bibliothèque d'empreintes en une transaction, relançable."""
    persons = _call(registry.import_people, body.model,
                    [person.model_dump() for person in body.persons], body.next_auto_number)
    return {"persons": persons}


@router.post("/{speaker_id}/merge", response_model=PersonOut)
def merge(speaker_id: UUID, body: MergeIn, registry: Registry):
    """Fusionne une personne dans `into` ; un alias garde l'ancien id valable."""
    return _call(registry.merge, str(speaker_id), str(body.into))


@router.patch("/{speaker_id}", response_model=PersonOut)
def rename(speaker_id: UUID, body: SpeakerName, registry: Registry):
    return _call(registry.rename, str(speaker_id), body.name)


@router.delete("/{speaker_id}", status_code=204)
def delete(speaker_id: UUID, registry: Registry):
    _call(registry.delete, str(speaker_id))
    return Response(status_code=204)
