"""Apprentissage, isolation des enregistrements et contrat HTTP sans GPU."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import (
    DiarizationSegments, SpeakerEmbedding, StubDiarizationEngine,
)
from transcription_server.diarization.pyannote_engine import _clean_speech_durations
from transcription_server.domain import SpeakerSegment, Turn, Word
from transcription_server.pipeline import TranscriptionRequest, TranscriptionResult, run_pipeline
from transcription_server.speakers import SpeakerRegistry

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"


def result(label=S0):
    return TranscriptionResult("Bonjour", "fr", 5.0, [label],
                               [Turn(label, 0.0, 5.0, "Bonjour", ())])


def voice(values=(1.0, 0.0), model="test", speech_s=5.0):
    return SpeakerEmbedding(model, tuple(values), speech_s)


def enroll(registry, name="Alice", embedding=None):
    pending = registry.recognize(result(), {S0: embedding or voice()})
    return registry.identify(pending.transcription_id, S0, name)


def test_persistence_et_labels_locaux(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    first = registry.recognize(result(), {S0: voice()})
    assert first.speakers == [S0]
    alice = registry.identify(first.transcription_id, S0, "Alice")
    restarted = SpeakerRegistry(tmp_path)
    recognized = restarted.recognize(result(S1), {S1: voice((0.99, 0.04))})
    assert recognized.transcription_id != first.transcription_id
    assert recognized.speakers == ["Alice"]
    assert recognized.turns[0].speaker == "Alice"
    assert recognized.turns[0].speaker_label == S1
    assert recognized.speaker_matches[S1]["speaker_id"] == alice["id"]
    assert recognized.speaker_matches[S1]["score"] > 0.99
    other = restarted.recognize(result(), {S0: voice((0.0, 1.0))})
    assert other.speakers == [S0]  # SPEAKER_00 ne designe pas toujours Alice.


@pytest.mark.parametrize("embedding", [
    voice((0, 0)), voice((float("nan"), 1)), voice((float("inf"), 1)),
    voice(speech_s=2.9), voice(speech_s=float("nan")), voice(()),
])
def test_empreinte_inutilisable_reste_anonyme(tmp_path, embedding):
    registry = SpeakerRegistry(tmp_path)
    enroll(registry)
    pending = registry.recognize(result(), {S0: embedding})
    assert pending.speakers == [S0]
    assert not pending.speaker_matches[S0]["enrollable"]
    with pytest.raises(ValueError, match="Empreinte indisponible"):
        registry.identify(pending.transcription_id, S0, "Bob")


@pytest.mark.parametrize("embedding", [voice(model="different"), voice((1, 0, 0))])
def test_modeles_ou_dimensions_incompatibles(tmp_path, embedding):
    registry = SpeakerRegistry(tmp_path)
    enroll(registry)
    assert registry.recognize(result(), {S0: embedding}).speakers == [S0]


def test_ambiguite_entre_personnes_et_pas_entre_references(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    enroll(registry)
    enroll(registry, "Alice", voice((0.99, 0.01)))
    assert registry.recognize(result(), {S0: voice()}).speakers == ["Alice"]
    enroll(registry, "Bob", voice((0.99, 0.02)))
    assert registry.recognize(result(), {S0: voice()}).speakers == [S0]


def test_correction_idempotence_renommage_et_suppression(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    pending = registry.recognize(result(), {S0: voice()})
    alice = registry.identify(pending.transcription_id, S0, "  Alice  ")
    assert registry.identify(pending.transcription_id, S0, "ALICE") == alice
    bob = registry.identify(pending.transcription_id, S0, "Bob")
    assert registry.list() == [bob]  # La référence erronée d'Alice est retirée.
    registry.rename(bob["id"], "Robert")
    assert registry.recognize(result(), {S0: voice()}).speakers == ["Robert"]
    registry.delete(bob["id"])
    assert registry.list() == []
    assert registry.recognize(result(), {S0: voice()}).speakers == [S0]


def test_limite_references_et_pas_apprentissage_automatique(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    for _ in range(7):
        person = enroll(registry)
    assert person["samples"] == 5
    for _ in range(3):
        registry.recognize(result(), {S0: voice()})
    assert registry.list()[0]["samples"] == 5


def test_expiration_et_limite_transcriptions_sans_perdre_profils(tmp_path, monkeypatch):
    monkeypatch.setattr("transcription_server.speakers.time.time", lambda: 1000.0)
    registry = SpeakerRegistry(tmp_path, pending_ttl_s=10)
    enroll(registry)
    pending = registry.recognize(result(), {S0: voice()})
    monkeypatch.setattr("transcription_server.speakers.time.time", lambda: 1011.0)
    with pytest.raises(KeyError, match="expirée"):
        registry.identify(pending.transcription_id, S0, "Bob")
    assert registry.recognize(result(), {S0: voice()}).speakers == ["Alice"]
    for _ in range(257):
        registry.recognize(result(), {S0: voice()})
    with sqlite3.connect(tmp_path / "speakers.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 256


def test_enrolement_concurrent_ne_duplique_pas_la_personne(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    pending = registry.recognize(result(), {S0: voice()})
    with ThreadPoolExecutor(max_workers=4) as pool:
        people = list(pool.map(lambda _: registry.identify(
            pending.transcription_id, S0, "Alice"), range(8)))
    assert len({person["id"] for person in people}) == 1
    assert registry.list()[0]["samples"] == 1


def test_duree_sans_chevauchement_ni_doublons():
    assert _clean_speech_durations([
        SpeakerSegment(S0, 0, 5), SpeakerSegment(S0, 1, 3),
        SpeakerSegment(S1, 2, 4), SpeakerSegment(S1, 5, 9),
    ]) == {S0: 3, S1: 4}


def test_pipeline_split_renumerote_aussi_les_empreintes(tmp_path, monkeypatch):
    registry = SpeakerRegistry(tmp_path)
    enroll(registry, "Alice", voice((1, 0)))
    enroll(registry, "Bob", voice((0, 1)))
    monkeypatch.setattr("transcription_server.pipeline.decode_channels",
                        lambda _: [np.zeros(80000), np.ones(80000)])

    class Diarizer:
        name = "test"

        def diarize(self, pcm, **kwargs):
            return DiarizationSegments([SpeakerSegment(S0, 0, 5)],
                                       {S0: voice((0, 1) if pcm[0] else (1, 0))})

    class Asr:
        name = "test"

        def transcribe(self, pcm, language):
            return [Word("Bob" if pcm[0] else "Alice", 0, 1)]

    output = run_pipeline("ignored", Asr(), Diarizer(),
                          TranscriptionRequest(channel_mode="split"),
                          480, 15, 1, speaker_registry=registry)
    assert output.speakers == ["Alice", "Bob"]
    assert [(t.speaker, t.speaker_label) for t in output.turns] == [("Alice", S0), ("Bob", S1)]
    corrected = registry.identify(output.transcription_id, S1, "Robert")
    assert corrected["name"] == "Robert"


@pytest.fixture
def client(tmp_path, monkeypatch):
    registry = SpeakerRegistry(tmp_path)
    monkeypatch.setattr("transcription_server.pipeline.decode_to_pcm", lambda _: np.zeros(80000))

    class Diarizer(StubDiarizationEngine):
        def diarize(self, *args, **kwargs):
            return DiarizationSegments(super().diarize(*args, **kwargs), {S0: voice()})

    app = create_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu", enable_tts=False),
        StubAsrEngine([Word("Bonjour", 0, 1)]),
        Diarizer([SpeakerSegment(S0, 0, 5)]), speaker_registry=registry,
    )
    with TestClient(app) as client:
        yield client


def transcribe(client, **data):
    response = client.post("/transcribe", files={"file": ("audio.wav", b"fixture")},
                           data={"diarize": "true", **data})
    assert response.status_code == 200, response.text
    return response


def test_cycle_http_et_exports(client):
    first = transcribe(client).json()
    assert first["speaker_matches"][S0]["enrollable"]
    person = client.post("/speakers/identify", json={
        "transcription_id": first["transcription_id"], "speaker": S0, "name": "Élodie",
    })
    assert person.status_code == 200
    person_id = person.json()["id"]
    second = transcribe(client).json()
    assert second["speakers"] == ["Élodie"]
    assert second["turns"][0]["speaker"] == "Élodie"
    assert second["turns"][0]["speaker_label"] == S0
    assert "embeddings" not in str(second)
    assert "Élodie" in transcribe(client, response_format="dialogue").text
    assert client.get("/speakers").json() == [person.json()]
    assert client.patch(f"/speakers/{person_id}", json={"name": "Alice"}).status_code == 200
    assert transcribe(client).json()["speakers"] == ["Alice"]
    assert client.delete(f"/speakers/{person_id}").status_code == 204
    assert transcribe(client).json()["speakers"] == [S0]
    assert client.get("/health").json()["speaker_recognition_enabled"] is True


@pytest.mark.parametrize("name", ["", "   ", "x" * 101, "Alice\nBob", "SPEAKER_01"])
def test_noms_invalides(client, name):
    first = transcribe(client).json()
    assert client.post("/speakers/identify", json={
        "transcription_id": first["transcription_id"], "speaker": S0, "name": name,
    }).status_code == 422


def test_requetes_inconnues_et_sans_diarization(client):
    body = {"transcription_id": str(uuid4()), "speaker": S0, "name": "Alice"}
    assert client.post("/speakers/identify", json=body).status_code == 404
    body["transcription_id"] = transcribe(client).json()["transcription_id"]
    body["speaker"] = S1
    assert client.post("/speakers/identify", json=body).status_code == 404
    body["transcription_id"] = "../../etc/passwd"
    assert client.post("/speakers/identify", json=body).status_code == 422
    assert client.delete(f"/speakers/{uuid4()}").status_code == 404
    plain = transcribe(client, diarize="false").json()
    assert plain["transcription_id"] is None
    assert plain["speaker_matches"] == {}


def test_renommage_en_doublon_refuse(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    enroll(registry)
    bob = enroll(registry, "Bob", voice((0, 1)))
    with pytest.raises(ValueError, match="déjà"):
        registry.rename(bob["id"], "ALICE")
    assert registry.list()[1]["name"] == "Bob"
