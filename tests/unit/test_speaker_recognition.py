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


# --- Registre central : references gardees par le client (TalkTrack) ---

MODEL = "test"


def reference(registry, source="talktrack:rec_a", speaker=S0, values=(1.0, 0.0), **target):
    return registry.add_reference(source, speaker, MODEL, list(values), **target)


def test_reference_par_nom_par_id_et_voix_automatique(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    alice = reference(registry, name="Alice")
    assert (alice["name"], alice["samples"], alice["auto"]) == ("Alice", 1, False)
    again = reference(registry, speaker=S1, values=(0.98, 0.1), person_id=alice["id"])
    assert again["samples"] == 2
    voix = reference(registry, source="talktrack:rec_b", values=(0, 1), auto=True)
    assert (voix["name"], voix["auto"]) == ("Voix 1", True)
    # Reconnue a la transcription suivante, comme une personne nommee.
    assert registry.recognize(result(), {S0: voice((0.0, 1.0))}).speakers == ["Voix 1"]
    assert registry.recognize(result(), {S0: voice((1.0, 0.0))}).speakers == ["Alice"]


def test_numero_automatique_jamais_reattribue(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    first = reference(registry, auto=True)
    registry.delete(first["id"])
    assert reference(registry, auto=True)["name"] == "Voix 2"
    reference(registry, source="s2", name="Voix 3")  # un nom deja pris est saute
    assert reference(registry, source="s3", auto=True)["name"] == "Voix 4"


def test_renvoyer_une_reference_la_deplace(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    voix = reference(registry, auto=True)
    bob = reference(registry, name="Bob")  # meme cle : la reference passe a Bob
    assert bob["samples"] == 1
    assert [p["name"] for p in registry.list()] == ["Bob"]  # la voix vide disparait
    assert voix["id"] not in {p["id"] for p in registry.list()}


@pytest.mark.parametrize("target", [{}, {"name": "A", "auto": True},
                                     {"name": "A", "person_id": str(uuid4())}])
def test_une_seule_cible(tmp_path, target):
    with pytest.raises(ValueError, match="une seule cible"):
        reference(SpeakerRegistry(tmp_path), **target)


def test_reference_inutilisable_ou_personne_inconnue(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    with pytest.raises(ValueError, match="inutilisable"):
        reference(registry, values=(0.0, 0.0), name="Alice")
    with pytest.raises(ValueError, match="inutilisable"):
        registry.add_reference("s", S0, MODEL, [1.0, 0.0], speech_s=1.0, name="Alice")
    with pytest.raises(KeyError):
        reference(registry, person_id=str(uuid4()))
    assert registry.list() == []


def test_detacher_une_reference(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    alice = reference(registry, name="Alice")
    reference(registry, speaker=S1, person_id=alice["id"])
    registry.remove_reference("talktrack:rec_a", S0)
    registry.remove_reference("talktrack:rec_a", S0)  # idempotent
    assert registry.list()[0]["samples"] == 1
    registry.remove_reference("talktrack:rec_a", S1)
    assert registry.list() == []


def test_fusion_alias_et_suppression(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    alice = reference(registry, name="Alice")
    voix = reference(registry, source="s2", values=(0.9, 0.3), auto=True)
    other = reference(registry, source="s3", values=(0.8, 0.5), auto=True)
    registry.merge(other["id"], voix["id"])
    merged = registry.merge(voix["id"], alice["id"])
    assert (merged["name"], merged["samples"]) == ("Alice", 3)
    assert registry.aliases() == {voix["id"]: alice["id"], other["id"]: alice["id"]}
    # Un ancien id reste une cible valable.
    assert reference(registry, source="s4", person_id=other["id"])["id"] == alice["id"]
    with pytest.raises(ValueError):
        registry.merge(alice["id"], alice["id"])
    with pytest.raises(KeyError):
        registry.merge(str(uuid4()), alice["id"])
    registry.delete(alice["id"])
    assert registry.aliases() == {}


def test_renommer_une_voix_automatique_la_nomme(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    voix = reference(registry, auto=True)
    assert registry.rename(voix["id"], "Corvus")["auto"] is False


def test_import_relancable_et_joint_les_memes_noms(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    existing = reference(registry, source="s0", name="Alice")
    persons = [
        {"key": "p_a", "name": "alice", "auto": False,
         "references": [{"source": "talktrack:r1", "speaker": S0, "vector": [1.0, 0.1]}]},
        {"key": "p_v", "name": "Voix 5", "auto": True, "auto_number": 5,
         "references": [{"source": "talktrack:r1", "speaker": S1, "vector": [0.1, 1.0]}]},
        {"key": "p_e", "name": "Sans empreinte", "references": []},
    ]
    first = registry.import_people(MODEL, persons, next_auto_number=6)
    assert first["p_a"]["id"] == existing["id"]
    assert first["p_a"]["samples"] == 2
    assert first["p_v"]["auto"] is True
    assert "p_e" not in first
    assert registry.import_people(MODEL, persons, next_auto_number=6) == first
    assert reference(registry, source="s9", auto=True)["name"] == "Voix 6"


def test_import_refuse_ne_change_rien(tmp_path):
    registry = SpeakerRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.import_people(MODEL, [
            {"key": "a", "name": "Alice", "references": [
                {"source": "s", "speaker": S0, "vector": [1.0, 0.0]}]},
            {"key": "b", "name": "Bob", "references": [
                {"source": "s", "speaker": S1, "vector": [0.0, 0.0]}]}])
    assert registry.list() == []


def test_registre_existant_sans_colonne_auto(tmp_path):
    with sqlite3.connect(tmp_path / "speakers.sqlite3") as db:
        db.executescript("""
            CREATE TABLE persons (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                                  name_key TEXT UNIQUE NOT NULL);
            INSERT INTO persons VALUES ('p1', 'Alice', 'alice');
            CREATE TABLE samples (transcription_id TEXT NOT NULL, label TEXT NOT NULL,
                person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                model TEXT NOT NULL, vector TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY (transcription_id, label));
            INSERT INTO samples VALUES ('t', 'SPEAKER_00', 'p1', 'test', '[1.0, 0.0]', 0);
        """)
    db.close()
    registry = SpeakerRegistry(tmp_path)
    assert registry.list() == [{"id": "p1", "name": "Alice", "samples": 1, "auto": False,
                                "last_seen": "1970-01-01T00:00:00+00:00"}]
    assert registry.recognize(result(), {S0: voice()}).speakers == ["Alice"]


def test_cycle_http_du_registre_central(client):
    info = client.get("/speakers/info").json()
    assert info["model"] == "pyannote/speaker-diarization-community-1|pyannote4-centroids-v1"
    assert info["max_references"] == 5
    body = {"source": "talktrack:rec", "speaker": S0, "model": "test", "vector": [1, 0],
            "speech_s": 12.0, "auto": True}
    voix = client.post("/speakers/references", json=body)
    assert voix.status_code == 200, voix.text
    assert voix.json()["name"] == "Voix 1"
    assert client.post("/speakers/references", json={**body, "name": "Alice"}).status_code == 422
    named = {k: v for k, v in body.items() if k != "auto"}
    alice = client.post("/speakers/references",
                        json={**named, "speaker": S1, "vector": [0.9, 0.1], "name": "Alice"}).json()
    merged = client.post(f"/speakers/{voix.json()['id']}/merge", json={"into": alice["id"]})
    assert merged.json()["samples"] == 2
    assert client.get("/speakers/aliases").json() == {voix.json()["id"]: alice["id"]}
    assert transcribe(client).json()["speakers"] == ["Alice"]
    gone = client.delete("/speakers/references",
                         params={"source": "talktrack:rec", "speaker": S0})
    assert gone.status_code == 204
    assert client.get("/speakers").json()[0]["samples"] == 1
    imported = client.post("/speakers/import", json={"model": "test", "persons": [
        {"key": "k", "name": "Bob", "references": [
            {"source": "talktrack:r2", "speaker": S0, "vector": [0, 1]}]}]})
    assert imported.status_code == 200, imported.text
    assert imported.json()["persons"]["k"]["name"] == "Bob"
    assert client.post(f"/speakers/{uuid4()}/merge", json={"into": alice["id"]}).status_code == 404


def test_empreintes_rendues_sur_demande(client):
    plain = transcribe(client).json()
    assert "speaker_embeddings" not in plain
    data = transcribe(client, return_embeddings="true").json()
    entry = data["speaker_embeddings"][S0]
    assert entry["model"] == "test"
    assert entry["speech_s"] == 5.0
    assert entry["vector"] == [1.0, 0.0]
    assert data["speaker_matches"][S0]["enrollable"]


def test_empreintes_trop_courtes_sans_vecteur():
    from transcription_server.speakers import export_embeddings
    exported = export_embeddings([S0, S1], {S0: voice((3.0, 4.0)), S1: voice(speech_s=1.0)}, 3.0)
    assert exported[S0]["vector"] == [0.6, 0.8]
    assert exported[S1] == {"model": "test", "vector": None, "speech_s": 1.0}
