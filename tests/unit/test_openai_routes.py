"""Endpoints compatibles OpenAI.

Ces tests pilotent toute la surface HTTP avec des moteurs factices : ni GPU,
ni conteneur, ni modele telecharge.
"""

import io
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import StubDiarizationEngine
from transcription_server.domain import Word

TOKEN = "hf_pour_les_tests"


def _wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    """Un wav mono 16 bits, silencieux : le moteur factice ignore le signal."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def client():
    settings = Settings(_env_file=None, enable_diarization=False, device="cpu")
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("tous", 0.6, 1.0)]),
        diarization=StubDiarizationEngine([]),
    )
    return TestClient(app)


def test_json_par_defaut(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "bonjour tous"}


def test_format_text(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "text"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "bonjour tous"


def test_format_srt(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "srt"},
    )
    assert response.status_code == 200
    assert response.text.startswith("1\n00:00:00,000 --> ")


def test_format_vtt(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "vtt"},
    )
    assert response.status_code == 200
    assert response.text.startswith("WEBVTT\n\n")


def test_verbose_json(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "verbose_json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["text"] == "bonjour tous"
    assert body["duration"] == pytest.approx(2.0, rel=0.05)
    assert body["segments"][0]["text"] == "bonjour tous"
    assert body["words"][0]["word"] == "bonjour"
    assert body["words"][0]["start"] == pytest.approx(0.0)


def test_verbose_json_echo_la_langue_demandee(client):
    """La langue detectee ne remonte pas : `transcribe` etant appele une fois
    par fenetre, un Protocol elargi rendrait N langues. On echo donc ce que
    l'appelant a demande, et `null` s'il n'a rien demande."""
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "verbose_json", "language": "fr"},
    )
    assert response.json()["language"] == "fr"

    sans = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "verbose_json"},
    )
    assert sans.json()["language"] is None


def test_la_langue_est_transmise_au_moteur():
    """Le stub ignorant son entree, seul un moteur qui enregistre ses appels
    peut prouver que `language` traverse reellement la route."""
    recu: list[str | None] = []

    class AsrQuiNoteLaLangue:
        name = "note-langue"

        def transcribe(self, audio, language):
            recu.append(language)
            return [Word("bonjour", 0.0, 0.5)]

    app = create_app(
        settings=Settings(_env_file=None, enable_diarization=False, device="cpu"),
        asr=AsrQuiNoteLaLangue(),
        diarization=StubDiarizationEngine([]),
    )
    TestClient(app).post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"language": "fr"},
    )
    assert recu == ["fr"]


def test_la_diarization_n_est_jamais_appelee():
    """L'endpoint OpenAI n'expose pas la diarization : le moteur ne doit pas
    etre sollicite, meme si le serveur en a un."""
    appels: list[int] = []

    class DiarQuiCompte:
        name = "compte"

        def diarize(self, audio, num_speakers, min_speakers, max_speakers):
            appels.append(1)
            return []

    app = create_app(
        settings=Settings(_env_file=None, hf_token=TOKEN, device="cpu"),
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.5)]),
        diarization=DiarQuiCompte(),
    )
    response = TestClient(app).post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert appels == []


def test_fichier_illisible_donne_400_au_format_openai(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("junk.wav", b"pas de l'audio", "audio/wav")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    # Le chemin du fichier temporaire et le stderr de ffmpeg restent au journal.
    assert "ffmpeg" not in body["error"]["message"].lower()
    assert "\\" not in body["error"]["message"]


def test_fichier_trop_gros_donne_413():
    settings = Settings(
        _env_file=None, enable_diarization=False, device="cpu", max_upload_mb=1
    )
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("a", 0.0, 0.5)]),
        diarization=StubDiarizationEngine([]),
    )
    response = TestClient(app).post(
        "/v1/audio/transcriptions",
        files={"file": ("gros.wav", b"\x00" * (2 * 1024 * 1024), "audio/wav")},
    )
    assert response.status_code == 413


def test_format_inconnu_donne_422(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "yaml"},
    )
    assert response.status_code == 422


def _espionner_le_fichier_temporaire(monkeypatch) -> list:
    """Intercepte le chemin cree par `_save_upload` pour verifier sa suppression.

    Sans cela, rien n'atteste le nettoyage : un serveur qui garde ses fichiers
    remplirait le disque du conteneur a raison d'environ 100 Mo par reunion.
    """
    from transcription_server.api import openai_routes

    vrai_save_upload = openai_routes._save_upload
    chemins = []

    async def espion(upload, max_bytes):
        chemin = await vrai_save_upload(upload, max_bytes)
        chemins.append(chemin)
        return chemin

    monkeypatch.setattr(openai_routes, "_save_upload", espion)
    return chemins


def test_le_fichier_temporaire_est_supprime_apres_succes(client, monkeypatch):
    chemins = _espionner_le_fichier_temporaire(monkeypatch)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert chemins, "_save_upload n'a pas ete appele"
    assert not chemins[0].exists()


def test_le_fichier_temporaire_est_supprime_apres_echec(client, monkeypatch):
    """Le chemin d'erreur est celui qui fuit le plus facilement : c'est aussi
    celui qu'un attaquant declencherait en boucle."""
    chemins = _espionner_le_fichier_temporaire(monkeypatch)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("junk.wav", b"pas de l'audio", "audio/wav")},
    )
    assert response.status_code == 400
    assert chemins, "_save_upload n'a pas ete appele"
    assert not chemins[0].exists()


def test_liste_des_modeles(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert any(entree["id"] == "stub-asr" for entree in body["data"])
    assert body["data"][0]["object"] == "model"
