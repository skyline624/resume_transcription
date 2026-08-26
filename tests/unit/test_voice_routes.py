import io
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import NullDiarizationEngine
from transcription_server.domain import Word
from transcription_server.tts.domain import SynthesisResult, WorkerHealth
from transcription_server.tts.profiles import VoiceProfileRepository
from transcription_server.tts.reference import NormalizedReference


def wav_bytes(rate=24000, seconds=4) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 100) * rate * seconds)
    return output.getvalue()


class FakeTts:
    def __init__(self):
        self.requests = []

    async def synthesize(self, request):
        self.requests.append(request)
        return SynthesisResult(wav_bytes(seconds=1), 24000, request.mode.value)

    async def unload(self, reason):
        return None

    async def health(self):
        return WorkerHealth(
            available=True, state="idle", speakers=("Ryan", "Vivian")
        )


@pytest.fixture
def voice_client(tmp_path, monkeypatch):
    normalized_paths = []

    def fake_prepare(source, output_directory, vad, limits):
        output_directory.mkdir(parents=True, exist_ok=True)
        path = output_directory / "normalized.wav"
        path.write_bytes(wav_bytes())
        normalized_paths.append(path)
        return NormalizedReference(path, 4.0, 0.1, -20.0, 0.0)

    import transcription_server.api.voice_routes as routes
    monkeypatch.setattr(routes, "prepare_reference", fake_prepare)
    tts = FakeTts()
    root = tmp_path / "voices"
    app = create_app(
        settings=Settings(
            _env_file=None, enable_diarization=False, device="cpu",
            voice_store_path=root,
        ),
        asr=StubAsrEngine([Word("bonjour", 0.0, 1.0)]),
        diarization=NullDiarizationEngine(),
        tts=tts,
        voice_profiles=VoiceProfileRepository(root),
    )
    return TestClient(app), tts, normalized_paths


def test_creation_exige_un_consentement_explicite(voice_client):
    client, _, _ = voice_client
    response = client.post(
        "/v1/voices",
        files={"file": ("voice.wav", wav_bytes(), "audio/wav")},
        data={"name": "Ma voix", "consent": "false"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "consent_required"


def test_creation_liste_et_suppression_d_un_clone(voice_client):
    client, _, normalized_paths = voice_client
    created = client.post(
        "/v1/voices",
        files={"file": ("voice.wav", wav_bytes(), "audio/wav")},
        data={
            "name": "Ma voix", "language": "fr", "transcript": "Bonjour",
            "consent": "true",
        },
    )
    assert created.status_code == 201
    voice_id = created.json()["id"]
    assert "audio_path" not in created.json()
    assert normalized_paths[0].exists() is False

    listed = client.get("/v1/voices")
    assert listed.status_code == 200
    assert {voice["kind"] for voice in listed.json()["data"]} == {"builtin", "clone"}

    deleted = client.delete(f"/v1/voices/{voice_id}")
    assert deleted.status_code == 204
    assert client.delete(f"/v1/voices/{voice_id}").status_code == 404


def test_clone_ponctuel_supprime_la_reference_normalisee(voice_client):
    client, tts, normalized_paths = voice_client
    response = client.post(
        "/v1/audio/speech/clone",
        files={"file": ("voice.wav", wav_bytes(), "audio/wav")},
        data={
            "input": "Bonjour", "transcript": "Bonjour", "consent": "true",
            "response_format": "wav",
        },
    )
    assert response.status_code == 200
    assert tts.requests[0].reference_path is not None
    assert normalized_paths[0].exists() is False


def test_clone_ponctuel_refuse_instructions(voice_client):
    client, _, _ = voice_client
    response = client.post(
        "/v1/audio/speech/clone",
        files={"file": ("voice.wav", wav_bytes(), "audio/wav")},
        data={"input": "Bonjour", "consent": "true", "instructions": "vite"},
    )
    assert response.status_code == 422
