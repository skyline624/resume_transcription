import io
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import NullDiarizationEngine
from transcription_server.tts.domain import SynthesisResult, TtsMode
from transcription_server.tts.profiles import VoiceProfileRepository


def wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack("<h", 100) * 2400)
    return output.getvalue()


class FakeTts:
    def __init__(self):
        self.requests = []

    async def synthesize(self, request):
        self.requests.append(request)
        return SynthesisResult(wav_bytes(), 24000, request.mode.value)

    async def unload(self, reason):
        return None

    async def health(self):
        raise AssertionError("health n'est pas utilisé par cette route")


@pytest.fixture
def tts_client(tmp_path):
    tts = FakeTts()
    app = create_app(
        settings=Settings(
            _env_file=None, enable_diarization=False, device="cpu",
            voice_store_path=tmp_path / "voices",
        ),
        asr=StubAsrEngine([]),
        diarization=NullDiarizationEngine(),
        tts=tts,
        voice_profiles=VoiceProfileRepository(tmp_path / "voices"),
    )
    return TestClient(app), tts


def test_alias_openai_utilise_custom_voice_en_francais(tts_client):
    client, tts = tts_client
    response = client.post("/v1/audio/speech", json={
        "model": "tts-1-hd",
        "input": "Bonjour",
        "voice": "Ryan",
        "response_format": "wav",
        "speed": 1.0,
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert tts.requests[0].mode is TtsMode.CUSTOM_VOICE
    assert tts.requests[0].language == "fr"


@pytest.mark.parametrize("alias", ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"])
def test_les_trois_alias_sont_acceptes(tts_client, alias):
    client, _ = tts_client
    response = client.post("/v1/audio/speech", json={
        "model": alias, "input": "Bonjour", "voice": "Ryan"
    })
    assert response.status_code == 200


def test_voice_design_exige_des_instructions(tts_client):
    client, _ = tts_client
    response = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-voice-design", "input": "Bonjour"
    })
    assert response.status_code == 422


def test_texte_superieur_a_4096_caracteres_est_refuse(tts_client):
    client, tts = tts_client
    response = client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "a" * 4097, "voice": "Ryan"
    })
    assert response.status_code == 422
    assert tts.requests == []


def test_instructions_sont_refusees_pour_clone(tts_client):
    client, _ = tts_client
    response = client.post("/v1/audio/speech", json={
        "model": "qwen3-tts-clone", "input": "Bonjour", "voice": "id",
        "instructions": "Parler vite",
    })
    assert response.status_code == 422
