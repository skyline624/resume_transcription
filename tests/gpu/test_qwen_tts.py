"""Preuves audio reelles des trois modes Qwen3-TTS."""

import io
import os
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1200)]
BASE_URL = os.getenv("TTS_TEST_BASE_URL", "http://127.0.0.1:8000")
FIXTURES = Path(__file__).parents[1] / "fixtures"
REFERENCE_TRANSCRIPT = "Bonjour à tous, merci de votre présence à cette réunion."


def assert_valid_wav(content: bytes) -> None:
    assert content.startswith(b"RIFF")
    with wave.open(io.BytesIO(content), "rb") as handle:
        assert handle.getframerate() == 24_000
        assert handle.getnchannels() == 1
        assert handle.getnframes() > 0
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert samples.size > 0
    assert np.isfinite(samples.astype(np.float32)).all()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "model": "tts-1-hd",
            "voice": "Ryan",
            "input": "Bonjour, ceci est une validation de la voix française.",
            "language": "fr",
            "response_format": "wav",
        },
        {
            "model": "qwen3-tts-voice-design",
            "input": "Le soleil se lève doucement sur la vallée.",
            "instructions": "Une voix française chaleureuse, calme et souriante.",
            "language": "fr",
            "response_format": "wav",
        },
    ],
    ids=["custom-voice", "voice-design"],
)
def test_custom_voice_et_voice_design_produisent_un_wav_24khz(payload):
    with httpx.Client(base_url=BASE_URL, timeout=1200) as client:
        response = client.post("/v1/audio/speech", json=payload)
    response.raise_for_status()
    assert_valid_wav(response.content)


def test_clone_produit_un_wav_24khz_depuis_reference_consentie():
    reference = FIXTURES / "echantillon_fr.wav"
    with reference.open("rb") as audio, httpx.Client(
        base_url=BASE_URL, timeout=1200
    ) as client:
        response = client.post(
            "/v1/audio/speech/clone",
            files={"file": (reference.name, audio, "audio/wav")},
            data={
                "input": "Merci de participer à cette validation vocale.",
                "transcript": REFERENCE_TRANSCRIPT,
                "language": "fr",
                "consent": "true",
                "response_format": "wav",
            },
        )
    response.raise_for_status()
    assert_valid_wav(response.content)
