"""Cycle VRAM et enchainement croise des charges GPU."""

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest
import torch

from transcription_server.tts.client import UnixTtsClient

from .test_qwen_tts import REFERENCE_TRANSCRIPT, assert_valid_wav


pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1200)]
BASE_URL = os.getenv("TTS_TEST_BASE_URL", "http://127.0.0.1:8000")
WORKER_SOCKET = Path(os.getenv("TTS_WORKER_SOCKET", "/run/qwen-tts/worker.sock"))
FIXTURES = Path(__file__).parents[1] / "fixtures"
MODELS = {
    "custom": os.getenv(
        "TTS_CUSTOM_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    ),
    "design": os.getenv(
        "TTS_VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    ),
    "clone": os.getenv("TTS_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
}


def unload_worker() -> None:
    client = UnixTtsClient(
        WORKER_SOCKET, load_timeout_s=120, generation_timeout_s=120
    )
    asyncio.run(client.unload(reason="gpu-lifecycle-test"))


def health(client: httpx.Client) -> dict:
    response = client.get("/health")
    response.raise_for_status()
    return response.json()


def synthesize(client: httpx.Client, model: str) -> bytes:
    payload = {
        "model": model,
        "input": "La qualité sonore reste stable pendant tout le test.",
        "language": "fr",
        "response_format": "wav",
    }
    if model == "tts-1-hd":
        payload["voice"] = "Ryan"
    else:
        payload["instructions"] = "Une voix française posée et naturelle."
    response = client.post("/v1/audio/speech", json=payload)
    response.raise_for_status()
    assert_valid_wav(response.content)
    return response.content


def clone(client: httpx.Client) -> bytes:
    reference = FIXTURES / "echantillon_fr.wav"
    with reference.open("rb") as audio:
        response = client.post(
            "/v1/audio/speech/clone",
            files={"file": (reference.name, audio, "audio/wav")},
            data={
                "input": "Le clonage vocal utilise une référence consentie.",
                "transcript": REFERENCE_TRANSCRIPT,
                "consent": "true",
                "response_format": "wav",
            },
        )
    response.raise_for_status()
    assert_valid_wav(response.content)
    return response.content


def assert_loaded(client: httpx.Client, expected_model: str) -> None:
    tts = health(client)["tts"]
    assert tts["state"] == "ready"
    assert tts["loaded_model"] == expected_model


def test_un_seul_checkpoint_est_charge_et_la_vram_est_restituee():
    assert torch.cuda.is_available()
    unload_worker()
    torch.cuda.synchronize()
    baseline_free, _ = torch.cuda.mem_get_info()

    with httpx.Client(base_url=BASE_URL, timeout=1200) as client:
        synthesize(client, "tts-1-hd")
        assert_loaded(client, MODELS["custom"])

        synthesize(client, "qwen3-tts-voice-design")
        assert_loaded(client, MODELS["design"])

        clone(client)
        assert_loaded(client, MODELS["clone"])

        unload_worker()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            torch.cuda.synchronize()
            free_after, _ = torch.cuda.mem_get_info()
            state = health(client)["tts"]["state"]
            if state == "idle" and baseline_free - free_after <= 512 * 2**20:
                break
            time.sleep(0.1)
        else:
            pytest.fail("La VRAM n'est pas revenue à moins de 512 Mio du niveau initial.")


def test_scenario_custom_transcription_clone_diarization_design():
    with httpx.Client(base_url=BASE_URL, timeout=1200) as client:
        custom_audio = synthesize(client, "tts-1-hd")
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("custom.wav", custom_audio, "audio/wav")},
            data={"model": "parakeet", "language": "fr"},
        )
        response.raise_for_status()
        assert response.json()["text"].strip()

        clone(client)

        diarization_reference = FIXTURES / "deux_voix.wav"
        with diarization_reference.open("rb") as audio:
            response = client.post(
                "/transcribe",
                files={"file": (diarization_reference.name, audio, "audio/wav")},
                data={"language": "fr", "diarize": "true"},
            )
        response.raise_for_status()

        synthesize(client, "qwen3-tts-voice-design")
        assert health(client)["tts"]["state"] in {"ready", "idle"}
