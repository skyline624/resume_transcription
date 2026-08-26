from pathlib import Path

import pytest

from transcription_server.tts.client import TransportResponse, UnixTtsClient
from transcription_server.tts.domain import (
    SynthesisRequest,
    TtsMode,
    TtsUnavailableError,
)


class FakeTransport:
    def __init__(self, response: TransportResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict | None, float]] = []

    async def request(self, method, path, payload, timeout_s):
        self.calls.append((method, path, payload, timeout_s))
        return self.response


def make_client(transport: FakeTransport) -> UnixTtsClient:
    return UnixTtsClient(
        socket_path=Path("/run/test.sock"),
        load_timeout_s=10.0,
        generation_timeout_s=20.0,
        transport=transport,
    )


@pytest.mark.asyncio
async def test_clone_envoie_un_chemin_interne_et_traduit_le_francais():
    transport = FakeTransport(
        TransportResponse(
            status=200,
            body=b"RIFFaudio",
            headers={
                "X-TTS-Sample-Rate": "24000",
                "X-TTS-Model": "base",
                "X-TTS-Load-Ms": "12.5",
                "X-TTS-Inference-Ms": "45.25",
            },
        )
    )
    client = make_client(transport)

    result = await client.synthesize(
        SynthesisRequest(
            text="Bonjour",
            mode=TtsMode.CLONE,
            reference_path=Path("/app/voices/audio/id.wav"),
            reference_text="Texte de référence",
        )
    )

    payload = transport.calls[0][2]
    assert payload is not None
    assert payload["reference_audio"] == "/app/voices/audio/id.wav"
    assert payload["language"] == "French"
    assert result.audio_wav == b"RIFFaudio"
    assert result.sample_rate == 24000
    assert result.load_ms == 12.5
    assert result.inference_ms == 45.25


@pytest.mark.asyncio
async def test_oom_du_worker_devient_une_indisponibilite_stable():
    transport = FakeTransport(
        TransportResponse(
            status=503,
            body=b'{"code":"cuda_oom","message":"interne"}',
            headers={"content-type": "application/json"},
        )
    )

    with pytest.raises(TtsUnavailableError) as error:
        await make_client(transport).synthesize(
            SynthesisRequest(
                text="Bonjour", mode=TtsMode.CUSTOM_VOICE, voice="Ryan"
            )
        )

    assert error.value.code == "cuda_oom"
    assert "interne" not in str(error.value)


@pytest.mark.asyncio
async def test_audio_vide_est_refuse():
    transport = FakeTransport(
        TransportResponse(
            status=200,
            body=b"",
            headers={"X-TTS-Sample-Rate": "24000", "X-TTS-Model": "custom"},
        )
    )
    with pytest.raises(TtsUnavailableError) as error:
        await make_client(transport).synthesize(
            SynthesisRequest(text="Bonjour", mode=TtsMode.CUSTOM_VOICE, voice="Ryan")
        )
    assert error.value.code == "empty_audio"


@pytest.mark.asyncio
async def test_unload_est_idempotent_quand_worker_repond_204():
    transport = FakeTransport(TransportResponse(status=204, body=b"", headers={}))
    await make_client(transport).unload(reason="diarization")
    assert transport.calls == [
        ("POST", "/unload", {"reason": "diarization"}, 10.0)
    ]
