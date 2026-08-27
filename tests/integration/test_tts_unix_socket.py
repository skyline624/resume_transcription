from pathlib import Path

import pytest

from transcription_server.tts.client import UnixTtsClient
from transcription_server.tts.domain import SynthesisRequest, TtsMode


@pytest.mark.asyncio
async def test_vrai_transport_unix_execute_health_generate_et_unload(live_uds_worker):
    client = UnixTtsClient(
        socket_path=live_uds_worker.socket_path,
        load_timeout_s=10,
        generation_timeout_s=10,
    )

    health = await client.health()
    result = await client.synthesize(
        SynthesisRequest(
            text="Bonjour depuis le socket Unix.",
            mode=TtsMode.CUSTOM_VOICE,
            voice="Ryan",
        )
    )
    await client.unload(reason="integration-test")

    output = Path(live_uds_worker.socket_path.parent) / "speech.wav"
    output.write_bytes(result.audio_wav)
    assert health.state == "idle"
    assert output.read_bytes().startswith(b"RIFF")
    assert result.sample_rate == 24_000
    assert [name for name, _ in live_uds_worker.calls] == [
        "health",
        "generate",
        "unload",
    ]
    assert live_uds_worker.calls[1][1]["language"] == "French"
    assert live_uds_worker.calls[2][1] == {"reason": "integration-test"}
