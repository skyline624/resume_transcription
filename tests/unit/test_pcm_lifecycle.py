"""Une interruption ferme le producteur avant de rendre le GPU disponible."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import HTTPException
from transcription_server.api.tts_streaming import pcm_response


@pytest.mark.asyncio
async def test_deconnexion_avant_premier_audio_annule_et_libere_verrou():
    started = anyio.Event()
    closed = []
    async def stream(_request):
        try:
            started.set()
            await anyio.sleep_forever()
            yield b"audio"
        finally:
            closed.append(True)
    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}
    state = SimpleNamespace(gpu_lock=asyncio.Lock(), prepare_tts=AsyncMock(), tts=SimpleNamespace(stream=stream))
    with pytest.raises(HTTPException) as error:
        await pcm_response(state, [(None, 0)], http_request=SimpleNamespace(receive=receive))
    assert error.value.status_code == 499
    assert closed == [True]
    assert not state.gpu_lock.locked()


@pytest.mark.asyncio
async def test_fermeture_pendant_lecture_ferme_source_et_reference():
    closed = []
    async def stream(_request):
        try:
            yield b"12"
            yield b"34"
        finally:
            closed.append("source")
    state = SimpleNamespace(gpu_lock=asyncio.Lock(), prepare_tts=AsyncMock(), tts=SimpleNamespace(stream=stream))
    response = await pcm_response(state, [(None, 0)], cleanup=lambda: closed.append("reference"))
    assert state.gpu_lock.locked()
    assert await anext(response.body_iterator) == b"12"
    await response.body_iterator.aclose()
    assert closed == ["source", "reference"]
    assert not state.gpu_lock.locked()


@pytest.mark.asyncio
async def test_producteur_arrete_avant_acquittement():
    import threading
    from qwen_tts_worker.streaming import StreamJob
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    def stream(_request):
        try:
            yield [0.0], 24000
            entered.set()
            assert release.wait(5)
            yield [1.0], 24000
        finally:
            closed.set()
    job = StreamJob(SimpleNamespace(stream=stream), None)
    assert await job.next_chunk() == b"\0\0"
    assert await asyncio.to_thread(entered.wait, 2)
    closing = asyncio.create_task(job.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    assert closed.is_set()
