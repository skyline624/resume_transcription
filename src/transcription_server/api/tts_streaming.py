"""Reponse publique PCM : premier audio avant les en-tetes, nettoyage garanti."""
from collections.abc import Callable, Iterable
from contextlib import aclosing
import time

import anyio
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from transcription_server.state import AppState
from transcription_server.tts.domain import SynthesisRequest, TtsUnavailableError


async def pcm_response(
    state: AppState, requests: Iterable[tuple[SynthesisRequest, int]],
    cleanup: Callable[[], None] | None = None,
    http_request: Request | None = None,
) -> StreamingResponse:
    async def generate():
        async with state.gpu_lock:
            await state.prepare_tts()
            pause_before = 0
            for request, pause_ms in requests:
                if pause_before:
                    yield b"\0\0" * (24 * pause_before)
                async with aclosing(state.tts.stream(request)) as chunks:
                    async for chunk in chunks:
                        yield chunk
                pause_before = pause_ms

    iterator = generate()
    closed = False

    async def close():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            with anyio.CancelScope(shield=True):
                await iterator.aclose()
        finally:
            if cleanup is not None:
                cleanup()

    started = time.perf_counter()
    try:
        if http_request is None:
            first = await anext(iterator)
        else:
            # StreamingResponse surveille ensuite la deconnexion. Avant les
            # en-tetes, la route doit elle-meme annuler une attente GPU longue.
            first = None
            failure = None
            async with anyio.create_task_group() as group:
                async def disconnected():
                    while True:
                        if (await http_request.receive())["type"] == "http.disconnect":
                            group.cancel_scope.cancel()
                            return
                group.start_soon(disconnected)
                try:
                    first = await anext(iterator)
                except Exception as exc:
                    failure = exc
                group.cancel_scope.cancel()
            if failure is not None:
                raise failure
            if first is None:
                raise HTTPException(status_code=499, detail="Lecture interrompue.")
    except BaseException as exc:
        await close()
        if isinstance(exc, StopAsyncIteration):
            raise TtsUnavailableError("empty_audio", "Aucun audio produit.") from exc
        raise

    async def body():
        try:
            yield first
            async for chunk in iterator:
                yield chunk
        finally:
            await close()

    return StreamingResponse(body(), media_type="audio/pcm", headers={
        "X-Audio-Sample-Rate": "24000", "X-Audio-Channels": "1",
        "X-TTS-First-Audio-Ms": f"{(time.perf_counter()-started)*1000:.3f}",
        "Cache-Control": "no-store", "X-Accel-Buffering": "no",
    }, background=BackgroundTask(close))
