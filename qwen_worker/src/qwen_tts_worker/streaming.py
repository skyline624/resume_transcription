"""Pont entre la generation GPU synchrone et le flux HTTP annullable."""
import asyncio
import queue
import threading
import time
from contextlib import closing
from typing import Annotated
from uuid import UUID, uuid4

import anyio
import numpy as np
from fastapi import Header
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from qwen_tts_worker.domain import GenerateCommand, Mode, WorkerModelError

MODES = {"qwen3-tts-custom-voice": Mode.CUSTOM, "qwen3-tts-clone": Mode.CLONE,
         "qwen3-tts-voice-design": Mode.DESIGN}


class StreamJob:
    """Un seul thread possede le generateur et son verrou jusqu'a fermeture."""
    def __init__(self, manager, command):
        self._stop = threading.Event()
        self._done = threading.Event()
        self._queue = queue.Queue(maxsize=4)
        self._task = asyncio.create_task(asyncio.to_thread(self._produce, manager, command))

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _produce(self, manager, command):
        try:
            if self._stop.is_set():
                return
            with closing(manager.stream(command)) as chunks:
                for waveform, rate in chunks:
                    if self._stop.is_set():
                        break
                    if rate != 24000:
                        raise ValueError("Frequence PCM inattendue")
                    samples = np.asarray(waveform, dtype=np.float32)
                    if not np.isfinite(samples).all():
                        raise ValueError("Audio non fini")
                    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                    if pcm:
                        self._put(pcm)
        except Exception as exc:
            error = exc if isinstance(exc, WorkerModelError) else WorkerModelError(
                "generation_failed", "La generation a echoue.")
            self._put(error)
        finally:
            self._done.set()

    def _next(self):
        while True:
            try:
                return self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._done.is_set():
                    return None

    async def next_chunk(self):
        item = await asyncio.to_thread(self._next)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self._stop.set()
        # Une deconnexion ne doit pas laisser une inference tourner sans suivi.
        with anyio.CancelScope(shield=True):
            await self._task


def register_stream_routes(app, manager, payload_type):
    jobs = {}
    app.state.stream_jobs = jobs

    @app.post("/generate/stream")
    async def generate_stream(
        payload: payload_type,
        request_id: Annotated[UUID | None, Header(alias="X-TTS-Request-Id")] = None,
    ):
        identifier = str(request_id or uuid4())
        if identifier in jobs:
            return JSONResponse(status_code=409, content={"code": "duplicate_request"})
        try:
            command = GenerateCommand(
                mode=MODES[payload.mode], text=payload.text, language=payload.language,
                speaker=payload.speaker, instruct=payload.instruct,
                reference_audio=payload.reference_audio, reference_text=payload.reference_text,
            )
        except (KeyError, ValueError):
            return JSONResponse(status_code=422, content={"code": "invalid_request"})
        started = time.perf_counter()
        job = jobs[identifier] = StreamJob(manager, command)

        async def cleanup():
            await job.close()
            jobs.pop(identifier, None)

        try:
            first = await job.next_chunk()
            if first is None:
                raise WorkerModelError("empty_audio", "Aucun audio produit.")
        except BaseException as exc:
            await cleanup()
            if not isinstance(exc, WorkerModelError):
                raise
            return JSONResponse(status_code=503, content={
                "code": exc.code, "message": "Le modele TTS est indisponible.",
            })

        async def body():
            try:
                yield first
                while (chunk := await job.next_chunk()) is not None:
                    yield chunk
            finally:
                await cleanup()

        return StreamingResponse(body(), media_type="audio/pcm", headers={
            "X-TTS-Sample-Rate": "24000", "X-TTS-Request-Id": identifier,
            "X-TTS-First-Audio-Ms": f"{(time.perf_counter()-started)*1000:.3f}",
        }, background=BackgroundTask(cleanup))

    @app.post("/cancel/{request_id}", status_code=204)
    async def cancel(request_id: UUID):
        job = jobs.get(str(request_id))
        if job is not None:
            await job.close()
        return Response(status_code=204)
