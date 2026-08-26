"""API privée du worker Qwen3-TTS."""

import asyncio
import io
import os
import wave
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from qwen_tts_worker.domain import GenerateCommand, Mode, WorkerModelError

SPEAKERS = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)
_PUBLIC_MODES = {
    "qwen3-tts-custom-voice": Mode.CUSTOM,
    "qwen3-tts-clone": Mode.CLONE,
    "qwen3-tts-voice-design": Mode.DESIGN,
}


class GeneratePayload(BaseModel):
    mode: str
    text: str
    language: str = "French"
    speaker: str | None = None
    instruct: str | None = None
    reference_audio: str | None = None
    reference_text: str | None = None


class UnloadPayload(BaseModel):
    reason: str = "explicit"


def create_worker_app(manager, downloaded_models: list[str], idle_poll_s: float = 30.0):
    @asynccontextmanager
    async def lifespan(app):
        async def monitor():
            while True:
                await asyncio.sleep(idle_poll_s)
                await asyncio.to_thread(manager.unload_if_idle)
        task = asyncio.create_task(monitor())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(manager.unload)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        state = manager.health()
        return {
            **state,
            "downloaded_models": downloaded_models,
            "precision": "bfloat16",
            "device": "cuda:0",
            "attention": "sdpa",
            "speakers": SPEAKERS,
            "features": ["custom_voice", "clone", "voice_design"],
            "pid": os.getpid(),
        }

    @app.post("/generate")
    async def generate(payload: GeneratePayload):
        try:
            mode = _PUBLIC_MODES[payload.mode]
            command = GenerateCommand(
                mode=mode, text=payload.text, language=payload.language,
                speaker=payload.speaker, instruct=payload.instruct,
                reference_audio=payload.reference_audio,
                reference_text=payload.reference_text,
            )
            waveform, rate, load_ms, inference_ms = await run_in_threadpool(
                manager.generate, command
            )
        except (KeyError, ValueError) as exc:
            return JSONResponse(
                status_code=422,
                content={"code": "invalid_request", "message": str(exc)},
            )
        except WorkerModelError as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "code": exc.code,
                    "message": "Le modèle TTS est indisponible.",
                },
            )
        model = manager.health().get("loaded_model") or payload.mode
        return Response(
            _wav_bytes(waveform, rate),
            media_type="audio/wav",
            headers={
                "X-TTS-Sample-Rate": str(rate),
                "X-TTS-Model": str(model),
                "X-TTS-Load-Ms": f"{load_ms:.3f}",
                "X-TTS-Inference-Ms": f"{inference_ms:.3f}",
            },
        )

    @app.post("/unload", status_code=204)
    async def unload(payload: UnloadPayload):
        await asyncio.to_thread(manager.unload)
        return Response(status_code=204)

    return app


def _wav_bytes(waveform, sample_rate: int) -> bytes:
    samples = np.asarray(waveform, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = np.round(samples * 32767).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()
