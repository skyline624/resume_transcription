"""Serveurs locaux utilises par les tests d'integration TTS."""

from __future__ import annotations

import io
import os
import socket
import struct
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import NullDiarizationEngine
from transcription_server.tts.client import UnixTtsClient
from transcription_server.tts.profiles import VoiceProfileRepository


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(struct.pack("<h", 100) * 2_400)
    return output.getvalue()


def _wait_until_started(server: uvicorn.Server, thread: threading.Thread) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if server.started:
            return
        if not thread.is_alive():
            break
        time.sleep(0.01)
    raise RuntimeError("Uvicorn n'a pas demarre dans le delai imparti.")


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("Uvicorn ne s'est pas arrete proprement.")


@dataclass
class LiveWorker:
    socket_path: Path
    calls: list[tuple[str, dict | None]]


@pytest.fixture
def live_uds_worker(tmp_path: Path) -> Iterator[LiveWorker]:
    if os.name == "nt":
        pytest.skip("Les sockets Unix sont verifies dans le conteneur Linux.")

    socket_path = tmp_path / "qwen-worker.sock"
    calls: list[tuple[str, dict | None]] = []
    app = FastAPI()

    @app.get("/health")
    async def health() -> JSONResponse:
        calls.append(("health", None))
        return JSONResponse(
            {
                "state": "idle",
                "downloaded_models": ["custom"],
                "loaded_model": None,
                "precision": "bfloat16",
                "device": "cuda",
                "attention": "sdpa",
                "speakers": ["Ryan"],
                "features": ["custom_voice", "voice_clone", "voice_design"],
                "pid": os.getpid(),
            }
        )

    @app.post("/generate")
    async def generate(payload: dict) -> Response:
        calls.append(("generate", payload))
        return Response(
            _wav_bytes(),
            media_type="audio/wav",
            headers={
                "X-TTS-Sample-Rate": "24000",
                "X-TTS-Model": "custom",
                "X-TTS-Load-Ms": "1.5",
                "X-TTS-Inference-Ms": "3.5",
            },
        )

    @app.post("/unload", status_code=204)
    async def unload(payload: dict) -> Response:
        calls.append(("unload", payload))
        return Response(status_code=204)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    _wait_until_started(server, thread)
    try:
        yield LiveWorker(socket_path=socket_path, calls=calls)
    finally:
        _stop_server(server, thread)
        listener.close()
        socket_path.unlink(missing_ok=True)


@dataclass
class LiveApi:
    base_url: str


@pytest.fixture
def live_tts_api(live_uds_worker: LiveWorker, tmp_path: Path) -> Iterator[LiveApi]:
    settings = Settings(
        _env_file=None,
        device="cpu",
        enable_diarization=False,
        enable_tts=True,
        tts_worker_socket=live_uds_worker.socket_path,
        voice_store_path=tmp_path / "voices",
    )
    tts = UnixTtsClient(
        socket_path=live_uds_worker.socket_path,
        load_timeout_s=10,
        generation_timeout_s=10,
    )
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([]),
        diarization=NullDiarizationEngine(),
        tts=tts,
        voice_profiles=VoiceProfileRepository(tmp_path / "voices"),
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    _wait_until_started(server, thread)
    try:
        yield LiveApi(base_url=f"http://127.0.0.1:{port}")
    finally:
        _stop_server(server, thread)
        listener.close()
