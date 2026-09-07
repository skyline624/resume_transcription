"""Client privé du worker Qwen via un socket Unix."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import aiohttp

from transcription_server.tts.domain import (
    SynthesisRequest,
    SynthesisResult,
    TtsUnavailableError,
    WorkerHealth,
)

_QWEN_LANGUAGES = {"fr": "French", "en": "English", "auto": "Auto"}


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes
    headers: dict[str, str]


class WorkerTransport(Protocol):
    async def request(
        self, method: str, path: str, payload: dict | None, timeout_s: float
    ) -> TransportResponse: ...

    def stream(self, payload: dict, timeout_s: float) -> AsyncIterator[bytes]: ...


class AiohttpUnixTransport:
    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path

    def stream(self, payload: dict, timeout_s: float) -> AsyncIterator[bytes]:
        from transcription_server.tts.streaming_client import stream_worker
        return stream_worker(self._socket_path, payload, timeout_s)

    async def request(
        self, method: str, path: str, payload: dict | None, timeout_s: float
    ) -> TransportResponse:
        connector = aiohttp.UnixConnector(path=str(self._socket_path))
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        try:
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout
            ) as session:
                async with session.request(
                    method, f"http://localhost{path}", json=payload
                ) as response:
                    return TransportResponse(
                        status=response.status,
                        body=await response.read(),
                        headers=dict(response.headers),
                    )
        except (aiohttp.ClientError, TimeoutError, OSError, NotImplementedError) as exc:
            raise TtsUnavailableError(
                "worker_unreachable", "Le worker TTS est indisponible."
            ) from exc


class TtsClient(Protocol):
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...
    def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]: ...
    async def health(self) -> WorkerHealth: ...
    async def unload(self, reason: str) -> None: ...


class UnixTtsClient:
    def __init__(
        self,
        socket_path: Path,
        load_timeout_s: float,
        generation_timeout_s: float,
        transport: WorkerTransport | None = None,
    ) -> None:
        self._load_timeout_s = load_timeout_s
        self._generation_timeout_s = generation_timeout_s
        self._transport = transport or AiohttpUnixTransport(socket_path)

    @staticmethod
    def _payload(request: SynthesisRequest) -> dict:
        return {
            "mode": request.mode.value,
            "text": request.text,
            "language": _QWEN_LANGUAGES.get(request.language, request.language),
            "speaker": request.voice,
            "instruct": request.instructions,
            "reference_audio": (
                request.reference_path.as_posix() if request.reference_path else None
            ),
            "reference_text": request.reference_text,
        }

    def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        return self._transport.stream(self._payload(request), self._generation_timeout_s)

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        payload = self._payload(request)
        response = await self._transport.request(
            "POST", "/generate", payload, self._generation_timeout_s
        )
        if response.status != 200:
            raise self._error_from_response(response)
        if not response.body:
            raise TtsUnavailableError(
                "empty_audio", "Le worker TTS n'a produit aucun audio."
            )
        headers = {name.lower(): value for name, value in response.headers.items()}
        try:
            return SynthesisResult(
                audio_wav=response.body,
                sample_rate=int(headers["x-tts-sample-rate"]),
                model=headers["x-tts-model"],
                load_ms=float(headers.get("x-tts-load-ms", 0)),
                inference_ms=float(headers.get("x-tts-inference-ms", 0)),
            )
        except (KeyError, ValueError) as exc:
            raise TtsUnavailableError(
                "invalid_worker_response", "La réponse du worker TTS est invalide."
            ) from exc

    async def health(self) -> WorkerHealth:
        response = await self._transport.request(
            "GET", "/health", None, self._load_timeout_s
        )
        if response.status != 200:
            raise self._error_from_response(response)
        try:
            payload = json.loads(response.body)
            return WorkerHealth(
                available=True,
                state=str(payload["state"]),
                downloaded_models=tuple(payload.get("downloaded_models", ())),
                loaded_model=payload.get("loaded_model"),
                precision=payload.get("precision"),
                device=payload.get("device"),
                attention=payload.get("attention"),
                speakers=tuple(payload.get("speakers", ())),
                features=tuple(payload.get("features", ())),
                last_error=payload.get("last_error"),
                pid=payload.get("pid"),
                vram_allocated_mib=payload.get("vram_allocated_mib"),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TtsUnavailableError(
                "invalid_worker_response", "La réponse du worker TTS est invalide."
            ) from exc

    async def unload(self, reason: str) -> None:
        response = await self._transport.request(
            "POST", "/unload", {"reason": reason}, self._load_timeout_s
        )
        if response.status not in (200, 204):
            raise self._error_from_response(response)

    @staticmethod
    def _error_from_response(response: TransportResponse) -> TtsUnavailableError:
        code = "worker_error"
        try:
            payload = json.loads(response.body)
            if isinstance(payload, dict) and isinstance(payload.get("code"), str):
                code = payload["code"]
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        return TtsUnavailableError(code, "Le worker TTS est indisponible.")


class UnavailableTtsClient:
    async def stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        raise TtsUnavailableError("tts_disabled", "La synthese vocale est desactivee.")
        yield b""

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        raise TtsUnavailableError("tts_disabled", "La synthèse vocale est désactivée.")

    async def health(self) -> WorkerHealth:
        return WorkerHealth(available=False, state="disabled")

    async def unload(self, reason: str) -> None:
        return None
