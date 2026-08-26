"""Client privé du worker Qwen via un socket Unix."""

import json
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


class AiohttpUnixTransport:
    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path

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
        except (aiohttp.ClientError, TimeoutError, OSError) as exc:
            raise TtsUnavailableError(
                "worker_unreachable", "Le worker TTS est indisponible."
            ) from exc


class TtsClient(Protocol):
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...
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

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        payload = {
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
        response = await self._transport.request(
            "POST", "/generate", payload, self._generation_timeout_s
        )
        if response.status != 200:
            raise self._error_from_response(response)
        if not response.body:
            raise TtsUnavailableError(
                "empty_audio", "Le worker TTS n'a produit aucun audio."
            )
        try:
            return SynthesisResult(
                audio_wav=response.body,
                sample_rate=int(response.headers["X-TTS-Sample-Rate"]),
                model=response.headers["X-TTS-Model"],
                load_ms=float(response.headers.get("X-TTS-Load-Ms", 0)),
                inference_ms=float(response.headers.get("X-TTS-Inference-Ms", 0)),
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
    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        raise TtsUnavailableError("tts_disabled", "La synthèse vocale est désactivée.")

    async def health(self) -> WorkerHealth:
        return WorkerHealth(available=False, state="disabled")

    async def unload(self, reason: str) -> None:
        return None
