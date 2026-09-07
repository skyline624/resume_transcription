"""Transport PCM progressif et annulation acquittee par le worker prive."""
from uuid import uuid4

import aiohttp
import anyio

from transcription_server.tts.domain import TtsUnavailableError


async def stream_worker(socket_path, payload: dict, timeout_s: float):
    from transcription_server.tts.client import TransportResponse, UnixTtsClient

    identifier = str(uuid4())
    connector = aiohttp.UnixConnector(path=str(socket_path))
    try:
        async with aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as session:
            try:
                async with session.post(
                    "http://localhost/generate/stream", json=payload,
                    headers={"X-TTS-Request-Id": identifier},
                ) as response:
                    if response.status != 200:
                        raise UnixTtsClient._error_from_response(TransportResponse(
                            response.status, await response.read(), dict(response.headers),
                        ))
                    if response.headers.get("X-TTS-Sample-Rate") != "24000":
                        raise TtsUnavailableError("invalid_worker_response", "Frequence PCM invalide.")
                    if not response.headers.get("Content-Type", "").startswith("audio/pcm"):
                        raise TtsUnavailableError("invalid_worker_response", "Format PCM invalide.")
                    received = False
                    async for chunk in response.content.iter_any():
                        if chunk:
                            received = True
                            yield chunk
                    if not received:
                        raise TtsUnavailableError("empty_audio", "Aucun audio produit.")
            finally:
                # Le verrou GPU applicatif reste reserve jusqu'a cet acquittement.
                with anyio.CancelScope(shield=True):
                    async with session.post(f"http://localhost/cancel/{identifier}") as cancelled:
                        await cancelled.read()
                        cancelled.raise_for_status()
    except (aiohttp.ClientError, TimeoutError, OSError, NotImplementedError) as exc:
        raise TtsUnavailableError("worker_unreachable", "Le flux TTS a ete interrompu.") from exc
