"""Etat partage de l'application, accessible depuis les routes."""

import asyncio
from dataclasses import dataclass, field

from fastapi import Request

from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.runtime import empty_cache
from transcription_server.summary.engine import SummaryEngine
from transcription_server.vad.engine import VadEngine
from transcription_server.tts.client import TtsClient, UnavailableTtsClient
from transcription_server.tts.profiles import VoiceProfileRepository


@dataclass
class AppState:
    settings: Settings
    asr: AsrEngine
    diarization: DiarizationEngine
    summary: SummaryEngine
    vad: VadEngine | None = None
    device_info: dict = field(default_factory=dict)
    tts: TtsClient = field(default_factory=UnavailableTtsClient)
    voice_profiles: VoiceProfileRepository | None = None

    # Un seul travail sur le GPU a la fois : deux inferences concurrentes se
    # disputeraient la VRAM et feraient tomber le serveur en OOM plutot que de
    # le ralentir. Construire le verrou hors d'une boucle d'evenements est sur
    # depuis Python 3.10 : asyncio.Lock ne resout la boucle qu'a la premiere
    # attente reelle, et s'y lie alors definitivement. Corollaire : une meme
    # application ne doit pas servir deux boucles distinctes.
    gpu_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def prepare_transcription(self) -> None:
        """Evacue le TTS avant toute transcription, avec ou sans diarization."""
        if self.settings.enable_tts:
            await self.tts.unload(reason="transcription")

    async def prepare_external_gpu(self, reason: str) -> None:
        """Libere la VRAM du conteneur avant un moteur GPU externe (Ollama)."""
        if self.settings.enable_tts:
            await self.tts.unload(reason=reason)
        await asyncio.to_thread(self._release_transcription_gpu)

    async def prepare_tts(self) -> None:
        """Libere Parakeet et pyannote avant de charger Qwen TTS."""
        await asyncio.to_thread(self._release_transcription_gpu)

    def _release_transcription_gpu(self) -> None:
        for engine in (self.asr, self.diarization):
            release = getattr(engine, "release_gpu", None)
            if release is not None:
                release()
        empty_cache()


def get_state(request: Request) -> AppState:
    """Dependance FastAPI : rend l'etat attache a l'application."""
    return request.app.state.app_state
