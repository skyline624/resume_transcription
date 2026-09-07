"""Etat partage de l'application, accessible depuis les routes."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import (
    DiarizationEngine,
    NullDiarizationEngine,
)
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
    lifecycle: Any = None

    # Un seul travail sur le GPU a la fois : deux inferences concurrentes se
    # disputeraient la VRAM et feraient tomber le serveur en OOM plutot que de
    # le ralentir. Construire le verrou hors d'une boucle d'evenements est sur
    # depuis Python 3.10 : asyncio.Lock ne resout la boucle qu'a la premiere
    # attente reelle, et s'y lie alors definitivement. Corollaire : une meme
    # application ne doit pas servir deux boucles distinctes.
    gpu_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _transcription_gpu_resident(self) -> bool:
        """Vrai quand les moteurs de transcription doivent rester en VRAM.

        Les delais d'inactivite nuls (`*_IDLE_UNLOAD_S=0`) signifient
        "residence permanente" : dans ce regime, liberer Parakeet/pyannote
        avant une synthese TTS (ou le TTS avant une transcription) couterait
        un rechargement complet a la requete suivante. Le ping-pong de
        dechargement n'a de sens que pour rendre la VRAM entre deux usages
        rares.
        """
        return self.settings.gpu_idle_unload_s <= 0

    async def prepare_transcription(self) -> None:
        """En mode lazy, laisse le timer gerer la residence du TTS."""
        if self.settings.enable_lazy_gpu:
            return
        if self.settings.enable_tts and not self._transcription_gpu_resident():
            await self.tts.unload(reason="transcription")

    async def prepare_external_gpu(self, reason: str) -> None:
        """Libere la VRAM du conteneur avant un moteur GPU externe (Ollama)."""
        if self.settings.enable_tts and not self._transcription_gpu_resident():
            await self.tts.unload(reason=reason)
        if not self._transcription_gpu_resident():
            await asyncio.to_thread(self._release_transcription_gpu)

    async def prepare_tts(self) -> None:
        """En mode lazy, conserve le STT jusqu'au delai d'inactivite."""
        if self.settings.enable_lazy_gpu:
            return
        if not self._transcription_gpu_resident():
            await asyncio.to_thread(self._release_transcription_gpu)

    def _release_transcription_gpu(self) -> None:
        for engine in (self.asr, self.diarization):
            release = getattr(engine, "release_gpu", None)
            if release is not None:
                release()
        empty_cache()

    def engine_state(self, engine: Any) -> str:
        if isinstance(engine, NullDiarizationEngine):
            return "unloaded"
        return str(getattr(engine, "state", "loaded"))

    def engine_display_name(self, engine: Any) -> str:
        """Nom affichable du moteur, en traversant l'enveloppe lazy."""
        return str(getattr(engine, "engine_name", None) or engine.name)


def get_state(request: Request) -> AppState:
    """Dependance FastAPI : rend l'etat attache a l'application."""
    return request.app.state.app_state
