"""Etat partage de l'application, accessible depuis les routes."""

import asyncio
from dataclasses import dataclass, field

from fastapi import Request

from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.summary.engine import SummaryEngine
from transcription_server.vad.engine import VadEngine


@dataclass
class AppState:
    settings: Settings
    asr: AsrEngine
    diarization: DiarizationEngine
    summary: SummaryEngine
    vad: VadEngine | None = None
    device_info: dict = field(default_factory=dict)

    # Un seul travail sur le GPU a la fois : deux inferences concurrentes se
    # disputeraient la VRAM et feraient tomber le serveur en OOM plutot que de
    # le ralentir. Construire le verrou hors d'une boucle d'evenements est sur
    # depuis Python 3.10 : asyncio.Lock ne resout la boucle qu'a la premiere
    # attente reelle, et s'y lie alors definitivement. Corollaire : une meme
    # application ne doit pas servir deux boucles distinctes.
    gpu_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def get_state(request: Request) -> AppState:
    """Dependance FastAPI : rend l'etat attache a l'application."""
    return request.app.state.app_state
