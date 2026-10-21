"""Synthese et transcription sur deux cartes (TTS_SEPARATE_GPU) : un verrou par carte.

Sur le serveur, Parakeet et pyannote tournent sur la carte 0 et le worker
Qwen3-TTS sur la carte 1. Avec un verrou commun, une transcription attendait
la fin de la voix en cours (3,9 s mesurees pendant une synthese en flux).
"""

import asyncio
import io
import struct
import wave

import httpx

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import StubDiarizationEngine
from transcription_server.domain import Word
from transcription_server.lifecycle import GpuLifecycleManager
from transcription_server.tts.domain import WorkerHealth


def _wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


class TtsNote:
    def __init__(self, events): self.events = events
    async def unload(self, reason): self.events.append(f"tts:{reason}")
    async def health(self): return WorkerHealth(True, "idle")


class AsrNote(StubAsrEngine):
    def __init__(self, events):
        super().__init__([Word("bonjour", 0.0, 0.5)])
        self.events = events
    def release_gpu(self): self.events.append("asr:cpu")


def _app(events, **reglages):
    settings = Settings(_env_file=None, enable_diarization=False, device="cpu", **reglages)
    return create_app(settings, AsrNote(events), StubDiarizationEngine([]), tts=TtsNote(events))


async def _transcrire(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://audio") as client:
        return await client.post("/v1/audio/transcriptions",
                                 files={"file": ("x.wav", _wav_bytes(), "audio/wav")})


def test_par_defaut_la_synthese_partage_le_verrou_de_la_transcription():
    state = _app([]).state.app_state
    assert state.tts_lock is state.gpu_lock


async def test_une_transcription_n_attend_pas_une_voix_sur_l_autre_carte():
    state_app = _app([], tts_separate_gpu=True)
    state = state_app.state.app_state
    assert state.tts_lock is not state.gpu_lock
    async with state.tts_lock:  # une voix en cours sur la carte 1
        reponse = await asyncio.wait_for(_transcrire(state_app), 5)
    assert reponse.status_code == 200 and reponse.json() == {"text": "bonjour"}


async def test_sur_une_seule_carte_la_transcription_attend_la_voix():
    app = _app([])
    state = app.state.app_state
    await state.tts_lock.acquire()
    transcription = asyncio.create_task(_transcrire(app))
    await asyncio.sleep(0.2)
    assert not transcription.done()
    state.tts_lock.release()
    assert (await asyncio.wait_for(transcription, 5)).status_code == 200


async def test_sur_deux_cartes_aucune_bascule_de_vram():
    # Hors mode paresseux, liberer l'autre famille ne rend rien sur sa carte, et
    # la ferait tomber en plein travail puisqu'elles tournent en meme temps.
    events = []
    state = _app(events, tts_separate_gpu=True, enable_lazy_gpu=False).state.app_state
    await state.prepare_transcription()
    await state.prepare_tts()
    assert events == []
    events.clear()
    partagee = _app(events, enable_lazy_gpu=False).state.app_state
    await partagee.prepare_transcription()
    await partagee.prepare_tts()
    assert events == ["tts:transcription", "asr:cpu"]


async def test_avant_ollama_le_worker_n_est_pas_decharge_en_pleine_voix():
    events = []
    state = _app(events, tts_separate_gpu=True).state.app_state
    await state.tts_lock.acquire()
    async with state.gpu_lock:
        preparation = asyncio.create_task(state.prepare_external_gpu("summary"))
        await asyncio.sleep(0.05)
        assert events == []  # la voix en cours garde son worker
        state.tts_lock.release()
        await asyncio.wait_for(preparation, 5)
    assert events == ["tts:summary", "asr:cpu"]


class MoteurInactif:
    name = "parakeet"
    def __init__(self, events): self.events = events
    def unload_if_idle(self):
        self.events.append("asr:decharge")
        return True


class TtsInactive:
    def __init__(self, events): self.events = events
    def unload_if_idle(self): return True
    async def unload(self, reason): self.events.append(f"tts:{reason}")


async def test_le_dechargement_de_parakeet_n_attend_pas_une_voix():
    events = []
    verrou_transcription, verrou_synthese = asyncio.Lock(), asyncio.Lock()
    manager = GpuLifecycleManager((MoteurInactif(events),), TtsInactive(events), interval_s=1,
                                  gpu_lock=verrou_transcription, tts_lock=verrou_synthese)
    await verrou_synthese.acquire()
    dechargement = asyncio.create_task(manager.unload_idle())
    await asyncio.sleep(0.05)
    assert events == ["asr:decharge"]  # Parakeet libere, la voix continue
    verrou_synthese.release()
    await asyncio.wait_for(dechargement, 5)
    assert events == ["asr:decharge", "tts:inactivite"]


async def test_un_seul_verrou_par_defaut_pour_le_dechargement():
    events = []
    verrou = asyncio.Lock()
    manager = GpuLifecycleManager((MoteurInactif(events),), TtsInactive(events), interval_s=1, gpu_lock=verrou)
    await verrou.acquire()
    dechargement = asyncio.create_task(manager.unload_idle())
    await asyncio.sleep(0.05)
    assert events == []
    verrou.release()
    await asyncio.wait_for(dechargement, 5)
    assert events == ["asr:decharge", "tts:inactivite"]
