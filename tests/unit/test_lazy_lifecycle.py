import asyncio
import time
import threading

import pytest
from fastapi.testclient import TestClient

from transcription_server import app as app_module
from transcription_server.config import Settings
from transcription_server.lifecycle import (
    GpuLifecycleManager,
    LazyEngine,
    TtsLifecycleClient,
)


class Horloge:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value


@pytest.fixture
def fake_clock():
    return Horloge()


class MoteurSimule:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.device = "cpu"

    def transcribe(self, audio, language):
        self.events.append("transcribe")
        return []

    def diarize(self, audio, num_speakers, min_speakers, max_speakers):
        self.events.append("diarize")
        return []


class Fabrique:
    def __init__(self, name, events, echec=False):
        self.name = name
        self.events = events
        self.echec = echec

    def __call__(self, **kwargs):
        self.events.append(f"load:{self.name}")
        if self.echec:
            raise RuntimeError("modele indisponible")
        return MoteurSimule(self.name, self.events)


def test_le_moteur_se_charge_a_la_premiere_requete(fake_clock):
    events = []
    fabrique = Fabrique("asr", events)
    moteur = LazyEngine("modele", fabrique, "transcribe", idle_s=900, clock=fake_clock)

    assert moteur.state == "unloaded"
    assert moteur.transcribe(b"audio", language="fr") == []

    assert events == ["load:asr", "transcribe"]
    assert moteur.state == "loaded"


def test_une_activite_continue_ne_recharge_pas(fake_clock):
    events = []
    fabrique = Fabrique("asr", events)
    moteur = LazyEngine("modele", fabrique, "transcribe", idle_s=900, clock=fake_clock)
    moteur.transcribe(b"audio", language=None)
    fake_clock.value += 600
    moteur.transcribe(b"audio", language=None)
    fake_clock.value += 600
    moteur.transcribe(b"audio", language=None)

    assert events == ["load:asr", "transcribe", "transcribe", "transcribe"]
    assert moteur.state == "loaded"


def test_l_inactivite_decharge_puis_la_requete_recharge(fake_clock):
    events = []
    fabrique = Fabrique("asr", events)
    moteur = LazyEngine("modele", fabrique, "transcribe", idle_s=900, clock=fake_clock)
    moteur.transcribe(b"audio", language=None)
    fake_clock.value += 899
    assert moteur.unload_if_idle() is False
    assert moteur.state == "loaded"
    fake_clock.value += 1
    assert moteur.unload_if_idle() is True
    assert moteur.state == "unloaded"
    moteur.transcribe(b"audio", language=None)

    assert events == ["load:asr", "transcribe", "load:asr", "transcribe"]


def test_les_requetes_concurrentes_ne_chargent_qu_une_fois(fake_clock):
    events = []
    fabrique = Fabrique("asr", events)
    moteur = LazyEngine("modele", fabrique, "transcribe", idle_s=900, clock=fake_clock)

    async def executer():
        results = await asyncio.gather(
            asyncio.to_thread(moteur.transcribe, b"audio", None),
            asyncio.to_thread(moteur.transcribe, b"audio", None),
            asyncio.to_thread(moteur.transcribe, b"audio", None),
        )
        return results

    asyncio.run(executer())
    assert events.count("load:asr") == 1
    assert events.count("transcribe") == 3


def test_le_monitor_decharge_pyannote_puis_parakeet(fake_clock):
    events = []
    diarization = LazyEngine("pyannote", Fabrique("diarization", events), "diarize", idle_s=900, clock=fake_clock)
    asr = LazyEngine("parakeet", Fabrique("asr", events), "transcribe", idle_s=900, clock=fake_clock)
    diarization.diarize(b"audio", None, None, None)
    asr.transcribe(b"audio", None)
    fake_clock.value += 900
    manager = GpuLifecycleManager((diarization, asr), None, interval_s=1, clock=fake_clock)

    asyncio.run(manager.unload_idle())

    assert moteur_est_decharge(diarization) is True
    assert moteur_est_decharge(asr) is True


def moteur_est_decharge(moteur):
    return moteur.state == "unloaded"


def test_le_worker_tts_est_decharge_par_le_monitor(fake_clock):
    events = []

    class ClientSimule:
        async def unload(self, reason):
            events.append(f"unload:{reason}")

    tts = TtsLifecycleClient(ClientSimule(), idle_s=300, clock=fake_clock)
    await_synth = False
    if await_synth:
        pass
    tts._last_used = fake_clock()
    manager = GpuLifecycleManager((), tts, interval_s=1, clock=fake_clock)
    fake_clock.value += 300
    asyncio.run(manager.unload_idle())

    assert events == ["unload:inactivite"]
    assert tts.last_used is None


def test_l_etat_est_expose_par_health(monkeypatch):
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_load_silero_vad_engine",
        lambda **kwargs: MoteurSimule("silero-vad", []),
    )
    application = app_module.build_app(
        Settings(_env_file=None, enable_diarization=True, hf_token="hf_test", device="cuda")
    )
    with TestClient(application) as client:
        corps = client.get("/health").json()

    assert corps["asr_state"] == "unloaded"
    assert corps["diarization_state"] == "unloaded"


def test_l_activation_du_gpu_ne_charche_pas_au_demarrage(monkeypatch):
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    events = []
    monkeypatch.setattr(app_module, "_load_nemo_engine", lambda **kwargs: events.append("load:asr"))
    monkeypatch.setattr(app_module, "_load_pyannote_engine", lambda **kwargs: events.append("load:pyannote"))
    app_module.build_app(
        Settings(_env_file=None, device="cuda", enable_diarization=True, hf_token="hf_test")
    )

    assert events == []


def test_l_lazy_desactive_conserve_le_chargement_initial(monkeypatch):
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    events = []
    monkeypatch.setattr(app_module, "_load_nemo_engine", lambda **kwargs: events.append("load:asr"))
    monkeypatch.setattr(app_module, "_load_pyannote_engine", lambda **kwargs: events.append("load:pyannote"))
    app_module.build_app(
        Settings(
            _env_file=None,
            device="cuda",
            enable_diarization=True,
            hf_token="hf_test",
            enable_lazy_gpu=False,
        )
    )

    assert events == ["load:asr", "load:pyannote"]


async def test_le_monitor_attend_la_fin_du_travail_gpu(monkeypatch):
    events = []
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    monkeypatch.setattr(app_module, "_warmup", lambda *args: None)
    monkeypatch.setattr(app_module, "_load_nemo_engine", Fabrique("asr", events))
    application = app_module.build_app(Settings(
        _env_file=None, enable_diarization=False, enable_vad=False,
        enable_tts=False, enable_summary=False, gpu_idle_unload_s=0.001,
    ))
    state = application.state.app_state
    await asyncio.to_thread(state.asr.transcribe, b"audio", None)
    async with state.gpu_lock:
        # Le timer expire pendant que la requete possede encore le GPU.
        await asyncio.sleep(0.01)
        monitor = asyncio.create_task(state.lifecycle.unload_idle())
        await asyncio.sleep(0.01)
        assert state.asr.state == "loaded"
    await monitor
    assert state.asr.state == "unloaded"


async def test_le_dechargement_ne_bloque_pas_la_boucle(fake_clock):
    thread_ids = []

    class Moteur(MoteurSimule):
        def release_gpu(self):
            thread_ids.append(threading.get_ident())

    moteur = LazyEngine(
        "asr", lambda: Moteur("asr", []), "transcribe",
        idle_s=900, clock=fake_clock,
    )
    moteur.transcribe(b"audio", None)
    fake_clock.value += 900
    manager = GpuLifecycleManager((moteur,), None, interval_s=1)
    await manager.unload_idle()
    assert moteur.state == "unloaded"
    assert thread_ids and thread_ids[0] != threading.get_ident()
def test_health_snapshot_ne_bloque_pas_pendant_chargement():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from transcription_server.lifecycle import LazyEngine
    entered, release = threading.Event(), threading.Event()
    def load():
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(name="asr", transcribe=lambda: None)
    engine = LazyEngine("asr", load, "transcribe", idle_s=300)
    with ThreadPoolExecutor(2) as pool:
        running = pool.submit(engine.transcribe)
        assert entered.wait(2)
        try:
            snapshot = pool.submit(lambda: (engine.state, engine.engine_name, engine.last_error))
            assert snapshot.result(timeout=0.2) == ("loading", "asr", None)
        finally:
            release.set()
            running.result(timeout=2)
