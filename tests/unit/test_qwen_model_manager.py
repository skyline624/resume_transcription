import pytest
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

from qwen_tts_worker.domain import GenerateCommand, Mode, WorkerModelError
from qwen_tts_worker.model_manager import QwenModelManager


class FakeModel:
    def __init__(self, mode, events, fail=False):
        self.mode, self.events, self.fail = mode, events, fail

    def generate(self, command):
        self.events.append(f"generate:{self.mode.value}")
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return [0.0, 0.1], 24000


class FakeLoader:
    def __init__(self):
        self.events = []
        self.fail = False

    def __call__(self, mode, model_id):
        self.events.append(f"load:{mode.value}")
        return FakeModel(mode, self.events, self.fail)


def command(mode):
    values = {"text": "Bonjour", "mode": mode, "language": "French"}
    if mode is Mode.CUSTOM:
        values["speaker"] = "Ryan"
    elif mode is Mode.DESIGN:
        values["instruct"] = "Voix chaleureuse"
    else:
        values["reference_audio"] = "/app/voices/id.wav"
        values["reference_text"] = "Bonjour"
    return GenerateCommand(**values)


def test_changement_de_mode_decharge_avant_de_charger(fake_clock):
    loader = FakeLoader()
    cleanups = []
    manager = QwenModelManager(
        {mode: mode.value for mode in Mode}, loader, lambda: cleanups.append("cleanup"),
        idle_s=300, clock=fake_clock,
    )
    manager.generate(command(Mode.CUSTOM))
    manager.generate(command(Mode.DESIGN))
    assert loader.events == [
        "load:custom", "generate:custom", "load:design", "generate:design"
    ]
    assert cleanups == ["cleanup"]


def test_mode_identique_reutilise_le_modele(fake_clock):
    loader = FakeLoader()
    manager = QwenModelManager(
        {mode: mode.value for mode in Mode}, loader, lambda: None, 300, fake_clock
    )
    manager.generate(command(Mode.CUSTOM))
    manager.generate(command(Mode.CUSTOM))
    assert loader.events.count("load:custom") == 1


def test_oom_invalide_le_modele_et_nettoie(fake_clock):
    loader = FakeLoader()
    loader.fail = True
    cleanups = []
    manager = QwenModelManager(
        {mode: mode.value for mode in Mode}, loader, lambda: cleanups.append(True),
        300, fake_clock,
    )
    with pytest.raises(WorkerModelError) as error:
        manager.generate(command(Mode.CUSTOM))
    assert error.value.code == "cuda_oom"
    assert manager.health()["loaded_model"] is None
    assert cleanups == [True]


def test_delai_inactif_decharge_le_modele(fake_clock):
    loader = FakeLoader()
    manager = QwenModelManager(
        {mode: mode.value for mode in Mode}, loader, lambda: None, 10, fake_clock
    )
    manager.generate(command(Mode.CUSTOM))
    fake_clock.value += 11
    assert manager.unload_if_idle() is True
    assert manager.health()["state"] == "idle"


def test_health_repond_pendant_le_chargement():
    loading = threading.Event()
    finish = threading.Event()

    def loader(mode, model_id):
        loading.set()
        finish.wait(3)
        return FakeModel(mode, [])

    manager = QwenModelManager({Mode.CUSTOM: "custom"}, loader, lambda: None, 300)
    with ThreadPoolExecutor(max_workers=2) as pool:
        generation = pool.submit(manager.generate, command(Mode.CUSTOM))
        assert loading.wait(2)
        try:
            snapshot = pool.submit(manager.health).result(timeout=0.2)
            assert snapshot["state"] == "loading"
        finally:
            finish.set()
            generation.result(timeout=2)


def test_stream_reutilise_le_modele_et_met_a_jour_l_inactivite(fake_clock):
    class StreamingModel(FakeModel):
        def stream(self, command):
            yield [0.1], 24000
            fake_clock.value += 10
            yield [0.2], 24000

    loads = []
    def loader(mode, model_id):
        loads.append(model_id)
        return StreamingModel(mode, [])

    manager = QwenModelManager({Mode.CUSTOM: "custom"}, loader, lambda: None, 10, fake_clock)
    stream = manager.stream(command(Mode.CUSTOM))
    assert next(stream) == ([0.1], 24000)
    assert manager.health()["state"] == "generating"
    assert list(stream) == [([0.2], 24000)]
    assert manager.health()["state"] == "ready"
    assert not manager.unload_if_idle()
    manager.generate(command(Mode.CUSTOM))
    assert loads == ["custom"]
    fake_clock.value += 11
    assert manager.unload_if_idle()


def test_fermer_le_flux_conserve_un_modele_reutilisable(fake_clock):
    closed = []
    class StreamingModel(FakeModel):
        def stream(self, command):
            try:
                yield [0.1], 24000
                raise AssertionError("La generation annulee doit s'arreter")
            finally:
                closed.append(True)
    manager = QwenModelManager(
        {Mode.CUSTOM: "custom"}, lambda mode, _: StreamingModel(mode, []),
        lambda: None, 10, fake_clock,
    )
    stream = manager.stream(command(Mode.CUSTOM))
    next(stream)
    stream.close()
    assert closed == [True]
    assert manager.health()["state"] == "ready"


@pytest.fixture
def fake_clock():
    class Clock:
        value = 100.0

        def __call__(self):
            return self.value
    return Clock()


@pytest.mark.parametrize("operation", ["generate", "stream", "load"])
def test_error_releases_model_before_cuda_cleanup(operation):
    refs = []
    alive_at_cleanup = []

    class FailingModel:
        def generate(self, command):
            raise RuntimeError("backend failure")

        def stream(self, command):
            yield from ()
            raise RuntimeError("backend failure")

    def loader(mode, model_id):
        model = FailingModel()
        refs.append(weakref.ref(model))
        if operation == "load":
            raise RuntimeError("load failure after allocation")
        return model

    manager = QwenModelManager(
        {Mode.CUSTOM: "custom"}, loader,
        lambda: alive_at_cleanup.append(any(ref() is not None for ref in refs)), 300,
    )
    with pytest.raises(WorkerModelError):
        if operation == "stream":
            list(manager.stream(command(Mode.CUSTOM)))
        else:
            manager.generate(command(Mode.CUSTOM))
    assert alive_at_cleanup == [False]
    assert all(ref() is None for ref in refs)
