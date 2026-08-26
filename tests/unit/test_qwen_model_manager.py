import pytest

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


@pytest.fixture
def fake_clock():
    class Clock:
        value = 100.0

        def __call__(self):
            return self.value
    return Clock()
