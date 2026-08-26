from fastapi.testclient import TestClient

from qwen_tts_worker.app import create_worker_app
from qwen_tts_worker.domain import WorkerModelError


class FakeManager:
    def __init__(self):
        self.error = None
        self.commands = []

    def generate(self, command):
        self.commands.append(command)
        if self.error:
            raise self.error
        return [0.0, 0.1, -0.1], 24000, 12.0, 34.0

    def health(self):
        return {"state": "idle", "loaded_model": None, "last_error": None}

    def unload(self):
        return None

    def unload_if_idle(self):
        return False


def payload():
    return {
        "mode": "qwen3-tts-custom-voice", "text": "Bonjour",
        "language": "French", "speaker": "Ryan",
    }


def test_generate_retourne_wav_et_metriques():
    manager = FakeManager()
    response = TestClient(create_worker_app(manager, ["custom"])).post(
        "/generate", json=payload()
    )
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert response.headers["X-TTS-Sample-Rate"] == "24000"
    assert response.headers["X-TTS-Load-Ms"] == "12.000"


def test_oom_est_assaini_en_503():
    manager = FakeManager()
    manager.error = WorkerModelError("cuda_oom", "chemin interne secret")
    response = TestClient(create_worker_app(manager, [])).post("/generate", json=payload())
    assert response.status_code == 503
    assert response.json() == {
        "code": "cuda_oom", "message": "Le modèle TTS est indisponible."
    }


def test_health_expose_modeles_et_voix_sans_charger():
    response = TestClient(create_worker_app(FakeManager(), ["custom", "base"])).get(
        "/health"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["downloaded_models"] == ["custom", "base"]
    assert "Ryan" in body["speakers"]
    assert body["precision"] == "bfloat16"


def test_unload_est_idempotent():
    response = TestClient(create_worker_app(FakeManager(), [])).post(
        "/unload", json={"reason": "diarization"}
    )
    assert response.status_code == 204
