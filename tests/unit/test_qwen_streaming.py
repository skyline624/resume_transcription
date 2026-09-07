"""Le worker rend du PCM et assainit les erreurs avant le premier morceau."""
import struct
from fastapi.testclient import TestClient
from qwen_tts_worker.app import create_worker_app
from qwen_tts_worker.model_manager import QwenModelManager


def app_with_model(model):
    manager = QwenModelManager({}, lambda *_: model, lambda: None, 300)
    from qwen_tts_worker.domain import Mode
    manager._model_ids = {Mode.CUSTOM: "custom"}
    return create_worker_app(manager, []), manager


def payload():
    return {"mode": "qwen3-tts-custom-voice", "text": "Bonjour", "speaker": "Ryan"}


def test_stream_pcm_et_modele_reutilisable():
    class Model:
        def stream(self, command):
            yield [0.0, 1.0], 24000
            yield [-1.0], 24000
    app, manager = app_with_model(Model())
    with TestClient(app) as client:
        response = client.post("/generate/stream", json=payload())
        assert response.status_code == 200
        assert response.headers["x-tts-sample-rate"] == "24000"
        assert response.content == struct.pack("<hhh", 0, 32767, -32767)
        assert manager.health()["state"] == "ready"


def test_erreur_avant_premier_morceau_est_un_503():
    class Model:
        def stream(self, command):
            raise RuntimeError("CUDA out of memory chemin secret")
            yield
    app, _ = app_with_model(Model())
    with TestClient(app) as client:
        response = client.post("/generate/stream", json=payload())
        assert response.status_code == 503
        assert response.json()["code"] == "cuda_oom"
        assert "secret" not in response.text
