from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import NullDiarizationEngine


def make_client(web_dist_path):
    settings = Settings(
        _env_file=None,
        enable_diarization=False,
        enable_summary=False,
        enable_tts=False,
        device="cpu",
        web_dist_path=web_dist_path,
    )
    app = create_app(settings, StubAsrEngine([]), NullDiarizationEngine())
    return TestClient(app)


def test_racine_sert_le_build_sans_masquer_docs_ni_api(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<main>VoxLab</main>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("export {};", encoding="utf-8")
    client = make_client(tmp_path)

    response = client.get("/")
    assert response.text == "<main>VoxLab</main>"
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/v1/inconnue").status_code == 404
    assert "default-src 'self'" in response.headers["content-security-policy"]


def test_build_absent_laisse_api_demarrer(tmp_path):
    client = make_client(tmp_path)

    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200
