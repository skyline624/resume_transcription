import io
import struct
import wave

from fastapi.testclient import TestClient

from transcription_server.app import _build_tts_dependencies, create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.domain import SpeakerSegment, Word
from transcription_server.tts.domain import TtsUnavailableError, WorkerHealth
from transcription_server.tts.client import UnixTtsClient


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
        handle.writeframes(struct.pack("<h", 0) * 16000)
    return output.getvalue()


class RecordingTts:
    def __init__(self, events): self.events = events
    async def unload(self, reason): self.events.append(f"tts:{reason}")
    async def health(self): return WorkerHealth(True, "idle")
    async def synthesize(self, request): raise AssertionError


class BrokenTts(RecordingTts):
    async def health(self):
        raise TtsUnavailableError("worker_unreachable", "indisponible")


class RecordingDiarization:
    name = "recording"
    def __init__(self, events): self.events = events
    def diarize(self, audio, num_speakers=None, min_speakers=None, max_speakers=None):
        self.events.append("diarization:start")
        return [SpeakerSegment("SPEAKER_00", 0, 1)]


def test_diarization_decharge_qwen_apres_acquisition_du_verrou():
    events = []
    app = create_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu"),
        StubAsrEngine([Word("bonjour", 0, 0.5)]),
        RecordingDiarization(events),
        tts=RecordingTts(events),
    )
    response = TestClient(app).post(
        "/transcribe", files={"file": ("x.wav", wav_bytes(), "audio/wav")},
        data={"diarize": "true"},
    )
    assert response.status_code == 200
    assert events[:2] == ["tts:diarization", "diarization:start"]


def test_asr_sans_diarization_garde_qwen_charge():
    events = []
    app = create_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu"),
        StubAsrEngine([Word("bonjour", 0, 0.5)]),
        RecordingDiarization(events),
        tts=RecordingTts(events),
    )
    response = TestClient(app).post(
        "/transcribe", files={"file": ("x.wav", wav_bytes(), "audio/wav")},
        data={"diarize": "false"},
    )
    assert response.status_code == 200
    assert events == []


def test_fabrique_de_production_cree_client_uds_et_registre(tmp_path):
    settings = Settings(
        _env_file=None, enable_diarization=False,
        voice_store_path=tmp_path / "voices",
    )
    client, repository = _build_tts_dependencies(settings)
    assert isinstance(client, UnixTtsClient)
    assert repository.list() == []


def test_health_expose_l_etat_complet_du_worker():
    events = []
    tts = RecordingTts(events)

    async def detailed_health():
        return WorkerHealth(
            available=True,
            state="ready",
            downloaded_models=("custom", "base"),
            loaded_model="custom",
            precision="bfloat16",
            device="cuda:0",
            attention="sdpa",
            speakers=("Ryan",),
            features=("custom_voice", "clone"),
            pid=42,
            vram_allocated_mib=4321.5,
        )

    tts.health = detailed_health
    app = create_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu"),
        StubAsrEngine([]), RecordingDiarization(events), tts=tts,
    )
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["tts"] == {
        "enabled": True,
        "worker": True,
        "state": "ready",
        "downloaded_models": ["custom", "base"],
        "loaded_model": "custom",
        "precision": "bfloat16",
        "device": "cuda:0",
        "attention": "sdpa",
        "features": ["custom_voice", "clone"],
        "last_error": None,
        "pid": 42,
        "vram_allocated_mib": 4321.5,
    }


def test_health_rend_503_si_le_worker_attendu_est_injoignable():
    events = []
    app = create_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu"),
        StubAsrEngine([]), RecordingDiarization(events), tts=BrokenTts(events),
    )
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["tts"]["last_error"] == "worker_unreachable"
