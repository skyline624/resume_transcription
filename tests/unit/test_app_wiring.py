"""Cablage de l'application : `build_app` et le point d'entree.

`build_app` charge les vrais moteurs, donc NeMo et pyannote. Ces tests
remplacent les deux fabriques par des doubles : ce qui est verifie ici est la
logique de cablage — quel moteur pour quelle configuration, dans quel ordre, et
ce qui doit echouer — pas l'inference, qui vit dans `tests/gpu/`.
"""

import pytest
from fastapi.testclient import TestClient

from transcription_server import app as app_module
from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import (
    NullDiarizationEngine,
    StubDiarizationEngine,
)
from transcription_server.domain import Word
from transcription_server.runtime import CudaUnavailableError

TOKEN = "hf_pour_les_tests"


@pytest.fixture
def moteurs_simules(monkeypatch):
    """Remplace les fabriques de modèles et enregistre leurs arguments."""
    appels = {"asr": [], "diarization": [], "vad": []}

    def faux_load_nemo(model_name, device, compute_type):
        appels["asr"].append(
            {"model_name": model_name, "device": device, "compute_type": compute_type}
        )
        return StubAsrEngine([Word("bonjour", 0.0, 0.5)], name=model_name)

    def faux_load_pyannote(model_name, hf_token, device):
        appels["diarization"].append({"model_name": model_name, "device": device})
        return StubDiarizationEngine([], name=model_name)

    class FauxVad:
        name = "silero-vad"

        def __init__(self, device):
            self.device = device

        def plan(self, audio):
            return [(0.0, len(audio) / 16000)]

    def faux_load_vad(device, max_segment_s):
        appels["vad"].append(
            {"device": device, "max_segment_s": max_segment_s}
        )
        return FauxVad(device)

    monkeypatch.setattr(app_module, "_load_nemo_engine", faux_load_nemo)
    monkeypatch.setattr(app_module, "_load_pyannote_engine", faux_load_pyannote)
    monkeypatch.setattr(
        app_module, "_load_silero_vad_engine", faux_load_vad, raising=False
    )
    return appels


def test_cuda_indisponible_fait_echouer_le_demarrage(monkeypatch, moteurs_simules):
    """Aucun repli CPU silencieux : mieux vaut refuser de demarrer que
    transcrire vingt fois plus lentement sans que personne ne le sache."""
    monkeypatch.setattr(app_module, "cuda_available", lambda: False)
    reglages = Settings(_env_file=None, hf_token=TOKEN, device="cuda")
    with pytest.raises(CudaUnavailableError):
        app_module.build_app(reglages)


def test_l_echec_cuda_precede_le_chargement_des_modeles(monkeypatch, moteurs_simules):
    """Echouer avant de telecharger 2,6 Go de poids, pas apres."""
    monkeypatch.setattr(app_module, "cuda_available", lambda: False)
    with pytest.raises(CudaUnavailableError):
        app_module.build_app(Settings(_env_file=None, hf_token=TOKEN, device="cuda"))
    assert moteurs_simules["asr"] == []
    assert moteurs_simules["diarization"] == []


def test_cpu_demande_demarre_sans_cuda(monkeypatch, moteurs_simules):
    monkeypatch.setattr(app_module, "cuda_available", lambda: False)
    application = app_module.build_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu")
    )
    assert moteurs_simules["asr"][0]["device"] == "cpu"
    assert TestClient(application).get("/health").json()["device"] == "cpu"


def test_diarization_desactivee_n_appelle_pas_pyannote(monkeypatch, moteurs_simules):
    """Le serveur doit demarrer sans token quand la diarization est eteinte."""
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    application = app_module.build_app(
        Settings(_env_file=None, enable_diarization=False, device="cuda")
    )
    assert moteurs_simules["diarization"] == []
    corps = TestClient(application).get("/health").json()
    assert corps["diarization_model"] == "none"
    assert corps["diarization_enabled"] is False


def test_diarization_activee_charge_pyannote(monkeypatch, moteurs_simules):
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    reglages = Settings(_env_file=None, hf_token=TOKEN, device="cuda")
    application = app_module.build_app(reglages)
    assert moteurs_simules["diarization"][0]["model_name"] == reglages.diarization_model
    assert moteurs_simules["diarization"][0]["device"] == "cuda"
    assert (
        TestClient(application).get("/health").json()["diarization_model"]
        == reglages.diarization_model
    )


def test_vad_est_charge_sur_cpu_par_defaut(monkeypatch, moteurs_simules):
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    application = app_module.build_app(
        Settings(_env_file=None, enable_diarization=False, device="cuda")
    )

    assert moteurs_simules["vad"] == [
        {"device": "cpu", "max_segment_s": 5.0}
    ]
    corps = TestClient(application).get("/health").json()
    assert corps["vad_model"] == "silero-vad"
    assert corps["vad_device"] == "cpu"


def test_echec_du_chargement_silero_garde_des_fenetres_courtes(
    monkeypatch, moteurs_simules
):
    monkeypatch.setattr(app_module, "cuda_available", lambda: False)

    def chargement_en_echec(**kwargs):
        raise RuntimeError("silero indisponible")

    monkeypatch.setattr(app_module, "_load_silero_vad_engine", chargement_en_echec)
    application = app_module.build_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu")
    )

    corps = TestClient(application).get("/health").json()
    assert corps["vad_model"] == "fixed-windows"
    assert corps["vad_enabled"] is True


def test_les_reglages_sont_transmis_aux_fabriques(monkeypatch, moteurs_simules):
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    reglages = Settings(
        _env_file=None, hf_token=TOKEN, device="cuda", compute_type="float32"
    )
    app_module.build_app(reglages)
    assert moteurs_simules["asr"][0] == {
        "model_name": reglages.asr_model,
        "device": "cuda",
        "compute_type": "float32",
    }


def test_le_warmup_a_lieu_sur_cuda(monkeypatch):
    """La premiere requete reelle ne doit pas payer la compilation des kernels."""
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)
    appels = []

    class AsrQuiCompte:
        name = "compte"

        def transcribe(self, audio, language):
            appels.append(len(audio))
            return []

    monkeypatch.setattr(
        app_module, "_load_nemo_engine", lambda **kw: AsrQuiCompte()
    )
    app_module.build_app(
        Settings(_env_file=None, enable_diarization=False, device="cuda")
    )
    assert appels == [16000], "le warmup doit passer 1 s de silence"


def test_pas_de_warmup_sur_cpu(monkeypatch):
    monkeypatch.setattr(app_module, "cuda_available", lambda: False)
    appels = []

    class AsrQuiCompte:
        name = "compte"

        def transcribe(self, audio, language):
            appels.append(len(audio))
            return []

    monkeypatch.setattr(
        app_module, "_load_nemo_engine", lambda **kw: AsrQuiCompte()
    )
    app_module.build_app(
        Settings(_env_file=None, enable_diarization=False, device="cpu")
    )
    assert appels == []


def test_un_warmup_qui_echoue_ne_bloque_pas_le_demarrage(monkeypatch):
    """Le warmup est un confort, pas une condition de service."""
    monkeypatch.setattr(app_module, "cuda_available", lambda: True)

    class AsrQuiExplose:
        name = "explose"

        def transcribe(self, audio, language):
            raise RuntimeError("kernel indisponible")

    monkeypatch.setattr(
        app_module, "_load_nemo_engine", lambda **kw: AsrQuiExplose()
    )
    application = app_module.build_app(
        Settings(_env_file=None, enable_diarization=False, device="cuda")
    )
    assert TestClient(application).get("/health").status_code == 200


# --- Contrats deja etablis, verifies ici sur l'application assemblee ----------


def test_device_info_est_expose_par_health():
    application = create_app(
        settings=Settings(_env_file=None, enable_diarization=False, device="cpu"),
        asr=StubAsrEngine([]),
        diarization=NullDiarizationEngine(),
        device_info={"name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576},
    )
    corps = TestClient(application).get("/health").json()
    assert corps["gpu"]["name"] == "NVIDIA GeForce RTX 3090"


def test_main_est_appelable_sans_lancer_le_serveur(monkeypatch):
    """Verifie le cablage de `main` sans ouvrir de socket."""
    from transcription_server import main as main_module

    lances = []
    monkeypatch.setattr(
        main_module, "build_app", lambda reglages: "application-simulee"
    )
    monkeypatch.setattr(
        main_module.uvicorn,
        "run",
        lambda application, **kw: lances.append((application, kw)),
    )
    monkeypatch.setattr(
        main_module, "get_settings", lambda: Settings(
            _env_file=None, enable_diarization=False, device="cpu", port=1234
        )
    )
    main_module.main()
    assert lances[0][0] == "application-simulee"
    assert lances[0][1]["port"] == 1234
    assert lances[0][1]["host"] == "0.0.0.0"
