"""Redaction de compte-rendu : gabarits, moteur Ollama, endpoint.

Aucun modele de langue ne tourne ici : le Protocol permet de piloter toute la
surface HTTP avec un moteur factice, comme pour l'ASR et la diarization.
"""

import io
import json
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import StubDiarizationEngine
from transcription_server.domain import Turn, Word
from transcription_server.summary.engine import (
    StubSummaryEngine,
    SummaryEngine,
    SummaryUnavailableError,
    UnavailableSummaryEngine,
)
from transcription_server.summary.ollama_engine import OllamaSummaryEngine
from transcription_server.summary.prompts import (
    FORMATS,
    construire_prompt,
    rendre_dialogue,
)

TOURS = [
    Turn("SPEAKER_00", 0.0, 4.0, "Bonjour à tous.", ()),
    Turn("SPEAKER_01", 65.5, 70.0, "Merci de votre présence.", ()),
    Turn(None, 3700.0, 3705.0, "Voix non attribuée.", ()),
]


def _wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


# --- Rendu du dialogue --------------------------------------------------------


def test_rendre_dialogue_horodate_en_heures_minutes_secondes():
    rendu = rendre_dialogue(TOURS).splitlines()
    assert rendu[0].startswith("[00:00:00] SPEAKER_00 : ")
    assert rendu[1].startswith("[00:01:05] SPEAKER_01 : ")
    assert rendu[2].startswith("[01:01:40] INCONNU : ")


def test_rendre_dialogue_ignore_les_tours_vides():
    tours = [*TOURS, Turn("SPEAKER_00", 10.0, 11.0, "", ())]
    assert len(rendre_dialogue(tours).splitlines()) == 3


def test_rendre_dialogue_sur_liste_vide():
    assert rendre_dialogue([]) == ""


# --- Gabarits -----------------------------------------------------------------


@pytest.mark.parametrize("format_", FORMATS)
def test_le_prompt_contient_la_transcription_et_les_garde_fous(format_):
    prompt = construire_prompt(rendre_dialogue(TOURS), format_)
    assert "SPEAKER_00" in prompt
    assert "Bonjour à tous." in prompt
    assert "N'invente rien" in prompt


def test_le_gabarit_structure_impose_ses_sections():
    prompt = construire_prompt("[00:00:00] SPEAKER_00 : test", "structure")
    for section in ("## Objet", "## Décisions", "## Actions à mener"):
        assert section in prompt


def test_le_gabarit_narratif_interdit_les_listes():
    prompt = construire_prompt("[00:00:00] SPEAKER_00 : test", "narratif")
    assert "prose continue" in prompt
    assert "## Décisions" not in prompt


def test_les_deux_gabarits_different_reellement():
    """Sans cette verification, une erreur d'aiguillage rendrait le meme texte
    pour les deux formats sans que rien ne le signale."""
    assert construire_prompt("x", "structure") != construire_prompt("x", "narratif")


def test_format_inconnu_est_rejete():
    with pytest.raises(ValueError, match="Format de compte-rendu inconnu"):
        construire_prompt("x", "haiku")


def test_transcription_vide_est_rejetee():
    with pytest.raises(ValueError, match="vide"):
        construire_prompt("   \n  ", "structure")


# --- Moteur Ollama ------------------------------------------------------------


class _FausseReponse:
    def __init__(self, charge):
        self._charge = json.dumps(charge).encode("utf-8")

    def read(self, *args):
        return self._charge

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_ollama_rend_le_texte_et_pose_une_temperature_basse(monkeypatch):
    """Une temperature elevee rendrait un compte-rendu different a chaque appel
    sur la meme reunion, et inviterait le modele a broder."""
    envoye = {}

    def faux_urlopen(requete, timeout=None):
        envoye["url"] = requete.full_url
        envoye["corps"] = json.loads(requete.data)
        return _FausseReponse({"response": "  Compte-rendu.  "})

    monkeypatch.setattr(
        "transcription_server.summary.ollama_engine.urllib.request.urlopen",
        faux_urlopen,
    )
    moteur = OllamaSummaryEngine("http://ollama:11434/", "qwen", 30.0)
    assert moteur.summarize("consigne") == "Compte-rendu."
    assert envoye["url"] == "http://ollama:11434/api/generate"
    assert envoye["corps"]["model"] == "qwen"
    assert envoye["corps"]["stream"] is False
    assert envoye["corps"]["options"]["temperature"] <= 0.3


def test_ollama_injoignable_donne_une_erreur_actionnable(monkeypatch):
    import urllib.error

    def faux_urlopen(requete, timeout=None):
        raise urllib.error.URLError("connexion refusée")

    monkeypatch.setattr(
        "transcription_server.summary.ollama_engine.urllib.request.urlopen",
        faux_urlopen,
    )
    moteur = OllamaSummaryEngine("http://absent:11434", "qwen", 1.0)
    with pytest.raises(SummaryUnavailableError) as excinfo:
        moteur.summarize("x")
    assert "OLLAMA_BASE_URL" in str(excinfo.value)


def test_ollama_reponse_vide_nomme_le_modele(monkeypatch):
    """Cas courant : le modele configure n'est pas installe."""

    def faux_urlopen(requete, timeout=None):
        return _FausseReponse({"response": ""})

    monkeypatch.setattr(
        "transcription_server.summary.ollama_engine.urllib.request.urlopen",
        faux_urlopen,
    )
    moteur = OllamaSummaryEngine("http://x", "modele-absent", 1.0)
    with pytest.raises(SummaryUnavailableError, match="modele-absent"):
        moteur.summarize("x")


def test_les_moteurs_respectent_le_protocol():
    for moteur in (StubSummaryEngine(), UnavailableSummaryEngine()):
        assert isinstance(moteur, SummaryEngine)


def test_le_moteur_indisponible_nomme_le_reglage():
    with pytest.raises(SummaryUnavailableError, match="ENABLE_SUMMARY"):
        UnavailableSummaryEngine().summarize("x")


# --- Endpoint -----------------------------------------------------------------


@pytest.fixture
def client_avec_redaction():
    moteur = StubSummaryEngine("## Objet\nUne réunion.")
    app = create_app(
        settings=Settings(_env_file=None, enable_diarization=False, device="cpu"),
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.5)]),
        diarization=StubDiarizationEngine([]),
        summary=moteur,
    )
    client = TestClient(app)
    client.moteur = moteur
    return client


def test_summarize_a_partir_d_une_transcription(client_avec_redaction):
    reponse = client_avec_redaction.post(
        "/summarize",
        data={"transcript": "[00:00:00] SPEAKER_00 : Bonjour.", "format": "structure"},
    )
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["summary"] == "## Objet\nUne réunion."
    assert corps["format"] == "structure"
    assert corps["model"] == "stub-summary"


def test_summarize_a_partir_d_un_audio(client_avec_redaction):
    reponse = client_avec_redaction.post(
        "/summarize",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
    )
    assert reponse.status_code == 200
    prompt = client_avec_redaction.moteur.prompts_recus[0]
    assert "bonjour" in prompt, "la transcription doit atteindre le rédacteur"


def test_summarize_audio_utilise_le_plan_vad():
    longueurs: list[int] = []

    class AsrQuiMesure:
        name = "asr-qui-mesure"

        def transcribe(self, audio, language):
            longueurs.append(len(audio))
            return [Word("bonjour", 0.0, 0.25)]

    class VadUneDemiSeconde:
        name = "vad-test"
        device = "cpu"

        def plan(self, audio):
            return [(0.5, 1.0)]

    app = create_app(
        settings=Settings(_env_file=None, enable_diarization=False, device="cpu"),
        asr=AsrQuiMesure(),
        diarization=StubDiarizationEngine([]),
        summary=StubSummaryEngine("Compte-rendu."),
        vad=VadUneDemiSeconde(),
    )
    response = TestClient(app).post(
        "/summarize", files={"file": ("test.wav", _wav_bytes(), "audio/wav")}
    )

    assert response.status_code == 200
    assert longueurs == [8000]


def test_fournir_les_deux_sources_est_refuse(client_avec_redaction):
    reponse = client_avec_redaction.post(
        "/summarize",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"transcript": "[00:00:00] SPEAKER_00 : Bonjour."},
    )
    assert reponse.status_code == 400
    assert "pas les deux" in reponse.json()["error"]["message"]


def test_ne_fournir_aucune_source_est_refuse(client_avec_redaction):
    reponse = client_avec_redaction.post("/summarize", data={"format": "narratif"})
    assert reponse.status_code == 400


def test_le_format_demande_atteint_le_gabarit(client_avec_redaction):
    client_avec_redaction.post(
        "/summarize",
        data={"transcript": "[00:00:00] SPEAKER_00 : Bonjour.", "format": "narratif"},
    )
    assert "prose continue" in client_avec_redaction.moteur.prompts_recus[0]


def test_format_inconnu_donne_422(client_avec_redaction):
    reponse = client_avec_redaction.post(
        "/summarize",
        data={"transcript": "[00:00:00] SPEAKER_00 : x", "format": "haiku"},
    )
    assert reponse.status_code == 422


def test_sortie_en_texte_brut(client_avec_redaction):
    reponse = client_avec_redaction.post(
        "/summarize",
        data={
            "transcript": "[00:00:00] SPEAKER_00 : Bonjour.",
            "response_format": "text",
        },
    )
    assert reponse.headers["content-type"].startswith("text/plain")
    assert reponse.text == "## Objet\nUne réunion."


def test_la_transcription_peut_etre_renvoyee(client_avec_redaction):
    corps = client_avec_redaction.post(
        "/summarize",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"include_transcript": "true"},
    ).json()
    assert "bonjour" in corps["transcript"]


def test_la_transcription_n_est_pas_renvoyee_par_defaut(client_avec_redaction):
    corps = client_avec_redaction.post(
        "/summarize",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
    ).json()
    assert "transcript" not in corps


def test_redaction_desactivee_donne_503():
    """Le serveur doit continuer a transcrire meme sans redacteur : c'est la
    route qui refuse, pas le demarrage."""
    app = create_app(
        settings=Settings(_env_file=None, enable_diarization=False, device="cpu"),
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.5)]),
        diarization=StubDiarizationEngine([]),
        summary=UnavailableSummaryEngine(),
    )
    client = TestClient(app)
    reponse = client.post(
        "/summarize", data={"transcript": "[00:00:00] SPEAKER_00 : x"}
    )
    assert reponse.status_code == 503
    assert reponse.json()["error"]["type"] == "service_unavailable"
    # La transcription, elle, doit rester disponible.
    assert (
        client.post(
            "/transcribe", files={"file": ("t.wav", _wav_bytes(), "audio/wav")}
        ).status_code
        == 200
    )


def test_health_annonce_le_redacteur(client_avec_redaction):
    corps = client_avec_redaction.get("/health").json()
    assert corps["summary_model"] == "stub-summary"
    assert corps["summary_enabled"] is True
