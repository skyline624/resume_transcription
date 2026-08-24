"""Tests de la surface HTTP native : POST /transcribe et GET /health.

Les moteurs sont injectes dans `create_app`, ce qui permet d'exercer toute la
route sur Windows, sans GPU ni telechargement de modele.
"""

import asyncio
import inspect
import io
import logging
import struct
import tempfile
import threading
import wave
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient

from transcription_server.api.native_routes import _TAILLE_MORCEAU, _save_upload
from transcription_server.api.schemas import result_to_out
from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import (
    NullDiarizationEngine,
    StubDiarizationEngine,
)
from transcription_server.domain import SpeakerSegment, Turn, Word
from transcription_server.pipeline import TranscriptionResult

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"

# Journal ou part le detail retire des reponses 400.
_JOURNAL_ROUTES = "transcription_server.api.native_routes"


def _wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def client():
    settings = Settings(_env_file=None, enable_diarization=False, device="cpu")
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("merci", 1.2, 1.6)]),
        diarization=StubDiarizationEngine(
            [SpeakerSegment(S0, 0.0, 1.0), SpeakerSegment(S1, 1.1, 2.0)]
        ),
    )
    return TestClient(app)


def test_health_repond(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert "asr_model" in body


def test_transcribe_json(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"diarize": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "bonjour merci"
    assert body["speakers"] == [S0, S1]
    assert len(body["turns"]) == 2
    assert body["turns"][0]["speaker"] == S0
    assert body["turns"][0]["words"][0]["word"] == "bonjour"


def test_transcribe_sans_diarization(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"diarize": "false"},
    )
    assert response.status_code == 200
    assert response.json()["speakers"] == []


def test_transcribe_format_dialogue(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"diarize": "true", "response_format": "dialogue"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "SPEAKER_00: bonjour" in response.text


def test_transcribe_format_srt(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "srt"},
    )
    assert response.status_code == 200
    assert response.text.startswith("1\n00:00:00,000 --> ")


def test_fichier_illisible_donne_400(client):
    response = client.post(
        "/transcribe",
        files={"file": ("junk.wav", b"pas de l'audio", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_num_speakers_avec_min_max_donne_400(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"num_speakers": "2", "min_speakers": "1"},
    )
    assert response.status_code == 400


def test_fichier_trop_gros_donne_413():
    settings = Settings(
        _env_file=None, enable_diarization=False, device="cpu", max_upload_mb=1
    )
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("a", 0.0, 0.5)]),
        diarization=StubDiarizationEngine([]),
    )
    client = TestClient(app)
    response = client.post(
        "/transcribe",
        files={"file": ("gros.wav", b"\x00" * (2 * 1024 * 1024), "audio/wav")},
    )
    assert response.status_code == 413


# --------------------------------------------------------------------------
# Outillage des tests supplementaires
# --------------------------------------------------------------------------

MOTS_PAR_DEFAUT = [Word("bonjour", 0.0, 0.5), Word("merci", 1.2, 1.6)]
SEGMENTS_PAR_DEFAUT = [SpeakerSegment(S0, 0.0, 1.0), SpeakerSegment(S1, 1.1, 2.0)]


def _creer_app(asr=None, diarization=None, settings=None, device_info=None):
    """Application de test, moteurs bouchonnes, diarization desactivee."""
    return create_app(
        settings=settings
        or Settings(_env_file=None, enable_diarization=False, device="cpu"),
        asr=asr if asr is not None else StubAsrEngine(list(MOTS_PAR_DEFAUT)),
        diarization=diarization
        if diarization is not None
        else StubDiarizationEngine(list(SEGMENTS_PAR_DEFAUT)),
        device_info=device_info,
    )


def _fichier(nom: str = "test.wav", contenu: bytes | None = None) -> dict:
    return {"file": (nom, _wav_bytes() if contenu is None else contenu, "audio/wav")}


def _wav_de_frames(frames: int, rate: int = 16000) -> bytes:
    """WAV d'un nombre exact d'echantillons, pour viser une duree precise."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack("<h", 0) * frames)
    return buffer.getvalue()


def _parametres_avec_diarization() -> Settings:
    return Settings(
        _env_file=None, enable_diarization=True, device="cpu", hf_token="hf_factice"
    )


class DiarizationEnregistreuse:
    """Retient les bornes de locuteurs recues, pour verifier leur transmission."""

    def __init__(self) -> None:
        self.appels: list[tuple[int | None, int | None, int | None]] = []

    @property
    def name(self) -> str:
        return "diarization-enregistreuse"

    def diarize(self, audio, num_speakers, min_speakers, max_speakers):
        self.appels.append((num_speakers, min_speakers, max_speakers))
        return [SpeakerSegment(S0, 0.0, 2.0)]


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def test_health_expose_les_modeles_et_le_gpu():
    app = _creer_app(device_info={"name": "RTX 4090", "vram_gb": 24})
    reponse = TestClient(app).get("/health")

    corps = reponse.json()
    assert corps["asr_model"] == "stub-asr"
    assert corps["diarization_model"] == "stub-diarization"
    assert corps["diarization_enabled"] is False
    assert corps["gpu"] == {"name": "RTX 4090", "vram_gb": 24}


def test_health_sans_device_info_rend_gpu_nul(client):
    assert client.get("/health").json()["gpu"] is None


def test_health_reflete_la_configuration_de_diarization():
    reponse = TestClient(_creer_app(settings=_parametres_avec_diarization())).get(
        "/health"
    )
    assert reponse.json()["diarization_enabled"] is True


# --------------------------------------------------------------------------
# Formats de sortie
# --------------------------------------------------------------------------


def test_transcribe_format_text(client):
    reponse = client.post(
        "/transcribe", files=_fichier(), data={"response_format": "text"}
    )
    assert reponse.status_code == 200
    assert reponse.headers["content-type"].startswith("text/plain")
    assert reponse.text == "bonjour merci"


def test_transcribe_format_vtt(client):
    reponse = client.post(
        "/transcribe", files=_fichier(), data={"response_format": "vtt"}
    )
    assert reponse.status_code == 200
    assert reponse.text.startswith("WEBVTT\n\n00:00:00.000 --> ")


def test_format_de_reponse_inconnu_donne_422(client):
    reponse = client.post(
        "/transcribe", files=_fichier(), data={"response_format": "yaml"}
    )
    assert reponse.status_code == 422


def test_transcribe_sans_horodatage_des_mots(client):
    reponse = client.post(
        "/transcribe",
        files=_fichier(),
        data={"diarize": "true", "word_timestamps": "false"},
    )
    corps = reponse.json()
    assert reponse.status_code == 200
    assert [t["text"] for t in corps["turns"]] == ["bonjour", "merci"]
    assert all(t["words"] == [] for t in corps["turns"])


# --------------------------------------------------------------------------
# Parametres de la requete
# --------------------------------------------------------------------------


def test_transcribe_rend_la_duree_et_le_chronometrage(client):
    corps = client.post("/transcribe", files=_fichier()).json()
    assert corps["duration"] == pytest.approx(2.0, abs=0.01)
    assert set(corps["timing"]) == {"decode", "asr", "diarization"}


def test_result_to_out_arrondit_a_la_frontiere():
    # `pipeline` arrondit deja son chronometrage, donc aucune requete HTTP ne
    # peut distinguer un `result_to_out` qui arrondit d'un qui se contente de
    # recopier. L'unite du corps JSON -- tous les flottants au millieme -- est
    # pourtant un contrat de l'API, pas une propriete empruntee au fournisseur :
    # on l'exerce donc directement, avec des valeurs brutes.
    resultat = TranscriptionResult(
        text="bonjour",
        language=None,
        duration=2.0000625,
        speakers=[],
        turns=[Turn(None, 0.0001234, 0.5006789, "bonjour", ())],
        timing={"decode": 0.014739990234375, "asr": 1.0 / 3},
    )

    sortie = result_to_out(resultat)

    assert sortie.duration == 2.0
    assert sortie.timing == {"decode": 0.015, "asr": 0.333}
    assert (sortie.turns[0].start, sortie.turns[0].end) == (0.0, 0.501)


def test_la_duree_est_arrondie_au_millieme(client):
    # 32 001 echantillons a 16 kHz font 2,0000625 s : sans arrondi, la reponse
    # porterait les chiffres parasites du calcul en virgule flottante.
    reponse = client.post(
        "/transcribe", files=_fichier("precis.wav", _wav_de_frames(32001))
    )
    assert reponse.json()["duration"] == 2.0


def test_les_horodatages_sont_arrondis_au_millieme():
    client = TestClient(
        _creer_app(asr=StubAsrEngine([Word("bonjour", 0.0001234, 0.5006789)]))
    )
    corps = client.post(
        "/transcribe", files=_fichier(), data={"diarize": "false"}
    ).json()

    tour = corps["turns"][0]
    assert (tour["start"], tour["end"]) == (0.0, 0.501)
    assert (tour["words"][0]["start"], tour["words"][0]["end"]) == (0.0, 0.501)


def test_transcribe_propage_la_langue(client):
    reponse = client.post("/transcribe", files=_fichier(), data={"language": "fr"})
    assert reponse.json()["language"] == "fr"


def test_langue_non_precisee_reste_nulle(client):
    # Limitation assumee : Parakeet v3 detecte la langue mais le contrat de
    # AsrEngine ne la fait pas remonter. Sans `language`, la reponse rend null.
    assert client.post("/transcribe", files=_fichier()).json()["language"] is None


def test_diarization_par_defaut_suit_la_configuration():
    client = TestClient(_creer_app(settings=_parametres_avec_diarization()))
    reponse = client.post("/transcribe", files=_fichier())
    assert reponse.json()["speakers"] == [S0, S1]


def test_diarize_false_prime_sur_la_configuration():
    client = TestClient(_creer_app(settings=_parametres_avec_diarization()))
    reponse = client.post("/transcribe", files=_fichier(), data={"diarize": "false"})
    assert reponse.json()["speakers"] == []


def test_num_speakers_est_transmis_au_moteur():
    moteur = DiarizationEnregistreuse()
    client = TestClient(_creer_app(diarization=moteur))
    reponse = client.post(
        "/transcribe", files=_fichier(), data={"diarize": "true", "num_speakers": "3"}
    )
    assert reponse.status_code == 200
    assert moteur.appels == [(3, None, None)]


def test_min_et_max_speakers_sont_transmis_au_moteur():
    moteur = DiarizationEnregistreuse()
    client = TestClient(_creer_app(diarization=moteur))
    reponse = client.post(
        "/transcribe",
        files=_fichier(),
        data={"diarize": "true", "min_speakers": "2", "max_speakers": "4"},
    )
    assert reponse.status_code == 200
    assert moteur.appels == [(None, 2, 4)]


def test_num_speakers_seul_est_accepte(client):
    reponse = client.post(
        "/transcribe", files=_fichier(), data={"diarize": "true", "num_speakers": "2"}
    )
    assert reponse.status_code == 200


def test_diarize_explicite_sur_moteur_nul_donne_400():
    # Sous Task 14, ENABLE_DIARIZATION=false injecte NullDiarizationEngine.
    # Rendre 200 avec `speakers: []` ferait croire a un audio mono-locuteur
    # alors que la fonction est simplement eteinte : echouer bruyamment.
    client = TestClient(_creer_app(diarization=NullDiarizationEngine()))
    reponse = client.post("/transcribe", files=_fichier(), data={"diarize": "true"})

    assert reponse.status_code == 400
    erreur = reponse.json()["error"]
    assert erreur["type"] == "invalid_request_error"
    assert "ENABLE_DIARIZATION" in erreur["message"]


def test_diarize_absent_sur_moteur_nul_reste_accepte():
    # Le cas non demande continue de suivre la configuration, sans erreur.
    client = TestClient(_creer_app(diarization=NullDiarizationEngine()))
    reponse = client.post("/transcribe", files=_fichier())

    assert reponse.status_code == 200
    assert reponse.json()["speakers"] == []


def test_diarize_false_sur_moteur_nul_reste_accepte():
    client = TestClient(_creer_app(diarization=NullDiarizationEngine()))
    reponse = client.post("/transcribe", files=_fichier(), data={"diarize": "false"})

    assert reponse.status_code == 200
    assert reponse.json()["speakers"] == []


def test_diarize_explicite_sur_moteur_reel_reste_accepte(client):
    # Le controle porte sur le moteur, pas sur ENABLE_DIARIZATION : ce serveur
    # a `enable_diarization=False` mais un vrai moteur, la demande est donc
    # legitime. C'est deja ce que fige `test_transcribe_json`.
    reponse = client.post("/transcribe", files=_fichier(), data={"diarize": "true"})
    assert reponse.status_code == 200
    assert reponse.json()["speakers"] == [S0, S1]


def test_num_speakers_avec_max_speakers_donne_400(client):
    reponse = client.post(
        "/transcribe",
        files=_fichier(),
        data={"num_speakers": "2", "max_speakers": "4"},
    )
    assert reponse.status_code == 400
    assert reponse.json()["error"]["type"] == "invalid_request_error"


# --------------------------------------------------------------------------
# Erreurs : forme et confidentialite
# --------------------------------------------------------------------------


def test_message_400_ne_divulgue_ni_chemin_ni_stderr(client, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    reponse = client.post("/transcribe", files=_fichier("junk.wav", b"pas un son"))

    assert reponse.status_code == 400
    corps = reponse.text
    assert "ffmpeg" not in corps.lower()
    assert str(tmp_path) not in corps
    assert tmp_path.name not in corps
    assert reponse.json()["error"]["message"] == (
        "Le fichier audio n'a pas pu être décodé."
    )


def test_message_413_ne_divulgue_pas_de_chemin(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    reponse = _client_limite_1mo().post(
        "/transcribe", files=_fichier("gros.wav", b"\x00" * (2 * 1024 * 1024))
    )

    assert reponse.status_code == 413
    assert reponse.json()["error"]["type"] == "invalid_request_error"
    assert str(tmp_path) not in reponse.text


def test_le_400_journalise_le_detail_retire_de_la_reponse(client, caplog):
    # `native_routes` est desormais le seul endroit ou le diagnostic survit :
    # sans cette assertion, le correctif degenererait en perte d'information.
    with caplog.at_level(logging.WARNING, logger=_JOURNAL_ROUTES):
        reponse = client.post("/transcribe", files=_fichier("junk.wav", b"pas un son"))

    assert reponse.status_code == 400
    avertissements = [
        enr
        for enr in caplog.records
        if enr.name == _JOURNAL_ROUTES and enr.levelno == logging.WARNING
    ]
    assert len(avertissements) == 1

    message = avertissements[0].getMessage()
    assert message.startswith("Échec de décodage :")
    assert "ffmpeg" in message.lower()


class AsrQuiExplose:
    """Moteur qui echoue comme le ferait un GPU tombant en panne."""

    @property
    def name(self) -> str:
        return "asr-qui-explose"

    def transcribe(self, audio, language):
        raise RuntimeError("le modèle a rendu l'âme sur /dev/nvidia0")


def test_erreur_non_geree_rend_lenveloppe_openai(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = TestClient(_creer_app(asr=AsrQuiExplose()), raise_server_exceptions=False)

    reponse = client.post("/transcribe", files=_fichier(), data={"diarize": "false"})

    assert reponse.status_code == 500
    assert reponse.headers["content-type"].startswith("application/json")
    assert reponse.json() == {
        "error": {"message": "Erreur interne du serveur.", "type": "server_error"}
    }
    # Ni la trace ni le message d'origine ne doivent transparaitre.
    assert "nvidia" not in reponse.text
    assert "RuntimeError" not in reponse.text
    # Et le temporaire disparait malgre l'exception : seul chemin d'erreur des
    # moteurs, jusqu'ici non figé.
    assert list(tmp_path.iterdir()) == []


def test_openapi_documente_la_reponse_de_transcribe(client):
    # La route rend soit un PlainTextResponse soit un TranscriptionOut, donc
    # pas de `response_model` ; `responses` documente le cas JSON sans
    # contraindre le retour reel.
    schema = client.get("/openapi.json").json()

    assert "TranscriptionOut" in schema["components"]["schemas"]
    contenu = schema["paths"]["/transcribe"]["post"]["responses"]["200"]["content"]
    assert contenu["application/json"]["schema"]["$ref"].endswith("/TranscriptionOut")


def test_les_trois_formes_derreur_de_lapi(client):
    """Fige chaque forme d'erreur exposee aujourd'hui.

    L'API n'en rend pas une seule : Task 10 tranchera leur unification. D'ici
    la, tout glissement de forme casse ce test au lieu de passer inapercu.
    """
    # 1. HTTPException levee par une route, et desormais toute exception non
    #    geree : enveloppe OpenAI.
    conflit = client.post(
        "/transcribe",
        files=_fichier(),
        data={"num_speakers": "2", "min_speakers": "1"},
    )
    assert conflit.status_code == 400
    assert set(conflit.json()) == {"error"}
    assert set(conflit.json()["error"]) == {"message", "type"}

    # 2. 404 de routage et 405 : forme FastAPI par defaut, hors du MRO de
    #    fastapi.HTTPException, donc non uniformisee.
    assert client.get("/inconnu").json() == {"detail": "Not Found"}
    assert client.get("/transcribe").json() == {"detail": "Method Not Allowed"}

    # 3. 422 de validation : liste de details FastAPI.
    invalide = client.post(
        "/transcribe", files=_fichier(), data={"response_format": "yaml"}
    )
    assert invalide.status_code == 422
    assert isinstance(invalide.json()["detail"], list)


def test_enveloppe_derreur_couvre_503_et_code_inconnu():
    app = _creer_app()

    @app.get("/_indisponible")
    async def _indisponible():
        raise HTTPException(status_code=503, detail="Moteur indisponible.")

    @app.get("/_bizarre")
    async def _bizarre():
        raise HTTPException(status_code=418, detail="Je suis une théière.")

    client = TestClient(app)

    indisponible = client.get("/_indisponible")
    assert indisponible.status_code == 503
    assert indisponible.json() == {
        "error": {"message": "Moteur indisponible.", "type": "service_unavailable"}
    }

    bizarre = client.get("/_bizarre")
    assert bizarre.status_code == 418
    assert bizarre.json() == {
        "error": {"message": "Je suis une théière.", "type": "server_error"}
    }


# --------------------------------------------------------------------------
# Limite de taille : bornes exactes
# --------------------------------------------------------------------------


def _client_limite_1mo() -> TestClient:
    parametres = Settings(
        _env_file=None, enable_diarization=False, device="cpu", max_upload_mb=1
    )
    return TestClient(_creer_app(settings=parametres))


def test_fichier_exactement_a_la_limite_nest_pas_refuse():
    # Exactement max_upload_bytes : accepte, puis rejete par le decodage (400).
    # Un `>=` a la place du `>` rendrait 413 ici.
    reponse = _client_limite_1mo().post(
        "/transcribe", files=_fichier("limite.wav", b"\x00" * (1024 * 1024))
    )
    assert reponse.status_code == 400


def test_un_octet_au_dessus_de_la_limite_donne_413():
    reponse = _client_limite_1mo().post(
        "/transcribe", files=_fichier("gros.wav", b"\x00" * (1024 * 1024 + 1))
    )
    assert reponse.status_code == 413


# --------------------------------------------------------------------------
# Fichiers temporaires : aucun reliquat, sur aucun chemin
# --------------------------------------------------------------------------


def test_aucun_fichier_temporaire_apres_succes(client, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert client.post("/transcribe", files=_fichier()).status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_aucun_fichier_temporaire_apres_400(client, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    reponse = client.post("/transcribe", files=_fichier("junk.wav", b"pas un son"))
    assert reponse.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_aucun_fichier_temporaire_apres_413(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    reponse = _client_limite_1mo().post(
        "/transcribe", files=_fichier("gros.wav", b"\x00" * (2 * 1024 * 1024))
    )
    assert reponse.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_une_extension_demesuree_ne_fait_pas_tomber_la_route(
    client, tmp_path, monkeypatch
):
    # Le nom du fichier est entierement controle par l'appelant : sans borne
    # sur le suffixe, la creation du temporaire echoue (nom trop long) et la
    # route rend un 500.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    reponse = client.post("/transcribe", files=_fichier("piege." + "a" * 300))

    assert reponse.status_code == 200
    assert reponse.json()["text"] == "bonjour merci"
    assert list(tmp_path.iterdir()) == []


def test_un_nom_sans_extension_est_accepte(client):
    reponse = client.post("/transcribe", files=_fichier("sans_extension"))
    assert reponse.status_code == 200


class UploadCassee:
    """Upload qui echoue en cours de lecture, apres un premier morceau."""

    filename = "casse.wav"

    def __init__(self) -> None:
        self.appels = 0

    async def read(self, taille: int) -> bytes:
        self.appels += 1
        if self.appels == 1:
            return b"\x00" * 32
        raise OSError("lecture interrompue")


async def test_save_upload_ne_laisse_rien_si_la_lecture_echoue(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(OSError):
        await _save_upload(UploadCassee(), 10 * 1024 * 1024)

    assert list(tmp_path.iterdir()) == []


class UploadSimple:
    """Upload minimal, qui rend son contenu puis la fin de flux."""

    filename = "petit.wav"

    def __init__(self, contenu: bytes = b"\x00" * 64) -> None:
        self._reste = contenu

    async def read(self, taille: int) -> bytes:
        morceau, self._reste = self._reste[:taille], self._reste[taille:]
        return morceau


async def test_save_upload_ecrit_tous_les_morceaux(tmp_path, monkeypatch):
    # `_save_upload` lit par blocs de 1 Mio. Tant qu'aucun test ne depasse un
    # bloc, la boucle ne tourne jamais deux fois : un `break` apres le premier
    # morceau, ou un `write` sorti de la boucle, passerait inapercu et
    # tronquerait silencieusement tout upload reel. 2,5 Mio, donc trois
    # iterations, avec un motif qui differe a chaque bloc pour attraper aussi
    # une ecriture desordonnee ou repetee.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    contenu = bytes(range(256)) * 10240
    assert len(contenu) == 2_621_440

    chemin = await _save_upload(UploadSimple(contenu), 10 * 1024 * 1024)

    try:
        assert chemin.stat().st_size == len(contenu)
        assert chemin.read_bytes() == contenu
    finally:
        chemin.unlink(missing_ok=True)


def test_transcribe_ne_tronque_pas_un_upload_multi_morceaux(client):
    # Pendant complet du test ci-dessus : de bout en bout, sur la vraie route.
    # 40 s a 16 kHz mono 16 bits font 1 280 044 octets, soit plus d'un bloc de
    # lecture. La duree l'atteste -- et referme du meme coup le fait que la
    # duree n'etait jamais assertee qu'autour de 2,0 s.
    audio = _wav_de_frames(40 * 16000)
    assert len(audio) > _TAILLE_MORCEAU

    reponse = client.post("/transcribe", files=_fichier("longue.wav", audio))

    assert reponse.status_code == 200
    assert reponse.json()["duration"] == pytest.approx(40.0, abs=0.01)


class FermetureCassee:
    """Emule un disque plein : le vidage du tampon echoue a la fermeture.

    Comme un vrai objet fichier, le second appel a close() est sans effet : io
    marque le fichier ferme meme quand le flush a leve.
    """

    def __init__(self, reel) -> None:
        self._reel = reel
        self.name = reel.name
        self.fermetures = 0

    def write(self, donnees: bytes) -> int:
        return self._reel.write(donnees)

    def close(self) -> None:
        self._reel.close()
        self.fermetures += 1
        if self.fermetures == 1:
            raise OSError("plus d'espace disponible sur le peripherique")


async def test_save_upload_ne_laisse_rien_si_la_fermeture_echoue(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    fabrique_reelle = tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: FermetureCassee(fabrique_reelle(*args, **kwargs)),
    )

    with pytest.raises(OSError):
        await _save_upload(UploadSimple(), 10 * 1024 * 1024)

    assert list(tmp_path.iterdir()) == []


def test_save_upload_garde_son_nom_et_sa_signature():
    # Task 10 importe `_save_upload` depuis ce module : le nom et la signature
    # font partie du contrat. Le nom et le module sont deja epingles par
    # l'import en tete de fichier, qui casse a la collecte.
    signature = inspect.signature(_save_upload)
    parametres = list(signature.parameters.values())

    assert [p.name for p in parametres] == ["upload", "max_bytes"]
    # Le kind compte autant que le nom : passer a `(upload, *, max_bytes)`
    # casserait un appel positionnel en Task 10 sans rien changer aux noms.
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parametres
    )
    # Et l'appelant construit un chemin, pas une chaine.
    assert signature.return_annotation is Path
    assert inspect.iscoroutinefunction(_save_upload)


# --------------------------------------------------------------------------
# Verrou GPU et disponibilite de /health
# --------------------------------------------------------------------------


class AsrBloquant:
    """Se met en attente dans le thread d'execution jusqu'a liberation."""

    def __init__(self, entre: threading.Event, liberer: threading.Event) -> None:
        self._entre = entre
        self._liberer = liberer

    @property
    def name(self) -> str:
        return "asr-bloquant"

    def transcribe(self, audio, language):
        self._entre.set()
        self._liberer.wait(10.0)
        return [Word("bonjour", 0.0, 0.5)]


async def test_health_reste_disponible_pendant_une_transcription():
    entre = threading.Event()
    liberer = threading.Event()
    app = _creer_app(asr=AsrBloquant(entre, liberer))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://serveur") as client:
        tache = asyncio.create_task(
            client.post(
                "/transcribe", files=_fichier(), data={"diarize": "false"}, timeout=30.0
            )
        )
        try:
            assert await asyncio.to_thread(entre.wait, 10.0), (
                "la transcription n'a jamais atteint l'ASR"
            )
            sante = await client.get("/health")
            assert sante.status_code == 200
            assert sante.json()["status"] == "ok"
        finally:
            liberer.set()
        assert (await tache).status_code == 200


class VerrouObserve(asyncio.Lock):
    """Verrou identique, qui compte les acquisitions ayant du attendre."""

    def __init__(self) -> None:
        super().__init__()
        self.attentes = 0

    async def acquire(self) -> bool:
        if self.locked():
            self.attentes += 1
        return await super().acquire()


class AsrRendezVous:
    """Le premier appel retient le verrou jusqu'a liberation, pas les suivants."""

    def __init__(self, entre: threading.Event, liberer: threading.Event) -> None:
        self._entre = entre
        self._liberer = liberer
        self._verrou = threading.Lock()
        self.appels = 0
        self.simultanes = 0
        self.maximum = 0

    @property
    def name(self) -> str:
        return "asr-rendez-vous"

    def transcribe(self, audio, language):
        with self._verrou:
            self.appels += 1
            premier = self.appels == 1
            self.simultanes += 1
            self.maximum = max(self.maximum, self.simultanes)
        if premier:
            self._entre.set()
            self._liberer.wait(10.0)
        with self._verrou:
            self.simultanes -= 1
        return [Word("bonjour", 0.0, 0.5)]


async def _attendre_contention(verrou: VerrouObserve) -> None:
    while verrou.attentes == 0:
        await asyncio.sleep(0.005)


async def test_le_verrou_gpu_serialise_les_transcriptions():
    entre = threading.Event()
    liberer = threading.Event()
    moteur = AsrRendezVous(entre, liberer)
    app = _creer_app(asr=moteur)
    verrou = VerrouObserve()
    app.state.app_state.gpu_lock = verrou

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://serveur") as client:

        def poster():
            return asyncio.create_task(
                client.post(
                    "/transcribe",
                    files=_fichier(),
                    data={"diarize": "false"},
                    timeout=30.0,
                )
            )

        premiere = poster()
        assert await asyncio.to_thread(entre.wait, 10.0), (
            "la premiere transcription n'a jamais atteint l'ASR"
        )

        seconde = poster()
        try:
            # Le point cle : on attend que la seconde requete se heurte
            # *reellement* au verrou. Sans verrou elle passerait sans jamais
            # attendre, cette attente expirerait, et le test echouerait -- au
            # lieu de reussir a vide faute de recouvrement, comme le ferait une
            # mesure fondee sur un `sleep`.
            await asyncio.wait_for(_attendre_contention(verrou), timeout=5.0)
            assert moteur.appels == 1, (
                "la seconde requete a atteint l'ASR malgre le verrou"
            )
        finally:
            liberer.set()

        reponses = await asyncio.gather(premiere, seconde)

    assert [r.status_code for r in reponses] == [200, 200]
    assert verrou.attentes == 1
    assert moteur.maximum == 1


def test_le_parametre_channels_est_transmis_au_pipeline(client, monkeypatch):
    """Sans ce test, un `channels=split` accepte puis ignore rendrait une
    transcription repliee en mono sans que rien ne le signale."""
    from transcription_server.api import native_routes

    recu = []
    vrai_run = native_routes.run_pipeline

    def espion(**kwargs):
        recu.append(kwargs["request"].channel_mode)
        return vrai_run(**kwargs)

    monkeypatch.setattr(native_routes, "run_pipeline", espion)
    client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"channels": "split"},
    )
    assert recu == ["split"]


def test_channels_vaut_mix_par_defaut(client, monkeypatch):
    from transcription_server.api import native_routes

    recu = []
    vrai_run = native_routes.run_pipeline

    def espion(**kwargs):
        recu.append(kwargs["request"].channel_mode)
        return vrai_run(**kwargs)

    monkeypatch.setattr(native_routes, "run_pipeline", espion)
    client.post(
        "/transcribe", files={"file": ("test.wav", _wav_bytes(), "audio/wav")}
    )
    assert recu == ["mix"]


def test_channels_inconnu_est_rejete(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"channels": "quadriphonie"},
    )
    assert response.status_code == 422


def test_la_reponse_dit_combien_de_canaux_ont_ete_transcrits(client):
    corps = client.post(
        "/transcribe", files={"file": ("test.wav", _wav_bytes(), "audio/wav")}
    ).json()
    assert corps["channels_used"] == 1
