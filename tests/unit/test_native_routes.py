"""Tests de la surface HTTP native : POST /transcribe et GET /health.

Les moteurs sont injectes dans `create_app`, ce qui permet d'exercer toute la
route sur Windows, sans GPU ni telechargement de modele.
"""

import asyncio
import inspect
import io
import struct
import tempfile
import threading
import time
import wave

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from transcription_server.api.native_routes import _save_upload
from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import StubDiarizationEngine
from transcription_server.domain import SpeakerSegment, Word

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"


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
    # font partie du contrat.
    signature = inspect.signature(_save_upload)
    assert list(signature.parameters) == ["upload", "max_bytes"]
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


class AsrCompteur:
    """Mesure le nombre maximal d'appels simultanes a l'ASR."""

    def __init__(self) -> None:
        self._verrou = threading.Lock()
        self.en_cours = 0
        self.maximum = 0

    @property
    def name(self) -> str:
        return "asr-compteur"

    def transcribe(self, audio, language):
        with self._verrou:
            self.en_cours += 1
            self.maximum = max(self.maximum, self.en_cours)
        time.sleep(0.15)
        with self._verrou:
            self.en_cours -= 1
        return [Word("bonjour", 0.0, 0.5)]


async def test_le_verrou_gpu_serialise_les_transcriptions():
    moteur = AsrCompteur()
    app = _creer_app(asr=moteur)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://serveur") as client:
        reponses = await asyncio.gather(
            *(
                client.post(
                    "/transcribe",
                    files=_fichier(),
                    data={"diarize": "false"},
                    timeout=30.0,
                )
                for _ in range(2)
            )
        )

    assert [r.status_code for r in reponses] == [200, 200]
    assert moteur.maximum == 1
