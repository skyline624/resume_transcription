from pathlib import Path

import pytest
from pydantic import ValidationError

from transcription_server.config import Settings, get_settings


# Les defauts activent la diarization, qui exige un token. Les tests qui ne
# portent pas sur cette regle en fournissent donc un factice.
TOKEN = "hf_pour_les_tests"


def test_defauts_conformes_a_la_spec():
    s = Settings(_env_file=None, hf_token=TOKEN)
    assert s.asr_model == "nvidia/parakeet-tdt-0.6b-v3"
    assert s.diarization_model == "pyannote/speaker-diarization-community-1"
    assert s.enable_diarization is True
    assert s.device == "cuda"
    assert s.compute_type == "float16"
    assert s.chunk_length_s == 480.0
    assert s.chunk_overlap_s == 15.0
    assert s.turn_gap_s == 1.0
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.max_upload_mb == 1024


def test_max_upload_bytes():
    s = Settings(_env_file=None, hf_token=TOKEN, max_upload_mb=2)
    assert s.max_upload_bytes == 2 * 1024 * 1024


def test_device_invalide_est_rejete():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hf_token=TOKEN, device="tpu")


def test_compute_type_invalide_est_rejete():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hf_token=TOKEN, compute_type="int4")


def test_recouvrement_superieur_au_chunk_est_rejete():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            hf_token=TOKEN,
            chunk_length_s=100.0,
            chunk_overlap_s=100.0,
        )


def test_diarization_active_sans_token_est_rejetee():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, enable_diarization=True, hf_token=None)


def test_diarization_desactivee_sans_token_est_acceptee():
    s = Settings(_env_file=None, enable_diarization=False, hf_token=None)
    assert s.enable_diarization is False


def test_lecture_depuis_l_environnement(monkeypatch):
    monkeypatch.setenv("CHUNK_LENGTH_S", "120")
    monkeypatch.setenv("ENABLE_DIARIZATION", "false")
    s = Settings(_env_file=None)
    assert s.chunk_length_s == 120.0
    assert s.enable_diarization is False


# Les tests ci-dessous completent ceux du brief : chacun tue une mutation
# plausible de l'implementation qui survivait a la premiere serie.


def test_recouvrement_strictement_superieur_au_chunk_est_rejete():
    # Sans ce cas, un `==` au lieu d'un `>=` passerait inapercu.
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            hf_token=TOKEN,
            chunk_length_s=10.0,
            chunk_overlap_s=20.0,
        )


def test_device_et_compute_type_alternatifs_sont_acceptes():
    # Le pendant des tests de rejet : les deux valeurs prevues par la spec
    # doivent bien etre acceptees, pas seulement celle par defaut.
    s = Settings(_env_file=None, hf_token=TOKEN, device="cpu", compute_type="float32")
    assert s.device == "cpu"
    assert s.compute_type == "float32"


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("chunk_length_s", 0.0),
        ("chunk_overlap_s", -1.0),
        ("turn_gap_s", -1.0),
        ("port", 0),
        ("port", 65536),
        ("max_upload_mb", 0),
    ],
)
def test_valeurs_hors_bornes_sont_rejetees(champ, valeur):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hf_token=TOKEN, **{champ: valeur})


def _ecrire_env(repertoire: Path) -> None:
    (repertoire / ".env").write_text(
        "HF_TOKEN=hf_depuis_le_fichier\nPORT=9001\n", encoding="utf-8"
    )


@pytest.fixture
def cache_get_settings_vide():
    """get_settings est memoisee : on vide son cache avant et apres, sinon
    l'instance construite ici fuirait vers les tests suivants."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_valeurs_lues_depuis_le_fichier_env(tmp_path, monkeypatch):
    # Seuls ce test et le suivant construisent Settings sans _env_file : ils
    # verifient justement que le .env par defaut est lu. L'isolation vient du
    # chdir, qui met le .env reel du depot hors de portee.
    _ecrire_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.hf_token == "hf_depuis_le_fichier"
    assert s.port == 9001


def test_get_settings_est_memoisee(tmp_path, monkeypatch, cache_get_settings_vide):
    _ecrire_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    premier = get_settings()
    assert premier.port == 9001
    assert get_settings() is premier
