"""Fixtures partagees par toute la suite.

Les deux fixtures sont autouse : elles couvrent aussi les tests a venir qui
construiront un Settings ou appelleront get_settings().
"""

import pytest

from transcription_server.config import get_settings

# Les douze variables lues par Settings. Une seule d'entre elles laissee dans
# l'environnement rend un test dependant de la machine ; pour HF_TOKEN l'echec
# afficherait en plus le secret en clair dans la sortie pytest. Docker Compose
# injecte les entrees de son env_file comme variables d'environnement, et
# os.environ prime sur le dotenv : le cas n'a rien de theorique.
VARIABLES_DE_CONFIGURATION = (
    "HF_TOKEN",
    "ASR_MODEL",
    "DIARIZATION_MODEL",
    "ENABLE_DIARIZATION",
    "DEVICE",
    "COMPUTE_TYPE",
    "CHUNK_LENGTH_S",
    "CHUNK_OVERLAP_S",
    "TURN_GAP_S",
    "HOST",
    "PORT",
    "MAX_UPLOAD_MB",
)


@pytest.fixture(autouse=True)
def environnement_neutre(monkeypatch):
    """Retire toute variable de configuration ambiante avant chaque test.

    Un test qui veut en poser une le fait ensuite dans son corps : monkeypatch
    est le meme objet, donc son setenv s'applique apres cette purge.
    """
    for nom in VARIABLES_DE_CONFIGURATION:
        monkeypatch.delenv(nom, raising=False)


@pytest.fixture(autouse=True)
def cache_de_get_settings_vide():
    """get_settings est memoisee : sans vidange avant et apres, la premiere
    instance construite fuirait vers tous les tests suivants."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
