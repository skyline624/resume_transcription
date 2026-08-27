"""Fixtures partagees par toute la suite.

Les deux fixtures sont autouse. Settings lit deux sources distinctes, et une
seule est neutralisee ici -- s'y fier pour l'autre serait une erreur :

- `os.environ` : neutralise par `environnement_neutre`.
- le fichier `.env`, via `env_file=".env"` dans `Settings.model_config` : **non
  neutralise**. Le chemin est relatif au CWD et aucune fixture ne fait de
  `chdir`. Un test lance depuis la racine du depot qui construit `Settings()`
  ou appelle `get_settings()` sans `_env_file=None` ni
  `monkeypatch.chdir(tmp_path)` chargera donc le vrai token du `.env`. Le champ
  `hf_token` porte `repr=False`, ce qui l'empeche de ressortir dans la sortie
  de pytest, mais il est bel et bien dans l'objet.

Pour se soustraire a la purge d'environnement -- par exemple pour exporter
DEVICE ou PORT avant de demarrer l'application -- il faut poser les variables
*apres* `environnement_neutre` : dans le corps du test, ou depuis une fixture
fonction-scope qui declare `environnement_neutre` dans ses arguments. Une
fixture de portee module ou session serait instanciee avant, donc silencieusement
effacee ; l'echec se presenterait comme un « field required » trompeur.
"""

import os

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
    "ENABLE_TTS",
    "TTS_WORKER_SOCKET",
    "TTS_CUSTOM_VOICE_MODEL",
    "TTS_CLONE_MODEL",
    "TTS_VOICE_DESIGN_MODEL",
    "TTS_DEFAULT_LANGUAGE",
    "TTS_PRECISION",
    "TTS_IDLE_UNLOAD_S",
    "TTS_LOAD_TIMEOUT_S",
    "TTS_GENERATION_TIMEOUT_S",
    "TTS_MAX_INPUT_CHARS",
    "TTS_REFERENCE_MIN_S",
    "TTS_REFERENCE_MAX_S",
    "TTS_REFERENCE_MIN_DBFS",
    "TTS_REFERENCE_MAX_CLIPPED_RATIO",
    "VOICE_STORE_PATH",
    "WEB_DIST_PATH",
)


@pytest.fixture(autouse=True)
def environnement_neutre(monkeypatch):
    """Retire toute variable de configuration ambiante avant chaque test.

    Un test qui veut en poser une le fait ensuite dans son corps : monkeypatch
    est le meme objet, donc son setenv s'applique apres cette purge.
    """
    for nom in VARIABLES_DE_CONFIGURATION:
        monkeypatch.delenv(nom, raising=False)
        # `case_sensitive=False` fait minusculer par pydantic-settings toutes
        # les cles de os.environ : sous POSIX, ou la casse est significative,
        # un `hf_token=` exporte en minuscules serait lu malgre la purge de la
        # forme majuscule. Sous Windows os.environ normalise deja, ce second
        # delenv y est simplement sans effet.
        monkeypatch.delenv(nom.lower(), raising=False)


# Photographie de l'environnement prise a l'import de ce module, donc AVANT
# que la moindre purge n'ait lieu. C'est la seule facon de rendre a un test la
# configuration reelle du conteneur : une fois `environnement_neutre` passee,
# l'information est perdue.
_ENVIRONNEMENT_INITIAL = {
    nom: os.environ[nom]
    for nom in VARIABLES_DE_CONFIGURATION
    if nom in os.environ
}


@pytest.fixture
def configuration_reelle(monkeypatch, environnement_neutre):
    """Rend a un test la configuration ambiante, HF_TOKEN compris.

    Reservee aux tests qui parlent aux vrais modeles : eux ont besoin du token
    et du peripherique reels, ce que la purge leur retire precisement.

    Declarer `environnement_neutre` en argument n'est pas decoratif : c'est ce
    qui garantit que la restauration a lieu APRES la purge. Une fixture de
    portee module ou session serait instanciee avant, donc silencieusement
    effacee — et l'echec se presenterait comme un « field required » trompeur,
    exactement ce qui est arrive aux premiers tests GPU.
    """
    for nom, valeur in _ENVIRONNEMENT_INITIAL.items():
        monkeypatch.setenv(nom, valeur)
    get_settings.cache_clear()
    return _ENVIRONNEMENT_INITIAL


@pytest.fixture(autouse=True)
def cache_de_get_settings_vide():
    """get_settings est memoisee : sans vidange avant et apres, la premiere
    instance construite fuirait vers tous les tests suivants."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
