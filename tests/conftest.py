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
        # `case_sensitive=False` fait minusculer par pydantic-settings toutes
        # les cles de os.environ : sous POSIX, ou la casse est significative,
        # un `hf_token=` exporte en minuscules serait lu malgre la purge de la
        # forme majuscule. Sous Windows os.environ normalise deja, ce second
        # delenv y est simplement sans effet.
        monkeypatch.delenv(nom.lower(), raising=False)


@pytest.fixture(autouse=True)
def cache_de_get_settings_vide():
    """get_settings est memoisee : sans vidange avant et apres, la premiere
    instance construite fuirait vers tous les tests suivants."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
