"""Inference reelle du moteur NeMo. Lancer avec : pytest -m gpu"""

import numpy as np
import pytest

from tests.gpu.conftest import echantillon

# Les avertissements tiers ne font pas echouer ces tests-la.
#
# `filterwarnings = ["error"]` discipline NOTRE code : un avertissement emis
# par transcription_server doit rester un echec. Mais ces tests-ci chargent
# NeMo, torch et pyannote, qui en emettent plusieurs a chaque import — jit.script
# deprecie, allocateur CUDA deprecie, TF32 desactive, torchcodec absent. Exiger
# leur silence reviendrait a exiger qu'elles soient a jour, ce qui n'est pas en
# notre pouvoir et n'apprend rien sur le serveur.
#
# Filtrer chacun par son module s'est revele intenable : les noms de modules
# emetteurs ne sont pas ceux qu'on croit, et chaque execution de huit minutes
# en decouvrait un nouveau.
pytestmark = [pytest.mark.gpu, pytest.mark.filterwarnings("default")]


@pytest.fixture
def moteur(configuration_reelle):
    from transcription_server.asr.nemo_parakeet import load_nemo_engine
    from transcription_server.config import get_settings

    reglages = get_settings()
    return load_nemo_engine(
        model_name=reglages.asr_model,
        device="cuda",
        compute_type=reglages.compute_type,
    )


def test_cuda_est_disponible():
    from transcription_server.runtime import cuda_available, gpu_info

    assert cuda_available() is True
    info = gpu_info()
    assert info["vram_total_mb"] > 8000, info


def test_le_moteur_expose_le_nom_du_modele(moteur):
    from transcription_server.config import get_settings

    assert moteur.name == get_settings().asr_model


def test_du_silence_ne_fait_pas_echouer_le_moteur(moteur):
    """Test de cablage, pas de qualite : le moteur doit rendre une liste."""
    mots = moteur.transcribe(np.zeros(16000 * 2, dtype=np.float32), language=None)
    assert isinstance(mots, list)


def test_les_mots_sont_horodates_de_maniere_croissante(moteur):
    from transcription_server.audio import decode_to_pcm

    pcm = decode_to_pcm(echantillon("echantillon_fr.wav"))
    mots = moteur.transcribe(pcm, language="fr")
    assert len(mots) > 0
    assert all(mot.start <= mot.end for mot in mots)
    assert [mot.start for mot in mots] == sorted(mot.start for mot in mots)


def test_les_timestamps_restent_dans_la_duree_de_l_audio(moteur):
    """Un decalage d'unite (millisecondes prises pour des secondes) sortirait
    silencieusement de l'intervalle et decalerait tous les sous-titres."""
    from transcription_server.audio import decode_to_pcm, duration_seconds

    pcm = decode_to_pcm(echantillon("echantillon_fr.wav"))
    duree = duration_seconds(pcm)
    mots = moteur.transcribe(pcm, language="fr")
    assert max(mot.end for mot in mots) <= duree + 0.5
