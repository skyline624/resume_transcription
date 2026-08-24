"""Inference reelle du moteur de diarization. Lancer avec : pytest -m gpu"""

import numpy as np
import pytest

from tests.gpu.conftest import echantillon

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def moteur():
    from transcription_server.config import get_settings
    from transcription_server.diarization.pyannote_engine import load_pyannote_engine

    reglages = get_settings()
    if not reglages.hf_token:
        pytest.skip("HF_TOKEN absent")
    # Le token n'est pas extrait dans une variable locale : Field(repr=False)
    # protege l'objet, pas ses copies, et --showlocals reexposerait la valeur.
    return load_pyannote_engine(
        model_name=reglages.diarization_model,
        hf_token=reglages.hf_token,
        device="cuda",
    )


def test_le_moteur_expose_le_nom_du_modele(moteur):
    from transcription_server.config import get_settings

    assert moteur.name == get_settings().diarization_model


def test_du_silence_ne_produit_aucun_segment(moteur):
    segments = moteur.diarize(
        np.zeros(16000 * 5, dtype=np.float32),
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )
    assert segments == []


def test_deux_voix_donnent_deux_locuteurs(moteur):
    from transcription_server.audio import decode_to_pcm

    pcm = decode_to_pcm(echantillon("deux_voix.wav"))
    segments = moteur.diarize(
        pcm, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert len({segment.speaker for segment in segments}) == 2
    assert all(segment.start < segment.end for segment in segments)
    assert segments == sorted(segments, key=lambda segment: segment.start)


def test_num_speakers_force_le_nombre(moteur):
    from transcription_server.audio import decode_to_pcm

    pcm = decode_to_pcm(echantillon("deux_voix.wav"))
    segments = moteur.diarize(
        pcm, num_speakers=1, min_speakers=None, max_speakers=None
    )
    assert len({segment.speaker for segment in segments}) == 1


def test_les_etiquettes_suivent_la_forme_attendue(moteur):
    """L'aval affiche ces etiquettes telles quelles dans le format dialogue."""
    import re

    from transcription_server.audio import decode_to_pcm

    pcm = decode_to_pcm(echantillon("deux_voix.wav"))
    segments = moteur.diarize(
        pcm, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert all(re.fullmatch(r"SPEAKER_\d{2}", s.speaker) for s in segments)
