import inspect

import numpy as np
import pytest

from transcription_server.asr.engine import AsrEngine, StubAsrEngine
from transcription_server.diarization.engine import (
    DiarizationEngine,
    NullDiarizationEngine,
    StubDiarizationEngine,
)
from transcription_server.domain import SpeakerSegment, Word

AUDIO = np.zeros(16000, dtype=np.float32)


def test_stub_asr_respecte_le_protocol():
    engine = StubAsrEngine([Word("bonjour", 0.0, 0.5)])
    assert isinstance(engine, AsrEngine)


def test_stub_asr_rend_les_mots_fournis():
    words = [Word("bonjour", 0.0, 0.5), Word("tous", 0.6, 1.0)]
    assert StubAsrEngine(words).transcribe(AUDIO, language=None) == words


def test_stub_diarization_respecte_le_protocol():
    engine = StubDiarizationEngine([SpeakerSegment("SPEAKER_00", 0.0, 1.0)])
    assert isinstance(engine, DiarizationEngine)


def test_stub_diarization_rend_les_segments_fournis():
    segs = [SpeakerSegment("SPEAKER_00", 0.0, 1.0)]
    out = StubDiarizationEngine(segs).diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert out == segs


def test_null_diarization_rend_une_liste_vide():
    out = NullDiarizationEngine().diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert out == []


def test_null_diarization_respecte_le_protocol():
    assert isinstance(NullDiarizationEngine(), DiarizationEngine)


def test_les_moteurs_exposent_un_nom():
    assert StubAsrEngine([]).name == "stub-asr"
    assert NullDiarizationEngine().name == "none"


# --- Ce que `isinstance` contre un Protocol ne verifie pas -------------------
#
# Un `Protocol` `runtime_checkable` ne compare que les *noms* d'attributs : ni
# les signatures, ni les types. Les tests ci-dessus prouvent donc qu'un moteur
# expose bien `name` et `transcribe`/`diarize`, rien de plus. Les tests qui
# suivent verrouillent ce qu'`isinstance` laisse passer et que les Tasks 12 et
# 13 devront respecter avec NeMo et pyannote.


def _parametres(fonction) -> list[tuple[str, object]]:
    """Noms et natures (positionnel, nomme...) des parametres, `self` exclu.

    Les annotations ne sont volontairement pas comparees : ce qui fait
    echouer un appel, c'est un nom ou une nature de parametre qui derive.
    """
    return [
        (p.name, p.kind)
        for p in inspect.signature(fonction).parameters.values()
        if p.name != "self"
    ]


# Parametres a une seule entree pour l'ASR : la Task 12 y ajoutera le moteur
# NeMo, la Task 13 le moteur pyannote dans la liste d'en dessous.
@pytest.mark.parametrize("implementation", [StubAsrEngine])
def test_la_signature_de_transcribe_suit_le_protocol(implementation):
    assert _parametres(implementation.transcribe) == _parametres(AsrEngine.transcribe)


@pytest.mark.parametrize(
    "implementation", [NullDiarizationEngine, StubDiarizationEngine]
)
def test_la_signature_de_diarize_suit_le_protocol(implementation):
    assert _parametres(implementation.diarize) == _parametres(DiarizationEngine.diarize)


def test_isinstance_ne_controle_pas_les_signatures():
    """Garde-fou documentaire : un moteur inutilisable passe quand meme
    `isinstance`. C'est la raison d'etre des deux tests de signature."""

    class MoteurMalForme:
        name = 42  # meme pas une chaine

        def transcribe(self):  # ni audio, ni language
            raise AssertionError("jamais appelable par une route")

    assert isinstance(MoteurMalForme(), AsrEngine)
    assert _parametres(MoteurMalForme.transcribe) != _parametres(AsrEngine.transcribe)


# --- Isolation des donnees rendues par les moteurs factices ------------------
#
# Un stub qui partage sa liste interne se laisse corrompre par son appelant :
# une route qui trie ou tronque la liste recue changerait la reponse de tous
# les appels suivants du meme moteur -- et de tous les tests qui le partagent.


def test_stub_asr_ne_rend_pas_sa_liste_interne():
    engine = StubAsrEngine([Word("bonjour", 0.0, 0.5)])
    engine.transcribe(AUDIO, language=None).clear()
    assert engine.transcribe(AUDIO, language=None) == [Word("bonjour", 0.0, 0.5)]


def test_stub_asr_copie_les_mots_a_la_construction():
    words = [Word("bonjour", 0.0, 0.5)]
    engine = StubAsrEngine(words)
    words.append(Word("tard", 9.0, 9.5))
    assert engine.transcribe(AUDIO, language=None) == [Word("bonjour", 0.0, 0.5)]


def test_stub_diarization_ne_rend_pas_sa_liste_interne():
    engine = StubDiarizationEngine([SpeakerSegment("SPEAKER_00", 0.0, 1.0)])
    engine.diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    ).clear()
    assert engine.diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    ) == [SpeakerSegment("SPEAKER_00", 0.0, 1.0)]


def test_stub_diarization_copie_les_segments_a_la_construction():
    segs = [SpeakerSegment("SPEAKER_00", 0.0, 1.0)]
    engine = StubDiarizationEngine(segs)
    segs.append(SpeakerSegment("SPEAKER_01", 2.0, 3.0))
    assert engine.diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    ) == [SpeakerSegment("SPEAKER_00", 0.0, 1.0)]


def test_null_diarization_rend_une_liste_neuve_a_chaque_appel():
    engine = NullDiarizationEngine()
    premier_appel = engine.diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    )
    premier_appel.append(SpeakerSegment("SPEAKER_00", 0.0, 1.0))
    assert engine.diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    ) == []


# --- Le nom expose par /health et /v1/models --------------------------------


def test_stub_diarization_a_un_nom_par_defaut():
    assert StubDiarizationEngine([]).name == "stub-diarization"


def test_les_stubs_honorent_le_nom_fourni():
    assert StubAsrEngine([], name="parakeet-factice").name == "parakeet-factice"
    assert StubDiarizationEngine([], name="pyannote-factice").name == "pyannote-factice"
