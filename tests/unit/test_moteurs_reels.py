"""Logique pure des adaptateurs NeMo et pyannote.

Ces moteurs n'ont besoin ni de GPU ni de modele telecharge pour que leur
logique de conversion soit verifiee : leurs imports lourds sont paresseux, et
ce qui reste — extraction des mots, tri, transmission des options — est du
Python ordinaire. Les tests d'inference reelle vivent dans `tests/gpu/`.
"""

import sys
import types

import numpy as np
import pytest

from transcription_server.asr.nemo_parakeet import NemoParakeetEngine, _extract_words
from transcription_server.diarization.pyannote_engine import PyannoteEngine
from transcription_server.domain import SpeakerSegment


class _HypotheseNeMo:
    """Reproduit la forme de ce que rend `model.transcribe(..., timestamps=True)`."""

    def __init__(self, mots=None, texte=""):
        self.text = texte
        self.timestamp = {"word": mots} if mots is not None else None


# --- `name` doit rester une propriete ----------------------------------------
#
# Les tests de signature de test_engines.py couvrent `transcribe` et `diarize`,
# mais pas `name` : le transformer en methode leur echappe, et `/health` comme
# `/v1/models` afficheraient alors « <bound method ...> » au lieu du modele.


@pytest.mark.parametrize(
    ("classe", "instance"),
    [
        (NemoParakeetEngine, NemoParakeetEngine(None, "nvidia/parakeet", "cpu")),
        (PyannoteEngine, PyannoteEngine(None, "pyannote/x")),
    ],
)
def test_name_est_une_propriete_pas_une_methode(classe, instance):
    assert isinstance(getattr(classe, "name"), property)
    assert isinstance(instance.name, str)


# --- Extraction des mots NeMo -------------------------------------------------


def test_extract_words_convertit_les_timestamps():
    mots = _extract_words(
        _HypotheseNeMo([
            {"word": "bonjour", "start": 0.1, "end": 0.6},
            {"word": "tous", "start": 0.7, "end": 1.0},
        ])
    )
    assert [(m.text, m.start, m.end) for m in mots] == [
        ("bonjour", 0.1, 0.6),
        ("tous", 0.7, 1.0),
    ]


def test_extract_words_trie_une_sortie_desordonnee():
    """`merge_windows` et `group_into_turns` ne trient pas : une sortie NeMo
    desordonnee produirait silencieusement des tours dont la fin precede le
    debut. NeMo les rend normalement dans l'ordre, mais rien ne le garantit."""
    mots = _extract_words(
        _HypotheseNeMo([
            {"word": "trois", "start": 2.0, "end": 2.5},
            {"word": "un", "start": 0.0, "end": 0.5},
            {"word": "deux", "start": 1.0, "end": 1.5},
        ])
    )
    assert [m.text for m in mots] == ["un", "deux", "trois"]


def test_extract_words_ignore_les_mots_vides():
    mots = _extract_words(
        _HypotheseNeMo([
            {"word": "  ", "start": 0.0, "end": 0.1},
            {"word": "bonjour", "start": 0.2, "end": 0.6},
            {"word": "", "start": 0.7, "end": 0.8},
        ])
    )
    assert [m.text for m in mots] == ["bonjour"]


def test_extract_words_rogne_les_espaces():
    mots = _extract_words(_HypotheseNeMo([{"word": " bonjour ", "start": 0.0, "end": 0.5}]))
    assert mots[0].text == "bonjour"


def test_extract_words_sans_timestamps_rend_le_texte_en_un_bloc():
    mots = _extract_words(_HypotheseNeMo(mots=None, texte="bonjour tous"))
    assert len(mots) == 1
    assert mots[0].text == "bonjour tous"
    assert mots[0].start == 0.0 and mots[0].end == 0.0


def test_extract_words_sans_timestamps_ni_texte_rend_une_liste_vide():
    assert _extract_words(_HypotheseNeMo(mots=None, texte="")) == []


def test_transcribe_sur_audio_vide_ne_touche_pas_au_modele():
    """Court-circuit avant toute ecriture de fichier temporaire."""

    class ModeleQuiRefuse:
        def transcribe(self, *args, **kwargs):
            raise AssertionError("le modèle ne doit pas être appelé")

    moteur = NemoParakeetEngine(ModeleQuiRefuse(), "nvidia/parakeet", "cpu")
    assert moteur.transcribe(np.array([], dtype=np.float32), language=None) == []


# --- Diarization pyannote -----------------------------------------------------


class _FauxTenseur:
    def __init__(self, tableau):
        self.tableau = tableau

    def unsqueeze(self, _dim):
        return self


@pytest.fixture
def faux_torch(monkeypatch):
    """`diarize` importe torch pour construire le tenseur d'entree ; torch n'est
    pas installe dans le venv de developpement, on injecte donc un double."""
    module = types.ModuleType("torch")
    module.from_numpy = _FauxTenseur
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


class _FauxTour:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FausseAnnotation:
    def __init__(self, pistes):
        self._pistes = pistes

    def itertracks(self, yield_label=False):
        return iter(self._pistes)


class _FauxPipeline:
    """Enregistre les options recues et rend une annotation fixe."""

    def __init__(self, pistes):
        self._pistes = pistes
        self.options_recues = None

    def __call__(self, entree, **options):
        self.entree_recue = entree
        self.options_recues = options
        return _FausseAnnotation(self._pistes)


def test_diarize_trie_les_segments(faux_torch):
    """pyannote ne garantit pas l'ordre de `itertracks` ; l'aval le suppose."""
    pipeline = _FauxPipeline([
        (_FauxTour(5.0, 6.0), None, "SPEAKER_01"),
        (_FauxTour(0.0, 1.0), None, "SPEAKER_00"),
        (_FauxTour(2.0, 3.0), None, "SPEAKER_00"),
    ])
    segments = PyannoteEngine(pipeline, "pyannote/x").diarize(
        np.zeros(16000, dtype=np.float32),
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )
    assert segments == [
        SpeakerSegment("SPEAKER_00", 0.0, 1.0),
        SpeakerSegment("SPEAKER_00", 2.0, 3.0),
        SpeakerSegment("SPEAKER_01", 5.0, 6.0),
    ]


def test_diarize_sur_audio_vide_ne_touche_pas_au_pipeline(faux_torch):
    class PipelineQuiRefuse:
        def __call__(self, *args, **kwargs):
            raise AssertionError("le pipeline ne doit pas être appelé")

    moteur = PyannoteEngine(PipelineQuiRefuse(), "pyannote/x")
    assert (
        moteur.diarize(
            np.array([], dtype=np.float32),
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
        )
        == []
    )


def test_num_speakers_est_transmis_seul(faux_torch):
    pipeline = _FauxPipeline([])
    PyannoteEngine(pipeline, "pyannote/x").diarize(
        np.zeros(160, dtype=np.float32),
        num_speakers=2,
        min_speakers=None,
        max_speakers=None,
    )
    assert pipeline.options_recues == {"num_speakers": 2}


def test_num_speakers_prime_sur_les_bornes(faux_torch):
    """Combiner les deux est contradictoire ; la route rejette deja ce cas en
    400, mais le moteur ne doit pas produire un appel incoherent s'il est
    utilise directement."""
    pipeline = _FauxPipeline([])
    PyannoteEngine(pipeline, "pyannote/x").diarize(
        np.zeros(160, dtype=np.float32),
        num_speakers=2,
        min_speakers=1,
        max_speakers=8,
    )
    assert pipeline.options_recues == {"num_speakers": 2}


def test_les_bornes_sont_transmises_ensemble(faux_torch):
    pipeline = _FauxPipeline([])
    PyannoteEngine(pipeline, "pyannote/x").diarize(
        np.zeros(160, dtype=np.float32),
        num_speakers=None,
        min_speakers=2,
        max_speakers=5,
    )
    assert pipeline.options_recues == {"min_speakers": 2, "max_speakers": 5}


def test_aucune_option_quand_rien_n_est_demande(faux_torch):
    pipeline = _FauxPipeline([])
    PyannoteEngine(pipeline, "pyannote/x").diarize(
        np.zeros(160, dtype=np.float32),
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )
    assert pipeline.options_recues == {}


def test_le_taux_d_echantillonnage_transmis_est_celui_du_decodeur(faux_torch):
    from transcription_server.audio import SAMPLE_RATE

    pipeline = _FauxPipeline([])
    PyannoteEngine(pipeline, "pyannote/x").diarize(
        np.zeros(160, dtype=np.float32),
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )
    assert pipeline.entree_recue["sample_rate"] == SAMPLE_RATE
