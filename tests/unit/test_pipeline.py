import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from transcription_server.asr.engine import StubAsrEngine
from transcription_server.diarization.engine import (
    NullDiarizationEngine,
    StubDiarizationEngine,
)
from transcription_server.domain import SpeakerSegment, Word
from transcription_server.pipeline import (
    TranscriptionRequest,
    run_pipeline,
)

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"


def _write_silence_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack("<h", 0) * int(seconds * rate))


@pytest.fixture
def short_wav(tmp_path):
    path = tmp_path / "court.wav"
    _write_silence_wav(path, seconds=3.0)
    return path


def test_pipeline_produit_du_texte(short_wav):
    asr = StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("tous", 0.6, 1.0)])
    result = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.text == "bonjour tous"
    assert result.duration == pytest.approx(3.0, rel=0.05)
    assert result.speakers == []


def test_pipeline_avec_diarization_rend_les_locuteurs(short_wav):
    asr = StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("merci", 2.0, 2.5)])
    diar = StubDiarizationEngine(
        [SpeakerSegment(S0, 0.0, 1.0), SpeakerSegment(S1, 1.5, 3.0)]
    )
    result = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=diar,
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.speakers == [S0, S1]
    assert [t.speaker for t in result.turns] == [S0, S1]


def test_diarize_false_ignore_le_moteur_de_diarization(short_wav):
    diar = StubDiarizationEngine([SpeakerSegment(S0, 0.0, 3.0)])
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.5)]),
        diarization=diar,
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.speakers == []
    assert result.turns[0].speaker is None


def test_les_speakers_sont_tries_et_dedupliques(short_wav):
    asr = StubAsrEngine(
        [Word("a", 0.0, 0.4), Word("b", 1.6, 2.0), Word("c", 2.2, 2.6)]
    )
    diar = StubDiarizationEngine(
        [
            SpeakerSegment(S1, 1.5, 2.1),
            SpeakerSegment(S0, 0.0, 1.0),
            SpeakerSegment(S1, 2.2, 3.0),
        ]
    )
    result = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=diar,
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.speakers == [S0, S1]


def test_le_timing_est_renseigne(short_wav):
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.5)]),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert set(result.timing) == {"decode", "asr", "diarization"}
    assert all(v >= 0.0 for v in result.timing.values())


def test_audio_long_appelle_l_asr_une_fois_par_fenetre(tmp_path):
    """Avec chunk_length_s=1.0 sur 3 s d'audio, l'ASR est appele plusieurs fois
    et les timestamps sont reoffsetes en absolu."""
    path = tmp_path / "long.wav"
    _write_silence_wav(path, seconds=3.0)

    calls: list[int] = []

    class CountingAsr:
        name = "counting"

        def transcribe(self, audio, language):
            calls.append(len(audio))
            # un mot au tout debut de chaque fenetre
            return [Word("mot", 0.1, 0.2)]

    result = run_pipeline(
        path=path,
        asr=CountingAsr(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=1.0,
        chunk_overlap_s=0.2,
        turn_gap_s=10.0,
    )
    assert len(calls) >= 3
    starts = [w.start for t in result.turns for w in t.words]
    assert starts == sorted(starts)
    assert max(starts) > 1.0  # les timestamps sont bien absolus


def test_fichier_invalide_remonte_l_erreur(tmp_path):
    from transcription_server.audio import AudioDecodeError

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"pas de l'audio")
    with pytest.raises(AudioDecodeError):
        run_pipeline(
            path=junk,
            asr=StubAsrEngine([]),
            diarization=NullDiarizationEngine(),
            request=TranscriptionRequest(diarize=False),
            chunk_length_s=480.0,
            chunk_overlap_s=15.0,
            turn_gap_s=1.0,
        )


# --- Tests de durcissement -------------------------------------------------
# Les sept tests ci-dessus laissaient survivre dix mutants du pipeline. Ceux
# qui suivent les tuent : ils verifient que chaque argument recu est bel et
# bien transmis au bon moteur, et que le recollage des fenetres a lieu.


def test_la_langue_demandee_est_transmise_au_moteur(short_wav):
    """request.language part vers l'ASR et revient dans le resultat."""
    recues: list[str | None] = []

    class AsrQuiNoteLaLangue:
        name = "langue"

        def transcribe(self, audio, language):
            recues.append(language)
            return [Word("bonjour", 0.0, 0.5)]

    result = run_pipeline(
        path=short_wav,
        asr=AsrQuiNoteLaLangue(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(language="fr", diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert recues == ["fr"]
    assert result.language == "fr"


def test_les_contraintes_de_locuteurs_sont_transmises(short_wav):
    """Les trois bornes de la requete arrivent a la diarization, chacune a sa
    place -- valeurs distinctes a dessein, une permutation serait visible."""
    recu: dict[str, int | None] = {}

    class DiarQuiNoteSesArguments:
        name = "notaire"

        def diarize(self, audio, num_speakers, min_speakers, max_speakers):
            recu.update(
                taille=len(audio),
                num=num_speakers,
                mini=min_speakers,
                maxi=max_speakers,
            )
            return [SpeakerSegment(S0, 0.0, 3.0)]

    run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.5)]),
        diarization=DiarQuiNoteSesArguments(),
        request=TranscriptionRequest(
            diarize=True, num_speakers=2, min_speakers=1, max_speakers=5
        ),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert recu["num"] == 2
    assert recu["mini"] == 1
    assert recu["maxi"] == 5
    # La diarization voit tout le fichier, jamais une fenetre.
    assert recu["taille"] == pytest.approx(3.0 * 16000, abs=200)


def test_chaque_fenetre_recoit_sa_tranche_de_pcm(tmp_path):
    """Le moteur recoit la tranche de sa fenetre, pas le PCM entier.

    3 s decoupees en fenetres de 1 s recouvrantes de 0.2 s donnent des bornes
    a 0.0 / 0.8 / 1.6 / 2.4 s, soit 16000 echantillons par fenetre pleine et
    9600 pour la derniere, tronquee par la fin du fichier.
    """
    path = tmp_path / "long.wav"
    _write_silence_wav(path, seconds=3.0)

    tailles: list[int] = []

    class AsrQuiMesureSaTranche:
        name = "mesure"

        def transcribe(self, audio, language):
            tailles.append(len(audio))
            return []

    run_pipeline(
        path=path,
        asr=AsrQuiMesureSaTranche(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=1.0,
        chunk_overlap_s=0.2,
        turn_gap_s=1.0,
    )
    assert tailles == [16000, 16000, 16000, 9600]
    # Le recouvrement en echantillons est exactement celui demande : les
    # tranches se chevauchent de 0.2 s et ne laissent aucun trou, donc leur
    # somme moins les trois recouvrements redonne la duree du fichier.
    assert sum(tailles) - 3 * int(0.2 * 16000) == 3 * 16000


def test_les_mots_sont_recales_en_temps_absolu(tmp_path):
    """Chaque fenetre rend un mot a 0.30 s de SON debut ; en sortie les
    horodatages doivent etre absolus depuis le debut du fichier."""
    path = tmp_path / "long.wav"
    _write_silence_wav(path, seconds=3.0)

    class AsrRelatif:
        name = "relatif"

        def transcribe(self, audio, language):
            return [Word("mot", 0.30, 0.40)]

    result = run_pipeline(
        path=path,
        asr=AsrRelatif(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=1.0,
        chunk_overlap_s=0.2,
        turn_gap_s=10.0,
    )
    starts = [round(w.start, 3) for t in result.turns for w in t.words]
    # Debuts de fenetre 0.0 / 0.8 / 1.6 / 2.4, plus 0.30 s.
    assert starts == [0.30, 1.10, 1.90, 2.70]
    assert result.text == "mot mot mot mot"


def test_un_mot_vu_par_deux_fenetres_ne_sort_qu_une_fois(tmp_path):
    """La zone de recouvrement est transcrite deux fois : merge_windows doit
    trancher, sinon le mot sort en double."""
    path = tmp_path / "long.wav"
    _write_silence_wav(path, seconds=3.0)

    class AsrDuRecouvrement:
        name = "recouvrement"

        def __init__(self) -> None:
            self.appel = -1

        def transcribe(self, audio, language):
            self.appel += 1
            # Les fenetres 0 ([0.0, 1.0]) et 1 ([0.8, 1.8]) transcrivent toutes
            # deux le meme mot, situe en absolu a [0.92, 0.98].
            if self.appel == 0:
                return [Word("commun", 0.92, 0.98)]
            if self.appel == 1:
                return [Word("commun", 0.12, 0.18)]
            return []

    result = run_pipeline(
        path=path,
        asr=AsrDuRecouvrement(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=1.0,
        chunk_overlap_s=0.2,
        turn_gap_s=10.0,
    )
    mots = [w for t in result.turns for w in t.words]
    assert [w.text for w in mots] == ["commun"]
    assert mots[0].start == pytest.approx(0.92)
    assert result.text == "commun"


@pytest.mark.parametrize("gap, tours_attendus", [(5.0, 1), (0.5, 2)])
def test_turn_gap_s_pilote_le_decoupage_en_tours(short_wav, gap, tours_attendus):
    """1.6 s separent les deux mots : le seuil recu decide de la coupure."""
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.4), Word("deux", 2.0, 2.4)]),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=gap,
    )
    assert len(result.turns) == tours_attendus
    # Les tours sont recolles par une espace : le texte reste le meme,
    # que la coupure ait lieu ou non.
    assert result.text == "un deux"


def test_un_locuteur_sans_aucun_mot_figure_quand_meme(short_wav):
    """speakers vient de la diarization et non des tours : un locuteur qui n'a
    recouvert aucun mot reste un locuteur detecte. Les quatre libelles sont
    fournis dans le desordre, la sortie doit etre triee."""
    diar = StubDiarizationEngine(
        [
            SpeakerSegment("SPEAKER_02", 2.0, 2.5),
            SpeakerSegment(S0, 0.0, 1.0),
            SpeakerSegment("SPEAKER_03", 2.6, 3.0),
            SpeakerSegment(S1, 1.2, 1.8),
        ]
    )
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.4)]),
        diarization=diar,
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.speakers == [S0, S1, "SPEAKER_02", "SPEAKER_03"]
    assert [t.speaker for t in result.turns] == [S0]


def test_le_texte_ignore_les_tours_sans_texte(short_wav):
    """Un moteur peut rendre un jeton vide ; le texte final n'y gagne pas une
    espace parasite."""
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.4), Word("", 2.0, 2.1)]),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert len(result.turns) == 2
    assert result.text == "bonjour"


def test_la_diarization_s_execute_avant_l_asr(short_wav):
    """Ordre delibere : le modele de diarization peut ainsi etre libere de la
    VRAM avant l'inference longue de l'ASR."""
    ordre: list[str] = []

    class DiarTracee:
        name = "diar-tracee"

        def diarize(self, audio, num_speakers, min_speakers, max_speakers):
            ordre.append("diarization")
            return []

    class AsrTrace:
        name = "asr-trace"

        def transcribe(self, audio, language):
            ordre.append("asr")
            return [Word("un", 0.0, 0.4)]

    run_pipeline(
        path=short_wav,
        asr=AsrTrace(),
        diarization=DiarTracee(),
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert ordre == ["diarization", "asr"]


def test_le_timing_de_diarization_est_nul_quand_elle_est_sautee(short_wav):
    """diarize=False n'appelle pas le moteur, donc ne facture aucun temps."""
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.5)]),
        diarization=StubDiarizationEngine([SpeakerSegment(S0, 0.0, 3.0)]),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.timing["diarization"] == 0.0


def test_le_pcm_transmis_est_un_tableau_mono_float32(short_wav):
    """Le moteur recoit ce que audio.decode_to_pcm promet, sans conversion."""
    vus: list[np.ndarray] = []

    class AsrQuiInspecte:
        name = "inspecteur"

        def transcribe(self, audio, language):
            vus.append(audio)
            return []

    run_pipeline(
        path=short_wav,
        asr=AsrQuiInspecte(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert len(vus) == 1
    assert vus[0].ndim == 1
    assert vus[0].dtype == np.float32


def test_un_audio_sans_aucun_mot_rend_un_resultat_vide_mais_valide(short_wav):
    """Silence complet : l'ASR ne rend rien. Le resultat reste exploitable --
    pas de tour, pas de texte -- et les locuteurs detectes restent annonces."""
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([]),
        diarization=StubDiarizationEngine([SpeakerSegment(S0, 0.0, 3.0)]),
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.text == ""
    assert result.turns == []
    assert result.speakers == [S0]
    assert result.duration == pytest.approx(3.0, rel=0.05)
