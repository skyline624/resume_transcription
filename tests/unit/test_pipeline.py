import shutil
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


def test_vad_limite_chaque_inference_et_preserve_les_offsets(short_wav):
    """Une regression vers la fenetre globale rappellerait l'ASR une seule fois
    et remettrait tous les mots au debut de l'enregistrement.
    """
    appels = []

    class VadDeuxPassages:
        name = "deux-passages"

        def plan(self, audio):
            return [
                # Fenetres disjointes : le silence intermediaire n'est pas
                # envoye au modele, comme avec Silero.
                (0.5, 1.0),
                (2.0, 2.5),
            ]

    class AsrLocal:
        name = "local"

        def transcribe(self, audio, language):
            appels.append(len(audio))
            return [Word(f"mot{len(appels)}", 0.0, 0.1)]

    result = run_pipeline(
        path=short_wav,
        asr=AsrLocal(),
        diarization=NullDiarizationEngine(),
        vad=VadDeuxPassages(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )

    assert appels == [8000, 8000]
    mots = [mot for tour in result.turns for mot in tour.words]
    assert [(mot.text, mot.start) for mot in mots] == [
        ("mot1", pytest.approx(0.5)),
        ("mot2", pytest.approx(2.0)),
    ]


def test_echec_du_vad_retombe_sur_des_fenetres_courtes(short_wav):
    """Une panne du VAD ne doit jamais restaurer la fenetre de huit minutes
    qui a provoque la mauvaise detection de langue.
    """
    appels = []

    class VadQuiExplose:
        name = "explose"

        def plan(self, audio):
            raise RuntimeError("modele VAD indisponible")

    class AsrQuiCompte:
        name = "compte"

        def transcribe(self, audio, language):
            appels.append(len(audio))
            return []

    run_pipeline(
        path=short_wav,
        asr=AsrQuiCompte(),
        diarization=NullDiarizationEngine(),
        vad=VadQuiExplose(),
        vad_fallback_length_s=1.0,
        vad_fallback_overlap_s=0.2,
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )

    assert appels == [16000, 16000, 16000, 9600]


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
    # Tranche = vue numpy et non copie : aucune duplication memoire par
    # fenetre, ce qui compte sur une heure d'audio (230 Mo de float32).
    assert vus[0].base is not None
    assert not vus[0].flags.owndata


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


# --- Correctifs de la revue -------------------------------------------------
# La revue a montre qu'un mutant « bonne longueur, mauvais decalage »
# (begin = 0) passait les vingt tests precedents : toutes les fixtures etaient
# du silence numerique, donc une tranche etait indiscernable d'une autre. Les
# tests ci-dessous ferment ce trou et verrouillent les contrats restes nus.


def _write_ramp_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    """WAV dont l'echantillon i vaut i % 30000 : chacun porte sa position.

    Le silence rend toutes les tranches identiques, donc invisible tout defaut
    de decalage. La rampe est le materiau minimal qui les distingue. Le modulo
    reste sous 32768 : pas de saturation en int16, et pas de normalisation par
    decode_to_pcm, dont le pic vaut 29999/32768 < 1.
    """
    total = int(seconds * rate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"".join(struct.pack("<h", i % 30000) for i in range(total)))


def test_chaque_fenetre_recoit_la_tranche_au_bon_decalage(tmp_path):
    """Le contenu recu identifie la position de la tranche, pas sa seule taille.

    Fenetres a 0.0 / 0.8 / 1.6 / 2.4 s, soit les echantillons 0 / 12800 /
    25600 / 38400 ; la rampe vaut i % 30000, d'ou 38400 -> 8400.
    """
    path = tmp_path / "rampe.wav"
    _write_ramp_wav(path, seconds=3.0)

    premiers: list[int] = []
    derniers: list[int] = []

    class AsrQuiLitSesBornes:
        name = "bornes"

        def transcribe(self, audio, language):
            premiers.append(int(round(float(audio[0]) * 32768)))
            derniers.append(int(round(float(audio[-1]) * 32768)))
            return []

    run_pipeline(
        path=path,
        asr=AsrQuiLitSesBornes(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=1.0,
        chunk_overlap_s=0.2,
        turn_gap_s=1.0,
    )
    assert premiers == [0, 12800, 25600, 8400]
    # La derniere fenetre atteint bien le dernier echantillon du fichier
    # (47999 % 30000), donc rien n'est perdu en fin de parcours.
    assert derniers[-1] == 17999


def test_le_timing_ventile_les_phases(short_wav):
    """decode, asr et diarization sont mesures separement, chacun sur sa phase.

    Les moteurs dorment de facon inegale : le timing doit refleter cet ecart,
    ce qu'un dict de zeros ne pourrait pas montrer.
    """
    import time

    class AsrLent:
        name = "asr-lent"

        def transcribe(self, audio, language):
            time.sleep(0.08)
            return [Word("un", 0.0, 0.4)]

    class DiarRapide:
        name = "diar-rapide"

        def diarize(self, audio, num_speakers, min_speakers, max_speakers):
            time.sleep(0.02)
            return [SpeakerSegment(S0, 0.0, 3.0)]

    result = run_pipeline(
        path=short_wav,
        asr=AsrLent(),
        diarization=DiarRapide(),
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    # Marges larges : on verrouille la ventilation, pas la precision d'horloge.
    assert result.timing["asr"] >= 0.06
    assert result.timing["diarization"] >= 0.01
    assert result.timing["asr"] > result.timing["diarization"]
    # ffmpeg est un sous-processus : son cout ne peut pas etre nul.
    assert result.timing["decode"] > 0.0


def test_les_defauts_de_la_requete_sont_le_contrat():
    """Les Tasks 9, 10, 12 et 13 consomment ces defauts ; ils sont verrouilles.

    diarize=True bascule a False passerait inapercu partout ailleurs : chaque
    appel de ce fichier fournit le drapeau explicitement.
    """
    assert TranscriptionRequest() == TranscriptionRequest(
        language=None,
        diarize=True,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )
    defauts = TranscriptionRequest()
    assert defauts.language is None
    assert defauts.diarize is True
    assert defauts.num_speakers is None
    assert defauts.min_speakers is None
    assert defauts.max_speakers is None


def test_la_requete_et_le_resultat_sont_immuables(short_wav):
    """frozen=True fait partie du contrat des deux dataclasses."""
    import dataclasses

    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.5)]),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.text = "autre chose"
    with pytest.raises(dataclasses.FrozenInstanceError):
        TranscriptionRequest().diarize = False


def test_des_mots_desordonnes_ressortent_tries(short_wav):
    """group_into_turns exige des mots tries et devient silencieusement faux
    sinon. Le pipeline ne trie pas lui-meme : il herite du tri final de
    merge_windows. Cette dependance implicite est verrouillee ici."""
    asr = StubAsrEngine(
        [Word("trois", 2.0, 2.4), Word("un", 0.0, 0.4), Word("deux", 1.0, 1.4)]
    )
    result = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=10.0,
    )
    assert result.text == "un deux trois"
    mots = [w for t in result.turns for w in t.words]
    assert [w.start for w in mots] == sorted(w.start for w in mots)
    # Sur une entree desordonnee, un tour sortirait avec end < start.
    assert all(t.end >= t.start for t in result.turns)


def test_un_jeton_vide_au_milieu_d_un_tour_ne_double_pas_l_espace(short_wav):
    """Cas complementaire du jeton vide formant un tour a lui seul : ici les
    trois mots tiennent dans un seul tour."""
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine(
            [Word("bonjour", 0.0, 0.4), Word("", 0.5, 0.6), Word("tous", 0.7, 0.9)]
        ),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert len(result.turns) == 1
    assert result.text == "bonjour tous"
    # Le jeton vide reste dans words : seul le rendu texte l'ecarte.
    assert len(result.turns[0].words) == 3


# --- Mode `split` : un canal par source ---------------------------------------
#
# Motivation reelle : un enregistrement de reunion peut porter le micro local
# sur une piste et les participants distants sur l'autre. Replie en mono, deux
# paroles simultanees se superposent et le modele rend une bouillie ou les deux
# sont perdues. Transcrire chaque canal separement les preserve toutes deux.


def _ecrire_wav_stereo(chemin, gauche, droit, rate=16000):
    """Ecrit un wav stereo 16 bits a partir de deux signaux float32."""
    import numpy as np

    entrelace = np.empty(len(gauche) * 2, dtype=np.int16)
    entrelace[0::2] = (np.clip(gauche, -1, 1) * 32767).astype(np.int16)
    entrelace[1::2] = (np.clip(droit, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(chemin), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(entrelace.tobytes())


class _AsrSelonLeCanal:
    """Rend des mots differents selon le niveau du signal recu.

    C'est ce qui permet de verifier que chaque canal est bien transcrit
    separement : un pipeline qui replierait tout en mono ne verrait qu'un seul
    des deux signaux.
    """

    name = "selon-canal"

    def __init__(self):
        self.appels = []

    def transcribe(self, audio, language):
        import numpy as np

        crete = float(np.abs(audio).max()) if audio.size else 0.0
        self.appels.append(round(crete, 2))
        if crete > 0.7:
            return [Word("fort", 0.0, 0.5)]
        if crete > 0.2:
            return [Word("faible", 1.0, 1.5)]
        return []


@pytest.fixture
def wav_stereo(tmp_path):
    import numpy as np

    n = 16000
    t = np.arange(n) / 16000
    gauche = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    droit = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    chemin = tmp_path / "stereo.wav"
    _ecrire_wav_stereo(chemin, gauche, droit)
    return chemin


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_split_transcrit_chaque_canal(wav_stereo):
    asr = _AsrSelonLeCanal()
    resultat = run_pipeline(
        path=wav_stereo,
        asr=asr,
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False, channel_mode="split"),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert resultat.channels_used == 2
    assert len(asr.appels) == 2, "un appel par canal attendu"
    assert "fort" in resultat.text and "faible" in resultat.text


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_mix_reste_le_defaut_et_ne_voit_qu_un_signal(wav_stereo):
    asr = _AsrSelonLeCanal()
    resultat = run_pipeline(
        path=wav_stereo,
        asr=asr,
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert resultat.channels_used == 1
    assert len(asr.appels) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_split_normalise_globalement_pas_par_canal(tmp_path):
    """Un canal quasi muet doit le rester.

    Normalise sur son propre pic, son bruit de fond monterait au niveau de la
    parole et le modele hallucinerait des mots dessus. C'est le cas courant
    du micro de celui qui parle peu.
    """
    import numpy as np

    n = 16000
    t = np.arange(n) / 16000
    fort = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    presque_muet = (0.001 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    chemin = tmp_path / "desequilibre.wav"
    _ecrire_wav_stereo(chemin, fort, presque_muet)

    asr = _AsrSelonLeCanal()
    run_pipeline(
        path=chemin,
        asr=asr,
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False, channel_mode="split"),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert asr.appels[0] > 0.7, "le canal fort doit rester fort"
    assert asr.appels[1] < 0.2, "le canal muet ne doit pas etre amplifie"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_split_supprime_les_doublons_entre_canaux(tmp_path):
    """Deux pistes portant le meme son ne doivent pas doubler la transcription."""
    import numpy as np

    n = 16000
    t = np.arange(n) / 16000
    identique = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    chemin = tmp_path / "identique.wav"
    _ecrire_wav_stereo(chemin, identique, identique)

    class AsrConstant:
        name = "constant"

        def transcribe(self, audio, language):
            return [Word("bonjour", 0.0, 0.5)]

    resultat = run_pipeline(
        path=chemin,
        asr=AsrConstant(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False, channel_mode="split"),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert resultat.text == "bonjour", f"doublon non supprime : {resultat.text!r}"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_split_renumerote_les_locuteurs_entre_canaux(wav_stereo):
    """Chaque canal repart de SPEAKER_00 : sans decalage, le premier locuteur
    du canal 1 se confondrait avec celui du canal 0."""

    class DiarConstante:
        name = "constante"

        def diarize(self, audio, num_speakers, min_speakers, max_speakers):
            return [SpeakerSegment("SPEAKER_00", 0.0, 2.0)]

    class AsrConstant:
        name = "constant"

        def transcribe(self, audio, language):
            return [Word("mot", 0.1, 0.4)]

    resultat = run_pipeline(
        path=wav_stereo,
        asr=AsrConstant(),
        diarization=DiarConstante(),
        request=TranscriptionRequest(diarize=True, channel_mode="split"),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert resultat.speakers == ["SPEAKER_00", "SPEAKER_01"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_split_sur_un_fichier_mono_se_comporte_comme_mix(short_wav):
    asr = _AsrSelonLeCanal()
    resultat = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False, channel_mode="split"),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert resultat.channels_used == 1
    assert len(asr.appels) == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_split_rend_les_tours_dans_l_ordre_chronologique(tmp_path):
    """Sans tri, tous les tours du canal 1 suivraient ceux du canal 0 et la
    lecture ne suivrait plus la conversation : on lirait une personne en
    entier, puis l'autre en entier."""
    import numpy as np

    n = 16000 * 4
    t = np.arange(n) / 16000
    gauche = (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    droit = (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    chemin = tmp_path / "alterne.wav"
    _ecrire_wav_stereo(chemin, gauche, droit)

    class AsrAlternant:
        """Le canal fort parle tot et tard, le canal faible au milieu."""

        name = "alternant"

        def transcribe(self, audio, language):
            import numpy as np

            crete = float(np.abs(audio).max()) if audio.size else 0.0
            if crete > 0.7:
                return [Word("debut", 0.0, 0.5), Word("fin", 3.0, 3.5)]
            return [Word("milieu", 1.5, 2.0)]

    resultat = run_pipeline(
        path=chemin,
        asr=AsrAlternant(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False, channel_mode="split"),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=0.5,
    )
    debuts = [tour.start for tour in resultat.turns]
    assert debuts == sorted(debuts), f"tours non chronologiques : {debuts}"
    assert resultat.text == "debut milieu fin", (
        f"ordre de lecture incorrect : {resultat.text!r}"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH")
def test_la_vram_est_rendue_entre_les_canaux_et_a_la_fin(wav_stereo, monkeypatch):
    """L'allocateur de torch garde la memoire liberee ; sans restitution, la
    diarization du canal suivant s'empile sur celle du precedent.

    Mesure sur un enregistrement reel de 110 min a deux canaux : 12,6 Go apres
    le premier canal, 21 Go apres le second, et 21 Go encore une fois la
    requete terminee. La requete suivante, sur un fichier a peine plus gros,
    aurait sature les 24 Go.
    """
    from transcription_server import pipeline as pipeline_module

    appels = []
    monkeypatch.setattr(
        pipeline_module, "empty_cache", lambda: appels.append(1)
    )

    class DiarConstante:
        name = "constante"

        def diarize(self, audio, num_speakers, min_speakers, max_speakers):
            return [SpeakerSegment("SPEAKER_00", 0.0, 1.0)]

    run_pipeline(
        path=wav_stereo,
        asr=StubAsrEngine([Word("mot", 0.1, 0.4)]),
        diarization=DiarConstante(),
        request=TranscriptionRequest(diarize=True, channel_mode="split"),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    # Deux canaux diarises, plus la restitution finale.
    assert len(appels) == 3, f"restitutions attendues : 3, obtenues : {len(appels)}"


def test_la_vram_est_rendue_meme_sans_diarization(short_wav, monkeypatch):
    from transcription_server import pipeline as pipeline_module

    appels = []
    monkeypatch.setattr(
        pipeline_module, "empty_cache", lambda: appels.append(1)
    )
    run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("mot", 0.1, 0.4)]),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert appels == [1], "la restitution finale doit avoir lieu sans diarization"
