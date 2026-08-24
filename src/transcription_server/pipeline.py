"""Orchestration : decodage, diarization, transcription, alignement.

Seul module a connaitre l'ordre des operations. Il ne depend que des
Protocol des moteurs, jamais de leurs implementations concretes.
"""

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from transcription_server.alignment import group_into_turns
from transcription_server.asr.engine import AsrEngine
from transcription_server.audio import (
    SAMPLE_RATE,
    decode_channels,
    decode_to_pcm,
    duration_seconds,
)
from transcription_server.chunking import merge_windows, offset_words, plan_windows
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.domain import SpeakerSegment, Turn, Word
from transcription_server.runtime import empty_cache

ChannelMode = Literal["mix", "split"]

#: Ecart en deca duquel deux mots identiques provenant de canaux differents
#: sont tenus pour une seule et meme source. Un locuteur ne repete pas le meme
#: mot a moins de 200 ms d'intervalle ; en revanche la diaphonie entre deux
#: pistes du meme enregistrement est simultanee a quelques dizaines de ms pres.
TOLERANCE_DOUBLON_S = 0.2


@dataclass(frozen=True)
class TranscriptionRequest:
    language: str | None = None
    diarize: bool = True
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    #: `mix` replie tous les canaux en mono avant transcription — le defaut,
    #: et le bon choix pour un enregistrement ordinaire. `split` transcrit
    #: chaque canal separement puis fusionne par horodatage : utile quand les
    #: pistes portent des sources distinctes, ou le repliement superposerait
    #: deux paroles simultanees et rendrait les deux inintelligibles.
    channel_mode: ChannelMode = "mix"


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float
    speakers: list[str]
    turns: list[Turn]
    timing: dict[str, float] = field(default_factory=dict)
    #: Nombre de canaux reellement transcrits separement. Vaut 1 en mode `mix`.
    channels_used: int = 1


def _transcrire(
    pcm,
    asr: AsrEngine,
    language: str | None,
    chunk_length_s: float,
    chunk_overlap_s: float,
) -> list[Word]:
    """Transcrit un signal mono, en le decoupant si besoin."""
    windows = plan_windows(duration_seconds(pcm), chunk_length_s, chunk_overlap_s)
    per_window: list[list[Word]] = []
    for window in windows:
        begin = int(window.start * SAMPLE_RATE)
        finish = int(window.end * SAMPLE_RATE)
        local_words = asr.transcribe(pcm[begin:finish], language=language)
        # Recalage en temps absolu ici, avant merge_windows : c'est chunking
        # qui raisonne en temps relatif a la fenetre, personne d'autre.
        per_window.append(offset_words(local_words, window.start))
    return merge_windows(per_window, windows)


def _retirer_les_doublons(par_canal: list[list[Word]]) -> list[list[Word]]:
    """Retire d'un canal les mots qu'un canal precedent a deja rendus.

    Deux pistes peuvent porter le meme son — stereo classique, ou diaphonie
    entre micros d'une meme piece. Sans cette passe, chaque mot commun
    apparaitrait deux fois dans la transcription finale.

    Le critere est volontairement strict : meme texte ET debuts distants de
    moins de `TOLERANCE_DOUBLON_S`. Deux locuteurs prononcant le meme mot a
    quelques secondes d'intervalle sont conserves tous les deux, ce qui est le
    comportement voulu — c'est une conversation, pas un doublon.
    """
    if len(par_canal) < 2:
        return par_canal

    retenus: list[list[Word]] = [list(par_canal[0])]
    deja_vus = [(mot.text, mot.start) for mot in par_canal[0]]

    for mots in par_canal[1:]:
        garde = []
        for mot in mots:
            double = any(
                texte == mot.text and abs(debut - mot.start) <= TOLERANCE_DOUBLON_S
                for texte, debut in deja_vus
            )
            if not double:
                garde.append(mot)
        retenus.append(garde)
        deja_vus.extend((mot.text, mot.start) for mot in garde)
    return retenus


def _renumeroter(turns: list[Turn], decalage: int) -> list[Turn]:
    """Decale les etiquettes de locuteurs pour qu'elles restent uniques.

    Chaque canal est diarise independamment et repart donc de SPEAKER_00.
    Sans decalage, le premier locuteur du canal 1 se confondrait avec celui du
    canal 0 — deux personnes differentes sous une seule etiquette.
    """
    if decalage == 0:
        return turns
    renumerotes = []
    for tour in turns:
        if tour.speaker is None:
            renumerotes.append(tour)
            continue
        numero = int(tour.speaker.rsplit("_", 1)[-1]) + decalage
        renumerotes.append(replace(tour, speaker=f"SPEAKER_{numero:02d}"))
    return renumerotes


def run_pipeline(
    path: str | Path,
    asr: AsrEngine,
    diarization: DiarizationEngine,
    request: TranscriptionRequest,
    chunk_length_s: float,
    chunk_overlap_s: float,
    turn_gap_s: float,
) -> TranscriptionResult:
    """Transcrit un fichier et rend des tours de parole."""
    started = time.perf_counter()
    if request.channel_mode == "split":
        canaux = decode_channels(path)
    else:
        canaux = [decode_to_pcm(path)]
    duration = max(duration_seconds(pcm) for pcm in canaux)
    decode_elapsed = time.perf_counter() - started

    # Diarization avant l'ASR : cela permet de liberer le modele de
    # diarization avant l'inference longue si la VRAM se tend.
    segments_par_canal: list[list[SpeakerSegment]] = [[] for _ in canaux]
    diarization_elapsed = 0.0
    if request.diarize:
        started = time.perf_counter()
        for indice, pcm in enumerate(canaux):
            # Chaque canal est diarise separement. Diariser le tout d'un bloc
            # attribuerait un mot du canal 0 a un locuteur detecte sur le
            # canal 1, par simple recouvrement temporel.
            segments_par_canal[indice] = diarization.diarize(
                pcm,
                num_speakers=request.num_speakers,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers,
            )
            # L'allocateur de torch garde par devers lui la memoire liberee.
            # Sans cette restitution, la diarization du canal suivant s'empile
            # sur celle du precedent : mesure sur un enregistrement de 110 min
            # a deux canaux, 12,6 Go apres le premier, 21 Go apres le second.
            # Le cout est une synchronisation, negligeable en regard des
            # minutes que dure une diarization.
            empty_cache()
        diarization_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    mots_par_canal = [
        _transcrire(pcm, asr, request.language, chunk_length_s, chunk_overlap_s)
        for pcm in canaux
    ]
    mots_par_canal = _retirer_les_doublons(mots_par_canal)
    asr_elapsed = time.perf_counter() - started

    turns: list[Turn] = []
    locuteurs: list[str] = []
    decalage = 0
    for mots, segments in zip(mots_par_canal, segments_par_canal):
        tours_du_canal = _renumeroter(
            group_into_turns(mots, segments, turn_gap_s=turn_gap_s), decalage
        )
        turns.extend(tours_du_canal)
        # Les locuteurs viennent des SEGMENTS, pas des tours : un locuteur que
        # la diarization a detecte mais dont l'ASR n'a tire aucun mot doit
        # quand meme figurer. Son absence de la liste laisserait croire qu'il
        # n'a pas parle, alors qu'il a parle sans etre transcrit.
        noms_du_canal = sorted({s.speaker for s in segments})
        locuteurs.extend(
            f"SPEAKER_{int(nom.rsplit('_', 1)[-1]) + decalage:02d}"
            if decalage
            else nom
            for nom in noms_du_canal
        )
        decalage += len(noms_du_canal)

    # Les canaux sont transcrits independamment : leurs tours doivent etre
    # remis dans l'ordre chronologique pour que la lecture suive la reunion.
    turns.sort(key=lambda tour: (tour.start, tour.end))

    # Le travail est fini : ce que l'allocateur retient encore ne servira
    # plus a cette requete, et le retenir ferait echouer la suivante sur un
    # fichier a peine plus gros.
    empty_cache()

    return TranscriptionResult(
        text=" ".join(t.text for t in turns if t.text),
        language=request.language,
        duration=duration,
        speakers=sorted(set(locuteurs)),
        turns=turns,
        timing={
            "decode": round(decode_elapsed, 3),
            "asr": round(asr_elapsed, 3),
            "diarization": round(diarization_elapsed, 3),
        },
        channels_used=len(canaux),
    )
