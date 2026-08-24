"""Orchestration : decodage, diarization, transcription, alignement.

Seul module a connaitre l'ordre des operations. Il ne depend que des
Protocol des moteurs, jamais de leurs implementations concretes.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from transcription_server.alignment import group_into_turns
from transcription_server.asr.engine import AsrEngine
from transcription_server.audio import SAMPLE_RATE, decode_to_pcm, duration_seconds
from transcription_server.chunking import merge_windows, offset_words, plan_windows
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.domain import SpeakerSegment, Turn, Word


@dataclass(frozen=True)
class TranscriptionRequest:
    language: str | None = None
    diarize: bool = True
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float
    speakers: list[str]
    turns: list[Turn]
    timing: dict[str, float] = field(default_factory=dict)


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
    pcm = decode_to_pcm(path)
    duration = duration_seconds(pcm)
    decode_elapsed = time.perf_counter() - started

    # Diarization avant l'ASR : cela permet de liberer le modele de
    # diarization avant l'inference longue si la VRAM se tend.
    segments: list[SpeakerSegment] = []
    diarization_elapsed = 0.0
    if request.diarize:
        started = time.perf_counter()
        segments = diarization.diarize(
            pcm,
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )
        diarization_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    windows = plan_windows(duration, chunk_length_s, chunk_overlap_s)
    per_window: list[list[Word]] = []
    for window in windows:
        begin = int(window.start * SAMPLE_RATE)
        finish = int(window.end * SAMPLE_RATE)
        local_words = asr.transcribe(pcm[begin:finish], language=request.language)
        # Recalage en temps absolu ici, avant merge_windows : c'est chunking
        # qui raisonne en temps relatif a la fenetre, personne d'autre.
        per_window.append(offset_words(local_words, window.start))
    words = merge_windows(per_window, windows)
    asr_elapsed = time.perf_counter() - started

    turns = group_into_turns(words, segments, turn_gap_s=turn_gap_s)
    speakers = sorted({s.speaker for s in segments})

    return TranscriptionResult(
        text=" ".join(t.text for t in turns if t.text),
        language=request.language,
        duration=duration,
        speakers=speakers,
        turns=turns,
        timing={
            "decode": round(decode_elapsed, 3),
            "asr": round(asr_elapsed, 3),
            "diarization": round(diarization_elapsed, 3),
        },
    )
