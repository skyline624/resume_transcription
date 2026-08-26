"""Préparation prudente d'une référence de clonage vocal."""

import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from transcription_server.audio import SAMPLE_RATE, decode_to_pcm
from transcription_server.vad.engine import VadEngine


class InvalidReferenceError(ValueError):
    """La référence ne permet pas un clonage fiable."""


@dataclass(frozen=True)
class ReferenceLimits:
    min_duration_s: float
    max_duration_s: float
    min_dbfs: float
    max_clipped_ratio: float


@dataclass(frozen=True)
class NormalizedReference:
    path: Path
    duration_s: float
    peak: float
    rms_dbfs: float
    clipped_ratio: float


Normalizer = Callable[[Path, Path, float, float], None]


def prepare_reference(
    source: Path,
    output_directory: Path,
    vad: VadEngine,
    limits: ReferenceLimits,
    decoder: Callable[[Path], np.ndarray] = decode_to_pcm,
    normalizer: Normalizer | None = None,
) -> NormalizedReference:
    pcm = decoder(source)
    windows = vad.plan(pcm)
    if not windows:
        raise InvalidReferenceError(
            "La référence ne contient aucune parole détectable."
        )
    total_s = len(pcm) / SAMPLE_RATE
    start_s = max(0.0, float(windows[0][0]))
    end_s = min(total_s, float(windows[-1][1]))
    duration_s = end_s - start_s
    if not limits.min_duration_s <= duration_s <= limits.max_duration_s:
        raise InvalidReferenceError(
            f"La durée de parole utile doit être comprise entre "
            f"{limits.min_duration_s:g} et {limits.max_duration_s:g} secondes."
        )
    useful = pcm[int(start_s * SAMPLE_RATE) : int(end_s * SAMPLE_RATE)]
    if useful.size == 0:
        raise InvalidReferenceError("La référence ne contient aucune parole utile.")
    peak = float(np.max(np.abs(useful)))
    rms = float(np.sqrt(np.mean(np.square(useful, dtype=np.float64))))
    rms_dbfs = 20.0 * math.log10(max(rms, np.finfo(float).tiny))
    if rms_dbfs < limits.min_dbfs:
        raise InvalidReferenceError("Le niveau de la référence est trop faible.")
    clipped_ratio = float(np.mean(np.abs(useful) >= 0.999))
    if clipped_ratio > limits.max_clipped_ratio:
        raise InvalidReferenceError("La référence est trop fortement écrêtée.")

    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"{uuid4()}.wav"
    normalize = normalizer or _normalize_with_ffmpeg
    try:
        normalize(source, destination, start_s, end_s)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise InvalidReferenceError(
                "La normalisation de la référence n'a produit aucun audio."
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return NormalizedReference(
        path=destination,
        duration_s=duration_s,
        peak=peak,
        rms_dbfs=rms_dbfs,
        clipped_ratio=clipped_ratio,
    )


def _normalize_with_ffmpeg(
    source: Path, destination: Path, start_s: float, end_s: float
) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file",
        "-ss",
        f"{start_s:.6f}",
        "-to",
        f"{end_s:.6f}",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, stdin=subprocess.DEVNULL
        )
    except OSError as exc:
        raise InvalidReferenceError("ffmpeg est indisponible.") from exc
    if result.returncode != 0:
        raise InvalidReferenceError("La référence audio n'a pas pu être normalisée.")
