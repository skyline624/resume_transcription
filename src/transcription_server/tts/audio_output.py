"""Assemblage et encodage final des segments produits par Qwen."""

import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from transcription_server.tts.domain import AudioFormat


class AudioRenderError(RuntimeError):
    """ffmpeg n'a pas pu produire la réponse demandée."""


@dataclass(frozen=True)
class EncodedAudio:
    data: bytes
    media_type: str


Runner = Callable[[list[str]], None]

_OUTPUTS = {
    AudioFormat.WAV: ("wav", "audio/wav", ["-c:a", "pcm_s16le"]),
    AudioFormat.MP3: ("mp3", "audio/mpeg", ["-c:a", "libmp3lame", "-b:a", "192k"]),
    AudioFormat.FLAC: ("flac", "audio/flac", ["-c:a", "flac"]),
    AudioFormat.OPUS: ("ogg", "audio/ogg", ["-c:a", "libopus", "-b:a", "128k"]),
    AudioFormat.AAC: ("aac", "audio/aac", ["-c:a", "aac", "-b:a", "192k"]),
    AudioFormat.PCM: ("pcm", "audio/pcm", ["-f", "s16le", "-c:a", "pcm_s16le"]),
}


def render_output(
    chunks: list[bytes],
    pauses_ms: list[int],
    speed: float,
    output_format: AudioFormat,
    runner: Runner | None = None,
) -> EncodedAudio:
    if not chunks or any(not chunk for chunk in chunks):
        raise ValueError("Au moins un segment audio non vide est requis.")
    if len(chunks) != len(pauses_ms):
        raise ValueError("Le nombre de pauses doit correspondre aux segments.")
    if not 0.25 <= speed <= 4.0:
        raise ValueError("La vitesse doit être comprise entre 0.25 et 4.0.")
    if any(pause < 0 for pause in pauses_ms):
        raise ValueError("Les pauses ne peuvent pas être négatives.")
    execute = runner or _run_ffmpeg
    extension, media_type, codec_args = _OUTPUTS[output_format]

    with tempfile.TemporaryDirectory(prefix="tts-render-") as directory:
        root = Path(directory)
        concat_entries: list[Path] = []
        for index, (chunk, pause_ms) in enumerate(zip(chunks, pauses_ms, strict=True)):
            raw = root / f"raw-{index}.wav"
            prepared = root / f"prepared-{index}.wav"
            raw.write_bytes(chunk)
            execute([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", str(raw), "-af",
                "aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono,"
                "afade=t=in:st=0:d=0.005,areverse,afade=t=in:st=0:d=0.005,areverse",
                "-c:a", "pcm_s16le", "-y", str(prepared),
            ])
            concat_entries.append(prepared)
            if pause_ms:
                silence = root / f"silence-{index}.wav"
                execute([
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-t", f"{pause_ms / 1000:.3f}", "-c:a", "pcm_s16le",
                    "-y", str(silence),
                ])
                concat_entries.append(silence)

        concat_file = root / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{_concat_path(path)}'\n" for path in concat_entries),
            encoding="utf-8",
        )
        output = root / f"output.{extension}"
        filters = [
            *_atempo_filters(speed),
            "loudnorm=I=-16:TP=-1.0:LRA=11",
            "aresample=24000",
        ]
        execute([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-af", ",".join(filters), *codec_args, "-y", str(output),
        ])
        try:
            data = output.read_bytes()
        except OSError as exc:
            raise AudioRenderError("Le rendu audio final est introuvable.") from exc
        if not data:
            raise AudioRenderError("Le rendu audio final est vide.")
        return EncodedAudio(data=data, media_type=media_type)


def _atempo_filters(speed: float) -> list[str]:
    factors: list[float] = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    if abs(remaining - 1.0) > 1e-9 or not factors:
        factors.append(remaining)
    return [f"atempo={factor:g}" for factor in factors if abs(factor - 1.0) > 1e-9]


def _concat_path(path: Path) -> str:
    return path.as_posix().replace("'", "'\\''")


def _run_ffmpeg(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, stdin=subprocess.DEVNULL
        )
    except OSError as exc:
        raise AudioRenderError("ffmpeg est indisponible.") from exc
    if result.returncode != 0:
        raise AudioRenderError("ffmpeg n'a pas pu encoder la synthèse vocale.")
