"""Contrats TTS légers, importables sans torch ni Qwen."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class TtsMode(StrEnum):
    CUSTOM_VOICE = "qwen3-tts-custom-voice"
    CLONE = "qwen3-tts-clone"
    VOICE_DESIGN = "qwen3-tts-voice-design"


class AudioFormat(StrEnum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OPUS = "opus"
    AAC = "aac"
    PCM = "pcm"


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    mode: TtsMode
    language: str = "fr"
    voice: str | None = None
    instructions: str | None = None
    reference_path: Path | None = None
    reference_text: str | None = None

    def __post_init__(self) -> None:
        if self.mode is TtsMode.CLONE and self.instructions:
            raise ValueError("instructions n'est pas pris en charge en mode clone.")
        if self.mode is TtsMode.CLONE and self.reference_path is None:
            raise ValueError("Une référence audio est requise en mode clone.")
        if self.mode is TtsMode.CUSTOM_VOICE and not self.voice:
            raise ValueError("Une voix prédéfinie est requise.")
        if self.mode is TtsMode.VOICE_DESIGN and not self.instructions:
            raise ValueError("Une description de voix est requise.")


@dataclass(frozen=True)
class SynthesisResult:
    audio_wav: bytes
    sample_rate: int
    model: str
    load_ms: float = 0.0
    inference_ms: float = 0.0


@dataclass(frozen=True)
class WorkerHealth:
    available: bool
    state: str
    downloaded_models: tuple[str, ...] = ()
    loaded_model: str | None = None
    precision: str | None = None
    device: str | None = None
    attention: str | None = None
    speakers: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    last_error: str | None = None
    pid: int | None = None
    vram_allocated_mib: float | None = None
    details: dict[str, object] = field(default_factory=dict)


class TtsError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TtsUnavailableError(TtsError):
    """Le worker ou son modèle ne peut pas servir la requête."""
