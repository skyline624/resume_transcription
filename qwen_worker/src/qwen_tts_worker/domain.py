"""Types internes du worker, sans import Qwen ou torch."""

from dataclasses import dataclass, field
from enum import StrEnum


class Mode(StrEnum):
    CUSTOM = "custom"
    CLONE = "clone"
    DESIGN = "design"


@dataclass(frozen=True)
class GenerateCommand:
    mode: Mode
    text: str = field(repr=False)
    language: str = "French"
    speaker: str | None = None
    instruct: str | None = field(default=None, repr=False)
    reference_audio: str | None = field(default=None, repr=False)
    reference_text: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.mode is Mode.CUSTOM and not self.speaker:
            raise ValueError("speaker est requis pour CustomVoice.")
        if self.mode is Mode.DESIGN and not self.instruct:
            raise ValueError("instruct est requis pour VoiceDesign.")
        if self.mode is Mode.CLONE and (
            not self.reference_audio or not self.reference_text
        ):
            raise ValueError("La référence audio et son texte sont requis.")


class WorkerModelError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
