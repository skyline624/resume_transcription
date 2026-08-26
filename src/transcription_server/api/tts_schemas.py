"""Schémas de l'API OpenAI de synthèse vocale."""

from pydantic import BaseModel, Field, model_validator

from transcription_server.tts.domain import AudioFormat, TtsMode

_ALIASES = {
    "tts-1": TtsMode.CUSTOM_VOICE,
    "tts-1-hd": TtsMode.CUSTOM_VOICE,
    "gpt-4o-mini-tts": TtsMode.CUSTOM_VOICE,
}


def resolve_model_alias(value: str) -> TtsMode:
    if value in _ALIASES:
        return _ALIASES[value]
    try:
        return TtsMode(value)
    except ValueError as exc:
        raise ValueError(f"Modèle TTS inconnu : {value}") from exc


class SpeechRequest(BaseModel):
    model: str = TtsMode.CUSTOM_VOICE.value
    input: str = Field(min_length=1, max_length=4096)
    voice: str | None = None
    instructions: str | None = None
    response_format: AudioFormat = AudioFormat.MP3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str = "fr"

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "SpeechRequest":
        mode = resolve_model_alias(self.model)
        if mode in (TtsMode.CUSTOM_VOICE, TtsMode.CLONE) and not self.voice:
            raise ValueError("voice est requis pour ce mode.")
        if mode is TtsMode.VOICE_DESIGN and not self.instructions:
            raise ValueError("instructions est requis pour VoiceDesign.")
        if mode is TtsMode.CLONE and self.instructions:
            raise ValueError("instructions n'est pas pris en charge pour clone.")
        return self

    @property
    def mode(self) -> TtsMode:
        return resolve_model_alias(self.model)
