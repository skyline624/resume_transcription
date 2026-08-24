"""Configuration du serveur, lue depuis l'environnement et le fichier .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    hf_token: str | None = None

    asr_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    enable_diarization: bool = True

    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: Literal["float16", "float32"] = "float16"

    chunk_length_s: float = Field(default=480.0, gt=0)
    chunk_overlap_s: float = Field(default=15.0, ge=0)
    turn_gap_s: float = Field(default=1.0, ge=0)

    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)
    max_upload_mb: int = Field(default=1024, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @model_validator(mode="after")
    def _check_coherence(self) -> "Settings":
        if self.chunk_overlap_s >= self.chunk_length_s:
            raise ValueError(
                "CHUNK_OVERLAP_S doit être strictement inférieur à CHUNK_LENGTH_S."
            )
        if self.enable_diarization and not self.hf_token:
            raise ValueError(
                "ENABLE_DIARIZATION=true exige un HF_TOKEN. Créez un token de "
                "type read sur huggingface.co et acceptez les conditions de "
                "https://huggingface.co/pyannote/speaker-diarization-community-1 "
                "ou mettez ENABLE_DIARIZATION=false."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
