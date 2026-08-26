"""Configuration du serveur, lue depuis l'environnement et le fichier .env."""

from functools import lru_cache
from pathlib import Path
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

    # repr=False masque le token dans repr(), str() et donc dans la sortie de
    # pytest : sans lui, une assertion qui echoue sur un Settings construit avec
    # le vrai token l'imprimerait en clair. Ne couvre ni model_dump() ni
    # ValidationError.errors(), qui exposent le token integral -- pour ceux-la,
    # utiliser errors(include_input=False) et ne jamais serialiser le modele.
    hf_token: str | None = Field(default=None, repr=False)

    asr_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    enable_diarization: bool = True

    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: Literal["float16", "float32"] = "float16"

    chunk_length_s: float = Field(default=480.0, gt=0)
    chunk_overlap_s: float = Field(default=15.0, ge=0)
    enable_vad: bool = True
    vad_device: Literal["cpu", "cuda"] = "cpu"
    vad_max_segment_s: float = Field(default=5.0, gt=0)
    vad_fallback_overlap_s: float = Field(default=1.0, ge=0)
    turn_gap_s: float = Field(default=1.0, ge=0)

    # --- Compte-rendu ---
    #
    # Le modele n'a pas de valeur imposee par le code : il depend de ce que
    # l'installation d'Ollama contient. Un modele « :cloud » fait transiter la
    # conversation par les serveurs d'Ollama — a ne poser qu'en connaissance
    # de cause.
    summary_model: str = "qwen3.8-27b-64k:latest"
    # Depuis le conteneur, l'hote se joint par ce nom : ni 172.17.0.1 ni
    # gateway.docker.internal ne repondent sous Docker Desktop.
    ollama_base_url: str = "http://host.docker.internal:11434"
    enable_summary: bool = True
    # Un modele de 27B redigeant un compte-rendu de reunion de deux heures
    # depasse largement les delais HTTP habituels.
    summary_timeout_s: float = Field(default=900.0, gt=0)

    # --- Synthèse vocale Qwen3-TTS ---
    enable_tts: bool = True
    tts_worker_socket: Path = Path("/run/qwen-tts/worker.sock")
    tts_custom_voice_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_clone_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    tts_voice_design_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    tts_default_language: str = "fr"
    tts_precision: Literal["bfloat16"] = "bfloat16"
    tts_idle_unload_s: float = Field(default=300.0, ge=0)
    tts_load_timeout_s: float = Field(default=300.0, gt=0)
    tts_generation_timeout_s: float = Field(default=900.0, gt=0)
    tts_max_input_chars: int = Field(default=4096, gt=0, le=4096)
    tts_reference_min_s: float = Field(default=3.0, gt=0)
    tts_reference_max_s: float = Field(default=30.0, gt=0)
    tts_reference_min_dbfs: float = Field(default=-60.0, lt=0)
    tts_reference_max_clipped_ratio: float = Field(default=0.01, ge=0, le=1)
    voice_store_path: Path = Path("/app/voices")

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
        if self.vad_fallback_overlap_s >= self.vad_max_segment_s:
            raise ValueError(
                "VAD_FALLBACK_OVERLAP_S doit être strictement inférieur à "
                "VAD_MAX_SEGMENT_S."
            )
        if self.tts_reference_min_s >= self.tts_reference_max_s:
            raise ValueError(
                "TTS_REFERENCE_MIN_S doit être strictement inférieur à "
                "TTS_REFERENCE_MAX_S."
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
