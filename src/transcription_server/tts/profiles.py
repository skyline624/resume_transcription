"""Registre local et atomique des références vocales consenties."""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4


class VoiceStoreError(RuntimeError):
    """Le registre est illisible ou contient une donnée dangereuse."""


class VoiceNotFoundError(VoiceStoreError):
    """L'identifiant de voix n'existe pas."""


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    language: str
    transcript: str
    audio_path: str
    duration_s: float
    created_at: str
    consent_at: str
    transcript_source: Literal["provided", "parakeet"]


class VoiceProfileRepository:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._audio_root = self._root / "audio"
        self._index = self._root / "profiles.json"
        self._lock = threading.RLock()
        self._audio_root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        name: str,
        language: str,
        transcript: str,
        source_path: Path,
        duration_s: float,
        transcript_source: Literal["provided", "parakeet"] = "provided",
        consent_at: str | None = None,
    ) -> VoiceProfile:
        source = source_path.resolve(strict=True)
        identifier = str(uuid4())
        target = (self._audio_root / f"{identifier}.wav").resolve()
        self._ensure_audio_path(target)
        temporary = target.with_suffix(".wav.tmp")
        now = datetime.now(UTC).isoformat()
        profile = VoiceProfile(
            id=identifier,
            name=name.strip(),
            language=language,
            transcript=transcript.strip(),
            audio_path=str(target),
            duration_s=float(duration_s),
            created_at=now,
            consent_at=consent_at or now,
            transcript_source=transcript_source,
        )
        with self._lock:
            try:
                shutil.copyfile(source, temporary)
                os.replace(temporary, target)
                profiles = self._load()
                profiles.append(profile)
                self._write_atomic(profiles)
            except BaseException:
                temporary.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                raise
        return profile

    def get(self, voice_id: str) -> VoiceProfile:
        self._validate_uuid(voice_id)
        with self._lock:
            for profile in self._load():
                if profile.id == voice_id:
                    return profile
        raise VoiceNotFoundError(f"Voix inconnue : {voice_id}")

    def list(self) -> list[VoiceProfile]:
        with self._lock:
            return sorted(self._load(), key=lambda item: (item.created_at, item.id))

    def delete(self, voice_id: str) -> None:
        self._validate_uuid(voice_id)
        with self._lock:
            profiles = self._load()
            selected = next((item for item in profiles if item.id == voice_id), None)
            if selected is None:
                raise VoiceNotFoundError(f"Voix inconnue : {voice_id}")
            remaining = [item for item in profiles if item.id != voice_id]
            self._write_atomic(remaining)
            Path(selected.audio_path).unlink(missing_ok=True)

    def _load(self) -> list[VoiceProfile]:
        if not self._index.exists():
            return []
        try:
            raw = json.loads(self._index.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise TypeError("la racine JSON n'est pas une liste")
            profiles = [VoiceProfile(**item) for item in raw]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VoiceStoreError("Le registre des voix est illisible.") from exc
        for profile in profiles:
            self._validate_uuid(profile.id)
            self._ensure_audio_path(Path(profile.audio_path).resolve())
        return profiles

    def _write_atomic(self, profiles: list[VoiceProfile]) -> None:
        temporary = self._root / "profiles.json.tmp"
        try:
            temporary.write_text(
                json.dumps(
                    [asdict(item) for item in profiles],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self._index)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_audio_path(self, path: Path) -> None:
        if not path.is_relative_to(self._audio_root):
            raise VoiceStoreError("Le fichier vocal sort du volume autorisé.")

    @staticmethod
    def _validate_uuid(value: str) -> None:
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError) as exc:
            raise VoiceNotFoundError("Identifiant de voix invalide.") from exc
        if str(parsed) != value:
            raise VoiceNotFoundError("Identifiant de voix invalide.")
