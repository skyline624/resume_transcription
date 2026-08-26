import json
from pathlib import Path
from uuid import UUID

import pytest

from transcription_server.tts.profiles import (
    VoiceNotFoundError,
    VoiceProfileRepository,
    VoiceStoreError,
)


@pytest.fixture
def reference_wav(tmp_path: Path) -> Path:
    path = tmp_path / "reference.wav"
    path.write_bytes(b"RIFFreference")
    return path


def test_creation_utilise_un_uuid_et_jamais_le_nom_affiche(tmp_path, reference_wav):
    root = tmp_path / "voices"
    profile = VoiceProfileRepository(root).create(
        name="../../ma voix",
        language="fr",
        transcript="bonjour",
        source_path=reference_wav,
        duration_s=4.2,
        transcript_source="provided",
    )

    assert str(UUID(profile.id)) == profile.id
    assert Path(profile.audio_path).parent == (root / "audio").resolve()
    assert "ma voix" not in Path(profile.audio_path).name
    assert profile.consent_at
    assert Path(profile.audio_path).read_bytes() == b"RIFFreference"


def test_suppression_retire_index_et_fichier(tmp_path, reference_wav):
    repo = VoiceProfileRepository(tmp_path / "voices")
    profile = repo.create(
        "Voix", "fr", "bonjour", reference_wav, 4.2, "provided"
    )

    repo.delete(profile.id)

    assert not Path(profile.audio_path).exists()
    with pytest.raises(VoiceNotFoundError):
        repo.get(profile.id)


def test_un_nouveau_depot_relit_les_metadonnees(tmp_path, reference_wav):
    root = tmp_path / "voices"
    created = VoiceProfileRepository(root).create(
        "Voix", "fr", "bonjour", reference_wav, 4.2, "parakeet"
    )

    loaded = VoiceProfileRepository(root).get(created.id)

    assert loaded == created
    assert loaded.transcript_source == "parakeet"


def test_un_chemin_hors_du_volume_est_refuse(tmp_path):
    root = tmp_path / "voices"
    root.mkdir()
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"secret")
    (root / "profiles.json").write_text(
        json.dumps([
            {
                "id": "d8b2ff96-a926-49b0-a735-8a6fa0e8a5ef",
                "name": "piège",
                "language": "fr",
                "transcript": "secret",
                "audio_path": str(outside),
                "duration_s": 4.0,
                "created_at": "2026-08-26T12:00:00+00:00",
                "consent_at": "2026-08-26T12:00:00+00:00",
                "transcript_source": "provided",
            }
        ]),
        encoding="utf-8",
    )

    with pytest.raises(VoiceStoreError, match="volume"):
        VoiceProfileRepository(root).list()


def test_index_json_tronque_est_signale_sans_etre_ecrase(tmp_path):
    root = tmp_path / "voices"
    root.mkdir()
    index = root / "profiles.json"
    index.write_text("[{", encoding="utf-8")

    with pytest.raises(VoiceStoreError, match="registre"):
        VoiceProfileRepository(root).list()
    assert index.read_text(encoding="utf-8") == "[{"
