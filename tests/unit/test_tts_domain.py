from pathlib import Path

import pytest

from transcription_server.tts.domain import (
    AudioFormat,
    SynthesisRequest,
    TtsMode,
)


def test_clone_refuse_des_instructions_non_prises_en_charge():
    with pytest.raises(ValueError, match="instructions"):
        SynthesisRequest(
            text="Bonjour",
            mode=TtsMode.CLONE,
            reference_path=Path("voice.wav"),
            reference_text="Bonjour",
            instructions="Parler doucement",
        )


def test_clone_exige_une_reference_audio():
    with pytest.raises(ValueError, match="référence"):
        SynthesisRequest(text="Bonjour", mode=TtsMode.CLONE)


def test_custom_voice_exige_un_identifiant_de_voix():
    with pytest.raises(ValueError, match="voix"):
        SynthesisRequest(text="Bonjour", mode=TtsMode.CUSTOM_VOICE)


def test_voice_design_exige_une_description():
    with pytest.raises(ValueError, match="description"):
        SynthesisRequest(text="Bonjour", mode=TtsMode.VOICE_DESIGN)


def test_les_six_formats_publics_sont_definis():
    assert {item.value for item in AudioFormat} == {
        "wav", "mp3", "flac", "opus", "aac", "pcm"
    }
