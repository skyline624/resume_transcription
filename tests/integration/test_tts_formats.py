import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from transcription_server.tts.audio_output import render_output
from transcription_server.tts.domain import AudioFormat


def make_wav() -> bytes:
    import io
    import math
    import struct

    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        samples = [
            struct.pack("<h", round(5000 * math.sin(2 * math.pi * 440 * i / 24000)))
            for i in range(2400)
        ]
        handle.writeframes(b"".join(samples))
    return output.getvalue()


@pytest.mark.parametrize("audio_format", list(AudioFormat))
def test_chaque_sortie_reelle_est_redecodable(tmp_path: Path, audio_format):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg absent")
    rendered = render_output([make_wav()], [0], 1.0, audio_format)
    suffix = "ogg" if audio_format is AudioFormat.OPUS else audio_format.value
    path = tmp_path / f"speech.{suffix}"
    path.write_bytes(rendered.data)
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if audio_format is AudioFormat.PCM:
        command.extend(["-f", "s16le", "-ar", "24000", "-ac", "1"])
    command.extend(["-i", str(path), "-f", "null", "-"])
    result = subprocess.run(command, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
