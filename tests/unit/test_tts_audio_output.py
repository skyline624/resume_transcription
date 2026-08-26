from pathlib import Path

import pytest

from transcription_server.tts.audio_output import render_output
from transcription_server.tts.domain import AudioFormat


class FakeRunner:
    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> None:
        self.commands.append(command)
        destination = Path(command[-1])
        if destination.suffix in {".wav", ".mp3", ".flac", ".ogg", ".aac", ".pcm"}:
            destination.write_bytes(b"encoded")


@pytest.mark.parametrize(
    ("format", "media_type"),
    [
        (AudioFormat.WAV, "audio/wav"),
        (AudioFormat.MP3, "audio/mpeg"),
        (AudioFormat.FLAC, "audio/flac"),
        (AudioFormat.OPUS, "audio/ogg"),
        (AudioFormat.AAC, "audio/aac"),
        (AudioFormat.PCM, "audio/pcm"),
    ],
)
def test_chaque_format_a_son_type_mime(format, media_type):
    result = render_output([b"RIFFchunk"], [0], 1.0, format, runner=FakeRunner())
    assert result.data == b"encoded"
    assert result.media_type == media_type


def test_vitesse_et_loudness_sont_appliques_au_rendu_final():
    runner = FakeRunner()
    render_output([b"RIFFchunk"], [350], 1.25, AudioFormat.WAV, runner=runner)
    final = runner.commands[-1]
    filter_graph = final[final.index("-af") + 1]
    assert "atempo=1.25" in filter_graph
    assert "loudnorm=I=-16:TP=-1.0:LRA=11" in filter_graph


def test_chaque_segment_recoit_un_fondu_aux_deux_extremites():
    runner = FakeRunner()
    render_output([b"a", b"b"], [100, 0], 1.0, AudioFormat.WAV, runner=runner)
    preparation_filters = [
        command[command.index("-af") + 1]
        for command in runner.commands
        if "-af" in command and "loudnorm" not in command[command.index("-af") + 1]
    ]
    assert len(preparation_filters) == 2
    assert all("afade=t=in" in graph and "areverse" in graph for graph in preparation_filters)


def test_nombre_de_pauses_doit_correspondre_aux_segments():
    with pytest.raises(ValueError, match="pauses"):
        render_output([b"a", b"b"], [0], 1.0, AudioFormat.WAV, runner=FakeRunner())


def test_vitesse_hors_bornes_est_refusee():
    with pytest.raises(ValueError, match="vitesse"):
        render_output([b"a"], [0], 4.1, AudioFormat.WAV, runner=FakeRunner())
