from pathlib import Path

import numpy as np
import pytest

from transcription_server.tts.reference import (
    InvalidReferenceError,
    ReferenceLimits,
    prepare_reference,
)


class FakeVad:
    def __init__(self, windows):
        self.windows = windows

    def plan(self, audio):
        return list(self.windows)


class FakeNormalizer:
    def __init__(self):
        self.call = None

    def __call__(self, source, destination, start_s, end_s):
        self.call = (source, destination, start_s, end_s)
        destination.write_bytes(b"RIFFnormalized")


def limits() -> ReferenceLimits:
    return ReferenceLimits(3.0, 30.0, -60.0, 0.01)


def test_conserve_uniquement_les_bornes_vad_exterieures(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    pcm = np.full(12 * 16000, 0.1, dtype=np.float32)
    normalizer = FakeNormalizer()

    result = prepare_reference(
        source,
        tmp_path / "normalized",
        FakeVad([(2.0, 5.0), (6.0, 9.0)]),
        limits(),
        decoder=lambda path: pcm,
        normalizer=normalizer,
    )

    assert normalizer.call[2:] == (2.0, 9.0)
    assert result.duration_s == 7.0
    assert result.path.read_bytes() == b"RIFFnormalized"


@pytest.mark.parametrize("bounds", [[(0.0, 2.99)], [(0.0, 30.01)]])
def test_refuse_une_duree_utile_hors_des_limites(tmp_path, bounds):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    pcm = np.full(31 * 16000, 0.1, dtype=np.float32)

    with pytest.raises(InvalidReferenceError, match="durée"):
        prepare_reference(
            source,
            tmp_path / "normalized",
            FakeVad(bounds),
            limits(),
            decoder=lambda path: pcm,
            normalizer=FakeNormalizer(),
        )


def test_refuse_le_silence_sans_creer_de_fichier(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    output = tmp_path / "normalized"
    with pytest.raises(InvalidReferenceError, match="parole"):
        prepare_reference(
            source,
            output,
            FakeVad([]),
            limits(),
            decoder=lambda path: np.zeros(160000, dtype=np.float32),
            normalizer=FakeNormalizer(),
        )
    assert not output.exists()


def test_refuse_un_signal_trop_faible(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    pcm = np.full(5 * 16000, 1e-5, dtype=np.float32)
    with pytest.raises(InvalidReferenceError, match="faible"):
        prepare_reference(
            source, tmp_path / "out", FakeVad([(0, 5)]), limits(),
            decoder=lambda path: pcm, normalizer=FakeNormalizer(),
        )


def test_refuse_plus_d_un_pourcent_d_echantillons_ecretes(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    pcm = np.full(5 * 16000, 0.1, dtype=np.float32)
    pcm[:1000] = 1.0
    with pytest.raises(InvalidReferenceError, match="écrêt"):
        prepare_reference(
            source, tmp_path / "out", FakeVad([(0, 5)]), limits(),
            decoder=lambda path: pcm, normalizer=FakeNormalizer(),
        )
