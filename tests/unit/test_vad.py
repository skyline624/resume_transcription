import numpy as np
import pytest


def test_silero_convertit_les_timestamps_en_fenetres_bornees():
    """Un retour hors limites ou désordonné ne doit jamais produire une
    tranche numpy inversée ni dépasser la durée du canal.
    """
    from transcription_server.vad.silero import SileroVadEngine

    parametres = {}
    tenseur = object()

    def faux_timestamps(audio, model, **kwargs):
        assert audio is tenseur
        parametres.update(kwargs)
        return [
            {"start": 1.5, "end": 2.5},
            {"start": -0.2, "end": 0.4},
            {"start": 0.8, "end": 0.7},
        ]

    moteur = SileroVadEngine(
        model=object(),
        get_speech_timestamps=faux_timestamps,
        tensor_factory=lambda audio, device: tenseur,
        device="cpu",
        max_segment_s=10.0,
    )

    fenetres = moteur.plan(np.zeros(2 * 16000, dtype=np.float32))

    assert fenetres == [(0.0, 0.4), (1.5, 2.0)]
    assert parametres == {
        "sampling_rate": 16000,
        "return_seconds": True,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 250,
        "max_speech_duration_s": pytest.approx(10.0),
    }


def test_silero_refuse_un_peripherique_inconnu():
    from transcription_server.vad.silero import SileroVadEngine

    with pytest.raises(ValueError, match="cpu.*cuda"):
        SileroVadEngine(
            model=object(),
            get_speech_timestamps=lambda *args, **kwargs: [],
            tensor_factory=lambda audio, device: audio,
            device="tpu",
            max_segment_s=10.0,
        )


def test_fenetres_fixes_restent_bornees_a_dix_secondes():
    from transcription_server.vad.engine import FixedWindowVadEngine

    moteur = FixedWindowVadEngine(max_segment_s=10.0, overlap_s=1.0)

    assert moteur.plan(np.zeros(25 * 16000, dtype=np.float32)) == [
        (0.0, 10.0),
        (9.0, 19.0),
        (18.0, 25.0),
    ]
