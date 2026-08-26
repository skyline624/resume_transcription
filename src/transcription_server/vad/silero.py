"""Adaptateur léger autour du paquet ``silero-vad``.

Les imports Torch et Silero restent dans la fabrique de production : le paquet
principal et les tests unitaires doivent pouvoir s'importer sans l'extra GPU.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from transcription_server.audio import SAMPLE_RATE, duration_seconds

VadDevice = Literal["cpu", "cuda"]


class SileroVadEngine:
    """Transforme les timestamps Silero en fenêtres absolues triées."""

    name = "silero-vad"

    def __init__(
        self,
        model: Any,
        get_speech_timestamps: Callable[..., list[dict]],
        tensor_factory: Callable[[np.ndarray, VadDevice], Any],
        device: VadDevice,
        max_segment_s: float,
    ) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError("Le périphérique VAD doit être 'cpu' ou 'cuda'.")
        if max_segment_s <= 0:
            raise ValueError("La durée maximale VAD doit être strictement positive.")
        self._model = model
        self._get_speech_timestamps = get_speech_timestamps
        self._tensor_factory = tensor_factory
        self._device = device
        self._max_segment_s = max_segment_s

    @property
    def device(self) -> VadDevice:
        return self._device

    def plan(self, audio: np.ndarray) -> list[tuple[float, float]]:
        """Rend les passages parlés en secondes, bornés à la durée audio."""
        total = duration_seconds(audio)
        tensor = self._tensor_factory(audio, self._device)
        timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=250,
            max_speech_duration_s=self._max_segment_s,
        )

        windows: list[tuple[float, float]] = []
        for timestamp in timestamps:
            start = max(0.0, float(timestamp["start"]))
            end = min(total, float(timestamp["end"]))
            if end > start:
                windows.append((start, end))
        windows.sort(key=lambda window: (window[0], window[1]))
        return windows


def load_silero_vad_engine(
    device: VadDevice = "cpu", max_segment_s: float = 5.0
) -> SileroVadEngine:
    """Charge le modèle JIT livré dans le paquet Silero, sans réseau."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=False).to(device)

    def tensor_factory(audio: np.ndarray, target: VadDevice):
        return torch.from_numpy(audio).to(target)

    return SileroVadEngine(
        model=model,
        get_speech_timestamps=get_speech_timestamps,
        tensor_factory=tensor_factory,
        device=device,
        max_segment_s=max_segment_s,
    )
