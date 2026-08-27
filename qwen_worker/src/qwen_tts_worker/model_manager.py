"""Cycle de vie d'au plus un checkpoint Qwen en VRAM."""

import gc
import threading
import time
from collections.abc import Callable
from typing import Any

from qwen_tts_worker.domain import GenerateCommand, Mode, WorkerModelError


class QwenModelManager:
    def __init__(
        self,
        model_ids: dict[Mode, str],
        loader: Callable[[Mode, str], Any],
        cuda_cleanup: Callable[[], None],
        idle_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._model_ids = model_ids
        self._loader = loader
        self._cuda_cleanup = cuda_cleanup
        self._idle_s = idle_s
        self._clock = clock
        self._lock = threading.RLock()
        self._model = None
        self._mode: Mode | None = None
        self._state = "idle"
        self._last_used: float | None = None
        self._last_error: str | None = None

    def generate(self, command: GenerateCommand):
        with self._lock:
            load_started = self._clock()
            self._ensure_loaded(command.mode)
            load_ms = max(0.0, (self._clock() - load_started) * 1000)
            self._state = "generating"
            started = self._clock()
            try:
                waveform, sample_rate = self._model.generate(command)
            except Exception as exc:
                code = "cuda_oom" if "out of memory" in str(exc).lower() else "generation_failed"
                self._last_error = code
                self._unload_locked(final_state="error")
                raise WorkerModelError(code, "La génération Qwen a échoué.") from exc
            inference_ms = max(0.0, (self._clock() - started) * 1000)
            self._state = "ready"
            self._last_used = self._clock()
            return waveform, sample_rate, load_ms, inference_ms

    def load(self, mode: Mode) -> None:
        with self._lock:
            self._ensure_loaded(mode)

    def unload(self) -> None:
        with self._lock:
            self._unload_locked(final_state="idle")

    def unload_if_idle(self) -> bool:
        with self._lock:
            if self._model is None or self._last_used is None:
                return False
            if self._clock() - self._last_used < self._idle_s:
                return False
            self._unload_locked(final_state="idle")
            return True

    def health(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "loaded_model": self._model_ids.get(self._mode) if self._mode else None,
                "last_error": self._last_error,
            }

    def _ensure_loaded(self, mode: Mode) -> None:
        if self._model is not None and self._mode is mode:
            return
        if self._model is not None:
            self._unload_locked(final_state="idle")
        self._state = "loading"
        try:
            self._model = self._loader(mode, self._model_ids[mode])
            self._mode = mode
            self._state = "ready"
            self._last_error = None
        except Exception as exc:
            self._last_error = "model_load_failed"
            self._unload_locked(final_state="error")
            raise WorkerModelError("model_load_failed", "Le modèle Qwen n'a pas pu être chargé.") from exc

    def _unload_locked(self, final_state: str) -> None:
        had_model = self._model is not None
        if had_model:
            self._state = "unloading"
        self._model = None
        self._mode = None
        self._last_used = None
        if had_model:
            gc.collect()
            self._cuda_cleanup()
        self._state = final_state


class QwenModelAdapter:
    def __init__(self, model, mode: Mode):
        self._model = model
        self._mode = mode

    def generate(self, command: GenerateCommand):
        if self._mode is Mode.CUSTOM:
            wavs, rate = self._model.generate_custom_voice(
                text=command.text, language=command.language,
                speaker=command.speaker, instruct=command.instruct,
            )
        elif self._mode is Mode.DESIGN:
            wavs, rate = self._model.generate_voice_design(
                text=command.text, language=command.language, instruct=command.instruct,
            )
        else:
            wavs, rate = self._model.generate_voice_clone(
                text=command.text, language=command.language,
                ref_audio=command.reference_audio, ref_text=command.reference_text,
                x_vector_only_mode=False,
            )
        return wavs[0], rate


def load_qwen_model(mode: Mode, model_id: str) -> QwenModelAdapter:
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return QwenModelAdapter(model, mode)


def cuda_cleanup() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def cuda_memory_allocated_mib() -> float:
    """Retourne la VRAM active vue par le processus worker Qwen."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 2**20
