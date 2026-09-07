"""Cycle de vie d'au plus un checkpoint Qwen en VRAM."""

import gc
import os
import threading
import time
from collections.abc import Callable
from contextlib import closing
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

    def stream(self, command: GenerateCommand):
        """Produit les morceaux sur le meme thread et garde le modele reserve."""
        with self._lock:
            self._ensure_loaded(command.mode)
            self._state = "generating"
            try:
                with closing(self._model.stream(command)) as chunks:
                    yield from chunks
            except Exception as exc:
                code = "cuda_oom" if "out of memory" in str(exc).lower() else "generation_failed"
                self._last_error = code
                self._unload_locked(final_state="error")
                raise WorkerModelError(code, "La generation Qwen a echoue.") from exc
            finally:
                if self._model is not None:
                    self._state = "ready"
                    self._last_used = self._clock()

    def unload(self) -> None:
        with self._lock:
            self._unload_locked(final_state="idle")

    def unload_if_idle(self) -> bool:
        """Decharge apres inactivite ; idle_s <= 0 desactive le dechargement."""
        with self._lock:
            if self._idle_s <= 0:
                return False
            if self._model is None or self._last_used is None:
                return False
            if self._clock() - self._last_used < self._idle_s:
                return False
            self._unload_locked(final_state="idle")
            return True

    def health(self) -> dict:
        # Ce snapshot ne doit jamais attendre le verrou d'une inference longue.
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
    def __init__(self, model, mode: Mode, *, backend: str = "standard", chunk_size: int = 8):
        self._model = model
        self._mode = mode
        self._backend = backend
        self._chunk_size = chunk_size

    def _arguments(self, command: GenerateCommand):
        options = {"text": command.text, "language": command.language}
        if self._mode is Mode.CUSTOM:
            return "generate_custom_voice", {**options, "speaker": command.speaker, "instruct": command.instruct}
        elif self._mode is Mode.DESIGN:
            return "generate_voice_design", {**options, "instruct": command.instruct}
        clone_flag = "xvec_only" if self._backend == "cuda_graphs" else "x_vector_only_mode"
        return "generate_voice_clone", {
            **options, "ref_audio": command.reference_audio, "ref_text": command.reference_text,
            clone_flag: False,
        }

    def generate(self, command: GenerateCommand):
        method, options = self._arguments(command)
        wavs, rate = getattr(self._model, method)(**options)
        return wavs[0], rate

    def stream(self, command: GenerateCommand):
        """Rend l'audio au fil de la generation, sans attendre la phrase entiere."""
        if self._backend != "cuda_graphs":
            raise WorkerModelError("streaming_unavailable", "Le streaming exige CUDA Graphs.")
        method, options = self._arguments(command)
        with closing(getattr(self._model, method + "_streaming")(
            **options, chunk_size=self._chunk_size,
        )) as chunks:
            for waveform, rate, _timing in chunks:
                yield waveform, rate


def load_qwen_model(mode: Mode, model_id: str) -> QwenModelAdapter:
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    threads = int(os.environ.get("TTS_CPU_THREADS", "1"))
    chunk_size = int(os.environ.get("TTS_STREAM_CHUNK_SIZE", "8"))
    if threads < 1 or not 1 <= chunk_size <= 32:
        raise ValueError("TTS_CPU_THREADS >= 1 et TTS_STREAM_CHUNK_SIZE entre 1 et 32 requis.")
    torch.set_num_threads(threads)
    try:
        model_path = snapshot_download(model_id, local_files_only=True)
    except LocalEntryNotFoundError:
        model_path = snapshot_download(model_id)
    backend = os.environ.get("TTS_BACKEND", "cuda_graphs")
    if backend == "cuda_graphs":
        from faster_qwen3_tts import FasterQwen3TTS
        model = FasterQwen3TTS.from_pretrained(
            model_path, device="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa", max_seq_len=2048,
        )
    elif backend == "standard":
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            model_path, device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        raise ValueError("TTS_BACKEND doit etre cuda_graphs ou standard.")
    return QwenModelAdapter(model, mode, backend=backend, chunk_size=chunk_size)


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
