"""Cycle de vie paresseux des moteurs GPU du processus principal."""

import asyncio
import gc
import logging
import threading
import time
from collections.abc import Callable
from contextlib import aclosing
from typing import Any

from transcription_server.runtime import empty_cache

logger = logging.getLogger(__name__)


class LazyEngine:
    """Enveloppe un constructeur de moteur et charge ce dernier a la demande.

    Le loader reste appele hors de la boucle d'evenements, via
    `run_in_threadpool` depuis les routes. Le verrou interne garantit qu'un
    seul appel declenche le chargement et que les concurrents attendent le
    meme resultat.
    """

    def __init__(
        self,
        name: str,
        loader: Callable[..., Any],
        method_name: str,
        *,
        idle_s: float,
        clock: Callable[[], float] | None = None,
        after_load: Callable[[Any], None] | None = None,
    ) -> None:
        self._name = name
        self._loader = loader
        self._method_name = method_name
        self._idle_s = idle_s
        self._clock = clock or time.monotonic
        self._after_load = after_load
        self._lock = threading.RLock()
        self._engine: Any | None = None
        self._state = "unloaded"
        self._last_used: float | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def engine_name(self) -> str:
        """Nom du moteur sous-jacent si charge, sinon l'identifiant du lazy."""
        engine = getattr(self._engine, "name", None)
        return str(engine) if engine is not None else self._name

    @property
    def state(self) -> str:
        # Instantane de supervision : ne jamais bloquer la boucle HTTP sur
        # le verrou tenu par un chargement ou une inference dans un thread.
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_used(self) -> float | None:
        return self._last_used

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        return lambda *args, **kwargs: self._call(item, *args, **kwargs)

    def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._ensure_loaded()
            started = self._clock()
            try:
                result = getattr(self._engine, method_name)(*args, **kwargs)
            except Exception as exc:
                self._state = "error"
                self._last_error = type(exc).__name__
                raise
            del started
            self._last_used = self._clock()
            return result

    def release_gpu(self) -> bool:
        """Decharge le moteur ; retourne False s'il etait deja absent."""
        with self._lock:
            if self._engine is None:
                self._state = "unloaded"
                return False
            self._state = "unloading"
            engine = self._engine
            self._engine = None
            self._last_used = None
            try:
                release = getattr(engine, "release_gpu", None)
                if release is not None:
                    release()
            finally:
                gc.collect()
                empty_cache()
            self._state = "unloaded"
            return True

    def unload_if_idle(self) -> bool:
        """Decharge uniquement si le moteur a depasse son delai d'inactivite.

        Une valeur `idle_s <= 0` desactive le dechargement automatique.
        """
        with self._lock:
            if self._idle_s <= 0:
                return False
            if self._engine is None or self._last_used is None:
                return False
            if self._clock() - self._last_used < self._idle_s:
                return False
            return self.release_gpu()

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return
        self._state = "loading"
        started = self._clock()
        try:
            engine = self._loader()
        except Exception as exc:
            self._state = "error"
            self._last_error = type(exc).__name__
            logger.exception("Chargement de %s echoue apres %.1f s.", self._name, self._clock() - started)
            raise
        self._engine = engine
        self._last_error = None
        self._state = "loaded"
        logger.info("Moteur %s charge en %.1f s.", self._name, self._clock() - started)
        if self._after_load is not None:
            try:
                self._after_load(engine)
            except Exception:
                logger.exception("Le warmup de %s a echoue ; le moteur reste charge.", self._name)


class TtsLifecycleClient:
    """Client TTS qui retient la derniere utilisation pour le monitor unifie."""

    def __init__(self, delegate: Any, *, idle_s: float, clock: Callable[[], float] | None = None) -> None:
        import time

        self._delegate = delegate
        self._idle_s = idle_s
        self._clock = clock or time.monotonic
        self._last_used: float | None = None

    @property
    def last_used(self) -> float | None:
        return self._last_used

    async def synthesize(self, request: Any) -> Any:
        result = await self._delegate.synthesize(request)
        self._last_used = self._clock()
        return result

    async def stream(self, request: Any):
        try:
            async with aclosing(self._delegate.stream(request)) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            self._last_used = self._clock()

    async def health(self) -> Any:
        return await self._delegate.health()

    async def unload(self, reason: str) -> None:
        await self._delegate.unload(reason)
        self._last_used = None

    def unload_if_idle(self) -> bool:
        if self._idle_s <= 0:
            return False
        if self._last_used is None or self._clock() - self._last_used < self._idle_s:
            return False
        self._last_used = None
        return True


class GpuLifecycleManager:
    """Verifie les trois familles GPU avec un seul timer applicatif."""

    def __init__(
        self,
        engines: tuple[LazyEngine, ...],
        tts: TtsLifecycleClient | None,
        *,
        interval_s: float,
        clock: Callable[[], float] | None = None,
        gpu_lock: asyncio.Lock | None = None,
    ) -> None:
        self._engines = engines
        self._tts = tts
        self._interval_s = interval_s
        self._gpu_lock = gpu_lock or asyncio.Lock()
        self._clock = clock or (lambda: __import__("time").monotonic())
        self._task = None
        self._stopping = False

    @property
    def interval_s(self) -> float:
        return self._interval_s

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._stopping = True
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

    async def unload_idle(self) -> None:
        """Decharge pyannote puis Parakeet, puis le worker TTS si inactif."""
        async with self._gpu_lock:
            await self._unload_idle_locked()

    async def _unload_idle_locked(self) -> None:
        """Reverifie l'inactivite apres la fin du travail GPU en cours."""
        for engine in self._engines:
            try:
                if await asyncio.to_thread(engine.unload_if_idle):
                    logger.info("Moteur %s decharge apres inactivite.", engine.name)
            except Exception:
                logger.exception("Le dechargement de %s a echoue.", engine.name)
        if self._tts is not None:
            try:
                if self._tts.unload_if_idle():
                    await self._tts.unload("inactivite")
                    logger.info("Worker TTS decharge apres inactivite.")
            except Exception:
                logger.exception("Le dechargement du worker TTS a echoue.")

    async def _run(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._interval_s)
            await self.unload_idle()
