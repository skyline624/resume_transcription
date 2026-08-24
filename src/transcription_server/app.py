"""Fabrique de l'application FastAPI."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from transcription_server.api import native_routes, openai_routes, summary_routes
from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import (
    DiarizationEngine,
    NullDiarizationEngine,
)
from transcription_server.runtime import cuda_available, gpu_info, resolve_device
from transcription_server.summary.engine import (
    SummaryEngine,
    UnavailableSummaryEngine,
)
from transcription_server.state import AppState

logger = logging.getLogger(__name__)

#: Duree du signal de chauffe, en secondes.
_WARMUP_S = 1.0

_ERROR_TYPES = {
    400: "invalid_request_error",
    413: "invalid_request_error",
    503: "service_unavailable",
}


def create_app(
    settings: Settings,
    asr: AsrEngine,
    diarization: DiarizationEngine,
    device_info: dict | None = None,
    summary: SummaryEngine | None = None,
) -> FastAPI:
    """Assemble l'application autour de moteurs deja construits.

    Les moteurs sont injectes plutot que charges ici : c'est ce qui rend
    l'ensemble de la surface HTTP testable sans GPU ni conteneur. La Task 14
    fournira un `build_app()` qui charge les vrais moteurs et appelle cette
    meme fonction.
    """
    app = FastAPI(title="Serveur de transcription Parakeet", version="0.1.0")
    app.state.app_state = AppState(
        settings=settings,
        asr=asr,
        diarization=diarization,
        summary=summary or UnavailableSummaryEngine(),
        device_info=device_info or {},
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Uniformise les erreurs au format OpenAI sur toutes les routes."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": _ERROR_TYPES.get(exc.status_code, "server_error"),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Meme enveloppe pour l'imprevu, sans rien divulguer de la trace.

        Sans ce gestionnaire, une exception echappee d'un moteur sort en
        `text/plain` sans enveloppe : une troisieme forme d'erreur, que la
        moindre panne du GPU suffirait a exposer. Le detail part au journal --
        Starlette releve ensuite l'exception, donc uvicorn la trace aussi.
        """
        logger.exception("Erreur non gérée sur %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Erreur interne du serveur.",
                    "type": "server_error",
                }
            },
        )

    app.include_router(native_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(summary_routes.router)
    return app


# Les deux fabriques ci-dessous existent pour que l'import de NeMo et de
# pyannote reste paresseux — le paquet doit s'importer sans l'extra `gpu` — et
# pour offrir un point de substitution aux tests, qui verifient le cablage sans
# telecharger 2,6 Go de poids.


def _load_nemo_engine(model_name: str, device: str, compute_type: str) -> AsrEngine:
    from transcription_server.asr.nemo_parakeet import load_nemo_engine

    return load_nemo_engine(
        model_name=model_name, device=device, compute_type=compute_type
    )


def _load_pyannote_engine(
    model_name: str, hf_token: str, device: str
) -> DiarizationEngine:
    from transcription_server.diarization.pyannote_engine import load_pyannote_engine

    return load_pyannote_engine(
        model_name=model_name, hf_token=hf_token, device=device
    )


def build_app(settings: Settings | None = None) -> FastAPI:
    """Construit l'application avec les vrais moteurs charges sur le GPU.

    Les moteurs sont charges ici, au demarrage du processus, et non a la
    premiere requete : mieux vaut echouer tout de suite qu'apres qu'un
    utilisateur a televerse une reunion d'une heure.

    L'ordre compte. La validation du peripherique passe avant tout chargement,
    afin qu'une erreur de configuration coute une seconde plutot que le
    telechargement de 2,6 Go de poids.
    """
    from transcription_server.config import get_settings

    settings = settings or get_settings()
    device = resolve_device(settings.device, cuda_available())

    logger.info("Chargement des moteurs sur %s…", device)
    asr = _load_nemo_engine(
        model_name=settings.asr_model,
        device=device,
        compute_type=settings.compute_type,
    )

    if settings.enable_diarization:
        diarization = _load_pyannote_engine(
            model_name=settings.diarization_model,
            # Le token n'est pas copie dans une variable locale : Field(repr=False)
            # protege l'objet, pas ses copies, et une trace le reexposerait.
            hf_token=settings.hf_token or "",
            device=device,
        )
    else:
        logger.info(
            "Diarization désactivée (ENABLE_DIARIZATION=false) : le serveur "
            "démarre sans token HuggingFace et ne séparera pas les locuteurs."
        )
        diarization = NullDiarizationEngine()

    if settings.enable_summary:
        from transcription_server.summary.ollama_engine import load_ollama_engine

        summary = load_ollama_engine(
            base_url=settings.ollama_base_url,
            model=settings.summary_model,
            timeout_s=settings.summary_timeout_s,
        )
    else:
        logger.info("Rédaction de compte-rendu désactivée (ENABLE_SUMMARY=false).")
        summary = UnavailableSummaryEngine()

    _warmup(asr, device)

    return create_app(
        settings=settings,
        asr=asr,
        diarization=diarization,
        summary=summary,
        device_info=gpu_info(),
    )


def _warmup(asr: AsrEngine, device: str) -> None:
    """Une inference sur une seconde de silence, avant d'accepter du trafic.

    Sans elle, la premiere vraie requete paierait la compilation des kernels
    CUDA. Le warmup est un confort : son echec est journalise et n'empeche pas
    le service de demarrer.
    """
    if device != "cuda":
        return
    import numpy as np

    from transcription_server.audio import SAMPLE_RATE

    try:
        asr.transcribe(
            np.zeros(int(_WARMUP_S * SAMPLE_RATE), dtype=np.float32), language=None
        )
        logger.info("Warmup terminé.")
    except Exception as exc:  # noqa: BLE001 — le warmup n'est pas critique
        logger.warning("Le warmup a échoué, le serveur démarre quand même : %s", exc)
