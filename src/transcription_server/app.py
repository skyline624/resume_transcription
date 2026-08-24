"""Fabrique de l'application FastAPI."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from transcription_server.api import native_routes
from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.state import AppState

logger = logging.getLogger(__name__)

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
    return app
