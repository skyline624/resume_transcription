"""Fabrique de l'application FastAPI."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from transcription_server.api import native_routes
from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.state import AppState

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

    app.include_router(native_routes.router)
    return app
