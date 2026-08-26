"""Route OpenAI de synthèse vocale."""

from typing import Annotated
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from transcription_server.api.tts_schemas import SpeechRequest
from transcription_server.state import AppState, get_state
from transcription_server.tts.audio_output import AudioRenderError, render_output
from transcription_server.tts.domain import SynthesisRequest, TtsMode, TtsUnavailableError
from transcription_server.tts.profiles import VoiceNotFoundError
from transcription_server.tts.text import segment_text

router = APIRouter(prefix="/v1")
_SEGMENT_MAX_CHARS = 500


@router.post("/audio/speech")
async def create_speech(
    body: SpeechRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> Response:
    if not state.settings.enable_tts:
        _raise_tts_error(503, "tts_disabled", "La synthèse vocale est désactivée.")
    profile = None
    if body.mode is TtsMode.CLONE:
        if state.voice_profiles is None:
            _raise_tts_error(503, "voice_store_unavailable", "Le registre vocal est indisponible.")
        try:
            profile = state.voice_profiles.get(body.voice or "")
        except VoiceNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={"message": "Voix inconnue.", "code": "voice_not_found"},
            ) from exc

    segments = segment_text(body.input, _SEGMENT_MAX_CHARS)
    chunks: list[bytes] = []
    try:
        async with state.gpu_lock:
            for segment in segments:
                request = SynthesisRequest(
                    text=segment.text,
                    mode=body.mode,
                    language=body.language,
                    voice=body.voice if body.mode is TtsMode.CUSTOM_VOICE else None,
                    instructions=body.instructions,
                    reference_path=(
                        Path(profile.audio_path) if profile else None
                    ),
                    reference_text=profile.transcript if profile else None,
                )
                result = await state.tts.synthesize(request)
                chunks.append(result.audio_wav)
        encoded = await run_in_threadpool(
            render_output,
            chunks,
            [segment.pause_after_ms for segment in segments],
            body.speed,
            body.response_format,
        )
    except TtsUnavailableError as exc:
        _raise_tts_error(503, exc.code, str(exc))
    except AudioRenderError as exc:
        _raise_tts_error(503, "audio_render_failed", str(exc))
    return Response(content=encoded.data, media_type=encoded.media_type)


def _raise_tts_error(status: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status,
        detail={"message": message, "code": code, "param": None},
    )
