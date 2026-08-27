"""Endpoints compatibles avec l'API audio d'OpenAI.

Objectif : qu'un client ecrit pour Whisper fonctionne sans modification.

La diarization n'est deliberement pas exposee ici — elle n'a pas d'equivalent
dans l'API d'OpenAI, et un champ supplementaire romprait la compatibilite que
cet endpoint existe precisement pour offrir. Elle reste disponible sur
`POST /transcribe`.
"""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from transcription_server.api.native_routes import _save_upload
from transcription_server.audio import AudioDecodeError
from transcription_server.formatting import to_plain_text, to_srt, to_vtt
from transcription_server.pipeline import TranscriptionRequest, run_pipeline
from transcription_server.state import AppState, get_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

OpenAIFormat = Literal["json", "text", "srt", "vtt", "verbose_json"]


@router.get("/models")
async def list_models(state: Annotated[AppState, Depends(get_state)]) -> dict:
    """Liste les modeles charges, au format attendu par les clients OpenAI."""
    return {
        "object": "list",
        "data": [
            {
                "id": state.asr.name,
                "object": "model",
                "owned_by": "nvidia",
            }
        ],
    }


@router.post("/audio/transcriptions")
async def create_transcription(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    response_format: Annotated[OpenAIFormat, Form()] = "json",
):
    """Transcrit un fichier audio.

    `model`, `prompt` et `temperature` sont acceptes pour que les clients
    Whisper existants n'aient rien a modifier, mais ils sont sans effet :
    le modele est celui charge au demarrage, Parakeet n'accepte pas de prompt
    de conditionnement, et son decodage TDT est deterministe.
    """
    settings = state.settings
    path = await _save_upload(file, settings.max_upload_bytes)

    try:
        # Le verrou serialise les travaux GPU ; run_in_threadpool rend la main
        # a la boucle pendant l'inference, donc /health repond toujours.
        async with state.gpu_lock:
            await state.prepare_transcription()
            result = await run_in_threadpool(
                run_pipeline,
                path=path,
                asr=state.asr,
                diarization=state.diarization,
                request=TranscriptionRequest(language=language, diarize=False),
                chunk_length_s=settings.chunk_length_s,
                chunk_overlap_s=settings.chunk_overlap_s,
                turn_gap_s=settings.turn_gap_s,
                vad=state.vad,
                vad_fallback_length_s=settings.vad_max_segment_s,
                vad_fallback_overlap_s=settings.vad_fallback_overlap_s,
            )
    except AudioDecodeError as exc:
        # str(exc) porte le chemin du fichier temporaire et le stderr brut de
        # ffmpeg : cela renseignerait un appelant sur l'arborescence du
        # conteneur. Le journal du serveur, oui ; le corps de la reponse, jamais.
        logger.warning("Échec de décodage : %s", exc)
        raise HTTPException(
            status_code=400, detail="Le fichier audio n'a pas pu être décodé."
        ) from exc
    finally:
        path.unlink(missing_ok=True)

    if response_format == "text":
        return PlainTextResponse(to_plain_text(result.turns))
    if response_format == "srt":
        return PlainTextResponse(to_srt(result.turns))
    if response_format == "vtt":
        return PlainTextResponse(to_vtt(result.turns))
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            # `language` echo ce que l'appelant a demande, et vaut null s'il
            # n'a rien demande : la langue detectee ne remonte pas, `transcribe`
            # etant appele une fois par fenetre — un Protocol elargi rendrait
            # N langues et exigerait une regle de reconciliation.
            "language": result.language,
            "duration": round(result.duration, 3),
            "text": result.text,
            "segments": [
                {
                    "id": index,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "text": turn.text,
                }
                for index, turn in enumerate(result.turns)
            ],
            "words": [
                {
                    "word": mot.text,
                    "start": round(mot.start, 3),
                    "end": round(mot.end, 3),
                }
                for turn in result.turns
                for mot in turn.words
            ],
        }
    return {"text": result.text}
