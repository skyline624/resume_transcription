"""Endpoint de redaction de compte-rendu.

Separe de /transcribe a dessein : une reunion de deux heures demande une
dizaine de minutes de transcription, et rejouer un compte-rendu avec d'autres
consignes ne doit pas obliger a les repayer.
"""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from transcription_server.api.native_routes import _save_upload
from transcription_server.audio import AudioDecodeError
from transcription_server.diarization.engine import NullDiarizationEngine
from transcription_server.pipeline import (
    ChannelMode,
    TranscriptionRequest,
    run_pipeline,
)
from transcription_server.summary.engine import SummaryUnavailableError
from transcription_server.summary.prompts import (
    FORMATS,
    construire_prompt,
    rendre_dialogue,
)
from transcription_server.state import AppState, get_state

logger = logging.getLogger(__name__)

router = APIRouter()

FormatCompteRendu = Literal["structure", "narratif"]


@router.post("/summarize")
async def summarize(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile | None, File()] = None,
    transcript: Annotated[str | None, Form()] = None,
    format: Annotated[FormatCompteRendu, Form()] = "structure",
    diarize: Annotated[bool | None, Form()] = None,
    channels: Annotated[ChannelMode, Form()] = "mix",
    language: Annotated[str | None, Form()] = None,
    include_transcript: Annotated[bool, Form()] = False,
    response_format: Annotated[Literal["json", "text"], Form()] = "json",
):
    """Rédige un compte-rendu, à partir d'un audio ou d'une transcription.

    Fournir `file` fait transcrire puis rédiger ; fournir `transcript` rédige
    directement. Les deux ensemble sont refusés : il faudrait choisir lequel
    fait foi, et deviner à la place de l'appelant serait le mauvais réflexe.
    """
    if (file is None) == (transcript is None):
        raise HTTPException(
            status_code=400,
            detail=(
                "Fournissez soit `file` (un fichier audio à transcrire), soit "
                "`transcript` (une transcription déjà produite) — mais pas les "
                "deux, et pas aucun des deux."
            ),
        )

    settings = state.settings
    dialogue = transcript or ""
    turns_rendus = None

    if file is not None:
        should_diarize = settings.enable_diarization if diarize is None else diarize
        if diarize and isinstance(state.diarization, NullDiarizationEngine):
            raise HTTPException(
                status_code=400,
                detail=(
                    "La diarization est désactivée sur ce serveur. Redémarrez-le "
                    "avec ENABLE_DIARIZATION=true et un HF_TOKEN valide, ou "
                    "omettez le paramètre diarize."
                ),
            )
        path = await _save_upload(file, settings.max_upload_bytes)
        try:
            async with state.gpu_lock:
                resultat = await run_in_threadpool(
                    run_pipeline,
                    path=path,
                    asr=state.asr,
                    diarization=state.diarization,
                    request=TranscriptionRequest(
                        language=language,
                        diarize=should_diarize,
                        channel_mode=channels,
                    ),
                    chunk_length_s=settings.chunk_length_s,
                    chunk_overlap_s=settings.chunk_overlap_s,
                    turn_gap_s=settings.turn_gap_s,
                )
        except AudioDecodeError as exc:
            logger.warning("Échec de décodage : %s", exc)
            raise HTTPException(
                status_code=400, detail="Le fichier audio n'a pas pu être décodé."
            ) from exc
        finally:
            path.unlink(missing_ok=True)
        dialogue = rendre_dialogue(resultat.turns)
        turns_rendus = dialogue

    try:
        prompt = construire_prompt(dialogue, format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        # La redaction ne prend pas le verrou GPU : elle se passe hors du
        # conteneur, dans Ollama. Le bloquer empecherait une transcription de
        # demarrer pendant les minutes que dure la redaction.
        texte = await run_in_threadpool(state.summary.summarize, prompt)
    except SummaryUnavailableError as exc:
        logger.warning("Rédaction indisponible : %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if response_format == "text":
        return PlainTextResponse(texte)

    corps = {"summary": texte, "format": format, "model": state.summary.name}
    if include_transcript and turns_rendus is not None:
        corps["transcript"] = turns_rendus
    return corps
