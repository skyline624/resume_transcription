"""Endpoints natifs : /transcribe et /health."""

import logging
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from transcription_server.api.schemas import (
    HealthOut,
    TranscriptionOut,
    result_to_out,
)
from transcription_server.audio import AudioDecodeError
from transcription_server.diarization.engine import NullDiarizationEngine
from transcription_server.formatting import to_dialogue, to_plain_text, to_srt, to_vtt
from transcription_server.pipeline import (
    ChannelMode,
    TranscriptionRequest,
    run_pipeline,
)
from transcription_server.state import AppState, get_state

logger = logging.getLogger(__name__)

router = APIRouter()

ResponseFormat = Literal["json", "text", "srt", "vtt", "dialogue"]

# Un mega-octet : compromis entre le nombre d'aller-retours et la memoire
# retenue par morceau lu.
_TAILLE_MORCEAU = 1024 * 1024

# Le suffixe du fichier temporaire vient du nom envoye par le client. Il ne
# sert qu'au confort de debogage -- ffmpeg reconnait le format au contenu, pas
# a l'extension -- mais il part tel quel dans un nom de fichier : une extension
# de plusieurs centaines de caracteres fait echouer la creation du temporaire
# et rend un 500 sur une entree entierement controlee par l'appelant.
_SUFFIXE_MAX = 16


async def _save_upload(upload: UploadFile, max_bytes: int) -> Path:
    """Ecrit l'upload dans un fichier temporaire, en refusant les trop gros.

    Le fichier est detruit sur *tout* chemin d'echec : depassement de taille,
    mais aussi erreur de lecture, disque plein, ou annulation de la requete par
    le client -- CancelledError derive de BaseException, pas de Exception.
    Sans ce filet, chacun de ces cas laisserait un reliquat de la taille de
    l'upload dans le repertoire temporaire du conteneur.

    La fermeture est *dans* le try : c'est elle qui vide le tampon, donc le
    dernier endroit ou un disque plein peut encore echouer. Une fermeture
    placee en dehors laisserait ce cas-la fuir. Fermer avant de delier est par
    ailleurs obligatoire sous Windows, ou l'on ne supprime pas un fichier
    encore ouvert.
    """
    suffix = (Path(upload.filename or "audio").suffix or ".bin")[:_SUFFIXE_MAX]
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = Path(handle.name)
    written = 0
    try:
        while chunk := await upload.read(_TAILLE_MORCEAU):
            written += len(chunk)
            # La borne porte sur l'upload compresse, pas sur sa dilatation en
            # PCM : voir la note de MAX_UPLOAD_MB dans la documentation.
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Fichier trop volumineux : maximum {max_bytes} octets.",
                )
            handle.write(chunk)
        # Vide le tampon : dernier endroit ou l'ecriture peut encore echouer.
        handle.close()
    except BaseException:
        # Sans effet si la fermeture ci-dessus a deja eu lieu ; io marque le
        # fichier ferme meme quand le vidage a leve.
        handle.close()
        path.unlink(missing_ok=True)
        raise
    return path


@router.get("/health", response_model=HealthOut)
async def health(state: Annotated[AppState, Depends(get_state)]) -> HealthOut:
    return HealthOut(
        status="ok",
        device=state.settings.device,
        asr_model=state.asr.name,
        diarization_model=state.diarization.name,
        diarization_enabled=state.settings.enable_diarization,
        vad_model=state.vad.name if state.vad is not None else "none",
        vad_device=state.vad.device if state.vad is not None else None,
        vad_enabled=state.settings.enable_vad,
        summary_model=state.summary.name,
        summary_enabled=state.settings.enable_summary,
        gpu=state.device_info or None,
    )


@router.post("/transcribe", responses={200: {"model": TranscriptionOut}})
async def transcribe(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = None,
    diarize: Annotated[bool | None, Form()] = None,
    num_speakers: Annotated[int | None, Form()] = None,
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = True,
    response_format: Annotated[ResponseFormat, Form()] = "json",
    channels: Annotated[ChannelMode, Form()] = "mix",
):
    if num_speakers is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="num_speakers et min_speakers/max_speakers s'excluent.",
        )

    # Demander explicitement la diarization a un serveur qui n'en a pas doit
    # echouer bruyamment. Le moteur nul rendrait une liste de locuteurs vide
    # avec un 200, indistinguable d'un enregistrement a un seul locuteur :
    # l'appelant conclurait a un audio mono-locuteur au lieu d'apprendre que la
    # fonction est eteinte. Le test porte sur le moteur et non sur
    # ENABLE_DIARIZATION, car un appelant peut legitimement demander la
    # diarization a un serveur dont le defaut est de ne pas la faire.
    if diarize and isinstance(state.diarization, NullDiarizationEngine):
        raise HTTPException(
            status_code=400,
            detail=(
                "La diarization est désactivée sur ce serveur. Redémarrez-le "
                "avec ENABLE_DIARIZATION=true et un HF_TOKEN valide, ou "
                "omettez le paramètre diarize."
            ),
        )

    settings = state.settings
    should_diarize = settings.enable_diarization if diarize is None else diarize
    path = await _save_upload(file, settings.max_upload_bytes)

    try:
        # Le verrou serialise les travaux GPU ; run_in_threadpool rend la main
        # a la boucle pendant l'inference, donc /health repond toujours.
        async with state.gpu_lock:
            result = await run_in_threadpool(
                run_pipeline,
                path=path,
                asr=state.asr,
                diarization=state.diarization,
                request=TranscriptionRequest(
                    language=language,
                    diarize=should_diarize,
                    channel_mode=channels,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                ),
                chunk_length_s=settings.chunk_length_s,
                chunk_overlap_s=settings.chunk_overlap_s,
                turn_gap_s=settings.turn_gap_s,
                vad=state.vad,
                vad_fallback_length_s=settings.vad_max_segment_s,
                vad_fallback_overlap_s=settings.vad_fallback_overlap_s,
            )
    except AudioDecodeError as exc:
        # str(exc) porte le chemin du fichier temporaire et le stderr brut de
        # ffmpeg : cela renseigne un appelant sur l'arborescence du conteneur.
        # Le journal du serveur, oui ; le corps de la reponse, jamais.
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
    if response_format == "dialogue":
        return PlainTextResponse(to_dialogue(result.turns))
    return result_to_out(result, include_words=word_timestamps)
