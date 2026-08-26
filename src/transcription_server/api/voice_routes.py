"""Inscription, liste, suppression et clonage vocal ponctuel."""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from transcription_server.api.native_routes import _save_upload
from transcription_server.pipeline import TranscriptionRequest, run_pipeline
from transcription_server.state import AppState, get_state
from transcription_server.tts.audio_output import AudioRenderError, render_output
from transcription_server.tts.domain import AudioFormat, SynthesisRequest, TtsMode, TtsUnavailableError
from transcription_server.tts.profiles import VoiceNotFoundError
from transcription_server.tts.reference import (
    InvalidReferenceError,
    ReferenceLimits,
    prepare_reference,
)

router = APIRouter(prefix="/v1")
BUILTIN_VOICES = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)


@router.post("/voices", status_code=201)
async def create_voice(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile, File()],
    name: Annotated[str, Form()],
    language: Annotated[str, Form()] = "fr",
    transcript: Annotated[str | None, Form()] = None,
    consent: Annotated[bool, Form()] = False,
) -> dict:
    if not consent:
        _error(400, "consent_required", "Le consentement explicite est requis.")
    repository = _repository(state)
    upload = await _save_upload(file, state.settings.max_upload_bytes)
    try:
        with tempfile.TemporaryDirectory(prefix="voice-enroll-") as directory:
            reference = await _prepare(state, upload, Path(directory))
            reference_text, source = await _reference_text(
                state, reference.path, transcript, language
            )
            profile = await run_in_threadpool(
                repository.create,
                name,
                language,
                reference_text,
                reference.path,
                reference.duration_s,
                source,
            )
    finally:
        upload.unlink(missing_ok=True)
    return _public_profile(profile)


@router.get("/voices")
async def list_voices(
    state: Annotated[AppState, Depends(get_state)],
) -> dict:
    repository = _repository(state)
    speakers = BUILTIN_VOICES
    try:
        health = await state.tts.health()
        if health.speakers:
            speakers = health.speakers
    except TtsUnavailableError:
        pass
    builtin = [
        {"id": speaker, "name": speaker, "kind": "builtin"}
        for speaker in speakers
    ]
    clones = [_public_profile(item) for item in await run_in_threadpool(repository.list)]
    return {"object": "list", "data": [*builtin, *clones]}


@router.delete("/voices/{voice_id}", status_code=204)
async def delete_voice(
    voice_id: str,
    state: Annotated[AppState, Depends(get_state)],
) -> Response:
    if voice_id in BUILTIN_VOICES:
        _error(405, "builtin_voice", "Une voix prédéfinie ne peut pas être supprimée.")
    try:
        await run_in_threadpool(_repository(state).delete, voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"message": "Voix inconnue.", "code": "voice_not_found"},
        ) from exc
    return Response(status_code=204)


@router.post("/audio/speech/clone")
async def clone_once(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile, File()],
    input: Annotated[str, Form(min_length=1, max_length=4096)],
    consent: Annotated[bool, Form()],
    transcript: Annotated[str | None, Form()] = None,
    language: Annotated[str, Form()] = "fr",
    instructions: Annotated[str | None, Form()] = None,
    response_format: Annotated[AudioFormat, Form()] = AudioFormat.MP3,
    speed: Annotated[float, Form(ge=0.25, le=4.0)] = 1.0,
) -> Response:
    if not consent:
        _error(400, "consent_required", "Le consentement explicite est requis.")
    if instructions:
        _error(422, "unsupported_parameter", "instructions n'est pas pris en charge pour clone.")
    upload = await _save_upload(file, state.settings.max_upload_bytes)
    try:
        with tempfile.TemporaryDirectory(prefix="voice-clone-") as directory:
            reference = await _prepare(state, upload, Path(directory))
            reference_text, _ = await _reference_text(
                state, reference.path, transcript, language
            )
            try:
                async with state.gpu_lock:
                    result = await state.tts.synthesize(SynthesisRequest(
                        text=input,
                        mode=TtsMode.CLONE,
                        language=language,
                        reference_path=reference.path,
                        reference_text=reference_text,
                    ))
                encoded = await run_in_threadpool(
                    render_output, [result.audio_wav], [0], speed, response_format
                )
            except TtsUnavailableError as exc:
                _error(503, exc.code, str(exc))
            except AudioRenderError as exc:
                _error(503, "audio_render_failed", str(exc))
    finally:
        upload.unlink(missing_ok=True)
    return Response(encoded.data, media_type=encoded.media_type)


async def _prepare(state: AppState, upload: Path, directory: Path):
    limits = ReferenceLimits(
        state.settings.tts_reference_min_s,
        state.settings.tts_reference_max_s,
        state.settings.tts_reference_min_dbfs,
        state.settings.tts_reference_max_clipped_ratio,
    )
    try:
        return await run_in_threadpool(
            prepare_reference, upload, directory, state.vad, limits
        )
    except InvalidReferenceError as exc:
        _error(400, "invalid_reference", str(exc))


async def _reference_text(
    state: AppState, path: Path, transcript: str | None, language: str
) -> tuple[str, str]:
    if transcript and transcript.strip():
        return transcript.strip(), "provided"
    async with state.gpu_lock:
        result = await run_in_threadpool(
            run_pipeline,
            path=path,
            asr=state.asr,
            diarization=state.diarization,
            request=TranscriptionRequest(language=language, diarize=False),
            chunk_length_s=state.settings.chunk_length_s,
            chunk_overlap_s=state.settings.chunk_overlap_s,
            turn_gap_s=state.settings.turn_gap_s,
            vad=state.vad,
            vad_fallback_length_s=state.settings.vad_max_segment_s,
            vad_fallback_overlap_s=state.settings.vad_fallback_overlap_s,
        )
    if not result.text.strip():
        _error(400, "empty_reference_transcript", "Parakeet n'a transcrit aucun texte.")
    return result.text.strip(), "parakeet"


def _repository(state: AppState):
    if state.voice_profiles is None:
        _error(503, "voice_store_unavailable", "Le registre vocal est indisponible.")
    return state.voice_profiles


def _public_profile(profile) -> dict:
    return {
        "id": profile.id,
        "name": profile.name,
        "language": profile.language,
        "duration": profile.duration_s,
        "created_at": profile.created_at,
        "transcript_source": profile.transcript_source,
        "kind": "clone",
    }


def _error(status: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status,
        detail={"message": message, "code": code, "param": None},
    )
