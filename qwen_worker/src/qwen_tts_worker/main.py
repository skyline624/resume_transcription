"""Point d'entrée Uvicorn du worker privé."""

import os
from pathlib import Path

import uvicorn

from qwen_tts_worker.app import create_worker_app
from qwen_tts_worker.domain import Mode
from qwen_tts_worker.model_manager import QwenModelManager, cuda_cleanup, load_qwen_model


def main() -> None:
    socket = Path(os.environ.get("TTS_WORKER_SOCKET", "/run/qwen-tts/worker.sock"))
    allowed = Path("/run/qwen-tts").resolve()
    socket = socket.resolve()
    if not socket.is_relative_to(allowed):
        raise RuntimeError("TTS_WORKER_SOCKET doit rester sous /run/qwen-tts.")
    socket.parent.mkdir(parents=True, exist_ok=True)
    socket.unlink(missing_ok=True)
    model_ids = {
        Mode.CUSTOM: os.environ.get(
            "TTS_CUSTOM_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ),
        Mode.CLONE: os.environ.get(
            "TTS_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ),
        Mode.DESIGN: os.environ.get(
            "TTS_VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ),
    }
    manager = QwenModelManager(
        model_ids, load_qwen_model, cuda_cleanup,
        float(os.environ.get("TTS_IDLE_UNLOAD_S", "300")),
    )
    downloaded = [item.strip() for item in os.environ.get("TTS_PRELOAD_MODELS", "").split(",") if item.strip()]
    app = create_worker_app(manager, downloaded)
    uvicorn.run(app, uds=str(socket), workers=1, log_level="info")


if __name__ == "__main__":
    main()
