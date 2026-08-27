"""Pré-télécharge les checkpoints sans les charger en VRAM."""

import os


DEFAULT_PRELOAD_MODELS = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)


def preload_model_ids() -> tuple[str, ...]:
    configured = os.environ.get("TTS_PRELOAD_MODELS")
    if configured is None:
        return DEFAULT_PRELOAD_MODELS
    return tuple(item.strip() for item in configured.split(",") if item.strip())


def main() -> None:
    from huggingface_hub import snapshot_download

    for model_id in preload_model_ids():
        print(f"[qwen-download] {model_id}", flush=True)
        snapshot_download(model_id)


if __name__ == "__main__":
    main()
