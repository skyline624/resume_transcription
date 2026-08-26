"""Pré-télécharge les checkpoints sans les charger en VRAM."""

import os


def main() -> None:
    from huggingface_hub import snapshot_download

    model_ids = [
        item.strip()
        for item in os.environ.get("TTS_PRELOAD_MODELS", "").split(",")
        if item.strip()
    ]
    for model_id in model_ids:
        print(f"[qwen-download] {model_id}", flush=True)
        snapshot_download(model_id)


if __name__ == "__main__":
    main()
