"""Configure l'installation studio-linux sans afficher les secrets."""
import os
import json
from pathlib import Path
from uuid import UUID

root = Path(__file__).resolve().parents[1]
values = {}
for filename in (".env.example", ".env"):
    for line in (root / filename).read_text(encoding="utf-8-sig").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
values.update({
    "HOST": "127.0.0.1",
    "PORT": "8000",
    "HF_HOME": str(root / "models"),
    "HF_HUB_DISABLE_XET": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "MAIN_PYTHON": str(root / ".venv/bin/python"),
    "QWEN_PYTHON": str(root / ".venv-qwen/bin/python"),
    "WEB_DIST_PATH": str(root / "web/dist"),
    "VOICE_STORE_PATH": str(root / "voices"),
    "NEMO_EXTRACTED_CACHE_DIR": str(root / "runtime/nemo-cache"),
    "NUMBA_CACHE_DIR": str(root / "runtime/numba-cache"),
    "TORCH_HOME": str(root / "runtime/torch"),
    "TRITON_CACHE_DIR": str(root / "runtime/triton"),
    "TMPDIR": str(root / "runtime/tmp"),
    "OLLAMA_BASE_URL": "http://127.0.0.1:11435",
    "TTS_WORKER_SOCKET": f"/run/user/{os.getuid()}/qwen-tts/worker.sock",
})
for dirname in ("voices", "runtime/tmp", "runtime/nemo-cache", "runtime/numba-cache", "runtime/torch", "runtime/triton"):
    (root / dirname).mkdir(parents=True, exist_ok=True)
destination = root / ".env.studio-linux"
destination.touch(mode=0o600, exist_ok=True)
destination.chmod(0o600)
destination.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
(root / ".env").chmod(0o600)

# Les profils Docker contiennent des chemins absolus vers /app/voices/audio.
# Conserver leurs identifiants et consentements, puis relocaliser les fichiers.
index = root / "voices/profiles.json"
if index.exists():
    original = index.read_text(encoding="utf-8")
    profiles = json.loads(original)
    changed = False
    for profile in profiles:
        identifier = str(UUID(profile["id"]))
        target = (root / "voices/audio" / f"{identifier}.wav").resolve(strict=True)
        assert target.is_relative_to((root / "voices/audio").resolve())
        previous = Path(profile["audio_path"])
        if previous == Path("/app/voices/audio") / target.name:
            profile["audio_path"] = str(target)
            changed = True
        elif previous != target:
            raise RuntimeError("Chemin vocal source inattendu : migration manuelle requise.")
    if changed:
        backup = root / "voices/profiles.before-ubuntu.json"
        if not backup.exists():
            backup.touch(mode=0o600)
            backup.write_text(original, encoding="utf-8")
        temporary = index.with_suffix(".migration.tmp")
        temporary.touch(mode=0o600, exist_ok=True)
        temporary.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(index)
    print(f"Profils vocaux verifies : {len(profiles)}")
print("Configuration Ubuntu ecrite, secrets masques.")
