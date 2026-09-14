"""Derive la configuration Docker sans modifier le service natif ni ses secrets."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
values = {}
for line in (root / '.env.studio-linux').read_text(encoding='utf-8-sig').splitlines():
    if line.strip() and not line.lstrip().startswith('#') and '=' in line:
        key, value = line.split('=', 1)
        values[key] = value
for key in ('MAIN_PYTHON', 'QWEN_PYTHON', 'XDG_RUNTIME_DIR', 'LD_LIBRARY_PATH'):
    values.pop(key, None)
values.update({
    'HOME': '/tmp',
    'HOST': '0.0.0.0',
    'PORT': '8000',
    'HF_HOME': '/app/models',
    'WEB_DIST_PATH': '/app/web-dist',
    'VOICE_STORE_PATH': '/mnt/donnees/resume_transcription/voices',
    'NEMO_EXTRACTED_CACHE_DIR': '/app/nemo-cache',
    'TORCH_HOME': '/app/torch-cache',
    'NUMBA_CACHE_DIR': '/tmp/numba_cache',
    'TRITON_CACHE_DIR': '/tmp/triton',
    'TMPDIR': '/tmp',
    'TTS_WORKER_SOCKET': '/run/qwen-tts/worker.sock',
    'OLLAMA_BASE_URL': 'http://ollama:11434',
})
destination = root / '.env.docker-studio'
destination.touch(mode=0o600, exist_ok=True)
destination.chmod(0o600)
destination.write_text(''.join(f'{key}={value}\n' for key, value in values.items()), encoding='utf-8')
(root / 'runtime/ollama-state').mkdir(parents=True, exist_ok=True)
(root / 'speakers').mkdir(mode=0o700, exist_ok=True)
print('Configuration Docker ecrite, secrets masques.')
