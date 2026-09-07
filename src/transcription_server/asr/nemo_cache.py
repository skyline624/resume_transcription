"""Persistent, disk-only cache of extracted NeMo checkpoints."""

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)
_MANIFEST = ".extracted-v1.json"


def _complete(folder: Path) -> bool:
    try:
        files = json.loads((folder / _MANIFEST).read_text())
        return (
            "model_config.yaml" in files
            and "model_weights.ckpt" in files
            and all(
                (folder / name).resolve().is_relative_to(folder.resolve())
                and (folder / name).is_file()
                and (folder / name).stat().st_size == size
                for name, size in files.items()
            )
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def extracted_checkpoint(source: Path, cache_root: Path) -> Path:
    """Extract once per archive revision, publishing only a complete directory.

    Extraction streams the archive to disk. Cache hits only check file metadata;
    no model weights or archive contents are retained in Python memory.
    """
    source = source.resolve()
    root = cache_root.resolve()
    info = source.stat()
    identity = f"v1:{source}:{info.st_size}:{info.st_mtime_ns}"
    key = hashlib.sha256(identity.encode()).hexdigest()
    target = root / key
    root.mkdir(parents=True, exist_ok=True)
    if _complete(target):
        logger.info("Cache NeMo extrait reutilise : %s", target)
        return target

    started = time.monotonic()
    logger.info("Extraction initiale du checkpoint NeMo vers %s", target)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=root) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        with tarfile.open(source) as archive:
            archive.extractall(staging, filter="data")
        files = {
            str(path.relative_to(staging)): path.stat().st_size
            for path in staging.rglob("*") if path.is_file()
        }
        (staging / _MANIFEST).write_text(json.dumps(files), encoding="utf-8")
        if not _complete(staging):
            raise ValueError("Checkpoint NeMo incomplet : configuration ou poids manquants.")
        if target.exists():
            if _complete(target):
                return target
            # Remove only this incomplete cache entry, never the source archive.
            if not target.resolve().is_relative_to(root):
                raise ValueError("Le cache NeMo doit rester dans son volume.")
            shutil.rmtree(target)
        try:
            staging.rename(target)
        except OSError:
            # Another process may have published the same revision first.
            if not _complete(target):
                raise
    logger.info("Cache NeMo extrait prepare en %.1f s", time.monotonic() - started)
    return target
