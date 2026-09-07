import io
import tarfile

import pytest

from transcription_server.asr.nemo_cache import extracted_checkpoint


def archive(path, weight=b"weights"):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in (("model_config.yaml", b"target: test"), ("model_weights.ckpt", weight), ("tokenizer.model", b"tokens")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_cache_reused_without_reading_archive(tmp_path, monkeypatch):
    source = archive(tmp_path / "model.nemo")
    first = extracted_checkpoint(source, tmp_path / "cache")
    assert (first / "model_weights.ckpt").read_bytes() == b"weights"
    monkeypatch.setattr(tarfile, "open", lambda *a, **kw: pytest.fail("Archive read on cache hit"))
    assert extracted_checkpoint(source, tmp_path / "cache") == first


def test_new_checkpoint_gets_new_cache(tmp_path):
    source = archive(tmp_path / "model.nemo")
    first = extracted_checkpoint(source, tmp_path / "cache")
    archive(source, b"updated weights")
    second = extracted_checkpoint(source, tmp_path / "cache")
    assert second != first
    assert (second / "model_weights.ckpt").read_bytes() == b"updated weights"


def test_incomplete_cache_is_rebuilt(tmp_path):
    source = archive(tmp_path / "model.nemo")
    first = extracted_checkpoint(source, tmp_path / "cache")
    (first / "tokenizer.model").unlink()
    restored = extracted_checkpoint(source, tmp_path / "cache")
    assert (restored / "tokenizer.model").read_bytes() == b"tokens"


def test_failed_extraction_can_be_retried(tmp_path):
    source = tmp_path / "broken.nemo"
    source.write_bytes(b"not a tar")
    with pytest.raises(tarfile.ReadError):
        extracted_checkpoint(source, tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*/model_config.yaml"))
    archive(source)
    assert (extracted_checkpoint(source, tmp_path / "cache") / "model_config.yaml").is_file()


def test_archive_cannot_escape_cache(tmp_path):
    source = tmp_path / "bad.nemo"
    with tarfile.open(source, "w") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(tarfile.TarError):
        extracted_checkpoint(source, tmp_path / "cache")
    assert not (tmp_path / "escape").exists()
