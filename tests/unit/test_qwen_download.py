from qwen_tts_worker.download import DEFAULT_PRELOAD_MODELS, preload_model_ids


def test_les_trois_checkpoints_sont_precharges_sans_variable(monkeypatch):
    monkeypatch.delenv("TTS_PRELOAD_MODELS", raising=False)
    assert preload_model_ids() == DEFAULT_PRELOAD_MODELS
    assert DEFAULT_PRELOAD_MODELS == (
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )


def test_liste_de_prechargement_reste_configurable(monkeypatch):
    monkeypatch.setenv("TTS_PRELOAD_MODELS", " model/a,model/b ,, ")
    assert preload_model_ids() == ("model/a", "model/b")
