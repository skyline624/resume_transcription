"""Contrat entre les commandes du worker et le moteur CUDA Graphs."""
import pytest

from qwen_tts_worker.domain import GenerateCommand, Mode
from qwen_tts_worker.model_manager import QwenModelAdapter


@pytest.mark.parametrize("mode,method,options", [
    (Mode.CUSTOM, "generate_custom_voice", {"speaker": "Ryan", "instruct": None}),
    (Mode.DESIGN, "generate_voice_design", {"instruct": "Voix calme"}),
    (Mode.CLONE, "generate_voice_clone", {
        "ref_audio": "/app/voices/a.wav", "ref_text": "Bonjour", "xvec_only": False,
    }),
])
def test_modes_cuda_graphs_et_flux(mode, method, options):
    calls = []
    class Model:
        def __getattr__(self, name):
            def generate(**kwargs):
                calls.append((name, kwargs))
                if name.endswith("_streaming"):
                    return (row for row in [([0.1], 24000, {}), ([0.2], 24000, {})])
                return [[0.1, 0.2]], 24000
            return generate
    command = GenerateCommand(
        mode=mode, text="Bonjour", speaker="Ryan" if mode is Mode.CUSTOM else None,
        instruct="Voix calme" if mode is Mode.DESIGN else None,
        reference_audio="/app/voices/a.wav" if mode is Mode.CLONE else None,
        reference_text="Bonjour" if mode is Mode.CLONE else None,
    )
    adapter = QwenModelAdapter(Model(), mode, backend="cuda_graphs", chunk_size=8)
    assert adapter.generate(command) == ([0.1, 0.2], 24000)
    assert list(adapter.stream(command)) == [([0.1], 24000), ([0.2], 24000)]
    assert calls == [
        (method, {"text": "Bonjour", "language": "French", **options}),
        (method + "_streaming", {"text": "Bonjour", "language": "French", **options, "chunk_size": 8}),
    ]
