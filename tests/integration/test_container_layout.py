from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]


def test_compose_publie_uniquement_api_principale_et_persiste_les_voix():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["transcription"]
    assert service["ports"] == ["127.0.0.1:8000:8000"]
    assert "tts-voices:/app/voices" in service["volumes"]
    assert compose["volumes"] == {"tts-voices": None}


def test_qwen_possede_un_fichier_de_versions_independant():
    requirements = (ROOT / "docker" / "requirements-qwen.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert "qwen-tts==0.1.1" in requirements
    assert "transformers==4.57.3" in requirements
    assert "accelerate==1.12.0" in requirements


def test_image_installe_le_support_venv_de_python_312():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "python3.12-venv" in dockerfile


def test_image_installe_le_binaire_sox_requis_par_qwen():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "        sox \\\n" in dockerfile
