from pathlib import Path
import tomllib

import pytest
import yaml


ROOT = Path(__file__).parents[2]
pytestmark = pytest.mark.skipif(
    not (ROOT / "docker-compose.yml").exists(),
    reason="Ces assertions exigent le contexte complet du depot.",
)


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


def test_sdk_openai_est_une_dependance_de_developpement_uniquement():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "openai>=2,<3" in project["project"]["optional-dependencies"]["dev"]
    assert "openai>=2,<3" not in project["project"]["dependencies"]


def test_scenarios_gpu_disposent_d_un_timeout_dur():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "pytest-timeout>=2,<3" in project["project"]["optional-dependencies"]["dev"]


def test_script_de_benchmark_est_inclus_dans_l_image():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY scripts/ ./scripts/" in dockerfile


def test_telechargement_huggingface_evite_le_transport_xet_instable():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "HF_HUB_DISABLE_XET=1" in dockerfile


def test_frontend_est_construit_dans_une_etape_node_ephemere():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM node:24-alpine AS web-builder" in dockerfile
    assert "RUN npm test" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "COPY --from=web-builder /web/dist /app/web-dist" in dockerfile
    final_stage = dockerfile.split("FROM pytorch/", 1)[1]
    assert "apt-get install -y nodejs" not in final_stage
    assert "apt-get install -y npm" not in final_stage
