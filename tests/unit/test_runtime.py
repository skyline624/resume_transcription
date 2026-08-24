"""Resolution du peripherique de calcul.

Ces tests tournent sans torch : `runtime.py` ne l'importe qu'a l'interieur de
ses fonctions, ce qui est precisement ce qu'ils verifient.
"""

import pytest

from transcription_server.runtime import (
    CudaUnavailableError,
    cuda_available,
    empty_cache,
    gpu_info,
    resolve_device,
)


def test_cuda_demande_et_disponible():
    assert resolve_device("cuda", cuda_available=True) == "cuda"


def test_cuda_demande_mais_indisponible_leve_une_erreur():
    """Aucun repli CPU silencieux : une transcription vingt fois plus lente
    doit etre un choix explicite, jamais une surprise a l'execution."""
    with pytest.raises(CudaUnavailableError) as excinfo:
        resolve_device("cuda", cuda_available=False)
    message = str(excinfo.value)
    assert "DEVICE=cpu" in message
    assert "--gpus all" in message


def test_cpu_demande_est_accepte_meme_avec_cuda():
    assert resolve_device("cpu", cuda_available=True) == "cpu"


def test_cpu_demande_sans_cuda():
    assert resolve_device("cpu", cuda_available=False) == "cpu"


@pytest.mark.parametrize("device", ["tpu", "CUDA", "gpu", "", "cuda:0"])
def test_device_inconnu_est_rejete(device):
    """La casse et les suffixes comptent : `Literal` dans Settings ne protege
    que la configuration, pas un appel direct."""
    with pytest.raises(ValueError):
        resolve_device(device, cuda_available=True)


def test_valueerror_n_est_pas_confondu_avec_cudaunavailable():
    """Un peripherique inconnu est une faute de programmation, pas une absence
    de GPU : les deux ne se rattrapent pas au meme endroit."""
    assert not issubclass(ValueError, CudaUnavailableError)
    assert issubclass(CudaUnavailableError, RuntimeError)


def test_le_module_n_importe_pas_torch_au_niveau_module():
    """Garde-fou structurel : un `import torch` en tete rendrait tout le paquet
    inimportable dans le venv de developpement Windows, ou torch est absent."""
    import ast
    import pathlib

    import transcription_server.runtime as runtime

    source = pathlib.Path(runtime.__file__).read_text(encoding="utf-8")
    arbre = ast.parse(source)
    for noeud in arbre.body:  # niveau module uniquement
        if isinstance(noeud, ast.Import):
            assert all(not a.name.startswith("torch") for a in noeud.names)
        if isinstance(noeud, ast.ImportFrom):
            assert not (noeud.module or "").startswith("torch")


def test_cuda_available_rend_faux_sans_torch():
    """Dans le venv de developpement torch est absent : la fonction doit rendre
    False plutot que lever, sinon le serveur ne pourrait pas diagnostiquer."""
    assert cuda_available() is False


def test_gpu_info_rend_un_dict_vide_sans_torch():
    assert gpu_info() == {}


def test_empty_cache_ne_leve_pas_sans_torch():
    empty_cache()
