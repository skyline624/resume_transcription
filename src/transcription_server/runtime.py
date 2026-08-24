"""Gestion du peripherique de calcul et de la memoire GPU.

torch est importe a l'interieur des fonctions, jamais au niveau du module.
C'est ce qui permet d'importer ce fichier — et donc `app.py` — depuis le venv
de developpement Windows, ou torch n'est pas installe. Un test structurel
verifie cette propriete par analyse syntaxique.
"""

_DEVICES_VALIDES = ("cuda", "cpu")


class CudaUnavailableError(RuntimeError):
    """CUDA a ete demande mais n'est pas accessible.

    Distincte de `ValueError` : un peripherique inconnu est une faute de
    programmation, tandis qu'un GPU absent est une condition d'environnement
    que l'exploitant peut corriger. Les deux ne se rattrapent pas au meme
    endroit.
    """


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Valide le peripherique demande. Ne se rabat jamais silencieusement.

    Le repli automatique sur CPU serait le pire service a rendre : la
    transcription fonctionnerait, vingt fois plus lentement, et personne ne
    saurait pourquoi avant d'avoir chronometre une reunion entiere.
    """
    if requested not in _DEVICES_VALIDES:
        raise ValueError(
            f"Périphérique inconnu : {requested!r}. Attendu 'cuda' ou 'cpu'."
        )
    if requested == "cuda" and not cuda_available:
        raise CudaUnavailableError(
            "DEVICE=cuda a été demandé mais torch.cuda.is_available() est faux. "
            "Vérifiez que le conteneur tourne avec --gpus all et que le runtime "
            "nvidia est actif, ou posez DEVICE=cpu pour accepter une exécution "
            "nettement plus lente."
        )
    return requested


def cuda_available() -> bool:
    """Rend False plutot que de lever quand torch est absent.

    Le serveur doit pouvoir diagnostiquer son environnement, pas s'y casser :
    c'est `resolve_device` qui decide si l'absence est fatale.
    """
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def gpu_info() -> dict:
    """Nom du GPU et etat de la VRAM, en megaoctets. Vide s'il n'y a pas de GPU."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    libre, total = torch.cuda.mem_get_info()
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_total_mb": round(total / (1024 * 1024)),
        "vram_free_mb": round(libre / (1024 * 1024)),
    }


def torch_dtype(compute_type: str):
    """Traduit `COMPUTE_TYPE` en dtype torch."""
    import torch

    try:
        return {"float16": torch.float16, "float32": torch.float32}[compute_type]
    except KeyError:
        raise ValueError(
            f"Type de calcul inconnu : {compute_type!r}. "
            "Attendu 'float16' ou 'float32'."
        ) from None


def empty_cache() -> None:
    """Rend au pilote la VRAM mise en cache par l'allocateur de torch."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
