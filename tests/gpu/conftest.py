"""Garde-fous communs aux tests GPU.

Ces tests ne tournent que dans le conteneur : `pytest` sans argument les
deselectionne (`addopts = -m 'not gpu'`). Les lancer avec `pytest -m gpu`.
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def echantillon(nom: str) -> str:
    """Chemin d'un echantillon audio, ou skip s'il n'a pas encore ete produit.

    Les echantillons sont generes a la Task 16 : sans eux, ces tests doivent
    s'abstenir plutot que d'echouer sur un fichier manquant, ce qui masquerait
    un vrai defaut du moteur.
    """
    chemin = FIXTURES / nom
    if not chemin.exists():
        pytest.skip(f"échantillon absent : {chemin} (généré à la Task 16)")
    return str(chemin)
