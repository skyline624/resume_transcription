"""Fige les versions de torch et torchaudio deja presentes dans l'image.

Ecrit un fichier de contraintes pip. Sans lui, l'installation de NeMo et de
pyannote peut remplacer le torch CUDA de l'image de base par une roue CPU pour
satisfaire une borne de version — le serveur demarrerait alors sans GPU, et
rien au build ne le signalerait.
"""

import sys

import torch
import torchaudio

destination = sys.argv[1]
lignes = [
    f"torch=={torch.__version__.split('+')[0]}",
    f"torchaudio=={torchaudio.__version__.split('+')[0]}",
]
with open(destination, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lignes) + "\n")

print("Contraintes figées :")
for ligne in lignes:
    print(f"  {ligne}")
