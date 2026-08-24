"""Decoupage de l'audio long en fenetres, puis recollage des mots.

Aucune dependance lourde : on ne manipule ici que des bornes temporelles et
des Word.
"""

from dataclasses import dataclass, replace

from transcription_server.domain import Word


@dataclass(frozen=True)
class Window:
    """Une fenetre d'inference, en secondes absolues."""

    index: int
    start: float
    end: float


def plan_windows(
    duration_s: float,
    chunk_length_s: float,
    overlap_s: float,
) -> list[Window]:
    """Decoupe [0, duration_s] en fenetres se recouvrant de overlap_s."""
    if duration_s <= 0.0:
        raise ValueError("La duree doit etre strictement positive.")
    if chunk_length_s <= 0.0:
        raise ValueError("La longueur de fenetre doit etre strictement positive.")
    if overlap_s < 0.0:
        raise ValueError("Le recouvrement ne peut pas etre negatif.")
    if overlap_s >= chunk_length_s:
        raise ValueError(
            "Le recouvrement doit etre strictement inferieur a la longueur "
            "de fenetre, sinon la progression serait nulle ou negative."
        )

    if duration_s <= chunk_length_s:
        return [Window(index=0, start=0.0, end=duration_s)]

    step = chunk_length_s - overlap_s
    windows: list[Window] = []
    start = 0.0
    index = 0
    while start < duration_s:
        end = min(start + chunk_length_s, duration_s)
        windows.append(Window(index=index, start=start, end=end))
        if end >= duration_s:
            break
        start += step
        index += 1
    return windows


def offset_words(words: list[Word], delta_s: float) -> list[Word]:
    """Decale une liste de mots dans le temps, sans muter l'entree."""
    return [replace(w, start=w.start + delta_s, end=w.end + delta_s) for w in words]


def _boundaries(windows: list[Window]) -> list[float]:
    """Points de bascule entre fenetres consecutives.

    La frontiere entre la fenetre i et i+1 est le milieu de leur zone de
    recouvrement, soit (windows[i+1].start + windows[i].end) / 2.
    """
    return [(b.start + a.end) / 2.0 for a, b in zip(windows, windows[1:])]


def merge_windows(
    per_window_words: list[list[Word]],
    windows: list[Window],
) -> list[Word]:
    """Recolle les mots produits par chaque fenetre.

    Un mot est attribue a la fenetre dont l'intervalle de frontieres contient
    le milieu du mot. Chaque mot est donc conserve exactement une fois : ni
    doublon dans la zone de recouvrement, ni troncature au raccord.
    """
    if len(per_window_words) != len(windows):
        raise ValueError(
            f"{len(per_window_words)} listes de mots pour {len(windows)} fenetres."
        )
    if not windows:
        return []
    if len(windows) == 1:
        return list(per_window_words[0])

    bounds = _boundaries(windows)
    merged: list[Word] = []
    for i, words in enumerate(per_window_words):
        lower = float("-inf") if i == 0 else bounds[i - 1]
        upper = float("inf") if i == len(windows) - 1 else bounds[i]
        for w in words:
            midpoint = (w.start + w.end) / 2.0
            if lower <= midpoint < upper:
                merged.append(w)
    merged.sort(key=lambda w: (w.start, w.end))
    return merged
