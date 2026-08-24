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
        index += 1
        # Forme indexee plutot que start += step : aucune accumulation d'erreur
        # flottante d'une fenetre a la suivante.
        start = index * step
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

    Precondition : les mots de per_window_words[i] sont deja en temps ABSOLU.
    Cette fonction ne recale rien -- windows ne sert qu'a calculer les
    frontieres. C'est a l'appelant d'appliquer offset_words(mots,
    windows[i].start) au prealable. Des mots restes en temps relatif se
    masseraient tous dans l'intervalle de la premiere fenetre et la fin de
    l'enregistrement disparaitrait sans la moindre erreur.

    Un mot est attribue a la fenetre dont l'intervalle de frontieres contient
    le milieu du mot. Chaque mot est ainsi conserve exactement une fois, mais
    a une condition : que les deux fenetres du recouvrement horodatent ce mot
    de facon identique. Elles transcrivent la zone commune independamment, donc
    une gigue de quelques dizaines de millisecondes pres d'une frontiere suffit
    a faire tomber le milieu du mauvais cote dans les deux fenetres (mot perdu)
    ou du bon cote dans les deux (mot duplique). L'aval ne doit donc pas
    traiter le "exactement une fois" comme un invariant absolu.
    """
    if len(per_window_words) != len(windows):
        raise ValueError(
            f"{len(per_window_words)} listes de mots pour {len(windows)} fenetres."
        )
    if any(b.start < a.start for a, b in zip(windows, windows[1:])):
        raise ValueError(
            "Les fenetres doivent etre triees par start croissant : sinon les "
            "frontieres ne sont plus monotones et le recollage perd et duplique "
            "des mots sans rien signaler."
        )
    if not windows:
        return []

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
