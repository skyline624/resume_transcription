"""Attribution des mots aux locuteurs et regroupement en tours de parole.

Aucune dependance lourde : ce module ne manipule que les types du domaine.
"""

from transcription_server.domain import SpeakerSegment, Turn, Word


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Duree de recouvrement entre deux intervalles, 0.0 s'ils sont disjoints."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(
    word: Word,
    segments: list[SpeakerSegment],
    previous: str | None,
) -> str | None:
    """Rend le locuteur d'un mot.

    Regle : recouvrement temporel maximal. A egalite stricte, le segment qui
    commence le plus tot l'emporte -- puis, a bornes identiques, celui dont le
    libelle de locuteur vient en premier --, afin que le resultat soit
    deterministe quel que soit l'ordre de la liste recue.

    Si aucun segment ne recouvre le mot -- liste vide comprise, qui en est le
    cas extreme --, celui-ci herite du locuteur precedent (continuite d'un tour
    de parole a travers un blanc de diarization), et vaut None s'il n'y a pas
    de precedent.
    """
    # Trier rend la regle de depart d'egalite independante de l'ordre d'arrivee
    # des segments. La cle inclut le locuteur pour rester totale : deux segments
    # de bornes identiques doivent eux aussi etre departages, sans quoi la
    # stabilite de sorted laisserait l'ordre d'arrivee decider.
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.speaker))

    best_speaker: str | None = None
    best_overlap = 0.0
    for seg in ordered:
        current_overlap = overlap(word.start, word.end, seg.start, seg.end)
        if current_overlap > best_overlap:
            best_overlap = current_overlap
            best_speaker = seg.speaker

    if best_speaker is not None:
        return best_speaker

    # Recouvrement nul partout. Un mot de duree nulle ne peut pas recouvrir
    # quoi que ce soit : on teste alors l'appartenance de son point.
    if word.start == word.end:
        for seg in ordered:
            if seg.start <= word.start <= seg.end:
                return seg.speaker

    return previous


def group_into_turns(
    words: list[Word],
    segments: list[SpeakerSegment],
    turn_gap_s: float = 1.0,
) -> list[Turn]:
    """Regroupe des mots horodates en tours de parole.

    Un nouveau tour demarre quand le locuteur change, ou quand le silence
    entre deux mots consecutifs depasse turn_gap_s.

    Precondition : words est trie par ordre chronologique croissant, ce que
    produit l'ASR. L'appelant doit garantir cet ordre. Sur une entree
    desordonnee les bornes d'un tour viennent quand meme de son premier et de
    son dernier mot, et les silences calcules deviennent negatifs donc ne
    coupent jamais : le resultat serait silencieusement faux.
    """
    if not words:
        return []

    turns: list[Turn] = []
    current: list[Word] = []
    current_speaker: str | None = None
    previous_speaker: str | None = None
    previous_end: float | None = None

    def flush() -> None:
        if not current:
            return
        turns.append(
            Turn(
                speaker=current_speaker,
                start=current[0].start,
                end=current[-1].end,
                # Le filtre ecarte les jetons vides : un moteur peut en
                # rendre, et sans lui deux mots encadrant un jeton vide se
                # retrouveraient separes par une double espace dans le
                # texte livre au client.
                text=" ".join(w.text for w in current if w.text),
                words=tuple(current),
            )
        )

    for word in words:
        speaker = assign_speaker(word, segments, previous_speaker)
        gap = 0.0 if previous_end is None else word.start - previous_end
        starts_new_turn = bool(current) and (
            speaker != current_speaker or gap > turn_gap_s
        )

        if starts_new_turn:
            flush()
            current = []

        if not current:
            current_speaker = speaker

        current.append(word)
        previous_speaker = speaker
        previous_end = word.end

    flush()
    return turns
