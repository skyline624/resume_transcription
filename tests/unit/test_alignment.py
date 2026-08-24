import itertools

import pytest

from transcription_server.alignment import assign_speaker, group_into_turns, overlap
from transcription_server.domain import SpeakerSegment, Word

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"
S2 = "SPEAKER_02"


def test_overlap_partiel():
    assert overlap(1.0, 3.0, 2.0, 5.0) == pytest.approx(1.0)


def test_overlap_nul_si_disjoint():
    assert overlap(1.0, 2.0, 3.0, 4.0) == 0.0


def test_overlap_nul_si_contigu():
    assert overlap(1.0, 2.0, 2.0, 3.0) == 0.0


def test_overlap_inclusion_totale_rend_la_duree_du_plus_petit():
    assert overlap(1.0, 5.0, 2.0, 3.0) == pytest.approx(1.0)


def test_mot_entierement_dans_un_segment():
    segs = [SpeakerSegment(S0, 0.0, 10.0)]
    w = Word("bonjour", 1.0, 2.0)
    assert assign_speaker(w, segs, None) == S0


def test_mot_a_cheval_le_recouvrement_maximal_gagne():
    # 0.3 s sur S0, 0.7 s sur S1 -> S1
    segs = [SpeakerSegment(S0, 0.0, 1.3), SpeakerSegment(S1, 1.3, 5.0)]
    w = Word("cheval", 1.0, 2.0)
    assert assign_speaker(w, segs, None) == S1


def test_egalite_stricte_le_segment_le_plus_precoce_gagne():
    # 0.5 s de chaque cote -> le segment qui commence le plus tot
    segs = [SpeakerSegment(S1, 1.5, 5.0), SpeakerSegment(S0, 0.0, 1.5)]
    w = Word("pile", 1.0, 2.0)
    assert assign_speaker(w, segs, None) == S0


# Trois segments a 0.5 s de recouvrement chacun avec le mot teste, dont deux de
# bornes strictement identiques : seule une cle de tri totale les departage.
_SEGMENTS_A_EGALITE = (
    SpeakerSegment(S1, 1.0, 1.5),
    SpeakerSegment(S0, 1.0, 1.5),
    SpeakerSegment(S2, 2.0, 2.5),
)


@pytest.mark.parametrize("ordre", list(itertools.permutations(_SEGMENTS_A_EGALITE)))
def test_egalite_stricte_le_resultat_ne_depend_pas_de_l_ordre(ordre):
    w = Word("pile", 0.0, 3.0)
    assert assign_speaker(w, list(ordre), None) == S0


def test_mot_dans_un_silence_herite_du_precedent():
    segs = [SpeakerSegment(S0, 0.0, 1.0)]
    w = Word("apres", 5.0, 5.5)
    assert assign_speaker(w, segs, previous=S0) == S0


def test_mot_dans_un_silence_sans_precedent_donne_none():
    segs = [SpeakerSegment(S0, 10.0, 12.0)]
    w = Word("avant", 1.0, 1.5)
    assert assign_speaker(w, segs, previous=None) is None


def test_mot_de_duree_nulle_teste_l_appartenance_du_point():
    segs = [SpeakerSegment(S0, 0.0, 5.0)]
    w = Word("point", 2.0, 2.0)
    assert assign_speaker(w, segs, None) == S0


def test_appartenance_du_point_prime_sur_le_precedent():
    segs = [SpeakerSegment(S0, 0.0, 5.0)]
    w = Word("point", 2.0, 2.0)
    assert assign_speaker(w, segs, previous=S1) == S0


def test_aucun_segment_donne_none():
    assert assign_speaker(Word("seul", 0.0, 1.0), [], None) is None


def test_aucun_segment_herite_du_precedent():
    # Une liste vide est le cas extreme de "ne recouvre rien" : la regle
    # d'heritage s'applique comme pour un mot tombe dans un blanc.
    assert assign_speaker(Word("seul", 0.0, 1.0), [], previous=S0) == S0


def test_group_into_turns_change_de_tour_au_changement_de_locuteur():
    segs = [SpeakerSegment(S0, 0.0, 2.0), SpeakerSegment(S1, 2.0, 4.0)]
    words = [
        Word("bonjour", 0.1, 0.6),
        Word("tous", 0.7, 1.2),
        Word("merci", 2.1, 2.6),
    ]
    turns = group_into_turns(words, segs, turn_gap_s=1.0)
    assert [t.speaker for t in turns] == [S0, S1]
    assert turns[0].text == "bonjour tous"
    assert turns[1].text == "merci"
    assert turns[0].start == pytest.approx(0.1)
    assert turns[0].end == pytest.approx(1.2)


def test_group_into_turns_coupe_sur_un_silence_long():
    segs = [SpeakerSegment(S0, 0.0, 20.0)]
    words = [
        Word("un", 0.0, 0.5),
        Word("deux", 0.6, 1.0),
        Word("trois", 9.0, 9.5),  # silence de 8 s
    ]
    turns = group_into_turns(words, segs, turn_gap_s=1.0)
    assert len(turns) == 2
    assert turns[0].text == "un deux"
    assert turns[1].text == "trois"
    assert turns[0].speaker == S0
    assert turns[1].speaker == S0


def test_group_into_turns_ne_coupe_pas_sur_un_silence_egal_au_seuil():
    # Le seuil doit etre depasse strictement : 1.0 s de silence pour
    # turn_gap_s=1.0 laisse les deux mots dans le meme tour.
    segs = [SpeakerSegment(S0, 0.0, 20.0)]
    words = [Word("un", 0.0, 1.0), Word("deux", 2.0, 3.0)]
    turns = group_into_turns(words, segs, turn_gap_s=1.0)
    assert len(turns) == 1
    assert turns[0].text == "un deux"


def test_group_into_turns_isole_un_mot_anterieur_a_toute_diarization():
    # Le silence est court : c'est le passage de None a S0 qui coupe le tour.
    segs = [SpeakerSegment(S0, 5.0, 9.0)]
    words = [Word("avant", 4.5, 4.9), Word("dedans", 5.1, 5.5)]
    turns = group_into_turns(words, segs, turn_gap_s=1.0)
    assert [(t.speaker, t.text) for t in turns] == [(None, "avant"), (S0, "dedans")]


def test_group_into_turns_sans_segments_produit_un_tour_sans_locuteur():
    words = [Word("un", 0.0, 0.5), Word("deux", 0.6, 1.0)]
    turns = group_into_turns(words, [], turn_gap_s=1.0)
    assert len(turns) == 1
    assert turns[0].speaker is None
    assert turns[0].text == "un deux"


def test_group_into_turns_sur_liste_vide():
    assert group_into_turns([], [], turn_gap_s=1.0) == []


def test_group_into_turns_conserve_les_mots():
    segs = [SpeakerSegment(S0, 0.0, 5.0)]
    words = [Word("un", 0.0, 0.5), Word("deux", 0.6, 1.0)]
    turns = group_into_turns(words, segs, turn_gap_s=1.0)
    assert turns[0].words == tuple(words)


def test_group_into_turns_ignore_les_jetons_vides_dans_le_texte():
    """Un jeton vide au milieu d'un tour ne doit pas laisser une double espace.

    Il reste dans words -- c'est le rendu texte, et lui seul, qui l'ecarte.
    """
    words = [Word("bonjour", 0.0, 0.4), Word("", 0.5, 0.6), Word("tous", 0.7, 0.9)]
    turns = group_into_turns(words, [], turn_gap_s=1.0)
    assert len(turns) == 1
    assert turns[0].text == "bonjour tous"
    assert turns[0].words == tuple(words)
