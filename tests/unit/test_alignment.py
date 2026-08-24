import pytest

from transcription_server.alignment import assign_speaker, group_into_turns, overlap
from transcription_server.domain import SpeakerSegment, Word

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"


def test_overlap_partiel():
    assert overlap(1.0, 3.0, 2.0, 5.0) == pytest.approx(1.0)


def test_overlap_nul_si_disjoint():
    assert overlap(1.0, 2.0, 3.0, 4.0) == 0.0


def test_overlap_nul_si_contigu():
    assert overlap(1.0, 2.0, 2.0, 3.0) == 0.0


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


def test_aucun_segment_donne_none():
    assert assign_speaker(Word("seul", 0.0, 1.0), [], None) is None


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
