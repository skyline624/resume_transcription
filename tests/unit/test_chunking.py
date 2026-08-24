import pytest

from transcription_server.chunking import (
    Window,
    merge_windows,
    offset_words,
    plan_windows,
)
from transcription_server.domain import Word


def test_audio_plus_court_qu_une_fenetre_donne_une_seule_fenetre():
    wins = plan_windows(duration_s=100.0, chunk_length_s=480.0, overlap_s=15.0)
    assert len(wins) == 1
    assert wins[0] == Window(index=0, start=0.0, end=100.0)


def test_audio_exactement_egal_a_une_fenetre():
    wins = plan_windows(duration_s=480.0, chunk_length_s=480.0, overlap_s=15.0)
    assert len(wins) == 1


def test_deux_fenetres_avec_recouvrement():
    wins = plan_windows(duration_s=900.0, chunk_length_s=480.0, overlap_s=15.0)
    assert len(wins) == 2
    assert wins[0].start == pytest.approx(0.0)
    assert wins[0].end == pytest.approx(480.0)
    # pas de 465 = 480 - 15
    assert wins[1].start == pytest.approx(465.0)
    assert wins[1].end == pytest.approx(900.0)


def test_les_fenetres_couvrent_tout_l_audio():
    wins = plan_windows(duration_s=2000.0, chunk_length_s=480.0, overlap_s=15.0)
    assert wins[0].start == 0.0
    assert wins[-1].end == pytest.approx(2000.0)
    for a, b in zip(wins, wins[1:]):
        assert b.start < a.end  # recouvrement effectif


def test_overlap_superieur_au_chunk_est_rejete():
    with pytest.raises(ValueError):
        plan_windows(duration_s=900.0, chunk_length_s=100.0, overlap_s=100.0)


def test_duree_nulle_est_rejetee():
    with pytest.raises(ValueError):
        plan_windows(duration_s=0.0, chunk_length_s=480.0, overlap_s=15.0)


def test_offset_words_decale_les_timestamps():
    words = [Word("a", 1.0, 2.0), Word("b", 3.0, 4.0)]
    out = offset_words(words, 10.0)
    assert [(w.start, w.end) for w in out] == [(11.0, 12.0), (13.0, 14.0)]
    assert [w.text for w in out] == ["a", "b"]


def test_offset_words_ne_modifie_pas_l_entree():
    words = [Word("a", 1.0, 2.0)]
    offset_words(words, 10.0)
    assert words[0].start == 1.0


def test_merge_une_seule_fenetre_rend_tout():
    wins = [Window(0, 0.0, 100.0)]
    words = [[Word("a", 1.0, 2.0), Word("b", 3.0, 4.0)]]
    assert merge_windows(words, wins) == words[0]


def test_merge_supprime_les_doublons_du_recouvrement():
    # Fenetres [0,480] et [465,900] -> frontiere au milieu de [465,480] = 472.5
    wins = [Window(0, 0.0, 480.0), Window(1, 465.0, 900.0)]
    w0 = [Word("avant", 400.0, 400.5), Word("commun", 470.0, 470.5)]
    w1 = [Word("commun", 470.0, 470.5), Word("apres", 500.0, 500.5)]
    out = merge_windows([w0, w1], wins)
    assert [w.text for w in out] == ["avant", "commun", "apres"]


def test_merge_mot_chevauchant_exactement_la_frontiere():
    # Frontiere a 472.5 ; un mot centre pile dessus va a la fenetre suivante.
    wins = [Window(0, 0.0, 480.0), Window(1, 465.0, 900.0)]
    pile = Word("pile", 472.0, 473.0)  # milieu = 472.5
    out = merge_windows([[pile], [pile]], wins)
    assert [w.text for w in out] == ["pile"]


def test_merge_conserve_l_ordre_chronologique():
    wins = [Window(0, 0.0, 480.0), Window(1, 465.0, 900.0)]
    w0 = [Word("a", 10.0, 11.0), Word("b", 200.0, 201.0)]
    w1 = [Word("c", 600.0, 601.0), Word("d", 800.0, 801.0)]
    out = merge_windows([w0, w1], wins)
    assert [w.text for w in out] == ["a", "b", "c", "d"]


def test_merge_rejette_un_desaccord_de_longueur():
    with pytest.raises(ValueError):
        merge_windows([[]], [Window(0, 0.0, 1.0), Window(1, 1.0, 2.0)])
