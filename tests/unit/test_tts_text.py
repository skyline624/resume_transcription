import pytest

from transcription_server.tts.text import normalize_text, segment_text


def test_normalisation_preserve_accents_nombres_et_ponctuation():
    assert normalize_text("  Échéance\u00a0: 12,5 %  ") == "Échéance : 12,5 %"


def test_segmentation_prefere_les_frontieres_de_phrase():
    text = "Bonjour à tous. Voici le second point ! Et la conclusion ?"
    segments = segment_text(text, max_chars=24)
    assert " ".join(item.text for item in segments) == text
    assert [item.pause_after_ms for item in segments] == [350, 450, 450]


def test_phrase_longue_se_coupe_d_abord_sur_une_proposition():
    segments = segment_text(
        "Une première idée assez longue ; une seconde idée utile.", max_chars=33
    )
    assert [item.text for item in segments] == [
        "Une première idée assez longue ;",
        "une seconde idée utile.",
    ]
    assert all(len(item.text) <= 33 for item in segments)


def test_dernier_recours_coupe_sur_un_espace_sans_perdre_de_mot():
    text = "alpha beta gamma delta epsilon"
    segments = segment_text(text, max_chars=12)
    assert " ".join(item.text for item in segments) == text
    assert all(len(item.text) <= 12 for item in segments)


def test_entree_vide_est_refusee():
    with pytest.raises(ValueError, match="vide"):
        segment_text(" \n\t ", max_chars=20)


def test_limite_non_positive_est_refusee():
    with pytest.raises(ValueError, match="strictement positive"):
        segment_text("Bonjour", max_chars=0)
