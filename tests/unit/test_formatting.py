from transcription_server.domain import Turn, Word
from transcription_server.formatting import (
    format_timestamp,
    to_dialogue,
    to_plain_text,
    to_srt,
    to_vtt,
)

TURNS = [
    Turn(
        speaker="SPEAKER_00",
        start=0.32,
        end=4.81,
        text="Bonjour a tous.",
        words=(Word("Bonjour", 0.32, 0.79),),
    ),
    Turn(
        speaker="SPEAKER_01",
        start=4.90,
        end=7.25,
        text="Merci de votre presence.",
        words=(Word("Merci", 4.90, 5.30),),
    ),
]


def test_format_timestamp_srt():
    assert format_timestamp(62.34, separator=",") == "00:01:02,340"


def test_format_timestamp_vtt():
    assert format_timestamp(62.34, separator=".") == "00:01:02.340"


def test_format_timestamp_heures():
    assert format_timestamp(3661.5, separator=",") == "01:01:01,500"


def test_format_timestamp_zero():
    assert format_timestamp(0.0, separator=",") == "00:00:00,000"


def test_format_timestamp_negatif_est_ramene_a_zero():
    assert format_timestamp(-1.0, separator=",") == "00:00:00,000"


def test_format_timestamp_separateur_par_defaut_est_la_virgule():
    assert format_timestamp(62.34) == "00:01:02,340"


def test_format_timestamp_arrondit_au_lieu_de_tronquer():
    assert format_timestamp(0.9999, separator=",") == "00:00:01,000"


def test_to_plain_text_joint_les_tours():
    assert to_plain_text(TURNS) == "Bonjour a tous. Merci de votre presence."


def test_to_plain_text_sur_liste_vide():
    assert to_plain_text([]) == ""


def test_to_srt_structure():
    out = to_srt(TURNS)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,320 --> 00:00:04,810"
    assert lines[2] == "SPEAKER_00: Bonjour a tous."
    assert lines[3] == ""
    assert lines[4] == "2"


def test_to_srt_sans_locuteur_n_ajoute_pas_de_prefixe():
    turns = [Turn(speaker=None, start=0.0, end=1.0, text="Seul.", words=())]
    assert to_srt(turns).splitlines()[2] == "Seul."


def test_to_vtt_commence_par_l_entete():
    out = to_vtt(TURNS)
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.320 --> 00:00:04.810" in out


def test_to_vtt_separe_les_cues_par_une_ligne_vide():
    assert to_vtt(TURNS).splitlines() == [
        "WEBVTT",
        "",
        "00:00:00.320 --> 00:00:04.810",
        "SPEAKER_00: Bonjour a tous.",
        "",
        "00:00:04.900 --> 00:00:07.250",
        "SPEAKER_01: Merci de votre presence.",
    ]


def test_srt_et_vtt_sur_liste_vide():
    assert to_srt([]) == ""
    assert to_vtt([]) == "WEBVTT\n\n"


def test_to_dialogue_format():
    out = to_dialogue(TURNS)
    assert out.splitlines() == [
        "[00:00:00.32] SPEAKER_00: Bonjour a tous.",
        "[00:00:04.90] SPEAKER_01: Merci de votre presence.",
    ]


def test_to_dialogue_sans_locuteur_utilise_un_marqueur():
    turns = [Turn(speaker=None, start=0.0, end=1.0, text="Seul.", words=())]
    assert to_dialogue(turns) == "[00:00:00.00] INCONNU: Seul."


def test_to_dialogue_arrondit_les_centiemes():
    turns = [Turn(speaker="SPEAKER_00", start=1.238, end=2.0, text="Vite.", words=())]
    assert to_dialogue(turns) == "[00:00:01.24] SPEAKER_00: Vite."


def test_to_dialogue_au_dela_d_une_heure():
    turns = [Turn(speaker="SPEAKER_00", start=3661.5, end=3662.0, text="Encore.", words=())]
    assert to_dialogue(turns) == "[01:01:01.50] SPEAKER_00: Encore."


def test_to_dialogue_sur_liste_vide():
    assert to_dialogue([]) == ""
