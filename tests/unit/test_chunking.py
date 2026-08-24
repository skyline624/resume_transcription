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


@pytest.mark.parametrize(
    ("duration_s", "chunk_length_s", "overlap_s"),
    [
        (0.0, 480.0, 15.0),  # duree nulle
        (-1.0, 480.0, 15.0),  # duree negative
        (900.0, 0.0, 15.0),  # fenetre de longueur nulle
        (900.0, -480.0, 15.0),  # fenetre de longueur negative
        (900.0, 480.0, -1.0),  # recouvrement negatif
        (900.0, 100.0, 100.0),  # recouvrement egal a la fenetre : pas nul
        (900.0, 100.0, 150.0),  # recouvrement superieur : pas negatif
    ],
)
def test_parametres_invalides_rejetes(duration_s, chunk_length_s, overlap_s):
    """Les quatre branches de validation de plan_windows, une a une."""
    with pytest.raises(ValueError):
        plan_windows(duration_s, chunk_length_s, overlap_s)


def test_recollage_multi_fenetres_restitue_toute_la_reunion():
    """Aller-retour complet sur 5 fenetres : le cas nominal d'une longue reunion.

    Deux fenetres suffisent a masquer une inversion de borne dans le calcul des
    frontieres, car bounds[0] et bounds[-1] y sont le meme element. Il faut au
    moins une fenetre INTERIEURE, bornee des deux cotes par des valeurs finies,
    pour attraper la faute : elle recevrait un intervalle vide et tout le centre
    de la reunion disparaitrait sans erreur.
    """
    duration_s = 2000.0
    wins = plan_windows(duration_s, 480.0, 15.0)
    assert len(wins) == 5
    assert sum(1 for w in wins if 0 < w.index < len(wins) - 1) == 3

    # Verite terrain : un mot de 0.6 s par seconde, en temps absolu.
    verite = [
        Word(f"mot{i}", float(i), i + 0.6) for i in range(int(duration_s) - 1)
    ]

    # Chaque fenetre ne rend que les mots qu'elle contient entierement, et les
    # horodate RELATIVEMENT a son propre debut, comme le fait le modele.
    per_window_words = [
        offset_words(
            [
                Word(m.text, m.start - w.start, m.end - w.start)
                for m in verite
                if m.start >= w.start and m.end <= w.end
            ],
            w.start,
        )
        for w in wins
    ]
    # Le recouvrement doit vraiment produire des doublons a eliminer.
    assert sum(len(mots) for mots in per_window_words) > len(verite)

    out = merge_windows(per_window_words, wins)
    assert [m.text for m in out] == [m.text for m in verite]
    assert [m.start for m in out] == pytest.approx([m.start for m in verite])
    assert [m.end for m in out] == pytest.approx([m.end for m in verite])


def test_mot_pile_sur_chacune_des_frontieres_interieures():
    """La convention semi-ouverte vaut pour toutes les frontieres, pas la seule."""
    wins = plan_windows(2000.0, 480.0, 15.0)
    frontieres = [(b.start + a.end) / 2.0 for a, b in zip(wins, wins[1:])]
    assert frontieres == [472.5, 937.5, 1402.5, 1867.5]

    for k, frontiere in enumerate(frontieres):
        # milieu == frontiere exactement (bornes multiples de 0.25)
        pile = Word("pile", frontiere - 0.25, frontiere + 0.25)
        per_window_words = [
            [pile] if i in (k, k + 1) else [] for i in range(len(wins))
        ]
        out = merge_windows(per_window_words, wins)
        assert [w.text for w in out] == ["pile"], f"frontiere {frontiere}"


def test_le_mot_est_attribue_par_son_milieu_pas_par_ses_bornes():
    """Un mot a cheval sur la frontiere suit son MILIEU (spec section 6).

    Compter les occurrences ne suffit pas a epingler la regle : une regle
    fondee sur w.start ou sur w.end garderait elle aussi chaque mot une fois.
    Chaque cas ci-dessous isole donc un mot qu'une seule fenetre a rendu, place
    de sorte que le milieu et la borne ne designent pas la meme fenetre.
    """
    wins = [Window(0, 0.0, 480.0), Window(1, 465.0, 900.0)]  # frontiere 472.5

    # Debut avant la frontiere mais milieu apres -> revient a la fenetre 1.
    a_droite = Word("droite", 472.0, 473.4)  # milieu 472.7
    assert merge_windows([[], [a_droite]], wins) == [a_droite]
    assert merge_windows([[a_droite], []], wins) == []

    # Fin apres la frontiere mais milieu avant -> revient a la fenetre 0.
    a_gauche = Word("gauche", 471.0, 472.6)  # milieu 471.8
    assert merge_windows([[a_gauche], []], wins) == [a_gauche]
    assert merge_windows([[], [a_gauche]], wins) == []


def test_merge_trie_une_fenetre_dont_les_mots_arrivent_desordonnes():
    """Le tri final est un filet reellement sollicite, pas un no-op.

    Rien dans le contrat public n'oblige l'appelant a fournir chaque liste deja
    ordonnee ; la sortie, elle, doit etre chronologique.
    """
    wins = [Window(0, 0.0, 480.0), Window(1, 465.0, 900.0)]
    w0 = [Word("b", 200.0, 201.0), Word("a", 10.0, 11.0)]
    w1 = [Word("d", 800.0, 801.0), Word("c", 600.0, 601.0)]
    out = merge_windows([w0, w1], wins)
    assert [w.text for w in out] == ["a", "b", "c", "d"]


def test_merge_trie_aussi_le_cas_a_une_seule_fenetre():
    """Une seule fenetre suit le meme chemin que n fenetres, tri compris."""
    wins = [Window(0, 0.0, 100.0)]
    desordre = [Word("b", 3.0, 4.0), Word("a", 1.0, 2.0)]
    out = merge_windows([desordre], wins)
    assert [w.text for w in out] == ["a", "b"]


def test_merge_rejette_des_fenetres_non_triees():
    """Sur des fenetres desordonnees les frontieres cessent d'etre monotones,
    ce qui produirait pertes et doublons silencieux plutot qu'une erreur."""
    desordre = [Window(1, 465.0, 900.0), Window(0, 0.0, 480.0)]
    with pytest.raises(ValueError):
        merge_windows([[], []], desordre)
