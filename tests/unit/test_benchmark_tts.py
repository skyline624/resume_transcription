import json
from pathlib import Path

import pytest

from scripts.benchmark_tts import (
    current_commit,
    load_corpus,
    normalize_text,
    word_error_rate,
)


ROOT = Path(__file__).parents[2]


def test_normalisation_est_commune_aux_accents_et_a_la_ponctuation():
    assert normalize_text("Élodie paie 12,50 € !") == "elodie paie 12 50"


def test_wer_compte_les_substitutions_insertions_et_suppressions():
    assert word_error_rate("un deux trois", "un quatre trois cinq") == pytest.approx(
        2 / 3
    )
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "mot") == 1.0


def test_corpus_francais_est_fixe_complet_et_borne():
    path = ROOT / "tests" / "fixtures" / "tts_corpus_fr.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    corpus = load_corpus(path)

    assert len(raw) == len(corpus) == 20
    assert len({item.id for item in corpus}) == 20
    assert all(item.id and item.category and item.text for item in corpus)
    assert max(map(lambda item: len(item.text), corpus)) <= 4096
    assert max(map(lambda item: len(item.text), corpus)) >= 3500
    categories = {item.category for item in corpus}
    assert {
        "nombres",
        "dates",
        "devises",
        "pourcentages",
        "sigles",
        "noms_propres",
        "questions",
        "exclamations",
        "parentheses",
        "discours_direct",
        "emotion",
        "long",
    } <= categories


def test_commit_explicite_n_exige_pas_de_depot_git(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("git ne doit pas etre appele")

    monkeypatch.setattr("scripts.benchmark_tts.subprocess.run", forbidden)
    assert current_commit("abc123") == "abc123"
