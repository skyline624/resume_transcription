# Serveur de transcription Parakeet — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Livrer un serveur HTTP conteneurisé qui transcrit un fichier audio avec NVIDIA Parakeet sur GPU CUDA et sépare les tours de parole par locuteur via pyannote.

**Architecture:** Serveur FastAPI mono-processus. Toute la logique métier (alignement mots/locuteurs, découpage et recollage des fenêtres, formatage) est pure et sans dépendance à torch, donc testable sur Windows sans GPU ni Docker. Les moteurs lourds sont derrière deux `Protocol`, ce qui permet de tester les routes avec des implémentations factices. Seuls les deux adaptateurs concrets (NeMo, pyannote) et le déploiement exigent le conteneur CUDA.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, NVIDIA NeMo 3.0.0, pyannote.audio 4.x, PyTorch 2.11 / CUDA 12.8, Docker Compose, ffmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-serveur-transcription-parakeet-design.md`

## Global Constraints

Ces contraintes s'appliquent à **toutes** les tâches. Valeurs reprises telles quelles de la spec.

- **Python 3.12**. Image de base : `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`.
- **Séparation stricte des dépendances** : `alignment.py`, `chunking.py`, `formatting.py`, `domain.py` n'importent **ni torch, ni NeMo, ni pyannote, ni numpy**. Toute violation casse la stratégie de test.
- **Extras du `pyproject.toml`** : les dépendances de base sont légères (FastAPI, numpy, pydantic) ; NeMo et pyannote vivent dans l'extra `gpu`, installé **uniquement dans l'image Docker**.
- **Modèles** : ASR `nvidia/parakeet-tdt-0.6b-v3`, diarization `pyannote/speaker-diarization-community-1`. Versions épinglées : `nemo_toolkit[asr]==3.0.0`, `pyannote.audio>=4.0,<5.0`.
- **Timestamps** : toujours en **secondes flottantes absolues** depuis le début du fichier. Aucune fonction ne rend de timestamp relatif à une fenêtre hors de `chunking.py`.
- **Défauts de configuration** : `CHUNK_LENGTH_S=480`, `CHUNK_OVERLAP_S=15`, `TURN_GAP_S=1.0`, `DEVICE=cuda`, `COMPUTE_TYPE=float16`, `MAX_UPLOAD_MB=1024`, `HOST=0.0.0.0`, `PORT=8000`.
- **Aucun repli CPU silencieux** : si `DEVICE=cuda` et CUDA est indisponible, le démarrage échoue.
- **Étiquettes de locuteurs** : `SPEAKER_00`, `SPEAKER_01`, … Jamais de noms réels.
- **`.env` sans BOM**, jamais commité, jamais inscrit dans une couche d'image.
- **Publication réseau** : `127.0.0.1:8000` côté hôte uniquement.
- **Ne jamais sérialiser une `ValidationError` pydantic.** Mesuré en Task 5 :
  `str(e)` tronque à 11 caractères, mais **`e.errors()` et `e.json()`
  contiennent le `HF_TOKEN` intégral**, et ni `SecretStr` ni
  `Field(repr=False)` n'y changent quoi que ce soit — un
  `model_validator(mode="after")` attache le dictionnaire d'entrée brut.
  Conséquence pour les tâches 9, 10 et 14 : l'idiome FastAPI courant
  `HTTPException(422, detail=e.errors())` **divulguerait le token dans une
  réponse HTTP**. N'utiliser que `e.errors(include_input=False)`, et ne jamais
  écrire `.errors()` ni `.json()` dans un log ou un corps de réponse.
  Même interdiction pour **`settings.model_dump()` et `model_dump_json()`**,
  qui exposent eux aussi le token intégral. En revanche `repr(settings)`,
  `str(settings)` et les f-strings sont sûrs depuis la Task 5
  (`Field(repr=False)` sur `hf_token`, vérifié sous `pytest --showlocals`).
  **Le masquage protège l'objet, pas ses copies** : mesuré en Task 5, un
  `jeton = settings.hf_token` réexpose la valeur sous `--showlocals`. Dans les
  tâches 9, 10, 13 et 14, ne jamais extraire le token dans une variable locale
  ni le passer à une fonction susceptible d'apparaître dans une trace —
  transmettre `settings` et lire l'attribut au point d'usage.
- **Français accentué** dans toute chaîne destinée à un humain, messages
  d'erreur compris.

### Prérequis avant la Task 1

Le dépôt n'est pas encore sous git. **Demander l'accord de l'utilisateur** avant
`git init`. Si l'accord est refusé, ignorer toutes les étapes « Commit » du plan
et poursuivre le reste à l'identique.

Le fichier `.env` existe déjà à la racine et contient un `HF_TOKEN` valide de
type *read*, dont l'accès au dépôt *gated* a été vérifié le 2026-08-24. **Ne
jamais l'écraser** : les tâches ne créent que `.env.example`.

## File Structure

| Fichier | Responsabilité | Dépendances lourdes |
|---|---|---|
| `src/transcription_server/domain.py` | Types `Word`, `SpeakerSegment`, `Turn` | aucune |
| `src/transcription_server/alignment.py` | Attribution mot → locuteur, regroupement en tours | aucune |
| `src/transcription_server/chunking.py` | Découpage en fenêtres, réoffset, recollage | aucune |
| `src/transcription_server/formatting.py` | SRT, VTT, dialogue, texte brut | aucune |
| `src/transcription_server/config.py` | `Settings` pydantic-settings | pydantic |
| `src/transcription_server/audio.py` | Décodage ffmpeg → PCM float32 | numpy |
| `src/transcription_server/asr/engine.py` | `Protocol AsrEngine` | aucune |
| `src/transcription_server/diarization/engine.py` | `Protocol DiarizationEngine` | aucune |
| `src/transcription_server/pipeline.py` | Orchestration décodage → diarization → ASR → tours | numpy |
| `src/transcription_server/api/schemas.py` | Modèles de requête/réponse | pydantic |
| `src/transcription_server/api/native_routes.py` | `POST /transcribe`, `GET /health` | fastapi |
| `src/transcription_server/api/openai_routes.py` | `POST /v1/audio/transcriptions`, `GET /v1/models` | fastapi |
| `src/transcription_server/runtime.py` | Device, fp16, verrou GPU | torch |
| `src/transcription_server/asr/nemo_parakeet.py` | Adaptateur NeMo | torch, NeMo |
| `src/transcription_server/diarization/pyannote_engine.py` | Adaptateur pyannote | torch, pyannote |
| `src/transcription_server/app.py` | Application FastAPI, lifespan | fastapi |
| `docker/Dockerfile`, `docker/entrypoint.sh`, `docker-compose.yml` | Conteneurisation | — |

Les quatre premiers fichiers sont le cœur testable sans GPU. C'est là que vit la
complexité réelle du projet.

---

## Task 1: Socle du projet, environnement de test, types du domaine

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.dockerignore`, `.env.example`
- Create: `src/transcription_server/__init__.py`, `src/transcription_server/domain.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`
- Test: `tests/unit/test_domain.py`

**Interfaces:**
- Consumes: rien.
- Produces: `Word(text: str, start: float, end: float)`, `SpeakerSegment(speaker: str, start: float, end: float)`, `Turn(speaker: str | None, start: float, end: float, text: str, words: tuple[Word, ...])`. Toutes trois `@dataclass(frozen=True)`, avec une propriété `duration: float` sur `Word` et `SpeakerSegment`.

- [ ] **Step 1: Créer `pyproject.toml`**

```toml
[project]
name = "transcription-server"
version = "0.1.0"
description = "Serveur de transcription Parakeet avec diarization"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "python-multipart>=0.0.12",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "numpy>=1.26",
]

[project.optional-dependencies]
gpu = [
    "nemo_toolkit[asr]==3.0.0",
    "pyannote.audio>=4.0,<5.0",
]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/transcription_server"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: nécessite CUDA et les dépendances de l'extra gpu (désélectionné par défaut)",
]
addopts = "-m 'not gpu'"
asyncio_mode = "auto"
```

`addopts = "-m 'not gpu'"` fait que `pytest` sans argument ignore les tests GPU.
Dans le conteneur, on les lance avec `pytest -m gpu`.

- [ ] **Step 2: Créer `.gitignore`**

```gitignore
.env
models/
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 3: Créer `.dockerignore`**

```dockerignore
.venv/
models/
.git/
.pytest_cache/
__pycache__/
*.egg-info/
docs/
.env
```

`.env` est exclu du contexte de build : le token n'entre dans aucune couche.

- [ ] **Step 4: Créer `.env.example`**

```dotenv
# Token HuggingFace, type "read".
# Requis uniquement si ENABLE_DIARIZATION=true.
# Accepter au prealable les conditions de
# https://huggingface.co/pyannote/speaker-diarization-community-1
HF_TOKEN=

ASR_MODEL=nvidia/parakeet-tdt-0.6b-v3
DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
ENABLE_DIARIZATION=true

DEVICE=cuda
COMPUTE_TYPE=float16

CHUNK_LENGTH_S=480
CHUNK_OVERLAP_S=15
TURN_GAP_S=1.0

HOST=0.0.0.0
PORT=8000
MAX_UPLOAD_MB=1024
```

- [ ] **Step 5: Créer le venv de développement Windows**

```powershell
cd D:\developpement\resume_transcription
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Attendu : installation en quelques secondes, ~50 Mo. NeMo et pyannote ne sont
**pas** installés — c'est voulu.

- [ ] **Step 6: Écrire le test qui échoue**

Créer `tests/__init__.py` et `tests/unit/__init__.py` vides, puis
`tests/unit/test_domain.py` :

```python
import pytest

from transcription_server.domain import SpeakerSegment, Turn, Word


def test_word_duration():
    w = Word(text="bonjour", start=1.0, end=1.75)
    assert w.duration == pytest.approx(0.75)


def test_word_is_immutable():
    w = Word(text="bonjour", start=1.0, end=1.75)
    with pytest.raises(Exception):
        w.text = "autre"


def test_speaker_segment_duration():
    s = SpeakerSegment(speaker="SPEAKER_00", start=2.0, end=5.5)
    assert s.duration == pytest.approx(3.5)


def test_turn_holds_words_as_tuple():
    words = (Word(text="a", start=0.0, end=0.5), Word(text="b", start=0.5, end=1.0))
    t = Turn(speaker="SPEAKER_00", start=0.0, end=1.0, text="a b", words=words)
    assert t.words == words
    assert t.speaker == "SPEAKER_00"


def test_turn_accepts_none_speaker():
    t = Turn(speaker=None, start=0.0, end=1.0, text="a", words=())
    assert t.speaker is None
```

- [ ] **Step 7: Lancer le test pour vérifier qu'il échoue**

Run: `.venv\Scripts\pytest tests/unit/test_domain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.domain'`

- [ ] **Step 8: Écrire l'implémentation minimale**

Créer `src/transcription_server/__init__.py` vide, puis
`src/transcription_server/domain.py` :

```python
"""Types du domaine, sans aucune dependance lourde.

Ce module ne doit jamais importer torch, NeMo, pyannote ni numpy :
c'est ce qui rend la logique metier testable hors du conteneur.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    """Un mot transcrit, horodate en secondes absolues."""

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SpeakerSegment:
    """Un intervalle attribue a un locuteur par la diarization."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Turn:
    """Un tour de parole : des mots consecutifs d'un meme locuteur."""

    speaker: str | None
    start: float
    end: float
    text: str
    words: tuple[Word, ...]
```

- [ ] **Step 9: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_domain.py -v`
Expected: PASS — 5 tests

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore .dockerignore .env.example src/ tests/
git commit -m "feat: socle du projet et types du domaine"
```

---

## Task 2: Alignement mots ↔ locuteurs

C'est le module le plus subtil du projet. Tous les cas limites de la spec
(section 7) sont couverts par des tests avant toute implémentation.

**Files:**
- Create: `src/transcription_server/alignment.py`
- Test: `tests/unit/test_alignment.py`

**Interfaces:**
- Consumes: `Word`, `SpeakerSegment`, `Turn` de `transcription_server.domain`.
- Produces:
  - `overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float`
  - `assign_speaker(word: Word, segments: list[SpeakerSegment], previous: str | None) -> str | None`
  - `group_into_turns(words: list[Word], segments: list[SpeakerSegment], turn_gap_s: float = 1.0) -> list[Turn]`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_alignment.py` :

```python
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
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_alignment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.alignment'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `src/transcription_server/alignment.py` :

```python
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
    commence le plus tot l'emporte, afin que le resultat soit deterministe
    quel que soit l'ordre de la liste recue.

    Si aucun segment ne recouvre le mot, celui-ci herite du locuteur precedent
    (continuite d'un tour de parole a travers un blanc de diarization), et vaut
    None s'il n'y a pas de precedent.
    """
    # Pas de retour anticipe sur une liste vide : la spec §7 ne prevoit aucune
    # exception, un mot que rien ne recouvre herite de `previous`. La logique
    # generale ci-dessous gere deja ce cas.

    # La cle de tri doit etre TOTALE. Avec (start, end) seuls, deux segments de
    # bornes identiques portant des locuteurs differents ne sont pas departages :
    # sorted etant stable, le resultat dependrait de l'ordre d'arrivee, ce que la
    # regle de determinisme interdit.
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.speaker))

    best_speaker: str | None = None
    best_overlap = 0.0
    for seg in ordered:
        current = overlap(word.start, word.end, seg.start, seg.end)
        if current > best_overlap:
            best_overlap = current
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
                text=" ".join(w.text for w in current),
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
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_alignment.py -v`
Expected: PASS — 15 tests

- [ ] **Step 5: Commit**

```bash
git add src/transcription_server/alignment.py tests/unit/test_alignment.py
git commit -m "feat: alignement mots-locuteurs et regroupement en tours"
```

---

## Task 3: Découpage et recollage des fenêtres

**Files:**
- Create: `src/transcription_server/chunking.py`
- Test: `tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `Word` de `transcription_server.domain`.
- Produces:
  - `Window(index: int, start: float, end: float)` — `@dataclass(frozen=True)`
  - `plan_windows(duration_s: float, chunk_length_s: float, overlap_s: float) -> list[Window]`
  - `offset_words(words: list[Word], delta_s: float) -> list[Word]`
  - `merge_windows(per_window_words: list[list[Word]], windows: list[Window]) -> list[Word]`

**Règle de recollage** (spec section 6) : la frontière entre la fenêtre *i* et
*i+1* est le milieu de leur zone de recouvrement. Un mot appartient à la fenêtre
dont l'intervalle de frontières contient **le milieu du mot**. Cela garantit
qu'aucun mot n'est ni dupliqué ni perdu.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_chunking.py` :

```python
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
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.chunking'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `src/transcription_server/chunking.py` :

```python
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
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_chunking.py -v`
Expected: PASS — 13 tests

- [ ] **Step 5: Commit**

```bash
git add src/transcription_server/chunking.py tests/unit/test_chunking.py
git commit -m "feat: decoupage en fenetres et recollage des mots"
```

---

## Task 4: Formatage des sorties

**Files:**
- Create: `src/transcription_server/formatting.py`
- Test: `tests/unit/test_formatting.py`

**Interfaces:**
- Consumes: `Turn`, `Word` de `transcription_server.domain`.
- Produces:
  - `format_timestamp(seconds: float, separator: str = ",") -> str` → `"00:01:02,340"`
  - `to_plain_text(turns: list[Turn]) -> str`
  - `to_srt(turns: list[Turn]) -> str`
  - `to_vtt(turns: list[Turn]) -> str`
  - `to_dialogue(turns: list[Turn]) -> str`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_formatting.py` :

```python
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


def test_to_dialogue_format():
    out = to_dialogue(TURNS)
    assert out.splitlines() == [
        "[00:00:00.32] SPEAKER_00: Bonjour a tous.",
        "[00:00:04.90] SPEAKER_01: Merci de votre presence.",
    ]


def test_to_dialogue_sans_locuteur_utilise_un_marqueur():
    turns = [Turn(speaker=None, start=0.0, end=1.0, text="Seul.", words=())]
    assert to_dialogue(turns) == "[00:00:00.00] INCONNU: Seul."


def test_to_dialogue_sur_liste_vide():
    assert to_dialogue([]) == ""
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_formatting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.formatting'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `src/transcription_server/formatting.py` :

```python
"""Rendu des tours de parole dans les formats de sortie supportes."""

from transcription_server.domain import Turn

UNKNOWN_SPEAKER = "INCONNU"


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """Formate des secondes en HH:MM:SS<sep>mmm.

    SRT attend une virgule avant les millisecondes, WebVTT un point.
    """
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _short_timestamp(seconds: float) -> str:
    """HH:MM:SS.cc, avec deux decimales, pour le format dialogue."""
    total_cs = max(0, round(seconds * 100))
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cents = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{cents:02d}"


def _labelled(turn: Turn) -> str:
    if turn.speaker is None:
        return turn.text
    return f"{turn.speaker}: {turn.text}"


def to_plain_text(turns: list[Turn]) -> str:
    return " ".join(t.text for t in turns if t.text)


def to_srt(turns: list[Turn]) -> str:
    blocks: list[str] = []
    for i, turn in enumerate(turns, start=1):
        start = format_timestamp(turn.start, separator=",")
        end = format_timestamp(turn.end, separator=",")
        blocks.append(f"{i}\n{start} --> {end}\n{_labelled(turn)}\n")
    return "\n".join(blocks)


def to_vtt(turns: list[Turn]) -> str:
    blocks: list[str] = []
    for turn in turns:
        start = format_timestamp(turn.start, separator=".")
        end = format_timestamp(turn.end, separator=".")
        blocks.append(f"{start} --> {end}\n{_labelled(turn)}\n")
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_dialogue(turns: list[Turn]) -> str:
    lines = []
    for turn in turns:
        speaker = turn.speaker or UNKNOWN_SPEAKER
        lines.append(f"[{_short_timestamp(turn.start)}] {speaker}: {turn.text}")
    return "\n".join(lines)
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_formatting.py -v`
Expected: PASS — 13 tests

- [ ] **Step 5: Commit**

```bash
git add src/transcription_server/formatting.py tests/unit/test_formatting.py
git commit -m "feat: formatage srt, vtt, dialogue et texte brut"
```

---

## Task 5: Configuration

**Files:**
- Create: `src/transcription_server/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: rien.
- Produces: `Settings` (pydantic-settings) avec les champs `hf_token`, `asr_model`, `diarization_model`, `enable_diarization`, `device`, `compute_type`, `chunk_length_s`, `chunk_overlap_s`, `turn_gap_s`, `host`, `port`, `max_upload_mb`. Plus `max_upload_bytes: int` (propriété) et `get_settings() -> Settings` (mémoïsée par `lru_cache`).

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_config.py` :

```python
import pytest
from pydantic import ValidationError

from transcription_server.config import Settings


# Les defauts activent la diarization, qui exige un token. Les tests qui ne
# portent pas sur cette regle en fournissent donc un factice.
TOKEN = "hf_pour_les_tests"


def test_defauts_conformes_a_la_spec():
    s = Settings(_env_file=None, hf_token=TOKEN)
    assert s.asr_model == "nvidia/parakeet-tdt-0.6b-v3"
    assert s.diarization_model == "pyannote/speaker-diarization-community-1"
    assert s.enable_diarization is True
    assert s.device == "cuda"
    assert s.compute_type == "float16"
    assert s.chunk_length_s == 480.0
    assert s.chunk_overlap_s == 15.0
    assert s.turn_gap_s == 1.0
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.max_upload_mb == 1024


def test_max_upload_bytes():
    s = Settings(_env_file=None, hf_token=TOKEN, max_upload_mb=2)
    assert s.max_upload_bytes == 2 * 1024 * 1024


def test_device_invalide_est_rejete():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hf_token=TOKEN, device="tpu")


def test_compute_type_invalide_est_rejete():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hf_token=TOKEN, compute_type="int4")


def test_recouvrement_superieur_au_chunk_est_rejete():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            hf_token=TOKEN,
            chunk_length_s=100.0,
            chunk_overlap_s=100.0,
        )


def test_diarization_active_sans_token_est_rejetee():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, enable_diarization=True, hf_token=None)


def test_diarization_desactivee_sans_token_est_acceptee():
    s = Settings(_env_file=None, enable_diarization=False, hf_token=None)
    assert s.enable_diarization is False


def test_lecture_depuis_l_environnement(monkeypatch):
    monkeypatch.setenv("CHUNK_LENGTH_S", "120")
    monkeypatch.setenv("ENABLE_DIARIZATION", "false")
    s = Settings(_env_file=None)
    assert s.chunk_length_s == 120.0
    assert s.enable_diarization is False
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.config'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `src/transcription_server/config.py` :

```python
"""Configuration du serveur, lue depuis l'environnement et le fichier .env."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    hf_token: str | None = None

    asr_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    enable_diarization: bool = True

    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: Literal["float16", "float32"] = "float16"

    chunk_length_s: float = Field(default=480.0, gt=0)
    chunk_overlap_s: float = Field(default=15.0, ge=0)
    turn_gap_s: float = Field(default=1.0, ge=0)

    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)
    max_upload_mb: int = Field(default=1024, gt=0)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @model_validator(mode="after")
    def _check_coherence(self) -> "Settings":
        if self.chunk_overlap_s >= self.chunk_length_s:
            raise ValueError(
                "CHUNK_OVERLAP_S doit etre strictement inferieur a CHUNK_LENGTH_S."
            )
        if self.enable_diarization and not self.hf_token:
            raise ValueError(
                "ENABLE_DIARIZATION=true exige un HF_TOKEN. Creez un token de "
                "type read sur huggingface.co et acceptez les conditions de "
                "https://huggingface.co/pyannote/speaker-diarization-community-1 "
                "ou mettez ENABLE_DIARIZATION=false."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_config.py -v`
Expected: PASS — 8 tests

Note : ces tests passent `_env_file=None` pour ignorer le `.env` réel de la
machine, sans quoi le token présent le rendrait dépendant de l'environnement.

- [ ] **Step 5: Commit**

```bash
git add src/transcription_server/config.py tests/unit/test_config.py
git commit -m "feat: configuration validee par pydantic-settings"
```

---

## Task 6: Décodage audio par ffmpeg

**Files:**
- Create: `src/transcription_server/audio.py`
- Test: `tests/unit/test_audio.py`

**Interfaces:**
- Consumes: rien du projet.
- Produces:
  - `SAMPLE_RATE: int = 16000`
  - `AudioDecodeError(Exception)`
  - `decode_to_pcm(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray` — mono float32 dans [-1, 1]
  - `duration_seconds(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float`

Ces tests utilisent ffmpeg, présent sur la machine de développement (8.1.1) et
installé dans l'image. Ils n'exigent **pas** de GPU.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_audio.py` :

```python
import shutil
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from transcription_server.audio import (
    SAMPLE_RATE,
    AudioDecodeError,
    decode_to_pcm,
    duration_seconds,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg absent du PATH"
)


def _write_sine_wav(path: Path, seconds: float = 1.0, rate: int = 44100) -> None:
    """Ecrit un wav mono 16 bits contenant un sinus a 440 Hz."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        data = b"".join(
            struct.pack("<h", int(20000 * np.sin(2 * np.pi * 440 * i / rate)))
            for i in range(frames)
        )
        f.writeframes(data)


def test_decode_rend_du_float32_mono(tmp_path):
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=1.0)
    pcm = decode_to_pcm(src)
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1


def test_decode_reechantillonne_a_16k(tmp_path):
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=2.0, rate=44100)
    pcm = decode_to_pcm(src)
    assert len(pcm) == pytest.approx(2 * SAMPLE_RATE, rel=0.02)


def test_decode_normalise_entre_moins_un_et_un(tmp_path):
    src = tmp_path / "sine.wav"
    _write_sine_wav(src, seconds=0.5)
    pcm = decode_to_pcm(src)
    assert np.abs(pcm).max() <= 1.0
    assert np.abs(pcm).max() > 0.1  # le signal n'est pas nul


def test_duration_seconds():
    pcm = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    assert duration_seconds(pcm) == pytest.approx(3.0)


def test_fichier_inexistant_leve_audio_decode_error(tmp_path):
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(tmp_path / "absent.wav")


def test_fichier_non_audio_leve_audio_decode_error(tmp_path):
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"ceci n'est pas de l'audio")
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(junk)


def test_fichier_audio_vide_leve_audio_decode_error(tmp_path):
    src = tmp_path / "vide.wav"
    _write_sine_wav(src, seconds=0.0)
    with pytest.raises(AudioDecodeError):
        decode_to_pcm(src)
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_audio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.audio'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `src/transcription_server/audio.py` :

```python
"""Decodage de n'importe quel format audio vers du PCM mono 16 kHz.

On delegue a ffmpeg plutot qu'a une bibliotheque Python : cela couvre mp3,
m4a, ogg, flac, mp4, webm et le reste sans dependance supplementaire, et
c'est le meme binaire dans le conteneur et sur la machine de developpement.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioDecodeError(Exception):
    """L'audio n'a pas pu etre decode. Correspond a un HTTP 400."""


def decode_to_pcm(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode un fichier audio en float32 mono, normalise dans [-1, 1]."""
    source = Path(path)
    if not source.exists():
        raise AudioDecodeError(f"Fichier introuvable : {source}")
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("ffmpeg est introuvable dans le PATH.")

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-f", "f32le",       # float32 little-endian brut
        "-acodec", "pcm_f32le",
        "-ac", "1",          # mono
        "-ar", str(sample_rate),
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:  # pragma: no cover - defense en profondeur
        raise AudioDecodeError(f"Echec de l'appel a ffmpeg : {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(f"ffmpeg n'a pas pu decoder le fichier : {detail}")

    pcm = np.frombuffer(result.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise AudioDecodeError("Le fichier ne contient aucun echantillon audio.")

    # ffmpeg rend deja du float normalise, mais un fichier deja sature
    # pourrait depasser legerement les bornes apres reechantillonnage.
    peak = float(np.abs(pcm).max())
    if peak > 1.0:
        pcm = pcm / peak
    return np.ascontiguousarray(pcm, dtype=np.float32)


def duration_seconds(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / float(sample_rate)
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_audio.py -v`
Expected: PASS — 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/transcription_server/audio.py tests/unit/test_audio.py
git commit -m "feat: decodage audio via ffmpeg"
```

---

## Task 7: Protocols des moteurs et implémentations factices

Sans cette tâche, aucune route ne serait testable hors du conteneur. Les
implémentations factices sont du **code de production** (elles vivent dans
`src/`), car elles servent aussi au mode `ENABLE_DIARIZATION=false`.

**Files:**
- Create: `src/transcription_server/asr/__init__.py`, `src/transcription_server/asr/engine.py`
- Create: `src/transcription_server/diarization/__init__.py`, `src/transcription_server/diarization/engine.py`
- Test: `tests/unit/test_engines.py`

**Interfaces:**
- Consumes: `Word`, `SpeakerSegment` de `transcription_server.domain`.
- Produces:
  - `AsrEngine` — `Protocol` avec `transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]` et la propriété `name: str`
  - `DiarizationEngine` — `Protocol` avec `diarize(self, audio: np.ndarray, num_speakers: int | None, min_speakers: int | None, max_speakers: int | None) -> list[SpeakerSegment]` et la propriété `name: str`
  - `NullDiarizationEngine` — rend toujours `[]`, utilisé quand la diarization est désactivée
  - `StubAsrEngine(words: list[Word], name: str = "stub-asr")` et `StubDiarizationEngine(segments: list[SpeakerSegment], name: str = "stub-diarization")` — rendent des données fixes, pour les tests

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_engines.py` :

```python
import numpy as np

from transcription_server.asr.engine import AsrEngine, StubAsrEngine
from transcription_server.diarization.engine import (
    DiarizationEngine,
    NullDiarizationEngine,
    StubDiarizationEngine,
)
from transcription_server.domain import SpeakerSegment, Word

AUDIO = np.zeros(16000, dtype=np.float32)


def test_stub_asr_respecte_le_protocol():
    engine = StubAsrEngine([Word("bonjour", 0.0, 0.5)])
    assert isinstance(engine, AsrEngine)


def test_stub_asr_rend_les_mots_fournis():
    words = [Word("bonjour", 0.0, 0.5), Word("tous", 0.6, 1.0)]
    assert StubAsrEngine(words).transcribe(AUDIO, language=None) == words


def test_stub_diarization_respecte_le_protocol():
    engine = StubDiarizationEngine([SpeakerSegment("SPEAKER_00", 0.0, 1.0)])
    assert isinstance(engine, DiarizationEngine)


def test_stub_diarization_rend_les_segments_fournis():
    segs = [SpeakerSegment("SPEAKER_00", 0.0, 1.0)]
    out = StubDiarizationEngine(segs).diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert out == segs


def test_null_diarization_rend_une_liste_vide():
    out = NullDiarizationEngine().diarize(
        AUDIO, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert out == []


def test_null_diarization_respecte_le_protocol():
    assert isinstance(NullDiarizationEngine(), DiarizationEngine)


def test_les_moteurs_exposent_un_nom():
    assert StubAsrEngine([]).name == "stub-asr"
    assert NullDiarizationEngine().name == "none"
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_engines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.asr'`

- [ ] **Step 3: Écrire `src/transcription_server/asr/engine.py`**

Créer d'abord `src/transcription_server/asr/__init__.py` vide, puis :

```python
"""Contrat des moteurs de transcription."""

from typing import Protocol, runtime_checkable

import numpy as np

from transcription_server.domain import Word


@runtime_checkable
class AsrEngine(Protocol):
    """Transforme une waveform mono 16 kHz en mots horodates."""

    @property
    def name(self) -> str:
        """Identifiant du modele, expose par /health et /v1/models."""
        ...

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]:
        """Rend les mots avec des timestamps relatifs au debut de `audio`."""
        ...


class StubAsrEngine:
    """Moteur a sortie fixe, pour tester les routes sans GPU."""

    def __init__(self, words: list[Word], name: str = "stub-asr") -> None:
        self._words = list(words)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]:
        return list(self._words)
```

- [ ] **Step 4: Écrire `src/transcription_server/diarization/engine.py`**

Créer d'abord `src/transcription_server/diarization/__init__.py` vide, puis :

```python
"""Contrat des moteurs de diarization."""

from typing import Protocol, runtime_checkable

import numpy as np

from transcription_server.domain import SpeakerSegment


@runtime_checkable
class DiarizationEngine(Protocol):
    """Decoupe une waveform en intervalles attribues a des locuteurs."""

    @property
    def name(self) -> str: ...

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        """Rend des segments non vides, tries par debut croissant."""
        ...


class NullDiarizationEngine:
    """Ne separe aucun locuteur.

    Utilise quand ENABLE_DIARIZATION=false : le serveur demarre alors sans
    token HuggingFace, et toutes les transcriptions sortent en un seul flux.
    """

    @property
    def name(self) -> str:
        return "none"

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        return []


class StubDiarizationEngine:
    """Moteur a sortie fixe, pour tester les routes sans GPU."""

    def __init__(
        self,
        segments: list[SpeakerSegment],
        name: str = "stub-diarization",
    ) -> None:
        self._segments = list(segments)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        return list(self._segments)
```

- [ ] **Step 5: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_engines.py -v`
Expected: PASS — 7 tests

- [ ] **Step 6: Commit**

```bash
git add src/transcription_server/asr/ src/transcription_server/diarization/ tests/unit/test_engines.py
git commit -m "feat: protocols des moteurs asr et diarization"
```

---

## Task 8: Pipeline d'orchestration

Relie décodage, diarization, ASR découpé et alignement. C'est le seul endroit
qui connaît l'ordre des opérations.

**Files:**
- Create: `src/transcription_server/pipeline.py`
- Test: `tests/unit/test_pipeline.py`

**Interfaces:**
- Consumes: `decode_to_pcm`, `duration_seconds`, `SAMPLE_RATE` (`audio`) ; `plan_windows`, `offset_words`, `merge_windows` (`chunking`) ; `group_into_turns` (`alignment`) ; `AsrEngine`, `DiarizationEngine`.
- Produces:
  - `TranscriptionResult(text: str, language: str | None, duration: float, speakers: list[str], turns: list[Turn], timing: dict[str, float])` — `@dataclass(frozen=True)`
  - `TranscriptionRequest(language: str | None = None, diarize: bool = True, num_speakers: int | None = None, min_speakers: int | None = None, max_speakers: int | None = None)` — `@dataclass(frozen=True)`
  - `run_pipeline(path, asr, diarization, request, chunk_length_s, chunk_overlap_s, turn_gap_s) -> TranscriptionResult`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_pipeline.py` :

```python
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from transcription_server.asr.engine import StubAsrEngine
from transcription_server.diarization.engine import (
    NullDiarizationEngine,
    StubDiarizationEngine,
)
from transcription_server.domain import SpeakerSegment, Word
from transcription_server.pipeline import (
    TranscriptionRequest,
    run_pipeline,
)

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"


def _write_silence_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack("<h", 0) * int(seconds * rate))


@pytest.fixture
def short_wav(tmp_path):
    path = tmp_path / "court.wav"
    _write_silence_wav(path, seconds=3.0)
    return path


def test_pipeline_produit_du_texte(short_wav):
    asr = StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("tous", 0.6, 1.0)])
    result = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.text == "bonjour tous"
    assert result.duration == pytest.approx(3.0, rel=0.05)
    assert result.speakers == []


def test_pipeline_avec_diarization_rend_les_locuteurs(short_wav):
    asr = StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("merci", 2.0, 2.5)])
    diar = StubDiarizationEngine(
        [SpeakerSegment(S0, 0.0, 1.0), SpeakerSegment(S1, 1.5, 3.0)]
    )
    result = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=diar,
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.speakers == [S0, S1]
    assert [t.speaker for t in result.turns] == [S0, S1]


def test_diarize_false_ignore_le_moteur_de_diarization(short_wav):
    diar = StubDiarizationEngine([SpeakerSegment(S0, 0.0, 3.0)])
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.5)]),
        diarization=diar,
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.speakers == []
    assert result.turns[0].speaker is None


def test_les_speakers_sont_tries_et_dedupliques(short_wav):
    asr = StubAsrEngine(
        [Word("a", 0.0, 0.4), Word("b", 1.6, 2.0), Word("c", 2.2, 2.6)]
    )
    diar = StubDiarizationEngine(
        [
            SpeakerSegment(S1, 1.5, 2.1),
            SpeakerSegment(S0, 0.0, 1.0),
            SpeakerSegment(S1, 2.2, 3.0),
        ]
    )
    result = run_pipeline(
        path=short_wav,
        asr=asr,
        diarization=diar,
        request=TranscriptionRequest(diarize=True),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert result.speakers == [S0, S1]


def test_le_timing_est_renseigne(short_wav):
    result = run_pipeline(
        path=short_wav,
        asr=StubAsrEngine([Word("un", 0.0, 0.5)]),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=480.0,
        chunk_overlap_s=15.0,
        turn_gap_s=1.0,
    )
    assert set(result.timing) == {"decode", "asr", "diarization"}
    assert all(v >= 0.0 for v in result.timing.values())


def test_audio_long_appelle_l_asr_une_fois_par_fenetre(tmp_path):
    """Avec chunk_length_s=1.0 sur 3 s d'audio, l'ASR est appele plusieurs fois
    et les timestamps sont reoffsetes en absolu."""
    path = tmp_path / "long.wav"
    _write_silence_wav(path, seconds=3.0)

    calls: list[int] = []

    class CountingAsr:
        name = "counting"

        def transcribe(self, audio, language):
            calls.append(len(audio))
            # un mot au tout debut de chaque fenetre
            return [Word("mot", 0.1, 0.2)]

    result = run_pipeline(
        path=path,
        asr=CountingAsr(),
        diarization=NullDiarizationEngine(),
        request=TranscriptionRequest(diarize=False),
        chunk_length_s=1.0,
        chunk_overlap_s=0.2,
        turn_gap_s=10.0,
    )
    assert len(calls) >= 3
    starts = [w.start for t in result.turns for w in t.words]
    assert starts == sorted(starts)
    assert max(starts) > 1.0  # les timestamps sont bien absolus


def test_fichier_invalide_remonte_l_erreur(tmp_path):
    from transcription_server.audio import AudioDecodeError

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"pas de l'audio")
    with pytest.raises(AudioDecodeError):
        run_pipeline(
            path=junk,
            asr=StubAsrEngine([]),
            diarization=NullDiarizationEngine(),
            request=TranscriptionRequest(diarize=False),
            chunk_length_s=480.0,
            chunk_overlap_s=15.0,
            turn_gap_s=1.0,
        )
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.pipeline'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `src/transcription_server/pipeline.py` :

```python
"""Orchestration : decodage, diarization, transcription, alignement.

Seul module a connaitre l'ordre des operations. Il ne depend que des
Protocol des moteurs, jamais de leurs implementations concretes.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from transcription_server.alignment import group_into_turns
from transcription_server.asr.engine import AsrEngine
from transcription_server.audio import SAMPLE_RATE, decode_to_pcm, duration_seconds
from transcription_server.chunking import merge_windows, offset_words, plan_windows
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.domain import Turn


@dataclass(frozen=True)
class TranscriptionRequest:
    language: str | None = None
    diarize: bool = True
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float
    speakers: list[str]
    turns: list[Turn]
    timing: dict[str, float] = field(default_factory=dict)


def run_pipeline(
    path: str | Path,
    asr: AsrEngine,
    diarization: DiarizationEngine,
    request: TranscriptionRequest,
    chunk_length_s: float,
    chunk_overlap_s: float,
    turn_gap_s: float,
) -> TranscriptionResult:
    """Transcrit un fichier et rend des tours de parole."""
    started = time.perf_counter()
    pcm = decode_to_pcm(path)
    duration = duration_seconds(pcm)
    decode_elapsed = time.perf_counter() - started

    # Diarization avant l'ASR : cela permet de liberer le modele de
    # diarization avant l'inference longue si la VRAM se tend.
    segments = []
    diarization_elapsed = 0.0
    if request.diarize:
        started = time.perf_counter()
        segments = diarization.diarize(
            pcm,
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )
        diarization_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    windows = plan_windows(duration, chunk_length_s, chunk_overlap_s)
    per_window: list[list[object]] = []
    for window in windows:
        begin = int(window.start * SAMPLE_RATE)
        finish = int(window.end * SAMPLE_RATE)
        local_words = asr.transcribe(pcm[begin:finish], language=request.language)
        per_window.append(offset_words(local_words, window.start))
    words = merge_windows(per_window, windows)
    asr_elapsed = time.perf_counter() - started

    turns = group_into_turns(words, segments, turn_gap_s=turn_gap_s)
    speakers = sorted({s.speaker for s in segments})

    return TranscriptionResult(
        text=" ".join(t.text for t in turns if t.text),
        language=request.language,
        duration=duration,
        speakers=speakers,
        turns=turns,
        timing={
            "decode": round(decode_elapsed, 3),
            "asr": round(asr_elapsed, 3),
            "diarization": round(diarization_elapsed, 3),
        },
    )
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_pipeline.py -v`
Expected: PASS — 7 tests

- [ ] **Step 5: Lancer toute la suite**

Run: `.venv\Scripts\pytest -v`
Expected: PASS — 68 tests

- [ ] **Step 6: Commit**

```bash
git add src/transcription_server/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: pipeline d'orchestration de la transcription"
```

---

## Task 9: Schémas d'API et routes natives

**Files:**
- Create: `src/transcription_server/api/__init__.py`, `src/transcription_server/api/schemas.py`, `src/transcription_server/api/native_routes.py`
- Create: `src/transcription_server/state.py`
- Test: `tests/unit/test_native_routes.py`

**Interfaces:**
- Consumes: `TranscriptionResult`, `TranscriptionRequest`, `run_pipeline` ; `Settings` ; les formateurs.
- Produces:
  - `state.AppState(asr: AsrEngine, diarization: DiarizationEngine, settings: Settings, gpu_lock: asyncio.Lock, device_info: dict)` et `get_state(request) -> AppState`
  - `schemas.WordOut`, `schemas.TurnOut`, `schemas.TranscriptionOut`, `schemas.HealthOut`
  - `native_routes.router` — `APIRouter` portant `POST /transcribe` et `GET /health`

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_native_routes.py` :

```python
import io
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import StubDiarizationEngine
from transcription_server.domain import SpeakerSegment, Word

S0 = "SPEAKER_00"
S1 = "SPEAKER_01"


def _wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def client():
    settings = Settings(_env_file=None, enable_diarization=False, device="cpu")
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("merci", 1.2, 1.6)]),
        diarization=StubDiarizationEngine(
            [SpeakerSegment(S0, 0.0, 1.0), SpeakerSegment(S1, 1.1, 2.0)]
        ),
    )
    return TestClient(app)


def test_health_repond(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert "asr_model" in body


def test_transcribe_json(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"diarize": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "bonjour merci"
    assert body["speakers"] == [S0, S1]
    assert len(body["turns"]) == 2
    assert body["turns"][0]["speaker"] == S0
    assert body["turns"][0]["words"][0]["word"] == "bonjour"


def test_transcribe_sans_diarization(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"diarize": "false"},
    )
    assert response.status_code == 200
    assert response.json()["speakers"] == []


def test_transcribe_format_dialogue(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"diarize": "true", "response_format": "dialogue"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "SPEAKER_00: bonjour" in response.text


def test_transcribe_format_srt(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "srt"},
    )
    assert response.status_code == 200
    assert response.text.startswith("1\n00:00:00,000 --> ")


def test_fichier_illisible_donne_400(client):
    response = client.post(
        "/transcribe",
        files={"file": ("junk.wav", b"pas de l'audio", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_num_speakers_avec_min_max_donne_400(client):
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"num_speakers": "2", "min_speakers": "1"},
    )
    assert response.status_code == 400


def test_fichier_trop_gros_donne_413():
    settings = Settings(
        _env_file=None, enable_diarization=False, device="cpu", max_upload_mb=1
    )
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("a", 0.0, 0.5)]),
        diarization=StubDiarizationEngine([]),
    )
    client = TestClient(app)
    response = client.post(
        "/transcribe",
        files={"file": ("gros.wav", b"\x00" * (2 * 1024 * 1024), "audio/wav")},
    )
    assert response.status_code == 413
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_native_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.app'`

- [ ] **Step 3: Écrire `src/transcription_server/state.py`**

```python
"""Etat partage de l'application, accessible depuis les routes."""

import asyncio
from dataclasses import dataclass, field

from fastapi import Request

from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import DiarizationEngine


@dataclass
class AppState:
    settings: Settings
    asr: AsrEngine
    diarization: DiarizationEngine
    device_info: dict = field(default_factory=dict)
    gpu_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def get_state(request: Request) -> AppState:
    """Dependance FastAPI : rend l'etat attache a l'application."""
    return request.app.state.app_state
```

- [ ] **Step 4: Écrire `src/transcription_server/api/schemas.py`**

Créer `src/transcription_server/api/__init__.py` vide, puis :

```python
"""Modeles de reponse de l'API."""

from pydantic import BaseModel, Field

from transcription_server.domain import Turn
from transcription_server.pipeline import TranscriptionResult


class WordOut(BaseModel):
    word: str
    start: float
    end: float


class TurnOut(BaseModel):
    speaker: str | None
    start: float
    end: float
    text: str
    words: list[WordOut] = Field(default_factory=list)


class TranscriptionOut(BaseModel):
    text: str
    language: str | None = None
    duration: float
    speakers: list[str] = Field(default_factory=list)
    turns: list[TurnOut] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)


class HealthOut(BaseModel):
    status: str
    device: str
    asr_model: str
    diarization_model: str
    diarization_enabled: bool
    gpu: dict | None = None


def turn_to_out(turn: Turn, include_words: bool) -> TurnOut:
    return TurnOut(
        speaker=turn.speaker,
        start=round(turn.start, 3),
        end=round(turn.end, 3),
        text=turn.text,
        words=[
            WordOut(word=w.text, start=round(w.start, 3), end=round(w.end, 3))
            for w in turn.words
        ]
        if include_words
        else [],
    )


def result_to_out(
    result: TranscriptionResult, include_words: bool = True
) -> TranscriptionOut:
    return TranscriptionOut(
        text=result.text,
        language=result.language,
        duration=round(result.duration, 3),
        speakers=result.speakers,
        turns=[turn_to_out(t, include_words) for t in result.turns],
        timing=result.timing,
    )
```

- [ ] **Step 5: Écrire `src/transcription_server/api/native_routes.py`**

```python
"""Endpoints natifs : /transcribe et /health."""

import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from transcription_server.api.schemas import HealthOut, TranscriptionOut, result_to_out
from transcription_server.audio import AudioDecodeError
from transcription_server.formatting import to_dialogue, to_plain_text, to_srt, to_vtt
from transcription_server.pipeline import TranscriptionRequest, run_pipeline
from transcription_server.state import AppState, get_state

router = APIRouter()

ResponseFormat = Literal["json", "text", "srt", "vtt", "dialogue"]


async def _save_upload(upload: UploadFile, max_bytes: int) -> Path:
    """Ecrit l'upload dans un fichier temporaire, en refusant les trop gros."""
    suffix = Path(upload.filename or "audio").suffix or ".bin"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    written = 0
    try:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                handle.close()
                Path(handle.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Fichier trop volumineux : maximum {max_bytes} octets.",
                )
            handle.write(chunk)
    finally:
        handle.close()
    return Path(handle.name)


@router.get("/health", response_model=HealthOut)
async def health(state: Annotated[AppState, Depends(get_state)]) -> HealthOut:
    return HealthOut(
        status="ok",
        device=state.settings.device,
        asr_model=state.asr.name,
        diarization_model=state.diarization.name,
        diarization_enabled=state.settings.enable_diarization,
        gpu=state.device_info or None,
    )


@router.post("/transcribe")
async def transcribe(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile, File()],
    language: Annotated[str | None, Form()] = None,
    diarize: Annotated[bool | None, Form()] = None,
    num_speakers: Annotated[int | None, Form()] = None,
    min_speakers: Annotated[int | None, Form()] = None,
    max_speakers: Annotated[int | None, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = True,
    response_format: Annotated[ResponseFormat, Form()] = "json",
):
    if num_speakers is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="num_speakers et min_speakers/max_speakers s'excluent.",
        )

    settings = state.settings
    should_diarize = settings.enable_diarization if diarize is None else diarize
    path = await _save_upload(file, settings.max_upload_bytes)

    try:
        async with state.gpu_lock:
            result = await run_in_threadpool(
                run_pipeline,
                path=path,
                asr=state.asr,
                diarization=state.diarization,
                request=TranscriptionRequest(
                    language=language,
                    diarize=should_diarize,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                ),
                chunk_length_s=settings.chunk_length_s,
                chunk_overlap_s=settings.chunk_overlap_s,
                turn_gap_s=settings.turn_gap_s,
            )
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)

    if response_format == "text":
        return PlainTextResponse(to_plain_text(result.turns))
    if response_format == "srt":
        return PlainTextResponse(to_srt(result.turns))
    if response_format == "vtt":
        return PlainTextResponse(to_vtt(result.turns))
    if response_format == "dialogue":
        return PlainTextResponse(to_dialogue(result.turns))
    return result_to_out(result, include_words=word_timestamps)
```

- [ ] **Step 6: Écrire `src/transcription_server/app.py` (version minimale)**

Cette version accepte des moteurs injectés, ce qui rend l'application testable
sans GPU. La Task 11 y ajoutera le `lifespan` qui charge les vrais moteurs.

```python
"""Fabrique de l'application FastAPI."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from transcription_server.api import native_routes
from transcription_server.asr.engine import AsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import DiarizationEngine
from transcription_server.state import AppState

_ERROR_TYPES = {
    400: "invalid_request_error",
    413: "invalid_request_error",
    503: "service_unavailable",
}


def create_app(
    settings: Settings,
    asr: AsrEngine,
    diarization: DiarizationEngine,
    device_info: dict | None = None,
) -> FastAPI:
    app = FastAPI(title="Serveur de transcription Parakeet", version="0.1.0")
    app.state.app_state = AppState(
        settings=settings,
        asr=asr,
        diarization=diarization,
        device_info=device_info or {},
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Uniformise les erreurs au format OpenAI sur toutes les routes."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": _ERROR_TYPES.get(exc.status_code, "server_error"),
                }
            },
        )

    app.include_router(native_routes.router)
    return app
```

- [ ] **Step 7: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_native_routes.py -v`
Expected: PASS — 8 tests

- [ ] **Step 8: Commit**

```bash
git add src/transcription_server/state.py src/transcription_server/api/ src/transcription_server/app.py tests/unit/test_native_routes.py
git commit -m "feat: endpoint natif /transcribe et /health"
```

---

## Task 10: Routes compatibles OpenAI

**Files:**
- Create: `src/transcription_server/api/openai_routes.py`
- Modify: `src/transcription_server/app.py` (ajouter `include_router`)
- Test: `tests/unit/test_openai_routes.py`

**Interfaces:**
- Consumes: `run_pipeline`, `TranscriptionRequest`, `AppState`, les formateurs, `_save_upload` de `native_routes`.
- Produces: `openai_routes.router` — `POST /v1/audio/transcriptions`, `GET /v1/models`.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_openai_routes.py` :

```python
import io
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import StubDiarizationEngine
from transcription_server.domain import Word


def _wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def client():
    settings = Settings(_env_file=None, enable_diarization=False, device="cpu")
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("bonjour", 0.0, 0.5), Word("tous", 0.6, 1.0)]),
        diarization=StubDiarizationEngine([]),
    )
    return TestClient(app)


def test_transcriptions_json_par_defaut(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "bonjour tous"}


def test_transcriptions_text(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "text"},
    )
    assert response.status_code == 200
    assert response.text == "bonjour tous"


def test_transcriptions_srt(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "srt"},
    )
    assert response.status_code == 200
    assert response.text.startswith("1\n00:00:00,000 --> ")


def test_transcriptions_verbose_json(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", _wav_bytes(), "audio/wav")},
        data={"response_format": "verbose_json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "transcribe"
    assert body["text"] == "bonjour tous"
    assert body["duration"] == pytest.approx(2.0, rel=0.05)
    assert body["segments"][0]["text"] == "bonjour tous"
    assert body["words"][0]["word"] == "bonjour"


def test_transcriptions_erreur_au_format_openai(client):
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("junk.wav", b"pas de l'audio", "audio/wav")},
    )
    assert response.status_code == 400
    assert "error" in response.json()
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_liste_des_modeles(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert any(m["id"] == "stub-asr" for m in body["data"])
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_openai_routes.py -v`
Expected: FAIL — 404 sur `/v1/audio/transcriptions`, le routeur n'existe pas

- [ ] **Step 3: Écrire `src/transcription_server/api/openai_routes.py`**

```python
"""Endpoints compatibles avec l'API audio d'OpenAI.

Objectif : qu'un client ecrit pour Whisper fonctionne sans modification.
La diarization n'est pas exposee ici, elle n'a pas d'equivalent OpenAI.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool

from transcription_server.api.native_routes import _save_upload
from transcription_server.audio import AudioDecodeError
from transcription_server.formatting import to_plain_text, to_srt, to_vtt
from transcription_server.pipeline import TranscriptionRequest, run_pipeline
from transcription_server.state import AppState, get_state

router = APIRouter(prefix="/v1")

OpenAIFormat = Literal["json", "text", "srt", "vtt", "verbose_json"]


@router.get("/models")
async def list_models(state: Annotated[AppState, Depends(get_state)]) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": state.asr.name,
                "object": "model",
                "owned_by": "nvidia",
            }
        ],
    }


@router.post("/audio/transcriptions")
async def create_transcription(
    state: Annotated[AppState, Depends(get_state)],
    file: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    response_format: Annotated[OpenAIFormat, Form()] = "json",
):
    settings = state.settings
    path = await _save_upload(file, settings.max_upload_bytes)

    try:
        async with state.gpu_lock:
            result = await run_in_threadpool(
                run_pipeline,
                path=path,
                asr=state.asr,
                diarization=state.diarization,
                request=TranscriptionRequest(language=language, diarize=False),
                chunk_length_s=settings.chunk_length_s,
                chunk_overlap_s=settings.chunk_overlap_s,
                turn_gap_s=settings.turn_gap_s,
            )
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)

    if response_format == "text":
        return PlainTextResponse(to_plain_text(result.turns))
    if response_format == "srt":
        return PlainTextResponse(to_srt(result.turns))
    if response_format == "vtt":
        return PlainTextResponse(to_vtt(result.turns))
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": result.language,
            "duration": round(result.duration, 3),
            "text": result.text,
            "segments": [
                {
                    "id": i,
                    "start": round(t.start, 3),
                    "end": round(t.end, 3),
                    "text": t.text,
                }
                for i, t in enumerate(result.turns)
            ],
            "words": [
                {
                    "word": w.text,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                }
                for t in result.turns
                for w in t.words
            ],
        }
    return {"text": result.text}
```

- [ ] **Step 4: Modifier `src/transcription_server/app.py`**

Ajouter l'import et l'enregistrement du routeur :

```python
from transcription_server.api import native_routes, openai_routes
```

puis, juste après `app.include_router(native_routes.router)` :

```python
    app.include_router(openai_routes.router)
```

- [ ] **Step 5: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_openai_routes.py -v`
Expected: PASS — 6 tests

- [ ] **Step 6: Lancer toute la suite**

Run: `.venv\Scripts\pytest -v`
Expected: PASS — 82 tests, aucun ignoré (les tests GPU ne sont pas encore écrits)

- [ ] **Step 7: Commit**

```bash
git add src/transcription_server/api/openai_routes.py src/transcription_server/app.py tests/unit/test_openai_routes.py
git commit -m "feat: endpoints compatibles openai"
```

---

## Task 11: Runtime GPU

**Files:**
- Create: `src/transcription_server/runtime.py`
- Test: `tests/unit/test_runtime.py`

**Interfaces:**
- Consumes: rien du projet.
- Produces:
  - `CudaUnavailableError(RuntimeError)`
  - `resolve_device(requested: str, cuda_available: bool) -> str` — **fonction pure**, testable sans torch
  - `cuda_available() -> bool` — importe torch paresseusement
  - `gpu_info() -> dict` — nom du GPU, VRAM totale et libre en Mo ; `{}` si pas de CUDA
  - `torch_dtype(compute_type: str)` — importe torch paresseusement
  - `empty_cache() -> None`

**Point clé** : `torch` est importé **à l'intérieur** des fonctions, jamais au
niveau du module. Sans cela, `import runtime` casserait le venv Windows de
développement, où torch n'est pas installé.

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_runtime.py` :

```python
import pytest

from transcription_server.runtime import CudaUnavailableError, resolve_device


def test_cuda_demande_et_disponible():
    assert resolve_device("cuda", cuda_available=True) == "cuda"


def test_cuda_demande_mais_indisponible_leve_une_erreur():
    """Aucun repli CPU silencieux : une transcription vingt fois plus lente
    doit etre un choix explicite, jamais une surprise."""
    with pytest.raises(CudaUnavailableError) as excinfo:
        resolve_device("cuda", cuda_available=False)
    assert "DEVICE=cpu" in str(excinfo.value)


def test_cpu_demande_est_accepte_meme_avec_cuda():
    assert resolve_device("cpu", cuda_available=True) == "cpu"


def test_cpu_demande_sans_cuda():
    assert resolve_device("cpu", cuda_available=False) == "cpu"


def test_device_inconnu_est_rejete():
    with pytest.raises(ValueError):
        resolve_device("tpu", cuda_available=True)


def test_le_module_s_importe_sans_torch():
    """Garde-fou : runtime.py ne doit pas importer torch au niveau module."""
    import transcription_server.runtime as runtime

    source = open(runtime.__file__, encoding="utf-8").read()
    lignes_module = [
        line
        for line in source.splitlines()
        if line.startswith("import torch") or line.startswith("from torch")
    ]
    assert lignes_module == []
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_runtime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transcription_server.runtime'`

- [ ] **Step 3: Écrire l'implémentation**

Créer `src/transcription_server/runtime.py` :

```python
"""Gestion du peripherique de calcul et de la memoire GPU.

torch est importe a l'interieur des fonctions, jamais au niveau du module :
cela permet d'importer ce fichier depuis le venv de developpement Windows,
ou torch n'est pas installe.
"""


class CudaUnavailableError(RuntimeError):
    """CUDA a ete demande mais n'est pas accessible."""


def resolve_device(requested: str, cuda_available: bool) -> str:
    """Valide le peripherique demande. Ne se rabat jamais silencieusement."""
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"Peripherique inconnu : {requested!r}. Attendu cuda ou cpu.")
    if requested == "cuda" and not cuda_available:
        raise CudaUnavailableError(
            "DEVICE=cuda a ete demande mais torch.cuda.is_available() est faux. "
            "Verifiez que le conteneur tourne avec --gpus all et que le runtime "
            "nvidia est actif, ou mettez DEVICE=cpu pour accepter une execution "
            "nettement plus lente."
        )
    return requested


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def gpu_info() -> dict:
    """Nom du GPU et etat de la VRAM, en megaoctets."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_total_mb": round(total / (1024 * 1024)),
        "vram_free_mb": round(free / (1024 * 1024)),
    }


def torch_dtype(compute_type: str):
    import torch

    return {"float16": torch.float16, "float32": torch.float32}[compute_type]


def empty_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_runtime.py -v`
Expected: PASS — 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/transcription_server/runtime.py tests/unit/test_runtime.py
git commit -m "feat: resolution du peripherique et informations gpu"
```

---

## Task 12: Adaptateur NeMo Parakeet

Premier module exigeant le conteneur. Ses tests sont marqués `gpu` et ignorés
sur Windows.

**Files:**
- Create: `src/transcription_server/asr/nemo_parakeet.py`
- Create: `tests/gpu/__init__.py`, `tests/gpu/test_nemo_engine.py`

**Interfaces:**
- Consumes: `Word`, `AsrEngine`, `runtime.torch_dtype`, `audio.SAMPLE_RATE`.
- Produces: `NemoParakeetEngine(model_name: str, device: str, compute_type: str)` avec `.name`, `.transcribe(audio, language)`, et la fabrique `load_nemo_engine(model_name, device, compute_type) -> NemoParakeetEngine`.

- [ ] **Step 1: Écrire le test GPU**

Créer `tests/gpu/__init__.py` vide, puis `tests/gpu/test_nemo_engine.py` :

```python
"""Tests exigeant CUDA et l'extra gpu. Lancer avec : pytest -m gpu"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_cuda_est_disponible():
    from transcription_server.runtime import cuda_available, gpu_info

    assert cuda_available() is True
    info = gpu_info()
    assert info["vram_total_mb"] > 8000


def test_le_moteur_se_charge_et_transcrit_du_silence():
    """Sur du silence, le moteur doit rendre une liste (vide ou non) sans
    lever d'exception. C'est le test de cablage, pas de qualite."""
    from transcription_server.asr.nemo_parakeet import load_nemo_engine

    engine = load_nemo_engine(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device="cuda",
        compute_type="float16",
    )
    assert engine.name == "nvidia/parakeet-tdt-0.6b-v3"
    audio = np.zeros(16000 * 2, dtype=np.float32)
    words = engine.transcribe(audio, language=None)
    assert isinstance(words, list)


def test_les_mots_sont_horodates_de_maniere_croissante():
    """Sur un vrai echantillon de parole, les timestamps doivent croitre."""
    from transcription_server.asr.nemo_parakeet import load_nemo_engine
    from transcription_server.audio import decode_to_pcm

    engine = load_nemo_engine(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device="cuda",
        compute_type="float16",
    )
    pcm = decode_to_pcm("tests/fixtures/echantillon_fr.wav")
    words = engine.transcribe(pcm, language="fr")
    assert len(words) > 0
    assert all(w.start <= w.end for w in words)
    assert [w.start for w in words] == sorted(w.start for w in words)
```

L'échantillon `tests/fixtures/echantillon_fr.wav` est produit à la Task 16.
Ce test échouera jusque-là — c'est attendu, il est marqué `gpu` et donc
désélectionné par défaut.

- [ ] **Step 2: Écrire l'implémentation**

Créer `src/transcription_server/asr/nemo_parakeet.py` :

```python
"""Adaptateur du modele Parakeet via NVIDIA NeMo.

NeMo et torch sont importes paresseusement pour que le reste du paquet
reste importable sans l'extra gpu.
"""

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

from transcription_server.audio import SAMPLE_RATE
from transcription_server.domain import Word

logger = logging.getLogger(__name__)


class NemoParakeetEngine:
    """Transcrit avec un modele NeMo ASR, timestamps mot a mot inclus."""

    def __init__(self, model, model_name: str, device: str) -> None:
        self._model = model
        self._name = model_name
        self._device = device

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]:
        """Rend les mots avec des timestamps relatifs au debut de `audio`."""
        if audio.size == 0:
            return []

        # NeMo lit un chemin de fichier de maniere fiable quelle que soit la
        # version ; passer un tableau change de signature d'une release a
        # l'autre. On ecrit donc un wav temporaire.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = Path(handle.name)
        try:
            _write_wav(path, audio)
            outputs = self._model.transcribe([str(path)], timestamps=True)
        finally:
            path.unlink(missing_ok=True)

        if not outputs:
            return []
        return _extract_words(outputs[0])


def _write_wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Ecrit un tableau float32 [-1, 1] en wav mono 16 bits."""
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm16.tobytes())


def _extract_words(hypothesis) -> list[Word]:
    """Extrait les mots horodates d'une hypothese NeMo.

    NeMo expose `hypothesis.timestamp["word"]`, une liste de dictionnaires
    contenant `word`, `start` et `end` en secondes.
    """
    timestamps = getattr(hypothesis, "timestamp", None)
    if not timestamps or "word" not in timestamps:
        text = getattr(hypothesis, "text", "") or ""
        if not text:
            return []
        logger.warning(
            "NeMo n'a pas rendu de timestamps mot a mot ; le texte est "
            "restitue en un seul bloc sans horodatage fin."
        )
        return [Word(text=text, start=0.0, end=0.0)]

    words: list[Word] = []
    for entry in timestamps["word"]:
        text = (entry.get("word") or "").strip()
        if not text:
            continue
        words.append(
            Word(
                text=text,
                start=float(entry["start"]),
                end=float(entry["end"]),
            )
        )
    return words


def load_nemo_engine(
    model_name: str,
    device: str,
    compute_type: str,
) -> NemoParakeetEngine:
    """Charge le modele et le place sur le peripherique demande."""
    import nemo.collections.asr as nemo_asr
    import torch

    logger.info("Chargement du modele ASR %s...", model_name)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(torch.device(device))
    if device == "cuda" and compute_type == "float16":
        model = model.half()
    model.eval()
    logger.info("Modele ASR %s charge sur %s.", model_name, device)
    return NemoParakeetEngine(model=model, model_name=model_name, device=device)
```

- [ ] **Step 3: Vérifier que le paquet s'importe toujours sans l'extra gpu**

Run: `.venv\Scripts\pytest -v`
Expected: PASS — 88 tests, les tests `gpu` désélectionnés

- [ ] **Step 4: Commit**

```bash
git add src/transcription_server/asr/nemo_parakeet.py tests/gpu/
git commit -m "feat: adaptateur nemo parakeet"
```

---

## Task 13: Adaptateur pyannote

**Files:**
- Create: `src/transcription_server/diarization/pyannote_engine.py`
- Create: `tests/gpu/test_pyannote_engine.py`

**Interfaces:**
- Consumes: `SpeakerSegment`, `DiarizationEngine`, `audio.SAMPLE_RATE`.
- Produces: `PyannoteEngine` avec `.name`, `.diarize(...)`, et `load_pyannote_engine(model_name: str, hf_token: str, device: str) -> PyannoteEngine`.

- [ ] **Step 1: Écrire le test GPU**

Créer `tests/gpu/test_pyannote_engine.py` :

```python
"""Tests exigeant CUDA, l'extra gpu et un HF_TOKEN valide."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def engine():
    from transcription_server.config import Settings
    from transcription_server.diarization.pyannote_engine import load_pyannote_engine

    settings = Settings()
    if not settings.hf_token:
        pytest.skip("HF_TOKEN absent")
    return load_pyannote_engine(
        model_name=settings.diarization_model,
        hf_token=settings.hf_token,
        device="cuda",
    )


def test_le_moteur_expose_son_nom(engine):
    assert engine.name == "pyannote/speaker-diarization-community-1"


def test_silence_ne_produit_aucun_segment(engine):
    audio = np.zeros(16000 * 5, dtype=np.float32)
    segments = engine.diarize(
        audio, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert segments == []


def test_deux_voix_donnent_deux_locuteurs(engine):
    """L'echantillon concatene deux voix distinctes (Task 16)."""
    from transcription_server.audio import decode_to_pcm

    pcm = decode_to_pcm("tests/fixtures/deux_voix.wav")
    segments = engine.diarize(
        pcm, num_speakers=None, min_speakers=None, max_speakers=None
    )
    assert len({s.speaker for s in segments}) == 2
    assert all(s.start < s.end for s in segments)
    assert segments == sorted(segments, key=lambda s: s.start)


def test_num_speakers_force_le_nombre(engine):
    from transcription_server.audio import decode_to_pcm

    pcm = decode_to_pcm("tests/fixtures/deux_voix.wav")
    segments = engine.diarize(
        pcm, num_speakers=1, min_speakers=None, max_speakers=None
    )
    assert len({s.speaker for s in segments}) == 1
```

- [ ] **Step 2: Écrire l'implémentation**

Créer `src/transcription_server/diarization/pyannote_engine.py` :

```python
"""Adaptateur du pipeline de diarization pyannote.audio."""

import logging

import numpy as np

from transcription_server.audio import SAMPLE_RATE
from transcription_server.domain import SpeakerSegment

logger = logging.getLogger(__name__)


class PyannoteEngine:
    """Separe les locuteurs avec un pipeline pyannote deja charge."""

    def __init__(self, pipeline, model_name: str) -> None:
        self._pipeline = pipeline
        self._name = model_name

    @property
    def name(self) -> str:
        return self._name

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        import torch

        if audio.size == 0:
            return []

        # pyannote attend un tenseur (canaux, echantillons).
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

        options: dict = {}
        if num_speakers is not None:
            options["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                options["min_speakers"] = min_speakers
            if max_speakers is not None:
                options["max_speakers"] = max_speakers

        annotation = self._pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            **options,
        )

        segments = [
            SpeakerSegment(
                speaker=str(speaker),
                start=float(turn.start),
                end=float(turn.end),
            )
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: (s.start, s.end))
        return segments


def load_pyannote_engine(
    model_name: str,
    hf_token: str,
    device: str,
) -> PyannoteEngine:
    """Charge le pipeline et le place sur le peripherique demande."""
    import torch
    from pyannote.audio import Pipeline

    logger.info("Chargement du pipeline de diarization %s...", model_name)
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    except TypeError:
        # pyannote < 4 utilisait use_auth_token.
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)

    if pipeline is None:
        raise RuntimeError(
            f"pyannote n'a pas pu charger {model_name}. Verifiez que le compte "
            "HuggingFace a bien accepte les conditions du modele et que "
            "HF_TOKEN est un token de type read valide."
        )

    pipeline.to(torch.device(device))
    logger.info("Pipeline de diarization charge sur %s.", device)
    return PyannoteEngine(pipeline=pipeline, model_name=model_name)
```

- [ ] **Step 3: Vérifier que la suite Windows passe toujours**

Run: `.venv\Scripts\pytest -v`
Expected: PASS — 88 tests, tests `gpu` désélectionnés

- [ ] **Step 4: Commit**

```bash
git add src/transcription_server/diarization/pyannote_engine.py tests/gpu/test_pyannote_engine.py
git commit -m "feat: adaptateur pyannote pour la diarization"
```

---

## Task 14: Chargement au démarrage et point d'entrée

**Files:**
- Modify: `src/transcription_server/app.py` (ajouter `build_app` et le lifespan)
- Create: `src/transcription_server/main.py`
- Test: `tests/unit/test_app_wiring.py`

**Interfaces:**
- Consumes: `load_nemo_engine`, `load_pyannote_engine`, `NullDiarizationEngine`, `resolve_device`, `gpu_info`, `Settings`.
- Produces:
  - `app.build_app(settings: Settings | None = None) -> FastAPI` — charge les vrais moteurs
  - `main.main() -> None` — lance uvicorn

- [ ] **Step 1: Écrire les tests qui échouent**

Créer `tests/unit/test_app_wiring.py` :

```python
import pytest
from fastapi.testclient import TestClient

from transcription_server.app import create_app
from transcription_server.asr.engine import StubAsrEngine
from transcription_server.config import Settings
from transcription_server.diarization.engine import NullDiarizationEngine
from transcription_server.domain import Word


def test_diarization_desactivee_expose_le_moteur_nul():
    settings = Settings(_env_file=None, enable_diarization=False, device="cpu")
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([Word("a", 0.0, 0.5)]),
        diarization=NullDiarizationEngine(),
    )
    body = TestClient(app).get("/health").json()
    assert body["diarization_model"] == "none"
    assert body["diarization_enabled"] is False


def test_device_info_est_expose_par_health():
    settings = Settings(_env_file=None, enable_diarization=False, device="cpu")
    app = create_app(
        settings=settings,
        asr=StubAsrEngine([]),
        diarization=NullDiarizationEngine(),
        device_info={"name": "RTX 3090", "vram_total_mb": 24576},
    )
    body = TestClient(app).get("/health").json()
    assert body["gpu"]["name"] == "RTX 3090"


def test_build_app_echoue_si_cuda_absent(monkeypatch):
    """build_app ne doit jamais se rabattre silencieusement sur le CPU."""
    from transcription_server import app as app_module
    from transcription_server.runtime import CudaUnavailableError

    monkeypatch.setattr(app_module, "cuda_available", lambda: False)
    settings = Settings(_env_file=None, enable_diarization=False, device="cuda")
    with pytest.raises(CudaUnavailableError):
        app_module.build_app(settings)
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `.venv\Scripts\pytest tests/unit/test_app_wiring.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'build_app'`

- [ ] **Step 3: Ajouter `build_app` à `src/transcription_server/app.py`**

Ajouter ces imports en tête du fichier :

```python
import logging

from transcription_server.diarization.engine import NullDiarizationEngine
from transcription_server.runtime import cuda_available, gpu_info, resolve_device

logger = logging.getLogger(__name__)
```

Puis ajouter cette fonction à la fin du fichier :

```python
def build_app(settings: Settings | None = None) -> FastAPI:
    """Construit l'application avec les vrais moteurs charges sur le GPU.

    Les moteurs sont charges ici, au demarrage du processus, et non a la
    premiere requete : mieux vaut echouer tout de suite qu'apres un upload.
    """
    from transcription_server.config import get_settings

    settings = settings or get_settings()
    device = resolve_device(settings.device, cuda_available())

    from transcription_server.asr.nemo_parakeet import load_nemo_engine

    asr = load_nemo_engine(
        model_name=settings.asr_model,
        device=device,
        compute_type=settings.compute_type,
    )

    if settings.enable_diarization:
        from transcription_server.diarization.pyannote_engine import (
            load_pyannote_engine,
        )

        diarization = load_pyannote_engine(
            model_name=settings.diarization_model,
            hf_token=settings.hf_token or "",
            device=device,
        )
    else:
        logger.info("Diarization desactivee (ENABLE_DIARIZATION=false).")
        diarization = NullDiarizationEngine()

    app = create_app(
        settings=settings,
        asr=asr,
        diarization=diarization,
        device_info=gpu_info(),
    )
    _warmup(asr, device)
    return app


def _warmup(asr, device: str) -> None:
    """Une inference sur 1 s de silence, pour que la premiere vraie requete
    n'encaisse pas la compilation des kernels CUDA."""
    if device != "cuda":
        return
    import numpy as np

    try:
        asr.transcribe(np.zeros(16000, dtype=np.float32), language=None)
        logger.info("Warmup termine.")
    except Exception as exc:  # pragma: no cover - le warmup n'est pas critique
        logger.warning("Le warmup a echoue, on continue : %s", exc)
```

- [ ] **Step 4: Créer `src/transcription_server/main.py`**

```python
"""Point d'entree du serveur."""

import logging

import uvicorn

from transcription_server.app import build_app
from transcription_server.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    settings = get_settings()
    app = build_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Lancer les tests pour vérifier qu'ils passent**

Run: `.venv\Scripts\pytest tests/unit/test_app_wiring.py -v`
Expected: PASS — 3 tests

- [ ] **Step 6: Lancer toute la suite**

Run: `.venv\Scripts\pytest -v`
Expected: PASS — 91 tests

- [ ] **Step 7: Commit**

```bash
git add src/transcription_server/app.py src/transcription_server/main.py tests/unit/test_app_wiring.py
git commit -m "feat: chargement des moteurs au demarrage et point d'entree"
```

---

## Task 15: Conteneurisation

**Files:**
- Create: `docker/Dockerfile`, `docker/pin_torch.py`, `docker/entrypoint.sh`, `docker-compose.yml`, `docker-compose.dev.yml`

**Interfaces:**
- Consumes: `main.main()`, `pyproject.toml` (extra `gpu`).
- Produces: le service Compose `transcription`, publié sur `127.0.0.1:8000`.

- [ ] **Step 1: Écrire `docker/Dockerfile`**

```dockerfile
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# ffmpeg pour le decodage, libsndfile1 pour soundfile (dependance de NeMo),
# git parce que certaines dependances de NeMo se resolvent via git.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

# On fige les versions de torch deja presentes dans l'image avant d'installer
# NeMo et pyannote. Sans cette contrainte, pip peut remplacer le torch CUDA
# par une roue CPU pour satisfaire une borne de version, et le serveur
# demarrerait sans GPU sans que rien ne le signale au build.
COPY docker/pin_torch.py /tmp/pin_torch.py
RUN python /tmp/pin_torch.py /tmp/constraints.txt

RUN pip install --no-cache-dir -c /tmp/constraints.txt -e ".[gpu,dev]"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY tests/ ./tests/

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# CMD indispensable : l'entrypoint fait `exec "$@"` sous `set -u`, donc un
# `docker run` sans commande echouerait sur une variable non liee.
CMD ["python", "-m", "transcription_server.main"]
```

- [ ] **Step 2: Écrire `docker/pin_torch.py`**

Un fichier plutôt qu'un `RUN python -c` : l'échappement des quotes dans une
commande d'une seule ligne est fragile, et un heredoc dans un `RUN` dépend de
la version de la syntaxe Dockerfile.

```python
"""Fige les versions de torch et torchaudio deja presentes dans l'image.

Ecrit un fichier de contraintes pip, pour que l'installation de NeMo et
pyannote ne puisse pas remplacer le torch CUDA de l'image par une roue CPU.
"""

import sys

import torch
import torchaudio

destination = sys.argv[1]
lignes = [
    f"torch=={torch.__version__.split('+')[0]}",
    f"torchaudio=={torchaudio.__version__.split('+')[0]}",
]
with open(destination, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lignes) + "\n")

print("Contraintes figees :")
for ligne in lignes:
    print(f"  {ligne}")
```

- [ ] **Step 3: Écrire `docker/entrypoint.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Verification de l'environnement CUDA..."
python - <<'PY'
import torch
print(f"  torch          : {torch.__version__}")
print(f"  cuda disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu            : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  vram           : {free // 1024**2} Mo libres / {total // 1024**2} Mo")
PY

echo "[entrypoint] Demarrage du serveur..."
exec "$@"
```

- [ ] **Step 4: Écrire `docker-compose.yml`**

```yaml
services:
  transcription:
    build:
      context: .
      dockerfile: docker/Dockerfile
    image: transcription-parakeet:latest
    env_file: .env
    # 127.0.0.1 uniquement : le serveur n'est pas joignable depuis le reseau local.
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      # Cache des poids HuggingFace : evite de retelecharger 2,6 Go a chaque rebuild.
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      start_period: 300s
      retries: 3
    restart: unless-stopped
```

`start_period: 300s` laisse au premier démarrage le temps de télécharger les
2,6 Go de modèles sans que Compose déclare le conteneur en échec.

Aucune `command` n'est déclarée : le `CMD` du Dockerfile lance déjà le serveur.

**Pas de profil `dev`.** Compose démarre les services sans profil **en plus** de
ceux du profil demandé, donc `--profile dev` lancerait deux conteneurs sur
`127.0.0.1:8000` et le bind échouerait. Le développement passe par un fichier
d'override, à l'étape suivante.

- [ ] **Step 5: Écrire `docker-compose.dev.yml`**

```yaml
# Surcharge de developpement : code monte depuis l'hote, rechargement a chaud.
# Usage : docker compose -f docker-compose.yml -f docker-compose.dev.yml up
services:
  transcription:
    volumes:
      - ./models:/app/models
      - ./src:/app/src
      - ./tests:/app/tests
    command:
      - "uvicorn"
      - "transcription_server.app:build_app"
      - "--factory"
      - "--host=0.0.0.0"
      - "--port=8000"
      - "--reload"
      - "--reload-dir=/app/src"
```

**À savoir** : `--reload` relance le processus a chaque sauvegarde dans `src/`,
ce qui **recharge les deux modeles sur le GPU** — environ une minute. Poser
`ENABLE_DIARIZATION=false` dans `.env` pendant le developpement divise cette
attente. Le README doit le dire.

- [ ] **Step 6: Valider la syntaxe Compose avant de construire**

Run: `docker compose config -q` puis
`docker compose -f docker-compose.yml -f docker-compose.dev.yml config -q`
Expected: aucune sortie dans les deux cas. Une erreur ici coûte deux secondes,
la même erreur après le build en coûte vingt minutes.

- [ ] **Step 7: Construire l'image**

Run: `docker compose build`
Expected: succès en 15-25 min. Surveiller la fin de l'installation pip : si
`torchcodec` échoue, appliquer le repli documenté à la Task 16, étape 5.

- [ ] **Step 8: Vérifier que le GPU est vu depuis le conteneur**

Run: `docker compose run --rm transcription python -c "import torch; print(torch.cuda.get_device_name(0))"`
Expected: `NVIDIA GeForce RTX 3090`

- [ ] **Step 9: Commit**

```bash
git add docker/ docker-compose.yml docker-compose.dev.yml
git commit -m "feat: conteneurisation cuda avec compose"
```

---

## Task 16: Documentation, échantillons de test et validation bout en bout

**Files:**
- Create: `README.md`
- Create: `tests/fixtures/echantillon_fr.wav`, `tests/fixtures/deux_voix.wav`
- Create: `scripts/make_fixtures.md` (procédure de génération)

**Interfaces:**
- Consumes: tout ce qui précède.
- Produces: rien de programmatique.

- [ ] **Step 1: Produire les échantillons de test**

Les deux fixtures GPU ont besoin de vraie parole ; du silence ne prouverait
rien. Deux voies, au choix de l'utilisateur :

**Voie A — enregistrements fournis par l'utilisateur.** Demander deux fichiers :
quelques secondes de français par une personne (`echantillon_fr.wav`), et un
fichier où deux personnes parlent l'une après l'autre (`deux_voix.wav`).
Les convertir :

```powershell
ffmpeg -i source1.m4a -ac 1 -ar 16000 tests/fixtures/echantillon_fr.wav
ffmpeg -i source2.m4a -ac 1 -ar 16000 tests/fixtures/deux_voix.wav
```

**Voie B — synthèse vocale Windows**, si l'utilisateur préfère ne rien
enregistrer :

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
Write-Host "Voix disponibles : $($voices -join ', ')"

$s.SetOutputToWaveFile("$PWD\tests\fixtures\voix_a.wav")
$s.SelectVoice($voices[0])
$s.Speak("Bonjour a tous, merci de votre presence a cette reunion.")
$s.SetOutputToWaveFile("$PWD\tests\fixtures\voix_b.wav")
$s.SelectVoice($voices[1])   # une voix differente
$s.Speak("Je vous propose de commencer par le premier point de l'ordre du jour.")
$s.Dispose()

ffmpeg -i tests\fixtures\voix_a.wav -ac 1 -ar 16000 tests\fixtures\echantillon_fr.wav
ffmpeg -i "concat:tests\fixtures\voix_a.wav|tests\fixtures\voix_b.wav" -ac 1 -ar 16000 tests\fixtures\deux_voix.wav
```

**Limite à connaître** : la synthèse vocale produit des voix moins variées que
de vraies personnes, et la diarization peut avoir plus de mal à les séparer. Si
`test_deux_voix_donnent_deux_locuteurs` échoue avec des fixtures synthétiques,
essayer la voie A avant de conclure à un défaut du code.

Documenter la voie retenue dans `scripts/make_fixtures.md`.

- [ ] **Step 2: Lancer les tests GPU dans le conteneur**

Run: `docker compose run --rm transcription pytest -m gpu -v`
Expected: PASS — les tests de `tests/gpu/`

- [ ] **Step 3: Démarrer le serveur et vérifier `/health`**

```powershell
docker compose up -d
Start-Sleep -Seconds 20
curl.exe -s http://127.0.0.1:8000/health
```

Expected: `"status":"ok"`, `"device":"cuda"`, `"name":"NVIDIA GeForce RTX 3090"`

- [ ] **Step 4: Vérifier chaque critère d'acceptation de la spec**

```powershell
# Transcription simple
curl.exe -s -F "file=@tests/fixtures/echantillon_fr.wav" http://127.0.0.1:8000/transcribe

# Diarization + dialogue
curl.exe -s -F "file=@tests/fixtures/deux_voix.wav" -F "diarize=true" `
  -F "response_format=dialogue" http://127.0.0.1:8000/transcribe

# Compatibilite OpenAI en SRT
curl.exe -s -F "file=@tests/fixtures/echantillon_fr.wav" -F "response_format=srt" `
  http://127.0.0.1:8000/v1/audio/transcriptions
```

Attendu, dans l'ordre : un JSON avec du texte non vide ; deux locuteurs
distincts au format `[00:00:00.32] SPEAKER_00: ...` ; un SRT valide.

- [ ] **Step 5: Valider le découpage sur un fichier long**

Concaténer l'échantillon jusqu'à dépasser 480 s, puis transcrire :

```powershell
$lignes = 1..120 | ForEach-Object { "file 'echantillon_fr.wav'" }
$lignes | Out-File -FilePath tests\fixtures\liste.txt -Encoding ascii
ffmpeg -f concat -safe 0 -i tests\fixtures\liste.txt -c copy tests\fixtures\long.wav
curl.exe -s -F "file=@tests/fixtures/long.wav" http://127.0.0.1:8000/transcribe > long.json
```

Vérifier qu'aucun mot n'est dupliqué ni tronqué aux raccords, en inspectant les
timestamps autour des multiples de 465 s (`480 - 15`).

**Repli si le build a échoué sur `torchcodec`** : remplacer dans
`pyproject.toml` la ligne `"pyannote.audio>=4.0,<5.0"` par
`"pyannote.audio>=3.3,<4.0"`, et dans `.env` le modèle par
`pyannote/speaker-diarization-3.1`. Accepter alors les conditions des trois
dépôts : `speaker-diarization-3.1`, `segmentation-3.0` et
`wespeaker-voxceleb-resnet34-LM`. Reconstruire.

- [ ] **Step 6: Écrire `README.md`**

Le README doit couvrir, dans cet ordre :

1. Ce que fait le serveur, en deux phrases.
2. Prérequis : Docker avec runtime nvidia, un GPU NVIDIA, un compte HuggingFace.
3. **Mise en place du token** : créer le compte, accepter les conditions sur
   `https://huggingface.co/pyannote/speaker-diarization-community-1`, créer un
   token de type **read**, puis :

   ```powershell
   Add-Content -Path .env -Value "HF_TOKEN=hf_VOTRE_TOKEN" -Encoding ascii
   ```

   **Avertir explicitement** que `>` et `Out-File` sans `-Encoding ascii`
   ajoutent un BOM sur Windows, ce qui fait lire la première variable comme
   `﻿HF_TOKEN` et l'ignorer silencieusement.
4. Démarrage : `docker compose build` puis `docker compose up`. Prévenir que le
   premier lancement télécharge 2,6 Go de modèles dans `./models/`.
5. Les endpoints, avec un exemple `curl` par format de réponse.
6. Le tableau complet des variables de `.env.example`.
7. Développement : `uv venv`, `uv pip install -e ".[dev]"`, `pytest` pour la
   logique métier sans GPU ;
   `docker compose -f docker-compose.yml -f docker-compose.dev.yml up` pour le
   rechargement à chaud, en avertissant que chaque sauvegarde recharge les
   modèles sur le GPU (environ une minute) et qu'`ENABLE_DIARIZATION=false`
   réduit cette attente ;
   `docker compose run --rm transcription pytest -m gpu` pour les tests GPU.
8. Dépannage : CUDA indisponible, 503 sur token invalide, 507 sur VRAM
   insuffisante avec le conseil de baisser `CHUNK_LENGTH_S`.

- [ ] **Step 7: Commit**

```bash
git add README.md tests/fixtures/ scripts/
git commit -m "docs: readme, echantillons de test et validation bout en bout"
```

---

## Notes d'auto-revue

Vérifications effectuées après rédaction, avec les corrections déjà appliquées
dans le plan ci-dessus.

**Couverture de la spec.** Chaque section de la spec est rattachée à une tâche :
objectif et périmètre (Tasks 8-10), environnement (Task 15), architecture et
frontières (Tasks 1, 7), flux (Task 8), découpage (Task 3), alignement
(Task 2), API (Tasks 9-10), GPU (Tasks 11, 14), erreurs (Tasks 9-10),
configuration (Task 5), conteneurisation (Task 15), tests (toutes), risques
(Task 16 étape 5), critères d'acceptation (Task 16 étape 4).

**Cohérence des noms entre tâches.** `Word.text` est le nom du champ dans le
domaine, mais l'API le sérialise en `word` — c'est voulu, pour rester conforme
à OpenAI ; la conversion est isolée dans `schemas.turn_to_out`. `AsrEngine` et
`DiarizationEngine` exposent tous deux `name`, utilisé par `/health` et
`/v1/models`. `_save_upload` est défini dans `native_routes` et réutilisé par
`openai_routes` : c'est un import de fonction privée entre modules voisins,
assumé pour éviter de dupliquer la logique de limite de taille.

**Un écart assumé par rapport à la spec.**

1. ~~Reprise automatique sur saturation VRAM~~ — **tranché le 2026-08-24** : le
   mécanisme est retiré de la spec comme du plan. Un `OutOfMemoryError` remonte
   en 500 ; le levier est de baisser `CHUNK_LENGTH_S`. Le code 507 a été
   supprimé de `_ERROR_TYPES` en Task 9.
2. La spec mentionne un `Protocol` avec `list[Word]` pour l'ASR ; l'annotation
   `per_window: list[list[object]]` dans `pipeline.py` est volontairement lâche
   pour éviter un import circulaire de `Word` déjà importé indirectement. Sans
   effet à l'exécution.

**Aucun placeholder.** Aucune occurrence de « TBD », « à compléter » ou de test
décrit sans son code.
