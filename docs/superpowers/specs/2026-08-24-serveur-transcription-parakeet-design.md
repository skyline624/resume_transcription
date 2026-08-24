# Serveur de transcription Parakeet avec diarization — Design

**Date** : 2026-08-24
**Statut** : validé, prêt pour le plan d'implémentation

## 1. Objectif

Fournir un serveur HTTP local qui transcrit un fichier audio avec le modèle
NVIDIA Parakeet sur GPU CUDA, et sépare les tours de parole par locuteur.

Le serveur tourne en conteneur Docker avec accès au GPU de la machine hôte.

### Hors périmètre

- Le résumé de la transcription. Les frontières du code doivent permettre de
  l'ajouter plus tard sans restructuration, mais rien n'est écrit pour lui.
- La transcription en temps réel (flux WebSocket).
- L'identification nominale des locuteurs par empreinte vocale. Les locuteurs
  sont étiquetés `SPEAKER_00`, `SPEAKER_01`, etc.
- L'authentification des clients. Le conteneur n'est publié que sur
  `127.0.0.1:8000` côté hôte, donc le serveur n'est pas joignable depuis le
  réseau local. Exposer le port au-delà de la machine exigerait d'ajouter une
  authentification, ce qui sort du périmètre.

## 2. Environnement cible

Relevé et vérifié sur la machine de développement le 2026-08-24 :

| Élément | Valeur |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 Go |
| Driver | 610.88 |
| OS hôte | Windows 11 Home 26200 |
| Docker | 29.5.2, Compose v5.1.4 |
| Runtime `nvidia` | enregistré dans le daemon |
| WSL2 | actif, distro `docker-desktop` |
| Espace disque | C: 106 Go libres, D: 146 Go libres |

Deux points ont été vérifiés par exécution, pas par supposition :

- **Passage du GPU** : `nvidia-smi` exécuté dans `ubuntu:24.04` avec
  `--gpus all` rend bien la RTX 3090.
- **Accès HuggingFace** : le token de type *read* présent dans `.env` télécharge
  réellement `config.yaml` depuis le dépôt *gated*
  `pyannote/speaker-diarization-community-1` (HTTP 200). Les conditions du
  modèle sont donc acceptées sur le compte.

## 3. Décisions techniques

| Sujet | Choix | Raison |
|---|---|---|
| Moteur ASR | NVIDIA NeMo (PyTorch/CUDA) | Implémentation officielle de Parakeet, timestamps mot à mot natifs, gestion de l'audio long |
| Modèle ASR | `nvidia/parakeet-tdt-0.6b-v3` | 25 langues européennes dont le français, accès libre sur HuggingFace |
| Moteur de diarization | `pyannote.audio` 4.x | Référence du domaine, nombre de locuteurs non borné et détecté automatiquement |
| Modèle de diarization | `pyannote/speaker-diarization-community-1` | Poids regroupés dans un seul dépôt, donc une seule acceptation de conditions. Son `config.yaml` déclare `pyannote.audio: 4.0.0` |
| API | Compatible OpenAI + endpoint natif | Compatibilité avec les clients Whisper existants, sans renoncer aux options propres à Parakeet |
| Fichiers longs | Synchrone avec découpage interne | Une seule requête, pas de file d'attente ni de suivi d'état à construire |
| Cible d'exécution | Docker uniquement | NeMo n'est pas officiellement supporté sur Windows ; le conteneur Linux supprime ce risque |
| Image de base | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` | torch 2.11 satisfait NeMo (≥ 2.6) et pyannote (≥ 2.8) ; CUDA 12.8 est le socle le mieux éprouvé pour NeMo |

### Options écartées

- **ONNX Runtime GPU** (`onnx-asr`) : installation plus légère et très stable
  sur Windows, mais dépend de conversions communautaires et offre des
  timestamps moins riches. Reste le repli si NeMo pose problème dans le
  conteneur.
- **NeMo Sortformer 4spk** pour la diarization : resterait dans une pile unique
  et sans token, mais plafonne à 4 locuteurs — rédhibitoire pour des réunions.
- **Image NGC NeMo officielle** : zéro risque d'installation, mais 25-30 Go et
  beaucoup d'outillage d'entraînement inutile ici.
- **`pyannote/speaker-diarization-3.1`** : son dépôt ne contient qu'un
  `config.yaml` renvoyant vers deux autres dépôts *gated*. Il aurait fallu
  accepter trois pages de conditions au lieu d'une.
- **Réutilisation de `ghcr.io/fqscfqj/parakeet-api-docker`**, déjà présente sur
  la machine : ne fait ni diarization pyannote ni API compatible OpenAI. Elle
  confirme en revanche que CUDA 12.8 fonctionne avec le driver 610.88.

## 4. Architecture

Serveur FastAPI servi par Uvicorn, mono-processus. Les deux modèles sont
chargés une fois au démarrage et restent résidents en VRAM.

```
D:\developpement\resume_transcription\
├─ docker/
│  ├─ Dockerfile              # multi-stage
│  └─ entrypoint.sh           # vérification CUDA, warmup, exec uvicorn
├─ docker-compose.yml         # profils dev / prod
├─ .dockerignore
├─ .env / .env.example
├─ pyproject.toml
├─ README.md
├─ models/                    # cache HuggingFace, bind-mount, gitignoré
├─ src/transcription_server/
│  ├─ config.py               # Settings (pydantic-settings)
│  ├─ app.py                  # FastAPI + lifespan
│  ├─ runtime.py              # device, fp16, verrou GPU, empty_cache
│  ├─ audio.py                # ffmpeg → PCM mono 16 kHz float32
│  ├─ domain.py               # dataclasses Word, SpeakerSegment, Turn
│  ├─ asr/
│  │  ├─ engine.py            # Protocol AsrEngine
│  │  └─ nemo_parakeet.py     # implémentation NeMo
│  ├─ chunking.py             # découpage et recollage des fenêtres
│  ├─ diarization/
│  │  ├─ engine.py            # Protocol DiarizationEngine
│  │  └─ pyannote_engine.py
│  ├─ alignment.py            # fusion mots ↔ locuteurs → tours de parole
│  ├─ formatting.py           # json / text / srt / vtt / verbose_json / dialogue
│  └─ api/
│     ├─ openai_routes.py
│     ├─ native_routes.py
│     └─ schemas.py
└─ tests/
   ├─ unit/                   # sans GPU ni Docker
   └─ gpu/                    # dans le conteneur
```

### Frontières

Deux `Protocol` isolent les moteurs lourds :

```python
class AsrEngine(Protocol):
    def transcribe(self, audio: np.ndarray, language: str | None) -> list[Word]: ...

class DiarizationEngine(Protocol):
    def diarize(self, audio: np.ndarray, num_speakers: int | None,
                min_speakers: int | None, max_speakers: int | None
                ) -> list[SpeakerSegment]: ...
```

C'est le seul point d'abstraction. Il permet de basculer sur ONNX, de simuler
les moteurs dans les tests, et de greffer une étape de résumé en aval sans
toucher aux routes. Aucune autre couche générique spéculative n'est introduite.

### Types du domaine

```python
@dataclass(frozen=True)
class Word:
    text: str
    start: float   # secondes, absolu depuis le début du fichier
    end: float

@dataclass(frozen=True)
class SpeakerSegment:
    speaker: str   # "SPEAKER_00"
    start: float
    end: float

@dataclass(frozen=True)
class Turn:
    speaker: str | None
    start: float
    end: float
    text: str
    words: list[Word]
```

`alignment.py`, `chunking.py` et `formatting.py` ne manipulent que ces types.
Ils n'importent ni torch, ni NeMo, ni pyannote — c'est ce qui les rend
testables hors Docker.

## 5. Flux de traitement

1. Réception du fichier en multipart, écriture dans un fichier temporaire.
2. `audio.py` : appel ffmpeg → PCM mono 16 kHz float32. Accepte mp3, wav, m4a,
   ogg, flac, mp4, webm. Rejette en 400 ce que ffmpeg ne décode pas.
3. Si `diarize=true` : pyannote sur la waveform → `list[SpeakerSegment]`.
4. Parakeet → `list[Word]` avec timestamps absolus. Au-delà de
   `CHUNK_LENGTH_S`, l'audio est découpé en fenêtres avec recouvrement, puis
   les mots sont recollés.
5. `alignment.py` : chaque mot reçoit le locuteur dont le segment recouvre le
   plus son intervalle ; les mots sont regroupés en `Turn`.
6. `formatting.py` : rendu dans le format demandé.
7. `finally` : suppression du fichier temporaire.

La diarization s'exécute **avant** l'ASR, ce qui permet de libérer le modèle
pyannote de la VRAM avant l'inférence longue de Parakeet si la mémoire se tend.

## 6. Découpage des fichiers longs

Parakeet TDT gère l'audio long via attention locale, mais la VRAM reste le
facteur limitant. Paramètres par défaut sur 24 Go : fenêtres de 480 s avec
15 s de recouvrement.

**Règle de recollage** : les mots du chunk *N* sont conservés jusqu'au **milieu**
de la zone de recouvrement ; au-delà, ceux du chunk *N+1* prennent le relais.
Cela évite à la fois les doublons et les mots tronqués en bord de fenêtre.

Les timestamps de chaque chunk sont réoffsetés en absolu avant recollage.

Ordre de grandeur attendu sur RTX 3090 : 1 h d'audio ≈ 1-2 min d'ASR et
environ 1 min de diarization.

## 7. Règle d'alignement mots ↔ locuteurs

Pour chaque mot, on calcule le recouvrement temporel avec chaque segment de
locuteur et on retient le locuteur au recouvrement maximal.

Cas limites à couvrir explicitement :

- **Aucun recouvrement** (mot dans un silence non attribué) : le mot hérite du
  locuteur du mot précédent ; s'il n'y en a pas, `speaker = None`.
- **Mot à cheval sur deux locuteurs** : le recouvrement maximal tranche ; à
  égalité stricte, le locuteur du segment le plus précoce l'emporte, afin que
  la règle soit déterministe.
- **Mot de durée nulle** (`start == end`) : on teste l'appartenance du point.

**Découpage en tours** : un nouveau `Turn` démarre quand le locuteur change, ou
quand le silence entre deux mots consécutifs dépasse `TURN_GAP_S` (défaut 1,0 s).

## 8. API

### `POST /v1/audio/transcriptions` — compatible OpenAI

Champs multipart : `file` (requis), `model`, `language`, `response_format`
(`json` | `text` | `srt` | `vtt` | `verbose_json`), `temperature`,
`timestamp_granularities[]`.

La réponse suit le schéma OpenAI. `verbose_json` inclut `segments` et `words`.
Cet endpoint n'expose pas la diarization : il existe pour que les clients
Whisper existants fonctionnent sans modification.

### `POST /transcribe` — natif

Champs multipart : `file` (requis), `language`, `diarize` (défaut : valeur de
`ENABLE_DIARIZATION`), `num_speakers`, `min_speakers`, `max_speakers`,
`word_timestamps`, `response_format` (`json` | `text` | `srt` | `vtt` |
`dialogue`).

```json
{
  "text": "Bonjour à tous. Merci de votre présence.",
  "language": "fr",
  "duration": 3612.4,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "turns": [
    { "speaker": "SPEAKER_00", "start": 0.32, "end": 4.81,
      "text": "Bonjour à tous.",
      "words": [{ "word": "Bonjour", "start": 0.32, "end": 0.79 }] }
  ],
  "timing": { "decode": 0.4, "asr": 18.2, "diarization": 9.1 }
}
```

Format `dialogue`, rendu en `text/plain` :

```
[00:00:00.32] SPEAKER_00: Bonjour à tous.
[00:00:04.90] SPEAKER_01: Merci de votre présence.
```

### `GET /health`

Rend le device effectif, la VRAM totale et libre, les modèles chargés et leur
état de warmup. Sert aussi de `healthcheck` Docker.

### `GET /v1/models`

Liste les modèles chargés, au format OpenAI.

## 9. Gestion du GPU

- `DEVICE` configurable. Si `cuda` est demandé mais indisponible, le serveur
  **échoue au démarrage** avec un message explicite. Aucun repli CPU
  silencieux : une transcription vingt fois plus lente que prévu doit être un
  choix, jamais une surprise.
- `COMPUTE_TYPE=float16` par défaut sur CUDA.
- Un `asyncio.Lock` sérialise les accès GPU. Les requêtes concurrentes font la
  queue au lieu de se disputer la VRAM.
- L'inférence s'exécute dans un threadpool (`run_in_threadpool`) pour ne pas
  bloquer la boucle événementielle.
- Warmup sur 1 s de silence au démarrage, afin que la première requête réelle
  n'encaisse pas la compilation des kernels.
- **Pas de reprise automatique sur saturation VRAM.** Un
  `torch.cuda.OutOfMemoryError` remonte en 500. Le mécanisme envisagé (réessayer
  avec des fenêtres divisées par deux) a été retiré : il est difficile à tester
  sans provoquer un vrai dépassement, et sa valeur reste théorique tant qu'aucun
  OOM n'a été observé sur 24 Go. À reconsidérer si le cas se présente
  réellement ; le levier immédiat est alors de baisser `CHUNK_LENGTH_S`.

## 10. Erreurs

| Code | Cas |
|---|---|
| 400 | Audio illisible ou vide, format non décodable par ffmpeg, paramètres incohérents (`num_speakers` combiné à `min_speakers`/`max_speakers`) |
| 413 | Fichier au-delà de `MAX_UPLOAD_MB` |
| 503 | Modèle non chargé, ou diarization demandée sans `HF_TOKEN` valide |
| 500 | Erreur d'inférence inattendue, saturation VRAM comprise, avec identifiant de corrélation en logs |

Les réponses d'erreur suivent le format OpenAI
(`{"error": {"message", "type"}}`) sur les deux familles d'endpoints, par
cohérence.

Le fichier temporaire est supprimé dans un `finally`, y compris en cas d'erreur.

## 11. Configuration

Variables lues depuis `.env` par `pydantic-settings` :

| Variable | Défaut | Rôle |
|---|---|---|
| `HF_TOKEN` | — | Token HuggingFace, type *read*. Requis seulement si la diarization est active |
| `ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Modèle de transcription |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | Modèle de diarization |
| `ENABLE_DIARIZATION` | `true` | Permet de démarrer sans token |
| `DEVICE` | `cuda` | `cuda` ou `cpu` |
| `COMPUTE_TYPE` | `float16` | `float16` ou `float32` |
| `CHUNK_LENGTH_S` | `480` | Longueur des fenêtres |
| `CHUNK_OVERLAP_S` | `15` | Recouvrement entre fenêtres |
| `TURN_GAP_S` | `1.0` | Silence déclenchant un nouveau tour de parole |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Écoute **dans** le conteneur |
| `MAX_UPLOAD_MB` | `1024` | Taille maximale acceptée |

`.env` est dans le `.gitignore`. Le token n'est jamais inscrit dans une couche
d'image : il est injecté à l'exécution par Compose.

Le fichier `.env` doit être écrit **sans BOM**. Sur Windows, `>` et `Out-File`
en ajoutent un par défaut, ce qui ferait lire la première variable comme
`\ufeffHF_TOKEN` et l'ignorer silencieusement. Le README documente ce piège.

## 12. Conteneurisation

**Image** : multi-stage à partir de
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`. `ffmpeg` est installé via
apt, ce qui supprime toute dépendance à l'installation Windows. Taille
attendue : 14-16 Go.

**Les poids ne sont pas dans l'image.** `./models` est monté sur `HF_HOME`. Les
~2,6 Go de modèles sont téléchargés au premier démarrage et survivent à tous
les rebuilds. Cela garde l'image plus petite et évite d'avoir besoin du token
au moment du build.

**Deux profils Compose** :

- `dev` : bind-mount de `src/`, `uvicorn --reload`. Édition sous Windows,
  rechargement sans rebuild.
- `prod` : code figé dans l'image.

Dans les deux cas : réservation GPU via
`deploy.resources.reservations.devices`, `healthcheck` sur `/health`, et
publication sur **`127.0.0.1:8000`** uniquement — le serveur n'est pas joignable
depuis le réseau local.

## 13. Stratégie de test

La logique délicate ne dépend ni de torch, ni du GPU, ni de Docker : elle prend
des `Word` et des `SpeakerSegment` et rend des `Turn`.

**Tests unitaires** (`tests/unit/`), exécutables dans un venv Windows contenant
seulement `pytest` et `pydantic` — cycle de quelques secondes :

- `alignment.py` : recouvrement partiel, mot à cheval, égalité stricte, mot
  dans un silence, mot de durée nulle, aucun segment de locuteur.
- `chunking.py` : recollage sans doublon ni troncature, réoffset des
  timestamps, audio plus court qu'une fenêtre, mot chevauchant exactement le
  point de bascule.
- `formatting.py` : SRT, VTT, `dialogue`, échappement, horodatage.
- Validation des paramètres d'API.

**Tests GPU** (`tests/gpu/`), marqués `@pytest.mark.gpu` et exécutés dans le
conteneur : transcription d'un court échantillon français, et détection de deux
locuteurs sur deux voix concaténées. Ils sont ignorés automatiquement si
`torch.cuda.is_available()` est faux.

C'est ce découpage qui rend le développement piloté par les tests praticable
malgré une image de 16 Go.

## 14. Risques connus

1. **`torchcodec`**, dépendance de pyannote 4, doit correspondre à la version
   de torch de l'image. Le couple sera verrouillé au build. Si le conflit
   résiste, le repli est `pyannote.audio` 3.x, qui ne l'exige pas — au prix de
   trois acceptations de conditions au lieu d'une.
2. **NeMo dans le conteneur** : le risque est fortement réduit par rapport à
   Windows, mais `lhotse` et `numba` restent les points de friction plausibles.
   Repli : ONNX Runtime, déjà vérifié installable.
3. **Premier build long** : 15-25 min selon la bande passante.

Le risque « token HuggingFace » est levé : la validité du token et l'accès au
dépôt *gated* ont été vérifiés par téléchargement réel le 2026-08-24.

## 15. Critères d'acceptation

- `docker compose up` démarre le serveur, `GET /health` répond `device: cuda`
  avec la RTX 3090 identifiée.
- `POST /transcribe` sur un fichier français rend une transcription non vide.
- Avec `diarize=true` sur un audio à deux voix, la réponse contient deux
  locuteurs distincts et des tours de parole cohérents.
- `POST /v1/audio/transcriptions` avec `response_format=srt` rend un SRT valide.
- Un fichier de plus de 480 s est transcrit sans doublon ni coupure au niveau
  des raccords de fenêtres.
- Les tests unitaires passent sur Windows sans GPU ni Docker.
