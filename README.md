# Serveur de transcription Parakeet

Serveur HTTP local qui transcrit un fichier audio avec **NVIDIA Parakeet** sur
GPU CUDA et sépare les tours de parole par locuteur avec **pyannote**.

Il tourne en conteneur Docker : NeMo n'est pas officiellement supporté sous
Windows, le conteneur Linux supprime ce risque.

---

## Prérequis

- Un GPU NVIDIA et son pilote à jour (développé sur une RTX 3090, 24 Go).
- **Docker** avec le runtime `nvidia` actif. Vérification :
  ```powershell
  docker run --rm --gpus all ubuntu:24.04 nvidia-smi
  ```
  Si cette commande n'affiche pas votre carte, rien d'autre ne fonctionnera.
- Un compte **HuggingFace**, uniquement si vous voulez la diarization.

## Mise en place du token HuggingFace

La diarization utilise un modèle *gated* : il faut accepter ses conditions une
fois, puis fournir un token.

1. Créez un compte sur [huggingface.co](https://huggingface.co).
2. Sur la page de
   [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1),
   remplissez le court formulaire. L'accès est accordé **immédiatement**, sans
   validation humaine.
3. *Settings → Access Tokens → New token*, de type **read**. Un token *write*
   n'apporte rien ici.
4. Écrivez-le dans `.env` :

   ```powershell
   Add-Content -Path .env -Value "HF_TOKEN=hf_VOTRE_TOKEN" -Encoding ascii
   ```

> **Le piège à connaître.** Sur Windows, `>` et `Out-File` écrivent un **BOM**
> UTF-8 en tête de fichier. Docker Compose lirait alors la première variable
> comme `﻿HF_TOKEN` et **ignorerait silencieusement votre token**.
> `Add-Content -Encoding ascii` l'évite ; un token HuggingFace est purement
> ASCII.

Vous pouvez partir de `.env.example` pour les autres variables.

**Sans token**, posez `ENABLE_DIARIZATION=false` : le serveur démarre et
transcrit, sans séparer les locuteurs.

## Démarrage

```powershell
docker compose build     # 15 à 25 min, une seule fois
docker compose up
```

Le **premier lancement télécharge environ 2,6 Go de modèles** dans `./models/`.
Ce dossier est monté depuis l'hôte : les poids survivent à toutes les
reconstructions, et le token n'entre dans aucune couche de l'image.

Vérifiez que tout est en place :

```powershell
curl.exe -s http://127.0.0.1:8000/health
```

Vous devez y lire `"device":"cuda"` et le nom de votre carte.

## Utilisation

### Transcription simple

```powershell
curl.exe -s -F "file=@reunion.mp3" http://127.0.0.1:8000/transcribe
```

### Avec séparation des locuteurs

```powershell
curl.exe -s -F "file=@reunion.mp3" -F "diarize=true" `
  -F "response_format=dialogue" http://127.0.0.1:8000/transcribe
```

```
[00:00:00.32] SPEAKER_00: Bonjour à tous.
[00:00:04.90] SPEAKER_01: Merci de votre présence.
```

### Endpoints

| Endpoint | Rôle |
|---|---|
| `POST /transcribe` | Transcription native, avec diarization et formats étendus |
| `POST /v1/audio/transcriptions` | **Compatible OpenAI** — vos clients Whisper fonctionnent sans modification |
| `GET /health` | État du service, périphérique, VRAM, modèles chargés |
| `GET /v1/models` | Liste des modèles, au format OpenAI |

**Paramètres de `/transcribe`** : `file` (requis), `language`, `diarize`,
`num_speakers`, `min_speakers`, `max_speakers`, `word_timestamps`,
`response_format` parmi `json`, `text`, `srt`, `vtt`, `dialogue`.

`num_speakers` fixe un nombre exact et s'exclut de `min_speakers`/`max_speakers`.

**`POST /v1/audio/transcriptions`** accepte les champs OpenAI habituels et rend
`json`, `text`, `srt`, `vtt` ou `verbose_json`. La diarization n'y est pas
exposée : elle n'a pas d'équivalent OpenAI, et un champ supplémentaire romprait
la compatibilité que cet endpoint existe pour offrir.

## Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `HF_TOKEN` | — | Token HuggingFace de type *read*. Requis si la diarization est active |
| `ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | 25 langues européennes, dont le français |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | Locuteurs non bornés |
| `ENABLE_DIARIZATION` | `true` | `false` permet de démarrer sans token |
| `DEVICE` | `cuda` | `cuda` ou `cpu` |
| `COMPUTE_TYPE` | `float16` | `float16` ou `float32` |
| `CHUNK_LENGTH_S` | `480` | Longueur des fenêtres d'inférence |
| `CHUNK_OVERLAP_S` | `15` | Recouvrement entre fenêtres |
| `TURN_GAP_S` | `1.0` | Silence déclenchant un nouveau tour de parole |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Écoute **dans** le conteneur |
| `MAX_UPLOAD_MB` | `1024` | Taille maximale acceptée |

`HOST=0.0.0.0` est l'adresse d'écoute interne au conteneur ; la publication
reste restreinte à `127.0.0.1:8000` côté hôte.

## Développement

Toute la logique métier — alignement mots/locuteurs, découpage des fenêtres,
formatage, routes HTTP — est testable **sans GPU ni Docker**, dans un
environnement d'une cinquantaine de mégaoctets :

```powershell
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
```

Les tests d'inférence réelle sont marqués `gpu` et désélectionnés par défaut.
Dans le conteneur :

```powershell
docker compose run --rm transcription pytest -m gpu
```

Rechargement à chaud pendant le développement :

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Chaque sauvegarde dans `src/` **recharge les modèles sur le GPU**, soit environ
une minute. `ENABLE_DIARIZATION=false` divise cette attente.

## Dépannage

**`/health` répond `"device":"cpu"` ou le serveur refuse de démarrer.**
C'est volontaire : il n'y a **aucun repli CPU silencieux**. Une transcription
vingt fois plus lente doit être un choix, jamais une surprise. Vérifiez que le
conteneur tourne avec le runtime `nvidia`, ou posez `DEVICE=cpu` pour accepter
explicitement la lenteur.

**Erreur au chargement de pyannote.**
Le compte HuggingFace n'a pas accepté les conditions du modèle, ou le token
n'est pas de type *read*. Vérifiez aussi l'absence de BOM dans `.env`.

**`400 : La diarization est désactivée sur ce serveur.`**
Vous avez demandé `diarize=true` sur une instance démarrée avec
`ENABLE_DIARIZATION=false`. Le serveur refuse explicitement plutôt que de
rendre une liste de locuteurs vide, qui vous ferait croire à un enregistrement
mono-locuteur.

**Mémoire GPU insuffisante sur un fichier long.**
Baissez `CHUNK_LENGTH_S`. La valeur par défaut de 480 s est calibrée pour
24 Go.

## Limites connues

- **La langue détectée ne remonte pas.** Parakeet v3 l'identifie, mais le
  moteur est appelé une fois par fenêtre : la faire remonter exigerait une
  règle de réconciliation entre N réponses. Le champ `language` renvoie donc ce
  que vous avez demandé, ou `null`.
- **Recollage des fenêtres.** Sur un fichier de plus de 8 minutes, un mot situé
  exactement à la jointure de deux fenêtres peut être perdu ou dupliqué si les
  deux fenêtres ne l'horodatent pas identiquement. Quelques mots par heure au
  pire.
- **Sous-titres non découpés.** Un tour de parole long produit un seul bloc
  SRT/VTT, ce qui déborde de l'écran. La sortie convient à une transcription
  horodatée, pas à du sous-titrage prêt à diffuser.
- **Taille décodée non bornée.** `MAX_UPLOAD_MB` plafonne le fichier compressé,
  pas son expansion en mémoire (1 Go de mp3 ≈ 17 h ≈ 3,9 Go de flottants).
  Acceptable pour un serveur local.
