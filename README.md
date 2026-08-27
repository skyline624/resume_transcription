# Serveur audio Parakeet et Qwen3-TTS

Serveur HTTP local qui transcrit avec **NVIDIA Parakeet**, sépare les locuteurs
avec **pyannote** et synthétise ou clone une voix avec **Qwen3-TTS 1.7B**.

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

Le **premier lancement télécharge environ 16 Go de modèles** dans `./models/`
(Parakeet, pyannote et les trois checkpoints Qwen spécialisés). Selon la
connexion, comptez plusieurs dizaines de minutes avant le premier état sain.
Ce dossier est monté depuis l'hôte : les poids survivent à toutes les
reconstructions, et le token n'entre dans aucune couche de l'image.

Vérifiez que tout est en place :

```powershell
curl.exe -s http://127.0.0.1:8000/health
```

Vous devez y lire `"device":"cuda"` et le nom de votre carte.

## Utilisation

### Synthèse vocale compatible OpenAI

Les trois checkpoints Qwen vivent dans le même conteneur que Parakeet, mais
dans un environnement Python et un processus séparés. Le worker n'écoute sur
aucun port réseau : l'API principale lui parle par le socket Unix privé
`/run/qwen-tts/worker.sock`. Un seul checkpoint occupe la VRAM à la fois.

```powershell
curl.exe -sS http://127.0.0.1:8000/v1/audio/speech `
  -H "Content-Type: application/json" `
  -d '{"model":"tts-1-hd","voice":"Ryan","input":"Bonjour depuis Qwen.","response_format":"wav"}' `
  --output parole.wav
```

Avec le SDK OpenAI 2.x :

```python
from openai import OpenAI

with OpenAI(api_key="local", base_url="http://127.0.0.1:8000/v1") as client:
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd",
        voice="Ryan",
        input="Bonjour depuis le client OpenAI.",
        response_format="wav",
    ) as response:
        response.stream_to_file("parole.wav")
```

| Mode | `model` | Champs spécifiques |
|---|---|---|
| Voix prédéfinie | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` ou `qwen3-tts-custom-voice` | `voice` parmi `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`; `instructions` facultatif |
| Création de voix | `qwen3-tts-voice-design` | `instructions` requis, par exemple « voix française chaleureuse et calme » |
| Clone persistant | `qwen3-tts-clone` | `voice` est l'identifiant créé par `POST /v1/voices` |

Formats de sortie : `mp3`, `opus` (conteneur Ogg), `aac`, `flac`, `wav` et
`pcm`. La vitesse va de `0.25` à `4.0`. Un texte de plus de 4096 caractères est
refusé explicitement.

Création et cycle de vie d'une voix consentie :

```powershell
curl.exe -sS http://127.0.0.1:8000/v1/voices `
  -F "file=@reference.wav" -F "name=Ma voix" -F "language=fr" `
  -F "transcript=Texte prononcé dans la référence." -F "consent=true"

curl.exe -sS http://127.0.0.1:8000/v1/voices
curl.exe -X DELETE http://127.0.0.1:8000/v1/voices/IDENTIFIANT
```

Le consentement explicite est obligatoire. La référence doit durer de 3 à
30 secondes, ne pas être silencieuse ni saturée, et est normalisée avant
stockage dans le volume `tts-voices`. Le clonage ponctuel est disponible via
`POST /v1/audio/speech/clone`; son fichier temporaire est supprimé après la
requête. Aucun texte synthétisé ni chemin de référence n'est journalisé.

`GET /health` expose l'état `idle`, `loading`, `ready`, `generating` ou
`error`, les checkpoints téléchargés, le checkpoint chargé, `bfloat16`, CUDA,
SDPA et le PID du worker. Une panne, un délai ou un manque de VRAM renvoie 503
avec un code stable ; il n'existe aucun repli CPU silencieux.

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
| `POST /v1/audio/speech` | Synthèse Qwen compatible avec le client audio OpenAI |
| `POST /v1/audio/speech/clone` | Clonage ponctuel avec référence consentie |
| `GET/POST /v1/voices`, `DELETE /v1/voices/{id}` | Registre des voix prédéfinies et clonées |
| `POST /transcribe` | Transcription native, avec diarization et formats étendus |
| `POST /summarize` | **Compte-rendu** rédigé par un modèle de langue local |
| `POST /v1/audio/transcriptions` | **Compatible OpenAI** — vos clients Whisper fonctionnent sans modification |
| `GET /health` | État du service, périphérique, VRAM, modèles chargés |
| `GET /v1/models` | Liste des modèles, au format OpenAI |

**Paramètres de `/transcribe`** : `file` (requis), `language`, `diarize`,
`num_speakers`, `min_speakers`, `max_speakers`, `word_timestamps`, `channels`,
`response_format` parmi `json`, `text`, `srt`, `vtt`, `dialogue`.

`num_speakers` fixe un nombre exact et s'exclut de `min_speakers`/`max_speakers`.

> Parakeet v3 détecte la langue lui-même et n'expose aucun moyen de la forcer.
> `language=fr` active néanmoins un garde-fou local : si une fenêtre de 5 s
> produit une sortie nettement anglaise, elle est retentée par sous-segments de
> 3 s. Les autres langues conservent l'auto-détection native du modèle.

### Enregistrements multi-pistes : `channels=split`

Certains outils enregistrent **une source par canal** — votre micro à gauche,
les participants distants à droite. Replié en mono, deux paroles simultanées se
superposent et **les deux deviennent inintelligibles**.

`channels=split` transcrit chaque canal séparément, diarise chacun de son côté,
puis refusionne par horodatage. Mesuré sur un enregistrement réel de ce type :

| | `mix` (défaut) | `split` |
|---|---|---|
| Mots récupérés | 54 | **207** |
| Locuteurs | 2 | **4** |
| Bascules parasites vers l'anglais | 14 | **0** |

```powershell
curl.exe -s -F "file=@appel.wav" -F "channels=split" -F "diarize=true" `
  -F "response_format=dialogue" http://127.0.0.1:8000/transcribe
```

**Aucune détection automatique** : le mode est explicite et vaut `mix` par
défaut. Une bascule automatique par corrélation des canaux a été essayée puis
abandonnée — elle confond « un canal est vide » et « les canaux portent des
contenus différents », les deux donnant une corrélation nulle.

Pour savoir si un fichier s'y prête, comparez les deux modes sur un extrait, ou
inspectez les canaux :

```powershell
ffmpeg -i enregistrement.wav -filter_complex "[0:a]channelsplit=channel_layout=stereo[g][d]" -map "[g]" g.wav -map "[d]" d.wav
```

La réponse indique toujours `channels_used`, pour que vous sachiez ce que le
serveur a réellement fait de votre fichier.

**`POST /v1/audio/transcriptions`** accepte les champs OpenAI habituels et rend
`json`, `text`, `srt`, `vtt` ou `verbose_json`. La diarization n'y est pas
exposée : elle n'a pas d'équivalent OpenAI, et un champ supplémentaire romprait
la compatibilité que cet endpoint existe pour offrir.

## Compte-rendu

`POST /summarize` rédige un compte-rendu, à partir d'un fichier audio ou d'une
transcription déjà produite. La rédaction passe par **Ollama**, sur votre
machine : rien ne sort, comme pour la transcription.

```powershell
# Depuis une transcription existante
curl.exe -s -X POST http://127.0.0.1:8000/summarize `
  -F "transcript=<reunion.txt" -F "format=structure" -F "response_format=text"

# Directement depuis l'audio
curl.exe -s -X POST http://127.0.0.1:8000/summarize `
  -F "file=@reunion.wav" -F "channels=split" -F "format=narratif"
```

Deux gabarits : `structure` produit des sections fixes — objet, participants,
sujets, **décisions**, **actions à mener**, points en suspens ; `narratif`
produit quelques paragraphes de prose.

**Fournissez soit `file`, soit `transcript`, jamais les deux.** L'endpoint est
séparé de `/transcribe` précisément pour que rejouer un compte-rendu avec un
autre gabarit ne coûte pas de retranscrire une réunion de deux heures.

Le modèle doit exister dans votre installation Ollama (`ollama list`). Un
modèle suffixé `:cloud` **ferait transiter la conversation par les serveurs
d'Ollama** — à ne poser qu'en connaissance de cause.

Mesuré avec `qwen3.8-27b` (Q4_K_M) : **1 min 53** pour une réunion de 51 min,
**5 min 43** pour une réunion de 110 min (17 400 tokens).

> **Les deux modèles ne tiennent pas ensemble sur 24 Go.** Ollama occupe 17,5 Go
> et la transcription en demande jusqu'à 21. Appelés l'un après l'autre, ils se
> séquencent naturellement — la transcription rend sa mémoire avant qu'Ollama
> charge la sienne. Mais **si Ollama tient encore son modèle quand vous lancez
> une transcription, celle-ci échouera faute de place**. Deux parades :
>
> ```powershell
> ollama stop qwen3.8-27b-64k:latest        # libérer immédiatement
> $env:OLLAMA_KEEP_ALIVE = "60"             # ou raccourcir la rétention à 60 s
> ```
>
> Cela concerne surtout `/summarize` appelé avec `file` : il transcrit puis
> rédige dans la même requête. Sur une carte plus petite, prenez un modèle de
> rédaction plus léger.

Si Ollama est absent ou le modèle introuvable, `/summarize` répond **503** avec
un message actionnable — et la transcription, elle, continue de fonctionner.

## Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `HF_TOKEN` | — | Token HuggingFace de type *read*. Requis si la diarization est active |
| `ASR_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | 25 langues européennes, dont le français |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` | Locuteurs non bornés |
| `ENABLE_DIARIZATION` | `true` | `false` permet de démarrer sans token |
| `DEVICE` | `cuda` | `cuda` ou `cpu` |
| `COMPUTE_TYPE` | `float16` | `float16` ou `float32` |
| `ENABLE_VAD` | `true` | Active la détection de parole Silero avant Parakeet |
| `VAD_DEVICE` | `cpu` | `cpu` recommandé ; `cuda` reste disponible |
| `VAD_MAX_SEGMENT_S` | `5` | Durée maximale envoyée à Parakeet par passage parlé |
| `VAD_FALLBACK_OVERLAP_S` | `1` | Recouvrement des fenêtres fixes si Silero échoue |
| `CHUNK_LENGTH_S` | `480` | Longueur des fenêtres si le VAD est désactivé |
| `CHUNK_OVERLAP_S` | `15` | Recouvrement si le VAD est désactivé |
| `TURN_GAP_S` | `1.0` | Silence déclenchant un nouveau tour de parole |
| `ENABLE_TTS` | `true` | Active Qwen3-TTS et le worker privé |
| `TTS_CUSTOM_VOICE_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Voix prédéfinies |
| `TTS_CLONE_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Clonage zero-shot |
| `TTS_VOICE_DESIGN_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Création de voix par instruction |
| `TTS_PRECISION` | `bfloat16` | Précision GPU du worker |
| `TTS_ATTENTION_IMPLEMENTATION` | `sdpa` | Attention PyTorch, sans FlashAttention obligatoire |
| `TTS_IDLE_UNLOAD_S` | `300` | Délai avant restitution automatique de la VRAM |
| `TTS_LOAD_TIMEOUT_S` / `TTS_GENERATION_TIMEOUT_S` | `300` / `900` | Délais worker |
| `VOICE_STORE_PATH` | `/app/voices` | Volume persistant des profils vocaux consentis |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Écoute **dans** le conteneur |
| `MAX_UPLOAD_MB` | `1024` | Taille maximale acceptée |
| `SUMMARY_MODEL` | `qwen3.8-27b-64k:latest` | Modèle Ollama rédigeant le compte-rendu |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama vu depuis le conteneur |
| `ENABLE_SUMMARY` | `true` | `false` désactive `/summarize` sans toucher au reste |
| `SUMMARY_TIMEOUT_S` | `900` | Délai maximal de rédaction |

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
Baissez `VAD_MAX_SEGMENT_S`. Si le VAD est désactivé, baissez plutôt
`CHUNK_LENGTH_S` — sa valeur par défaut de 480 s est calibrée pour 24 Go.
Sachez cependant que **c'est la diarization qui domine, pas la transcription** :
elle traite le fichier d'un seul tenant, sans découpage. Sur 110 minutes elle
occupe environ 12 Go et représente 83 % du temps de traitement. Si la mémoire
manque, `ENABLE_DIARIZATION=false` libère bien plus que n'importe quel réglage
de fenêtre.

**Trop de locuteurs détectés.**
Sur une réunion de 110 minutes à 6-7 personnes, pyannote en a rapporté 11 : il
sur-segmente quand les voix se ressemblent ou que la qualité varie. Bornez-le
avec `max_speakers`, ou fixez `num_speakers` si vous connaissez le nombre
exact.

## Performances mesurées

Relevé sur une RTX 3090, modèles déjà téléchargés :

| | |
|---|---|
| Démarrage (chargement des deux modèles + préchauffage) | ~150 s |
| Transcription seule | ~95× le temps réel |
| Diarization | ~12× le temps réel |
| VRAM au repos | ~2,7 Go sur 24 |
| VRAM après une requête | ~3,1 Go, **stable** sur des appels répétés |

Relevés bout en bout, sur des enregistrements réels :

| Fichier | Mode | Transcription | Compte-rendu |
|---|---|---|---|
| 51 min, stéréo | `mix` | 2 min 28 | — |
| 51 min, stéréo | `split` | 3 min 09 | 1 min 53 |
| 110 min, stéréo | `split` | 6 min 29 | 5 min 43 |

**La diarization domine largement** : 557 s sur les 667 s des 110 minutes, soit
83 %. La transcription elle-même n'en prend que 70. C'est aussi elle qui
consomme la VRAM, car elle traite le fichier entier sans le découper.

Le démarrage (~150 s) est payé une fois : gardez le conteneur allumé plutôt que
de le relancer à chaque fichier.

**Limite d'upload.** Un enregistrement de deux heures en WAV 48 kHz stéréo pèse
1,2 Go et dépasse le plafond de 1 Go. Convertissez-le d'abord — le serveur
ramène de toute façon tout à 16 kHz, donc **sans aucune perte de qualité** :

```powershell
ffmpeg -i long.wav -ac 2 -ar 16000 -c:a pcm_s16le pret.wav
```

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
