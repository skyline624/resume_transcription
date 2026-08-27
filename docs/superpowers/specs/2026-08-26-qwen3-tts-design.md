# Synthèse vocale Qwen3-TTS dans le serveur de transcription — Design

**Date** : 2026-08-26  
**Statut** : validé, prêt pour le plan d'implémentation

## 1. Objectif

Étendre le serveur existant avec une synthèse vocale française de meilleure
qualité que le module TTS de VoxMind, tout en conservant :

- une API compatible OpenAI ;
- le clonage de voix ;
- la création de voix par instruction ;
- un unique conteneur Docker et un unique port HTTP public ;
- l'exécution CUDA en précision réduite sur la RTX 3090 de 24 Go.

Le moteur retenu est **Qwen3-TTS 12 Hz 1.7B**. Trois checkpoints spécialisés
sont disponibles sur disque, mais un seul est chargé en VRAM à la fois :

| Mode d'API | Checkpoint | Usage |
|---|---|---|
| `qwen3-tts-custom-voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | voix prédéfinies et contrôle par instructions |
| `qwen3-tts-clone` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | clonage à partir d'une référence vocale |
| `qwen3-tts-voice-design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | création d'une voix décrite en langage naturel |

La langue par défaut est le français (`fr`). Le client peut demander une autre
langue supportée ou `auto` explicitement.

## 2. Périmètre

### Inclus

- synthèse Qwen3-TTS CustomVoice, Base/clone et VoiceDesign ;
- profils vocaux persistants créés à partir d'un fichier de référence ;
- clonage ponctuel sans persistance ;
- transcription automatique de la référence par Parakeet lorsque le texte
  exact n'est pas fourni ;
- réponses WAV, MP3, FLAC, OGG, AAC et PCM ;
- gestion des textes longs par segmentation linguistique ;
- supervision des processus FastAPI/NeMo et Qwen dans le même conteneur ;
- mesures de qualité, latence et consommation VRAM face à VoxMind.

### Hors périmètre de la première version

- flux audio progressif phrase par phrase : la réponse est envoyée après la
  génération complète ;
- exécution CPU silencieuse lorsque CUDA est indisponible ;
- repli automatique vers Magpie, Kokoro ou un autre moteur ;
- entraînement ou fine-tuning d'un modèle ;
- authentification réseau : le service reste publié sur
  `127.0.0.1:8000` comme aujourd'hui ;
- évaluation automatique de la similarité du locuteur avec un nouveau modèle
  d'embedding, qui alourdirait inutilement le conteneur pour la V1.

## 3. Choix du moteur

Qwen3-TTS est retenu face aux alternatives examinées :

- le Magpie TTS multilingue 357M actuel de NeMo est plus simple à intégrer,
  mais son dernier checkpoint public ne fournit plus le clonage zero-shot ;
- Kokoro et F5-TTS sont déjà présents dans VoxMind, mais l'objectif est
  précisément d'améliorer la qualité perçue ;
- Fish Audio S2 Pro demande au moins 24 Go de VRAM selon sa documentation et
  sa licence est moins adaptée ;
- Higgs Audio est plus volumineux et soumis à une licence de recherche ;
- le classement public Hugging Face TTS Arena V2 ne contient pas encore
  Qwen3-TTS, de sorte que la supériorité sur VoxMind doit être mesurée dans ce
  projet et non présumée à partir de chiffres incomparables.

Qwen3-TTS apporte le meilleur compromis entre qualité annoncée, français,
clonage, VoiceDesign, taille 1.7B et licence Apache 2.0.

## 4. Contrainte de dépendances

Le serveur NeMo opérationnel utilise actuellement Python 3.12, NeMo 3.0.0,
PyTorch 2.11.0+cu128 et Transformers 5.16.1. Le paquet Qwen TTS officiel fixe
notamment Transformers 4.57.3 et Accelerate 1.12.0. Installer les deux piles
dans le même environnement Python risquerait donc de casser le serveur de
transcription existant.

La solution conserve un seul conteneur mais crée deux environnements Python :

- `/opt/venv-main` pour FastAPI, NeMo, Parakeet, Silero et pyannote ;
- `/opt/venv-qwen` pour le worker Qwen et ses versions compatibles.

Cette séparation est une frontière de compatibilité, pas un second service
Docker. Le conteneur, son système de fichiers, le cache Hugging Face et le GPU
restent communs.

## 5. Architecture d'exécution

```text
Client OpenAI ou HTTP
        |
        v
127.0.0.1:8000 — FastAPI principal (/opt/venv-main)
        |             |             |
        |             |             +-- Parakeet / Silero / pyannote / Ollama
        |             +-- stockage des profils vocaux
        +-- arbitre GPU unique
                    |
                    v
          socket Unix privé /run/qwen-tts/worker.sock
                    |
                    v
          Worker Qwen (/opt/venv-qwen)
          zéro ou un checkpoint en VRAM
```

Le point d'entrée du conteneur supervise les deux processus. Il démarre le
worker, attend que son socket soit prêt, puis démarre l'API principale. Il
propage les signaux d'arrêt aux deux processus. La mort inattendue de l'un des
processus rend le conteneur défaillant et provoque sa sortie, afin que la
politique `restart` de Docker le relance au lieu de laisser une API
partiellement fonctionnelle.

Le worker expose un petit protocole HTTP/JSON sur socket Unix. Aucun port TCP
supplémentaire n'est écouté ni publié. Seul le processus principal accepte des
requêtes externes.

### Responsabilités du processus principal

- validation de l'API et des fichiers ;
- profils vocaux, consentement et durée de vie des fichiers temporaires ;
- segmentation du texte et assemblage des segments audio ;
- arbitrage de toutes les opérations GPU ;
- conversion ffmpeg, vitesse, loudness et format final ;
- traduction des erreurs du worker au format OpenAI ;
- état de santé global.

### Responsabilités du worker Qwen

- chargement, réutilisation, changement et déchargement des checkpoints ;
- prétraitement propre au paquet Qwen ;
- génération BF16 sur CUDA ;
- retour de PCM/WAV 24 kHz et des métriques d'inférence ;
- nettoyage CUDA sur changement de modèle, timeout et erreur mémoire.

Le worker ne connaît ni les routes publiques, ni la base de profils, ni les
chemins choisis par les clients.

## 6. Cycle de vie GPU

Le verrou GPU déjà utilisé par le serveur devient l'arbitre unique. Toutes les
requêtes ASR, diarization et TTS sont sérialisées par le processus principal.
Le worker ne reçoit donc jamais deux générations concurrentes.

Les règles sont les suivantes :

1. zéro ou un checkpoint Qwen est chargé à un instant donné ;
2. deux appels consécutifs au même mode réutilisent le checkpoint chargé ;
3. un changement de mode décharge le checkpoint courant, exécute
   `gc.collect()` et `torch.cuda.empty_cache()`, puis charge le suivant ;
4. Qwen est impérativement déchargé avant une diarization ;
5. Qwen est déchargé après une durée d'inactivité configurable ;
6. une exception CUDA, un OOM ou un timeout invalide le modèle courant et
   déclenche le nettoyage avant la requête suivante ;
7. aucune génération ne bascule automatiquement sur le CPU.

La précision cible est BF16 avec SDPA. FlashAttention 2 n'est activé que si un
benchmark local démontre un gain utile sans régression de stabilité ou de
qualité.

L'image contient les dépendances des trois variantes. Au premier démarrage,
les checkpoints configurés sont téléchargés dans un volume de modèles
persistant, sans être tous chargés en VRAM. Le démarrage peut être configuré
pour vérifier seulement leur présence après le premier téléchargement.

## 7. API publique

### `POST /v1/audio/speech`

Endpoint JSON compatible avec l'API OpenAI de synthèse vocale.

Champs :

| Champ | Règle |
|---|---|
| `input` | requis, 1 à 4096 caractères |
| `model` | un mode Qwen ou un alias OpenAI |
| `voice` | requis pour CustomVoice et les profils clonés persistants |
| `instructions` | optionnel pour CustomVoice, requis pour VoiceDesign |
| `response_format` | `wav`, `mp3`, `flac`, `opus`, `aac` ou `pcm` |
| `speed` | multiplicateur validé puis appliqué avec ffmpeg |
| `language` | extension locale, `fr` par défaut ou `auto` explicite |

Les alias `tts-1`, `tts-1-hd` et `gpt-4o-mini-tts` sont acceptés et dirigés
vers `qwen3-tts-custom-voice`. Aucun alias ne change silencieusement de moteur
en cas d'échec.

Pour `qwen3-tts-clone`, `voice` désigne l'identifiant UUID d'un profil
persistant. Pour `qwen3-tts-voice-design`, la description de la voix est portée
par `instructions` et `voice` n'est pas obligatoire.

La réponse contient directement le flux audio avec le type MIME correspondant.
Les erreurs suivent l'enveloppe OpenAI :

```json
{
  "error": {
    "message": "Le modèle TTS est indisponible.",
    "type": "tts_unavailable",
    "param": null,
    "code": "model_load_failed"
  }
}
```

Les erreurs de validation donnent 400/422. Un modèle indisponible, un OOM, un
timeout ou la mort du worker donne 503. Une voix inconnue donne 404.

### `POST /v1/voices`

Création multipart d'un profil de clonage persistant :

- `file` : référence audio requise ;
- `name` : nom d'affichage requis ;
- `language` : `fr` par défaut ;
- `transcript` : texte exact facultatif ;
- `consent` : booléen obligatoirement vrai.

Le serveur normalise et valide l'audio. Si `transcript` manque, il appelle
Parakeet sous le même verrou GPU. La réponse fournit un UUID, le nom, la
langue, la durée, la date de création et indique si le texte a été fourni ou
transcrit. Le chemin interne et le contenu de la référence ne sont jamais
retournés.

### `GET /v1/voices`

Retourne les voix Qwen prédéfinies disponibles et les profils persistants,
avec un champ `kind` permettant de distinguer `builtin` et `clone`.

### `DELETE /v1/voices/{voice_id}`

Supprime les métadonnées et l'audio du profil UUID. Les voix prédéfinies ne
peuvent pas être supprimées. Une suppression réussie est définitive et donne
204.

### `POST /v1/audio/speech/clone`

Endpoint multipart de clonage ponctuel :

- `file`, `input` et `consent=true` sont requis ;
- `transcript`, `language`, `response_format` et `speed` sont facultatifs ;
- `instructions` est refusé en mode clone : l'API officielle du checkpoint
  Base ne le prend pas en charge et le serveur ne doit pas prétendre
  l'appliquer ;
- la référence normalisée et sa transcription éventuelle sont supprimées dans
  un `finally`, succès ou échec ;
- aucun profil n'est enregistré.

## 8. Stockage et confidentialité

Les profils sont stockés dans un volume Docker persistant distinct du cache de
modèles. Un registre de métadonnées atomique associe chaque UUID à : nom
d'affichage, langue, transcription de référence, durée, fichier audio interne,
date de création et date du consentement explicite.

Règles de sécurité :

- les noms de fichiers sont des UUID générés par le serveur ;
- le nom d'affichage fourni par le client n'est jamais utilisé comme chemin ;
- tout chemin résolu doit rester sous la racine configurée ;
- les écritures de métadonnées passent par un remplacement atomique ;
- le consentement explicite est requis pour toute opération de clonage ;
- ni la transcription de référence, ni son chemin, ni le texte synthétisé ne
  sont inscrits dans les logs ordinaires ;
- les fichiers temporaires sont supprimés dans tous les chemins d'erreur ;
- taille, durée, type décodable, silence et écrêtage sont validés avant
  persistance.

Cette API est locale. Si le port devait être exposé au réseau, authentification,
chiffrement et contrôle d'accès aux profils deviendraient des prérequis.

## 9. Chaîne de préparation audio

### Référence de clonage

1. décodage par ffmpeg ;
2. conversion mono PCM 24 kHz ;
3. détection Silero pour retirer uniquement les silences extérieurs ;
4. mesure de la durée utile, du niveau et de l'écrêtage ;
5. rejet si la parole utile dure moins de 3 s ou plus de 30 s, si le signal est
   silencieux, illisible, extrêmement faible ou fortement écrêté ;
6. conservation sans débruitage agressif ni traitement susceptible de changer
   le timbre.

Le transcript fourni est préféré car il est plus fiable. Sinon, la référence
est convertie à l'entrée attendue par Parakeet et transcrite en français par
défaut.

### Texte à synthétiser

Le texte subit seulement une normalisation Unicode, des espaces et de la
ponctuation. La langue n'est jamais détectée implicitement sauf si le client a
demandé `auto`.

Un texte long est divisé aux frontières de phrases, puis aux frontières de
propositions si nécessaire. Chaque segment conserve exactement le même mode,
la même voix, la même langue et les mêmes instructions. Les segments audio sont
assemblés avec :

- de courts fondus pour éviter les clics ;
- des pauses déterminées par la ponctuation ;
- une normalisation finale proche de -16 LUFS avec protection des crêtes.

La vitesse est appliquée après génération avec le filtre ffmpeg approprié. La
V1 privilégie la cohérence et produit le fichier complet avant de répondre.

## 10. État de santé et observabilité

`GET /health` conserve les informations actuelles et ajoute :

- disponibilité et PID du worker ;
- socket joignable ;
- checkpoints téléchargés ;
- checkpoint actuellement chargé ou `null` ;
- état `idle`, `loading`, `ready`, `generating`, `unloading` ou `error` ;
- device, précision et backend d'attention ;
- mémoire GPU libre/totale ;
- disponibilité de CustomVoice, clone et VoiceDesign ;
- dernière erreur sous forme de code non sensible.

Les logs structurés contiennent mode, identifiant de requête, durées de
préparation/chargement/génération/encodage, RTF et pic VRAM. Ils ne contiennent
pas les textes ni les références vocales.

## 11. Configuration

Les noms exacts pourront suivre les conventions déjà présentes dans
`Settings`, mais la configuration doit couvrir au minimum :

- activation TTS ;
- chemin de `/opt/venv-qwen` et du socket Unix ;
- identifiants des trois checkpoints ;
- liste des checkpoints à pré-télécharger ;
- répertoires persistants des modèles et des profils vocaux ;
- langue, modèle, voix et format par défaut ;
- délai d'inactivité avant déchargement ;
- timeout de chargement et de génération ;
- limites de fichier, de référence utile et de texte ;
- paramètres de segmentation, pauses, loudness et crêtes ;
- activation éventuelle de FlashAttention 2 après benchmark.

Les valeurs par défaut doivent démarrer sur la RTX 3090 sans configuration
supplémentaire, hormis le token Hugging Face déjà utilisé si nécessaire.

## 12. Tests

### Tests unitaires, sans GPU ni téléchargement

- schémas, routes, alias OpenAI, paramètres conditionnels et enveloppes
  d'erreur ;
- registre des profils, consentement, UUID, protection contre la traversée de
  chemins, suppression et nettoyage temporaire ;
- worker simulé : chargement, réutilisation, changement, déchargement, timeout,
  mort et OOM ;
- ordre du verrou GPU et déchargement obligatoire avant diarization ;
- validation des références, segmentation de texte, assemblage, vitesse et
  formats de sortie ;
- contenu de `/health` dans chaque état.

### Tests d'intégration

- API principale reliée à un worker simulé par socket Unix ;
- appel réel avec le client Python OpenAI et
  `base_url=http://127.0.0.1:8000/v1` ;
- décodage de chaque format avec ffprobe/ffmpeg ;
- construction et démarrage d'un unique conteneur supervisant exactement les
  deux processus ;
- vérification qu'aucun port du worker n'est exposé.

### Tests GPU optionnels, marqués `gpu`

- chargement séquentiel de chaque checkpoint 1.7B sur la RTX 3090 ;
- synthèse française CustomVoice, VoiceDesign et clone avec une référence de
  test consentie ;
- changement Base → CustomVoice → VoiceDesign et vérification de la restitution
  de VRAM ;
- transcription après TTS puis TTS après diarization, sans OOM ni deadlock ;
- mesures de latence froide/chaude, RTF et pic VRAM.

## 13. Validation de la qualité

Un corpus français fixe d'environ vingt phrases couvre nombres, dates,
abréviations, noms propres, ponctuation, émotion et texte long.

La comparaison porte sur Qwen3-TTS et les sorties VoxMind disponibles
(Kokoro/F5-TTS) :

1. Parakeet retranscrit toutes les sorties pour mesurer l'intelligibilité ;
2. Qwen ne doit pas dégrader matériellement le taux d'erreur par rapport à la
   meilleure base VoxMind ;
3. une écoute A/B en aveugle compare naturel, prosodie, artefacts et fidélité
   du clonage ;
4. Qwen doit être préféré par une majorité des évaluations pour que le projet
   affirme une meilleure qualité ;
5. les résultats, paramètres, versions, latences et métriques VRAM sont
   conservés dans un rapport reproductible.

Un seuil indicatif de WER inférieur ou égal à 8 % est suivi sur le corpus
propre, mais la comparaison relative et l'écoute priment : ce seuil ne doit pas
être présenté comme comparable aux benchmarks éditeurs.

## 14. Critères d'acceptation

L'implémentation est acceptée lorsque :

1. CustomVoice, clonage persistant, clonage ponctuel et VoiceDesign produisent
   un audio français intelligible ;
2. le client Python OpenAI peut appeler `/v1/audio/speech` sans adaptation
   spécifique, hors extensions facultatives ;
3. un seul conteneur et `127.0.0.1:8000` sont nécessaires ;
4. le worker Qwen n'est pas accessible depuis le réseau ;
5. un seul checkpoint Qwen occupe la VRAM à la fois ;
6. l'alternance TTS, ASR et diarization ne provoque ni OOM ni deadlock sur la
   RTX 3090 lorsque les requêtes sont sérialisées ;
7. les six formats annoncés sont décodables et respectent leur type MIME ;
8. les profils sont protégés par UUID, soumis au consentement, listables et
   supprimables ;
9. les erreurs GPU ne déclenchent ni repli CPU silencieux ni réponse audio
   produite par un autre moteur ;
10. les 338 tests existants continuent de passer, ainsi que les nouveaux tests
    unitaires, d'intégration et GPU applicables ;
11. `git diff --check`, la configuration Compose et le healthcheck passent ;
12. le benchmark documenté justifie toute affirmation d'amélioration de
    qualité face à VoxMind.

## 15. Risques et mesures

| Risque | Mesure |
|---|---|
| conflit Transformers entre NeMo et Qwen | environnements Python séparés dans le même conteneur |
| OOM lors d'un changement de variante | un seul checkpoint, arbitrage central, déchargement et test GPU séquentiel |
| premier démarrage très long | cache persistant, healthcheck avec période de démarrage adaptée, progression dans les logs |
| référence de clonage médiocre | validation 3–30 s, Silero, rejet du silence/écrêtage, transcript explicite recommandé |
| fuite de données vocales | stockage local UUID, logs expurgés, consentement, suppression et nettoyage `finally` |
| latence sur texte long | segmentation contrôlée, réutilisation du modèle chaud, mesures cold/warm |
| qualité française non supérieure | benchmark comparatif avant toute revendication |
| worker mort mais API encore active | supervision couplée et healthcheck global défaillant |

## 16. Déploiement et compatibilité

Le service Compose existant reste l'unique unité de déploiement. Le Dockerfile
construit les deux environnements, copie le worker et installe ffmpeg. Les caches
de modèles et de voix sont montés dans des volumes persistants. Le README doit
indiquer la taille des trois checkpoints, le temps du premier démarrage, les
commandes OpenAI/curl, la création et la suppression d'une voix, ainsi que le
fait que le clonage exige le consentement du locuteur.

L'ASR, la diarization, le résumé Ollama et leurs endpoints existants ne changent
pas de contrat. Toute modification de leur cycle GPU doit être couverte par les
338 tests de régression et par le scénario GPU croisé TTS/ASR/diarization.
