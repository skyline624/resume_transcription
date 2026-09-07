# Tâche — Cycle de vie GPU paresseux (chargement à la demande + déchargement après inactivité)

## Objectif

Rendre le serveur capable de **charger les moteurs GPU (ASR Parakeet, diarization
pyannote) à la première requête** plutôt qu'au démarrage, puis de **décharger
automatiquement ASR, diarization ET TTS Qwen après une période d'inactivité
configurable**, afin de libérer la VRAM quand le serveur n'est pas utilisé.
Le worker TTS Qwen a déjà ce cycle de vie — l'objectif est d'étendre le même
comportement aux moteurs du processus principal et d'unifier la configuration.

## Contraintes d'architecture (NON négociables)

1. **L'API doit rester disponible pendant qu'un moteur est déchargé.** `/health`
   et les routes de statut répondent toujours (nouveau champ d'état par moteur :
   `loaded` / `unloaded` / `loading`), mais une requête de transcription pendant
   un rechargement doit **attendre** le chargement (pas d'échec 503 tant que le
   chargement réussit dans le timeout).
2. **Verrou de rechargement** : une seule requête déclenche le chargement ; les
   autres attendent le même chargement (pas de double chargement concurrent ni
   de course). Le chargement se fait hors event-loop (thread) comme aujourd'hui.
3. **Le warmup existant** (inference de 1 s de silence au démarrage,
   `app.py::_warmup`) ne doit PAS faire charger les modèles au démarrage quand
   le mode lazy est actif — le warmup devient conditionnel ou se fait après un
   chargement à la demande.
4. **Comportement par défaut préservé quand le lazy est désactivé**
   (`LAZY_GPU=false` par défaut ? Non — cf. §Config) : les tests existants
   `tests/unit/test_app*.py`, `tests/unit/test_transcribe*.py` doivent continuer
   à passer sans modification de leur sémantique (les moteurs injectés par les
   tests sont des stubs — le cycle de vie ne doit pas les décharger ni les
   envelopper quand le lazy est off).
5. **Timeouts** : réutiliser `tts_load_timeout_s`/`tts_generation_timeout_s`
   comme modèle ; ajouter des timeouts de chargement ASR/diarization distincts
   et généreux (défaut 600 s — le téléchargement initial de 2,6 Go peut être
   long).
6. **Diarization désactivée** (`ENABLE_DIARIZATION=false`) : pas de changement —
   `NullDiarizationEngine` n'a rien à charger/décharger.
7. **VAD** : hors périmètre (CPU, quelques Mo, chargé au démarrage comme
   aujourd'hui).
8. **Style du repo** : docstrings/commentaires en français, typos complètes,
   dataclasses/Protocols comme le code existant, fichiers < 500 lignes.
   Regarde `src/transcription_server/tts/` et `qwen_worker/` pour le style.

## Design attendu

- Nouveau module `src/transcription_server/runtime/lifecycle.py` (ou proche) :
  un `LazyEngine`/`GpuLifecycleManager` générique qui enveloppe un
  constructeur de moteur (loader), expose la même interface (`AsrEngine`,
  `DiarizationEngine`), traque `last_used` via un `clock` injectable, et expose
  `unload_if_idle()` + verrou d'unicité du rechargement.
- `build_app()` : si lazy actif, envelopper les loaders NeMo/pyannote dans le
  manager lazy au lieu de charger immédiatement. Le monitor d'inactivité du
  worker Qwen devient un monitor **unifié** : un seul timer vérifie ASR,
  diarization et (via le client TTS existant, `POST /unload` déjà implémenté)
  le worker Qwen.
- Le worker Qwen garde son code existant — la coordination se fait via
  `UnixTtsClient.unload()` (déjà écrit) et son `/health` déjà exposé.
- `/health` du serveur principal doit refléter l'état réel de chaque moteur
  (champs existants + états de cycle de vie), sans casser le schéma actuel
  (ajouts de champs uniquement).

## Config (Settings, config.py)

- `enable_lazy_gpu: bool = True` — le comportement demandé par l'utilisateur
  (libérer la VRAM) est le défaut.
- `gpu_idle_unload_s: float = 900` (15 min) — s'applique à ASR/diarization ;
  le worker Qwen continue d'utiliser `tts_idle_unload_s` (300 s, inchangé).
- `asr_load_timeout_s: float = 600.0`, `diarization_load_timeout_s: float = 600.0`.
- Documenter chaque clé avec un commentaire français court, comme les autres.

## Comportement de déchargement

- Après `gpu_idle_unload_s` sans requête : décharger pyannote PUIS Parakeet
  (ordre sans importance, mais libérer `torch.cuda.empty_cache()` après chaque
  unload — s'inspirer de `qwen_worker.model_manager.cuda_cleanup`).
- Déchargé ≠ cassé : la requête suivante re-déclenche un chargement complet
  (avec re-téléchargement depuis le cache HF local `./models`, donc ~10-40 s,
  pas 2,6 Go à re-télécharger).
- Journaliser chaque chargement/déchargement (logger.info, français, avec
  durées).

## Tests attendus (pytest, style existant, répertoire tests/unit/)

- `test_lazy_lifecycle.py` : chargement à la 1re requête ; pas de rechargement
  si activité continue ; déchargement après idle (fake clock comme dans
  `tests/unit/test_qwen_model_manager.py`) ; rechargement sur requête suivante ;
  requêtes concurrentes → un seul chargement ; lazy off → comportement
  identique à l'actuel ; `/health` expose les états.
- Les tests existants doivent rester verts : `python -m pytest tests/unit -q`
  via le venv du projet (`.venv/Scripts/python.exe` sous Windows).

## Hors périmètre

- Pas de modification du Dockerfile/entrypoint (le worker garde son propre
  monitor ; le monitor unifié côté serveur appelle simplement `/unload`).
- Pas de changement de l'API HTTP publique en dehors des ajouts à `/health`.
- Pas de commit — laisser le working tree sale pour review.

## Critères d'acceptation

1. `./.venv/Scripts/python.exe -m pytest tests/unit -q` : tous les tests
   passent (anciens + nouveaux).
2. `LAZY` par défaut actif : avec le lazy, `/health` au démarrage montre ASR et
   diarization non chargés, et la première requête de transcription les charge.
3. Aucune régression du contrat d'erreur OpenAI des routes.