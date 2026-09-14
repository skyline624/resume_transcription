# Installation Ubuntu sur studio-linux

Depuis le 14 septembre 2026, le service utilise Docker. Voir
[le déploiement Docker](docker-studio-linux.md) pour les commandes actuelles.
L'installation native décrite ci-dessous est conservée comme solution de retour
arrière ; ses deux services sont arrêtés et désactivés.

Le projet est installé dans `/mnt/donnees/resume_transcription`, sur le SSD
de données `/dev/sda1`. Python 3.12 et la RTX 3090 sont utilisés directement.
Les environnements `.venv` et `.venv-qwen` séparent NeMo et Qwen3-TTS.

## Accès depuis Windows

Ouvrir un tunnel SSH et le conserver pendant l'utilisation :

```powershell
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 studio-linux
```

Puis ouvrir <http://127.0.0.1:8000/>. Le port reste privé sur le serveur.
L'historique est dans le navigateur Windows (IndexedDB) : conserver le même
navigateur et la même adresse pour retrouver les entrées existantes.

## Services

```bash
systemctl --user status resume-transcription resume-transcription-ollama
systemctl --user restart resume-transcription
journalctl --user -u resume-transcription -n 100 --no-pager
curl -fsS http://127.0.0.1:8000/health
```

Les deux services sont activés au démarrage avec `loginctl enable-linger pc`.
Ollama utilise le port privé `11435` et le modèle local
`qwen3.8-27b-unc-32k:latest`. Les services du studio déjà présents conservent
leurs ports et leurs données.

La configuration effective est `.env.studio-linux` (permissions 600).
Les modèles Hugging Face sont dans `models/`, les profils vocaux dans
`voices/`, les modèles Ollama dans `runtime/ollama/`. Les caches et fichiers
temporaires sont dans `runtime/`. Le socket privé Qwen se trouve dans
`/run/user/1000/qwen-tts/worker.sock`.

Les modèles occupent la VRAM à la demande et se déchargent après 15 minutes
d'inactivité. ComfyUI et les autres applications partagent la même RTX 3090 ;
leur consommation mémoire reste indépendante de la file GPU de ce projet.

## Réinstallation

Avec `uv`, Node.js, Python 3.12, FFmpeg et le pilote NVIDIA disponibles :

```bash
cd /mnt/donnees/resume_transcription
bash scripts/install-ubuntu.sh
python3 scripts/configure-ubuntu.py
mkdir -p ~/.config/systemd/user
cp scripts/resume-transcription*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now resume-transcription-ollama resume-transcription
```

SoX est fourni dans `runtime/vendor/` à partir des paquets Ubuntu, sans
installation système. La copie Windows d'origine est conservée.

## Validation du 14 septembre 2026

- Les 49 fichiers de modèles contrôlés correspondent à leurs empreintes.
- 459 tests unitaires Python et 39 tests frontend réussissent sur Ubuntu ;
  la compilation de l'interface réussit également. Un avertissement connu
  de Starlette concernant un alias AnyIO reste visible dans les tests.
- Transcription avec diarization, génération d'un WAV 24 kHz et résumé local
  ont été exécutés avec succès sur la RTX 3090.
- Le profil vocal existant a été conservé avec son identifiant et son
  consentement. Son chemin Docker a été adapté ; l'index original est gardé
  dans `voices/profiles.before-ubuntu.json`.
- Environ 20 Go restent disponibles après installation. Le premier chargement
  des modèles inclut l'extraction et les préparations CUDA.

Les résultats détaillés sont dans `runtime/logs/` et les sorties des essais
dans `runtime/verification-smoke/` sur le serveur.
