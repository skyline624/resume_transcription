# Deploiement Docker sur studio-linux

Deploiement actif depuis le 14 septembre 2026. Les services natifs
`resume-transcription` et `resume-transcription-ollama` sont arretes et
desactives. Docker demarre automatiquement et les deux conteneurs utilisent
la politique `unless-stopped`.

Le fichier `docker-compose.studio-linux.yml` definit deux conteneurs :
`transcription` (Parakeet, pyannote, Qwen3-TTS et interface web) et `ollama`
(comptes-rendus locaux). Les deux accedent a la RTX 3090 via NVIDIA Container
Toolkit. Le port 8000 est publie sur `192.168.1.141` pour le reseau local et
sur `127.0.0.1` pour les connexions locales et le tunnel SSH.

Le code, les modeles et les voix restent sur le second disque, dans
`/mnt/donnees/resume_transcription`. Docker stocke ses images sur le disque
systeme (`/var/lib/docker` et le stockage containerd), qui dispose de plus
d'espace. Les conteneurs utilisent l'UID 1000 pour conserver des fichiers
accessibles au compte `pc`.

## Commandes

```bash
cd /mnt/donnees/resume_transcription
docker compose -f docker-compose.studio-linux.yml ps
docker compose -f docker-compose.studio-linux.yml logs --tail 100
docker compose -f docker-compose.studio-linux.yml restart transcription
docker compose -f docker-compose.studio-linux.yml exec ollama ollama list
curl -fsS http://127.0.0.1:8000/health
```

Depuis un autre ordinateur du reseau local, ouvrir directement
<http://192.168.1.141:8000/>. Le tunnel SSH est facultatif pour cet acces.
Cette adresse privee est egalement utilisable via un VPN donnant acces au LAN.

Pour conserver l'acces par tunnel SSH depuis Windows :

```powershell
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 studio-linux
```

Interface : <http://127.0.0.1:8000/>. Le navigateur conserve son historique
IndexedDB a cette adresse. L'adresse LAN utilise un historique distinct : les
anciens enregistrements du navigateur restent associes a l'adresse locale.
L'enregistrement au microphone exige un contexte securise : utiliser HTTPS ou
l'adresse localhost via le tunnel SSH. L'import de fichiers fonctionne sur
l'adresse HTTP du LAN.

La configuration effective est `.env.docker-studio` (permissions 600),
initialement generee par `scripts/configure-docker-studio.py` a partir de
la configuration native. Ollama est joint par `http://ollama:11434` sur le
reseau prive Compose. Son cache est monte depuis `runtime/ollama/` et le
modele est `qwen3.8-27b-unc-32k:latest`.

Les modeles Hugging Face sont montes depuis `models/`, le cache NeMo depuis
`runtime/nemo-cache/`, et les profils vocaux depuis `voices/`. Le chemin
absolu des references vocales est conserve a l'interieur du conteneur.

Les personnes reconnues par pyannote sont enregistrées séparément dans
`speakers/speakers.sqlite3`, monté sur `/app/speakers`. Le répertoire est créé
par `scripts/configure-docker-studio.py`. Pour une installation existante,
exécuter `mkdir -p speakers && chmod 700 speakers` avant la mise à jour Compose.
Voir [le contrat API de reconnaissance](speaker-recognition.md).

## Images et reconstruction

L'image existante Windows a ete transferee sous le nom
`transcription-parakeet:windows-migration-2904b2f0` (image
`sha256:2904b2f0bb634132349fb463932277ba67c7906b1fa563ca4224ea88ff1d442f`).
Elle fournit les dependances Python deja validees avec PyTorch 2.11 / CUDA
12.8. `docker/Dockerfile.studio-linux` actualise le code et recompile le web :

```bash
docker compose -f docker-compose.studio-linux.yml build transcription
docker compose -f docker-compose.studio-linux.yml up -d
```

Pour reconstruire aussi les dependances a partir du Dockerfile complet :

```bash
docker build -f docker/Dockerfile -t transcription-parakeet:base-rebuilt .
docker compose -f docker-compose.studio-linux.yml build --build-arg BASE_IMAGE=transcription-parakeet:base-rebuilt transcription
docker compose -f docker-compose.studio-linux.yml up -d
```

## Retour a l'installation native

Les environnements Python et fichiers du service natif sont conserves.
Ces commandes arretent Docker pour ce projet puis restaurent le service :

```bash
docker compose -f docker-compose.studio-linux.yml down
systemctl --user enable --now resume-transcription-ollama resume-transcription
```

Pour basculer de nouveau vers Docker apres verification de l'image :

```bash
systemctl --user disable --now resume-transcription resume-transcription-ollama
docker compose -f docker-compose.studio-linux.yml up -d --wait --wait-timeout 600
```

Les donnees persistantes ne sont pas supprimees par `down`.

## Verification de la bascule

- 459 tests Python et 39 tests frontend reussis dans la construction Docker.
  Un avertissement tiers Starlette/AnyIO reste visible dans les tests Python.
- Essais GPU reussis : transcription avec diarization, WAV Qwen a 24 kHz et
  compte-rendu Ollama. Sorties : `runtime/verification-docker/`.
- Deux conteneurs `healthy`, aucun redemarrage intempestif au controle final.
- Interface et API verifiees depuis Windows via le tunnel SSH ; dix voix
  disponibles, dont le profil migre.
- Environ 165 Go libres sur le disque systeme et 18 Go sur le disque de donnees.

Les journaux de construction et de validation sont dans `runtime/logs/docker-*.log`.
Pour rejouer les tests unitaires avec le compte non privilegie du conteneur :

```bash
docker run --rm --user 1000:1000 --entrypoint /opt/venv-main/bin/python transcription-parakeet:studio-linux -m pytest tests/unit -q -o cache_dir=/tmp/pytest-cache
```
