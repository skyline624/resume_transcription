# Acceleration et lecture progressive Qwen TTS

Le profilage du moteur Qwen standard sur la RTX 3090 montre surtout un cout de
pilotage CPU des operations CUDA : environ 160 805 lancements de kernels pour
24 pas profiles, avec 42,56 % du temps CPU propre dans `cudaLaunchKernel`.
Garder les poids en VRAM supprime le rechargement, mais pas ce cout d'inference.

Le worker utilise maintenant `faster-qwen3-tts==0.3.0` (CUDA Graphs), compatible
avec Qwen TTS 0.1.1 et Transformers 4.57.3. Le chargement prend d'abord le snapshot
Hugging Face local. Une absence de cache declenche le telechargement habituel.
Le premier appel d'un modele inclut son chargement et la capture des graphes.
Le passage entre CustomVoice, VoiceDesign et Base change de checkpoint ; un seul
checkpoint Qwen reste resident a la fois.

Configuration du worker :

| Variable | Defaut | Effet |
|---|---|---|
| `TTS_BACKEND` | `cuda_graphs` | Acceleration et streaming ; `standard` permet de revenir au moteur precedent pour les requetes completes |
| `TTS_CPU_THREADS` | `1` | Nombre de threads PyTorch du worker |
| `TTS_STREAM_CHUNK_SIZE` | `8` | Pas de codec par morceau ; intervalle admis 1 a 32 |

Le cycle d'inactivite conserve ses reglages : TTS 300 s, ASR/diarisation 900 s
dans l'environnement local. Une requete reserve le verrou GPU jusqu'a la fin
ou jusqu'a l'acquittement de son annulation. Le monitor attend ce verrou avant
de decharger. La supervision lit des instantanes sans attendre les verrous
d'inference des moteurs.

`POST /v1/audio/speech` accepte `stream: true` en JSON. Le clone ponctuel accepte
le champ multipart `stream=true`. Le streaming exige `response_format=pcm` et
`speed=1` : mono PCM signe 16 bits little-endian, 24 kHz. Les en-tetes
`X-Audio-Sample-Rate`, `X-Audio-Channels` et `X-TTS-First-Audio-Ms` decrivent le flux.
Les erreurs precedant le premier son restent des reponses HTTP 503. Une erreur
ulterieure interrompt le transfert. Aucun fichier incomplet n'est presente
comme un resultat termine par l'interface.

L'interface propose « Ecouter en direct » et « Arreter » pour les trois modes,
ainsi que pour le clone ponctuel. Web Audio lit les morceaux a leur arrivee ;
le resultat termine est assemble en WAV pour le lecteur et le telechargement.
La conservation de l'audio dans l'historique reste explicite. Les exports
complets conservent leurs formats et leur reglage de vitesse.

Les textes sont segmentes a 500 caracteres avant synthese, y compris le clone
ponctuel, pour rester dans la capacite du cache des graphes. La reference d'un
clone ponctuel reste presente jusqu'a la fin effective du flux puis est effacee.
Une deconnexion avant les premiers en-tetes annule aussi l'attente cote serveur ;
un chargement ou calcul GPU deja engage doit atteindre un point d'arret sur.

Validation : tests Python de cycle de vie, annulation avant/apres premier son,
erreur initiale, nettoyage des references et modes de l'adaptateur ; tests web
de lecture avant fin de reponse, raccord des echantillons PCM, arret et absence
de conservation d'un resultat partiel. Les mesures reelles et fichiers audio
sont dans `benchmark-results/streaming-verification/` (sorties locales ignorees).

Source du moteur : https://github.com/andimarafioti/faster-qwen3-tts (licence MIT).

## Mesures du conteneur reconstruit le 7 septembre 2026

RTX 3090 24 Go, phrase « Bonjour, ceci est un test de rapidité. » ; appels HTTP
depuis Windows vers Docker Desktop. Les durees audio varient selon le tirage
du modele : ces valeurs sont des essais ponctuels, pas des percentiles.

| Mode | Premier son a chaud | Fin du flux a chaud | Export complet a chaud | Premier appel apres chargement |
|---|---:|---:|---:|---:|
| CustomVoice | 0,95–1,01 s | 3,69–4,97 s | 5,21–8,82 s | 81,82 s |
| VoiceDesign | 1,27 s | 4,94 s | 3,83 s | 73,93 s |
| Clone ponctuel | 1,56 s | 4,89 s | 4,99 s | 74,19 s |

Reference precedente sans acceleration : 13–32 s a chaud pour produire le
fichier complet. Le profil isole avec graine identique mesurait environ
12,8 s avec le moteur standard, puis 2,7 s avec CUDA Graphs ; les appels HTTP
ci-dessus incluent les autres etapes et n'utilisent pas de graine imposee.

Le premier STT avec diarisation a pris 157,4 s (chargement inclus). Les STT
suivants, sans diarisation, prennent 0,35–2,70 s selon l'audio. ASR, diarisation
et un checkpoint Qwen coexistent apres les appels ; la VRAM active du worker
Qwen reste autour de 4,3 Go apres les changements de mode. La supervision
repond en 0,2 s environ pendant le chargement et en 6–14 ms avec une connexion
HTTP deja ouverte hors chargement.

L'annulation apres premier morceau et le retour a la synthese passent. Une
annulation avant les en-tetes suivie d'une transcription prend 1,46 s au total.
Aucune reference temporaire de clone ne reste dans `/tmp` apres ces essais.
Le navigateur a termine une lecture progressive, propose `speech.wav`, puis
arrete une autre generation sans erreur console.

**Limite observee :** deux exports CustomVoice n'ont pas passe le controle
automatique par Parakeet (un ajout « Hehehe » et une phrase mal reconnue).
Les trois essais VoiceDesign et les trois essais clone ont ete correctement
retranscrits. Douze audios supplementaires de CustomVoice, sur plusieurs
phrases et reglages avec graines fixees, ont egalement ete correctement
retranscrits. Cela ne suffit pas a garantir une qualite vocale constante ni a
attribuer avec certitude les deux ecarts au TTS ou au STT. Les reglages
d'echantillonnage d'origine ont ete conserves ; les sorties en echec restent
dans le rapport brut pour comparaison.

Validation finale : 445 tests unitaires Python et 39 tests web passent ; build
TypeScript/Vite valide. 34 tests cibles passent aussi dans le conteneur, avec
un avertissement de deprecation connu de Starlette/AnyIO. `pip check` du worker
ne signale aucun conflit. Image deployee :
`sha256:cc32237a541016c81dc925f6ec58d000709c8e4e54145977201dcce41ea0cdfd`.
