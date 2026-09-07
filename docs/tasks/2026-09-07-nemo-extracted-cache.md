# Cache disque des checkpoints NeMo extraits

Le service Compose monte le volume nomme `nemo-extracted` dans `/app/nemo-cache`
et definit `NEMO_EXTRACTED_CACHE_DIR` sur ce chemin. Les archives d'origine
restent dans le cache HuggingFace existant, `./models:/app/models`.

Au premier chargement d'une archive Parakeet, le serveur extrait les fichiers
directement dans le volume Linux de Docker. Les restaurations suivantes
utilisent le support `SaveRestoreConnector.model_extracted_dir` de NeMo et ne
reextraient plus l'archive dans `/tmp` depuis le dossier partage Windows.

Le cache est indexe par chemin reel, taille et date de modification de
l'archive. Un manifeste controle la presence et la taille des fichiers,
configuration et poids compris. L'extraction est publiee par renommage apres
validation ; un cache incomplet est reconstruit. Le filtre `data` de tarfile
empeche les chemins d'archive de sortir du dossier cible.

Ce cache ne conserve aucun modele en RAM CPU ou en VRAM. Le dechargement
complet apres inactivite reste actif. Le systeme peut conserver des pages de
fichiers en cache disque RAM reclamable. Le volume consomme de l'espace disque
supplementaire ; il survit aux reconstructions et recreations du conteneur.
Les anciennes revisions restent disponibles, sans suppression automatique.

Sans `NEMO_EXTRACTED_CACHE_DIR`, le chargement NeMo habituel reste disponible,
notamment pour les executions locales hors Compose. Un depot NeMo deja
distribue sous forme de dossier extrait est utilise directement.

## Verification

- 459 tests unitaires et 9 tests de structure Docker passent sur l'hote.
- 38 tests cibles passent dans l'image (filtre de depreciation Starlette/AnyIO
  existant uniquement).
- Premiere extraction mesuree : 100,1 s.
- Premiere requete, extraction et imports inclus : 145,29 s ; suivante : 0,73 s.
- Apres recreation du conteneur avec le meme volume : premiere requete 45,43 s
  (chargement et warmup 43,5 s), suivante 0,39 s. Aucun nouvel extracteur execute.
- Cache extrait sur disque : 2,4 Gio. Environ 100 s economisees sur ce scenario.
- Deux cycles reels de chargement/dechargement avec une horloge de test :
  premier chargement 44,15 s, rechargement dans le meme processus 23,40 s.
  Les references faibles aux modeles sont mortes apres chaque delai d'inactivite.
  VRAM active apres dechargement : 36,5 puis 63,88 Mio, contre environ 2,4 Gio
  modele charge. Le processus de test s'est ensuite termine normalement.
- Image deployee : `sha256:2904b2f0bb634132349fb463932277ba67c7906b1fa563ca4224ea88ff1d442f`.
- La phrase de test de 3,63 s est correctement transcrite.

Scripts et mesures ignores par Git : `benchmark-results/nemo-cache/`.
