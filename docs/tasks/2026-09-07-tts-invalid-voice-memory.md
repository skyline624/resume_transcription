# Echecs TTS et retention de VRAM

## Diagnostic

Hermes Desktop utilisait `tts.openai.voice: verse` avec le modele
`qwen3-tts-custom-voice`. Cette voix OpenAI ne fait pas partie des neuf voix
Qwen. La validation etait effectuee par le backend seulement apres chargement
du checkpoint. Une mauvaise voix provoquait donc un HTTP 503, invalidait le
modele, puis imposait un nouveau chargement a la requete suivante.

Les logs initiaux masquaient l'exception. Une reproduction instrumentee avec
`alloy`, puis avec `verse` sur le worker prive, donne
`ValueError: Unsupported speakers`. Les requetes avec Ryan reussissent.

Le nettoyage CUDA etait execute alors que les tracebacks de l'exception
conservaient encore les objets du backend. Un echec pendant le chargement
n'effectuait aucun nettoyage si `self._model` n'avait pas encore ete assigne.
Trois tests avec references faibles reproduisent ces defauts pour generation,
streaming et chargement : le modele etait encore vivant au nettoyage, ou le
nettoyage n'etait pas appele.

## Correction

- Valider les voix CustomVoice a l'entree de l'API, en reutilisant le catalogue
  de `/v1/voices`, sans changer les identifiants de voix clonees.
- Renvoyer HTTP 422 avec les voix disponibles, sans calcul GPU ni dechargement.
- Journaliser les exceptions completes cote worker.
- Vider les frames terminees des exceptions et de leurs causes avant la
  collecte des objets GPU. Nettoyer aussi un chargement partiellement echoue.
- Configuration locale Hermes corrigee de `verse` vers `Ryan`, avec sauvegarde
  `config.yaml.before-qwen-voice-fix-20260907`. Son resolveur confirme Ryan.
  Ce fichier de configuration externe ne fait pas partie du depot ni de l'image.

## Verification

- 454 tests unitaires passent sur l'hote.
- 39 tests cibles passent dans l'image. Filtre limite a la depreciation connue
  de l'alias AnyIO BlockingPortal utilise par Starlette.
- Six tests de validation de voix et trois tests de liberation memoire ont ete
  observes en echec avant leur correction.
- Image Docker reconstruite et service recree.
- Test GPU : voix invalide refusee en 4 a 17 ms a froid, sans chargement.
- Streaming Ryan : premier audio a 77,43 s a froid et 0,67 s a chaud.
- WAV complet : 2,98 s ; annulation du flux suivie d'une generation reussie.
- Voix invalide a chaud : modele conserve, allocation VRAM inchangee.
- Erreur volontaire sur le worker prive : allocation active de 4263,28 Mio
  avant erreur a 53,75 Mio apres nettoyage (environ 4 Go retenus auparavant).
- Reprise apres erreur reussie sans redemarrage du worker : premier audio
  apres rechargement en 69,86 s. Dechargement explicite final confirme : etat
  `idle`, aucun modele, aucune erreur, 80,13 Mio encore alloues par PyTorch.
- Le WAV verifie contient 3,63 s d'audio mono 24 kHz non silencieux.
- Image deployee : `sha256:afb5bc112935a90ba49e3ee5e1bff6d970580d9d414f593857c61b408bc264c2`.

Les scripts et mesures brutes sont dans `benchmark-results/debug-memory/`
(ignore par Git). La verification GPU provoque volontairement une erreur
privee `Unsupported speakers: ['verse']` pour tester la recuperation.
