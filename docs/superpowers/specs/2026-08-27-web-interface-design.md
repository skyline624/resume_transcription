# Interface web locale — Spécification de conception

Date : 2026-08-27
Statut : validé

## Objectif

Ajouter au serveur une interface web locale, simple et intuitive, couvrant
toute l'API existante : état du service, transcription, diarisation, résumé,
synthèse vocale, clonage ponctuel et gestion des profils vocaux. L'interface
est livrée dans la même image Docker et utilise l'API HTTP existante sur la
même origine.

L'application vise un seul utilisateur local. Elle ne comporte ni compte, ni
authentification, ni base de données serveur supplémentaire.

## Hors périmètre

- accès multi-utilisateur ou distant sécurisé ;
- authentification et gestion de comptes ;
- progression GPU chiffrée lorsque le serveur n'en fournit pas ;
- conservation automatique des fichiers importés ou des prises micro ;
- modification des contrats publics de l'API audio ;
- nouveau service, nouveau port ou processus frontend en production.

## Architecture retenue

Le frontend est une application Preact écrite en TypeScript et construite par
Vite dans `web/`. Le développement utilise le serveur Vite avec un proxy vers
FastAPI. En production, une étape Node multistage exécute les tests et le build
Vite ; seuls les fichiers compilés sont copiés dans l'image finale.

FastAPI sert :

- `GET /` : le document `index.html` ;
- `/assets/*` : les ressources Vite immuables et nommées par leur hash ;
- les routes API, `/docs` et `/openapi.json` sans changement.

La navigation utilise des ancres (`/#/transcribe`, `/#/summarize`, etc.). Elle
ne nécessite donc aucun fallback universel susceptible de masquer un 404 API.
Le chemin des ressources compilées est configurable pour les tests et le
développement, avec `/app/web-dist` comme valeur de production.

## Découpage du frontend

Chaque unité possède une responsabilité unique :

- `app/` : coque, navigation et composition des écrans ;
- `api/` : client HTTP typé, sérialisation multipart et normalisation des
  erreurs OpenAI/FastAPI ;
- `features/health/` : état du serveur et du GPU ;
- `features/transcription/` : source audio, options, résultat et exports ;
- `features/summary/` : résumé depuis audio, texte ou historique ;
- `features/speech/` : CustomVoice, VoiceDesign, clone enregistré et clone
  ponctuel ;
- `features/voices/` : liste, inscription et suppression des voix ;
- `features/history/` : historique, détail, reprise et suppression ;
- `media/` : capture micro, URLs d'objet et lecture audio ;
- `storage/` : dépôt IndexedDB versionné ;
- `ui/` : composants accessibles communs, sans logique métier ;
- `utils/exports/` : génération locale de texte, dialogue, SRT, VTT et JSON.

Les composants ne font pas de `fetch` direct et n'accèdent pas directement à
IndexedDB. Ils passent par le client API et le dépôt de stockage afin de rendre
les comportements testables sans navigateur réel ni GPU.

## Structure de navigation

La barre supérieure affiche le nom du produit et un état compact : service,
GPU, Parakeet, pyannote, résumé et Qwen. Elle ouvre un panneau de détails sans
occuper l'espace de travail.

La navigation principale comporte cinq destinations :

1. Transcrire
2. Résumer
3. Synthétiser
4. Voix
5. Historique

Sur ordinateur, elle forme une colonne étroite à gauche. Sur petit écran, elle
devient une barre inférieure. Le contenu conserve une seule action principale
visible par écran.

## Direction visuelle

L'identité est celle d'un établi audio clair : l'interface emprunte la
lisibilité d'une console d'enregistrement sans reproduire sa densité.

Palette :

- brume `#EEF3F4` : fond général ;
- blanc `#FFFFFF` : surfaces de travail ;
- encre `#172428` : texte principal ;
- ardoise `#607177` : texte secondaire ;
- pétrole `#176B75` : actions et focus ;
- enregistrement `#C94F46` : capture active et erreurs critiques.

`Bahnschrift` sert aux titres, `Segoe UI` au texte courant et `Consolas` aux
durées, tailles et métriques. Des familles système équivalentes assurent le
repli multiplateforme, sans téléchargement de police.

La forme d'onde est l'unique signature visuelle. Elle sert réellement à la
capture, à la lecture et à l'état du média ; aucune onde décorative n'est
ajoutée ailleurs. Les animations sont rares, fonctionnelles et désactivées par
`prefers-reduced-motion`.

## Écran Transcrire

L'utilisateur peut déposer un fichier, le choisir sur disque ou enregistrer
depuis le microphone. L'aperçu affiche le nom, la durée et un lecteur avant
envoi.

Les options secondaires restent repliées par défaut :

- langue automatique, français ou anglais ;
- diarisation ;
- mélange, canal gauche, canal droit ou canaux séparés ;
- nombre exact de locuteurs, ou bornes minimale et maximale ;
- horodatage des mots.

L'interface applique les mêmes exclusions que le serveur entre nombre exact et
bornes de locuteurs. Elle appelle `POST /transcribe` au format JSON détaillé.
À partir de cette réponse unique, elle produit localement les vues et exports
texte, dialogue, SRT, VTT et JSON. Aucun export ne relance l'inférence.

Le résultat affiche texte complet, tours de parole et durées. Les actions
principales sont copier, télécharger et « Résumer cette transcription ».

## Écran Résumer

Trois sources sont possibles : fichier audio, texte collé ou transcription de
l'historique. Une seule source peut être active. L'utilisateur choisit le
format structuré ou narratif et, pour un fichier, les options de langue,
canaux et diarisation.

L'écran appelle `POST /summarize`. Le résultat indique le modèle utilisé, est
automatiquement ajouté à l'historique et peut être copié ou téléchargé. Si le
service de résumé est désactivé ou indisponible, le bandeau de santé et l'écran
expliquent comment l'activer au lieu de laisser un formulaire voué à échouer.

## Écran Synthétiser

Le formulaire propose les capacités supportées par `/v1/audio/speech` :

- CustomVoice avec une voix prédéfinie ;
- VoiceDesign avec une instruction obligatoire ;
- clone persistant avec un profil vocal existant ;
- langue, vitesse de 0,25 à 4 et format de sortie ;
- texte borné à 4096 caractères avec compteur visible.

Les champs incompatibles disparaissent lorsque le mode change. La liste des
voix provient de `GET /v1/voices`, jamais d'une liste dupliquée dans le client.
Le résultat possède un lecteur, un téléchargement et une action explicite
« Conserver dans l'historique ».

Une section distincte « Clone ponctuel » appelle
`POST /v1/audio/speech/clone`. Elle accepte un fichier ou une prise micro, le
texte à prononcer, une transcription de référence facultative et le
consentement obligatoire. La référence n'est jamais enregistrée dans
l'historique.

## Écran Voix

Les voix prédéfinies et les clones personnels sont présentés dans deux groupes.
Seuls les clones personnels disposent d'une action de suppression.

La création d'un profil accepte un fichier ou une prise micro de 3 à 30
secondes, un nom, une langue, une transcription facultative et une case de
consentement non cochée par défaut. Le formulaire explique que Parakeet
transcrira la référence lorsque le texte est absent.

Une suppression exige une confirmation nommant la voix concernée. La liste est
rafraîchie après création ou suppression réussie.

## Écran Historique

L'historique est stocké exclusivement dans IndexedDB. Chaque entrée contient :

- un identifiant local et un horodatage ;
- le type d'opération ;
- les paramètres non sensibles ;
- le résultat textuel et ses métadonnées ;
- éventuellement un audio généré conservé explicitement.

Une transcription ou un résumé réussi crée automatiquement une entrée. Une
synthèse réussie enregistre automatiquement son texte et ses paramètres, mais
pas son contenu audio : le blob n'est ajouté que par l'action « Conserver ».
Les créations et suppressions de profils vocaux ne sont pas historisées.

Les fichiers importés, prises micro et références de clonage ne sont jamais
persistés automatiquement. Les chemins internes, tokens et références vocales
ne font partie d'aucun enregistrement.

La rétention par défaut est de 100 opérations et 250 MiB d'audios conservés.
L'utilisateur peut modifier ces limites dans les bornes du quota accordé par
le navigateur. Si conserver un nouvel audio impose une éviction, l'interface
propose de supprimer d'abord les audios les plus anciens ou d'annuler. Un bouton
« Effacer l'historique local » supprime toute la base IndexedDB après
confirmation.

L'historique permet de rouvrir un résultat, reprendre une transcription pour
la résumer, réutiliser un texte pour la synthèse, télécharger un audio conservé
ou supprimer une entrée.

## Capture et lecture audio

`MediaRecorder` est créé uniquement après une action explicite. L'interface :

1. demande la permission du microphone ;
2. choisit le premier format supporté parmi WebM/Opus et Ogg/Opus ;
3. affiche durée et forme d'onde pendant la capture ;
4. arrête toutes les pistes sur arrêt, annulation, erreur ou démontage ;
5. fournit un aperçu avant tout envoi.

Les URLs créées avec `URL.createObjectURL` sont révoquées lorsqu'elles ne sont
plus utilisées. Les captures destinées à une référence vocale appliquent la
borne de 3 à 30 secondes avant l'appel API ; le serveur reste l'autorité finale
pour silence, saturation et validité du signal.

## Traitements longs et concurrence

Une requête en cours affiche son étape et le temps écoulé, sans inventer de
pourcentage. La navigation, l'historique et le panneau de santé restent
disponibles. Le bouton déclencheur est protégé contre les doubles soumissions.

Le frontend ne prétend pas annuler une inférence GPU déjà commencée. Une perte
de connexion ou une fermeture de page arrête seulement l'attente côté client ;
le serveur peut terminer le travail déjà confié à un thread.

Le bandeau interroge `GET /health` toutes les 10 secondes lorsque la page est
visible et toutes les 30 secondes en arrière-plan. Une erreur de santé ne bloque
pas l'accès aux résultats déjà stockés.

## Erreurs

Le client normalise les enveloppes d'erreur OpenAI et les erreurs FastAPI. Une
erreur visible contient :

- une phrase concrète ;
- le champ concerné lorsque le serveur l'indique ;
- une action corrective possible ;
- un détail technique replié avec code et statut HTTP.

Les erreurs de validation restent près du champ. Les erreurs réseau ou serveur
apparaissent dans la zone de résultat. Aucun toast éphémère ne porte seul une
information nécessaire à la correction.

## Sécurité et confidentialité

- aucun CDN, service analytique ou appel tiers depuis le navigateur ;
- aucune variable secrète injectée dans le bundle Vite ;
- aucune exposition du token Hugging Face ou du socket Qwen ;
- consentement explicite obligatoire pour les deux parcours de clonage ;
- rendu de tous les textes comme contenu, jamais comme HTML non assaini ;
- politique CSP compatible avec les bundles locaux ;
- écoute sur `127.0.0.1` conservée dans Docker Compose ;
- historique effaçable et non synchronisé.

Le modèle de sécurité reste celui d'un outil local. Exposer le port sur le
réseau nécessitera une conception d'authentification distincte.

## Accessibilité et responsive

Toutes les actions sont atteignables au clavier et possèdent un focus visible.
Les formulaires ont des libellés persistants, les états ne reposent jamais sur
la couleur seule et les mises à jour importantes utilisent une région ARIA
appropriée. Le contraste vise WCAG AA.

L'application fonctionne à partir de 360 px. Les tableaux deviennent des
cartes, la navigation passe en bas et les actions restent dans l'ordre du
document. La cible principale demeure un écran d'ordinateur local.

## Build et distribution Docker

Le Dockerfile ajoute une étape `node:24-alpine` :

1. copie `package.json` et `package-lock.json` ;
2. exécute `npm ci` ;
3. copie `web/`, lance les tests puis `npm run build` ;
4. copie `dist/` vers `/app/web-dist` dans l'étape PyTorch finale.

Node, npm, le cache et les sources TypeScript ne restent pas dans l'image
d'exécution. Le healthcheck, le port 8000 et les deux processus Python restent
inchangés.

## Stratégie de test

### Frontend

Vitest, Testing Library et un environnement DOM couvrent :

- le routage et la navigation clavier ;
- la sérialisation de chaque contrat API ;
- les champs conditionnels et validations ;
- les formats d'erreur ;
- la génération locale des exports ;
- le dépôt IndexedDB, ses limites et sa purge ;
- `MediaRecorder`, arrêt des pistes et révocation des URLs ;
- les reprises depuis l'historique ;
- les états désactivés issus de `/health`.

### Backend et conteneur

Pytest vérifie :

- `GET /` et les ressources statiques ;
- la conservation de `/docs`, `/openapi.json` et des 404 API ;
- le comportement lorsque les ressources web sont absentes hors image ;
- la présence du build dans l'image et l'absence de Node dans l'étape finale.

La validation finale inclut le build Docker, la suite Python complète, les
tests frontend, une inspection responsive et les parcours réels suivants :

1. capture micro puis transcription avec diarisation ;
2. résumé de la transcription produite ;
3. synthèse CustomVoice et lecture du résultat ;
4. création, utilisation et suppression d'un clone consenti ;
5. clonage ponctuel sans persistance de la référence ;
6. rechargement de la page et récupération de l'historique local.

## Critères d'acceptation

- l'interface complète est disponible sur `/` dans le conteneur unique ;
- aucune route API existante ni `/docs` ne régresse ;
- aucune dépendance réseau frontend n'est nécessaire à l'exécution ;
- les cinq espaces couvrent toute l'API convenue ;
- le micro et l'import de fichier fonctionnent ;
- l'historique survit au rechargement sans base serveur ;
- aucun audio source n'est persisté sans action explicite ;
- l'interface est utilisable au clavier et à 360 px ;
- l'image finale ne contient ni runtime Node ni serveur web supplémentaire ;
- toutes les suites de tests et les parcours conteneur passent.
