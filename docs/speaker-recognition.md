# Reconnaissance des personnes par la voix

L'API utilise les empreintes déjà calculées par pyannote 4. Aucun modèle
supplémentaire n'est nécessaire. Les profils sont distincts des voix de synthèse
Qwen : nommer un locuteur ne crée pas de voix TTS.

## Associer un nom

1. Envoyer l'audio à `POST /transcribe` avec `diarize=true` et
   `response_format=json` (valeur par défaut).
2. Dans la réponse, récupérer `transcription_id` et le label du locuteur dans
   `turns[].speaker_label` ou les clés de `speaker_matches`.
3. Envoyer le nom :

```bash
curl -X POST http://192.168.1.141:8000/speakers/identify \
  -H 'Content-Type: application/json' \
  -d '{"transcription_id":"UUID_RETOURNE_PAR_TRANSCRIBE","speaker":"SPEAKER_00","name":"Jean Dupont"}'
```

Réponse HTTP 200 :

```json
{"id":"UUID_DE_LA_PERSONNE","name":"Jean Dupont","samples":1}
```

Le label `SPEAKER_00` est propre à cet enregistrement : il ne désigne pas la
même personne dans tous les fichiers. L'identifiant de transcription évite
d'associer accidentellement la voix d'un autre fichier. Les anciennes réponses
produites avant cette fonctionnalité doivent être retranscrites.

## Reconnaissance automatique

Les prochains appels à `/transcribe` avec diarisation renvoient directement
le nom reconnu dans `speakers` et `turns[].speaker`. Exemple partiel :

```json
{
  "transcription_id": "NOUVEL_UUID",
  "speakers": ["Jean Dupont"],
  "speaker_matches": {
    "SPEAKER_01": {
      "speaker_id": "UUID_DE_LA_PERSONNE",
      "name": "Jean Dupont",
      "score": 0.91,
      "enrollable": true
    }
  },
  "turns": [{
    "speaker": "Jean Dupont",
    "speaker_label": "SPEAKER_01",
    "start": 0.5,
    "end": 4.2,
    "text": "Bonjour à tous.",
    "words": []
  }]
}
```

`score` est une similarité cosinus, **pas un pourcentage de certitude**.
Sans correspondance suffisante, le label reste `SPEAKER_XX` et les champs
`name`, `speaker_id`, `score` sont `null`. `enrollable=false` indique une
empreinte absente/invalide ou moins de trois secondes de parole sans
chevauchement. Les modèles anciens ne fournissant pas d'empreintes continuent
à transcrire et diariser, sans reconnaissance.

Les exports dialogue, SRT et VTT ainsi que `/summarize` à partir d'un audio
utilisent aussi les noms reconnus. Les formats texte brut et les routes
OpenAI sans diarisation conservent leur comportement. L'association ne
modifie pas rétroactivement les réponses déjà reçues ni l'historique du navigateur.

## Gérer et corriger les personnes

| Route | Usage |
| --- | --- |
| `GET /speakers` | Liste des personnes (`id`, `name`, `samples`) |
| `POST /speakers/identify` | Associer un label d'une transcription à un nom |
| `PATCH /speakers/{id}` avec `{"name":"Nouveau nom"}` | Renommer une personne |
| `DELETE /speakers/{id}` | Supprimer la personne et ses références (HTTP 204) |

Les noms sont uniques sans distinction de casse. Renvoyer un nom existant
avec un nouvel extrait ajoute une référence à cette personne (les cinq plus
récentes sont conservées). Répéter le même appel ne duplique pas l'extrait.
Renvoyer **le même identifiant de transcription et le même label** avec un
autre nom réattribue cet extrait ; une personne sans référence est retirée.
Pour corriger une reconnaissance automatique, utiliser `speaker_label`,
même quand `speaker` contient déjà un nom.

Seuls les appels explicites à `/speakers/identify` enregistrent des références.
Une reconnaissance automatique ne modifie pas le profil. Supprimer une
personne empêche sa reconnaissance sur les nouvelles requêtes ; les réponses
déjà reçues restent inchangées. Un appel explicite peut la réenregistrer.

Les erreurs donnent 404 pour une transcription expirée/inconnue ou un label
absent, 400 pour une empreinte inexploitable ou un nom déjà pris au renommage,
422 pour un corps invalide, et 503 si le registre est désactivé.

## Conservation et réglages

Sur `studio-linux`, la base SQLite est conservée dans
`/mnt/donnees/resume_transcription/speakers/speakers.sqlite3`, sur le second
disque et montée dans Docker. Elle survit aux redémarrages et reconstructions.
Sauvegarder ce répertoire avec les autres données du service.

Seules les empreintes et les noms sont conservés : aucun audio supplémentaire
n'est archivé. Les empreintes en attente d'un nom sont disponibles pendant
24 heures, dans la limite des 256 transcriptions les plus récentes. Les
enregistrements expirés sont purgés lors du prochain accès au registre.
Les références nommées restent jusqu'à leur suppression. Limites : 1000 personnes,
cinq références par personne et 256 empreintes candidates par transcription.

| Variable | Défaut | Sens |
| --- | --- | --- |
| `ENABLE_SPEAKER_RECOGNITION` | `true` | Actif si la diarisation est également activée |
| `SPEAKER_STORE_PATH` | `speakers` | Répertoire persistant (`/app/speakers` dans Compose) |
| `SPEAKER_MATCH_THRESHOLD` | `0.75` | Similarité minimale |
| `SPEAKER_MATCH_MARGIN` | `0.08` | Écart minimal entre les deux personnes les plus proches |
| `SPEAKER_MIN_SPEECH_S` | `3` | Durée minimale de parole sans chevauchement |
| `SPEAKER_PENDING_TTL_S` | `86400` | Délai pour nommer un extrait, en secondes |

Ces seuils sont des valeurs initiales à évaluer sur vos enregistrements,
pas une garantie d'identification. Utiliser des extraits propres et assez longs,
avec plusieurs références si les micros ou conditions changent. Les empreintes
de modèles différents ou de dimensions différentes ne sont pas comparées.
Après remplacement des poids d'un modèle sous le même nom, supprimer les
anciennes références et réenregistrer les personnes.

`GET /health` expose `speaker_recognition_enabled`. Le schéma interactif
est disponible sur [la documentation de l'API](http://192.168.1.141:8000/docs).
Comme les autres routes de ce serveur, ce registre est partagé entre ses
utilisateurs ; les personnes ayant accès à l'API peuvent le consulter et le modifier.
