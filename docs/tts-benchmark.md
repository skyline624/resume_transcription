# Benchmark de qualité Qwen3-TTS / VoxMind

Ce protocole sépare trois questions qui ne doivent pas être confondues :
l'intelligibilité mesurée, la performance matérielle et la préférence humaine.
Un WER inférieur ne suffit jamais à déclarer une voix meilleure.

## Corpus et exécution

Le corpus versionné `tests/fixtures/tts_corpus_fr.json` contient 20 textes
français fixes : nombres, dates, devises, pourcentages, sigles, noms propres,
questions, exclamations, parenthèses, discours direct, émotion et un texte
multi-phrase proche de la limite de 4096 caractères.

Dans le conteneur en fonctionnement :

```bash
/opt/venv-main/bin/python scripts/benchmark_tts.py \
  --output-dir /app/benchmark-results \
  --voxmind-dir /app/voxmind-exports
```

Les exports VoxMind sont facultatifs et portent le nom de l'identifiant du
corpus (`dates-01.wav`, par exemple). Le script :

1. décharge Qwen avant la mesure à froid ;
2. génère une seconde fois pour la latence à chaud ;
3. mesure durée, RTF et variation de VRAM ;
4. retranscrit chaque sortie avec `/v1/audio/transcriptions` ;
5. applique la même normalisation et le même WER aux deux moteurs ;
6. écrit `results.csv` et `results.json` avec le commit et les paramètres.

`--limit N` permet un smoke test sans modifier le corpus. Les fichiers de
résultats et les audios ne doivent pas être commités s'ils contiennent une voix
personnelle.

## Écoute A/B aveugle

Pour chaque texte, copier les sorties Qwen et VoxMind sous des identifiants
aléatoires sans nom de moteur. Produire un ordre différent par évaluateur et
ne révéler la correspondance qu'après verrouillage des notes. Utiliser un
casque identique, un niveau sonore normalisé et interdire l'accès aux métriques
pendant l'écoute.

Chaque critère reçoit une note entière de 1 (inacceptable) à 5 (excellent) :

| Critère | Question posée |
|---|---|
| Naturel | La voix semble-t-elle humaine plutôt que synthétique ? |
| Prosodie | Rythme, accentuation, pauses et intonation conviennent-ils au texte ? |
| Artefacts | Entend-on clics, souffle, répétitions, coupures ou instabilité du timbre ? |
| Intelligibilité | Tous les mots et nombres sont-ils compris sans effort ? |
| Fidélité du clone | Identité, accent et couleur restent-ils proches de la référence consentie ? |

Ajouter une préférence forcée A/B et un commentaire libre. La fidélité de
clone est notée uniquement par une personne autorisée à écouter la référence.

## Critères de décision

L'amélioration est acceptée si Qwen obtient une préférence majoritaire en
aveugle, un WER indicatif inférieur ou égal à 8 % sur le corpus propre et
aucune régression matérielle bloquante face à la meilleure base VoxMind. Toute
exception doit être documentée par catégorie ; la moyenne seule ne doit pas
masquer les nombres, les noms propres ou le texte long.

Les mesures RTX 3090 produites pendant la validation sont conservées avec les
artefacts `results.csv` et `results.json`. Les notes d'écoute exigent plusieurs
évaluateurs humains et ne sont donc pas fabriquées par le script.
