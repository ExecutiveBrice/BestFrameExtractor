# Ranking personnel local

Le modèle personnel BestShotAI apprend des comparaisons relatives, jamais des règles
esthétiques universelles. Tout le flux reste local : vidéos, previews, embeddings DINOv2,
préférences SQLite et poids du head entraîné.

## Flux de travail

```bash
bestshot models download embedding
bestshot embeddings VIDEO
bestshot preferences generate VIDEO
bestshot preferences review VIDEO   # nécessite l'extra PySide6
bestshot preferences review VIDEO --reviewed  # revoit et modifie des réponses
bestshot preferences stats
bestshot train-ranking
bestshot ranking-score IMAGE
```

Installez l'interface facultative avec :

```bash
pip install -e ".[desktop]"
```

L'écran propose deux aperçus réduits et les actions `Premier`, `Second`, `Égal` et
`Passer` (`←`, `→`, `Espace`, `Échap`). Il écrit chaque réponse immédiatement dans SQLite
depuis un thread de travail, puis charge la paire suivante sans bloquer l'interface.

## Sémantique des réponses

| Choix | Effet d'entraînement |
| --- | --- |
| `FIRST` | le score de la première frame doit dépasser le second |
| `SECOND` | le score de la seconde frame doit dépasser le premier |
| `EQUAL` | les deux scores sont rapprochés par une perte quadratique pondérée |
| `SKIP` | aucune donnée ; exclue du training et des métriques |

Les identifiants d'une paire sont canoniques en base. Enregistrer `A > B` puis `B < A`
met à jour une seule ligne : il n'existe pas de vote inverse dupliqué.

## Modèle et validation

Le backbone DINOv2 ViT-S/14 produit des vecteurs L2 normalisés et reste frozen. Seule une
couche linéaire est optimisée avec des préférences RankNet. Le training ne lit aucun pixel
et ne télécharge aucun modèle.

Le split est fait par vidéo entière. Lorsqu'une comparaison porte sur deux vidéos qui
tombent de part et d'autre du split, elle est exclue plutôt que de créer une fuite. Les
résultats écrivent `metrics.json` avec les précisions globales et par type de préférence,
la marge d'égalité, les comparaisons et vidéos train/validation.

## Artefacts et compatibilité

Chaque exécution crée un nouveau répertoire, sans écraser les précédents :

```text
.bestshot/models/personal/model-0001/
  model.pt
  metadata.json
  metrics.json
```

`current.json` désigne la version active. `metadata.json` inclut la date, la version
d'embedding, la dimension, les compteurs, les hyperparamètres, le seed et le format.

Les anciens `FrameLabel` (`KEEP`, `REJECT`, `SKIP`) restent dans la base pour compatibilité
et administration, mais aucun entraînement pairwise ne les consomme automatiquement.
