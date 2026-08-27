# Architecture V2 — analyse et candidates locales

## Objectif

BestShotAI V2 produit des candidates stables à partir de vidéos locales. Il ne cherche
pas à évaluer une qualité esthétique : dans chaque courte fenêtre, il conserve seulement
les frames les moins floues relativement à leurs voisines. Le dataset local conserve les
labels personnels, utilisés par une tête de décision locale sans modifier DINOv2.

## Pipeline

```text
vidéo locale
  │
  ├─ PyAVTemporalSamplingBackend : décodage séquentiel
  ├─ TemporalSampler : environ 8 instants/s, gris ciblé ≤ 640 px
  ├─ CandidateGenerator : fenêtres fixes de 1 seconde
  ├─ SharpnessRanker : variance du Laplacien, classement dans la fenêtre
  ├─ DINOv2 frozen : embeddings locaux et cache persistant
  └─ export JPEG : bestshot-candidates/ à côté de la vidéo
```

Le présampling ne fait ni détection de scènes, ni analyse de visages, ni score esthétique
ou composite. La netteté ne sert qu'à classer les frames d'une même fenêtre ; elle n'a ni
seuil absolu ni comparaison entre vidéos.

## Couches

La logique métier de présampling est dans `sampling/` :

```text
sampling/
  temporal_sampler.py
  sharpness_ranker.py
  candidate_generator.py
infrastructure/temporal_sampling.py
infrastructure/embedding_frames.py
infrastructure/selection_export.py
services/presampling.py
services/embeddings.py
```

`desktop/application.py` n'affiche qu'un choix de dossier et le démarrage du traitement.
Son onglet **Apprentissage IA** ouvre ensuite un dossier `bestshot-candidates/`, affiche
les aperçus externes et enregistre les labels `KEEP`, `REJECT` ou `SKIP`. Les accès vidéo,
DINOv2, cache et SQLite s'exécutent dans un worker Qt ; seuls les signaux de progression
modifient l'interface.

L'onglet **Sélection IA** ne contient qu'un choix de dossier vidéo et le lancement du
traitement. Son worker entraîne une couche linéaire binaire sur tous les embeddings du
dataset associés aux labels `KEEP` et `REJECT` — `SKIP` est exclu — puis présample et
embedde les vidéos choisies pour l'inférence. Ces vidéos, leurs candidates et leurs aperçus
ne sont pas ajoutés à SQLite : seules les candidates prédites `KEEP` sont exportées dans
`bestshot-selection/`. DINOv2 n'est pas chargé par l'entraînement et ses poids ne sont
jamais modifiés.

## Données locales

SQLite garde l'identité des vidéos et les métadonnées des candidates. Les pixels restent
hors de SQLite : `preview_reference` et `embedding_reference` pointent vers des caches
locaux. L'export JPEG est aussi local, dans `bestshot-candidates/` à côté de chaque source.

Le schéma SQLite crée `videos`, `frames` et `training_models`. Une candidate contient la
référence de sa vidéo et son hash, son timestamp, son index, sa netteté et ses références
d'aperçu et d'embedding. `FrameLabel` accepte `KEEP`, `REJECT` et `SKIP` ; `SKIP` est
stocké en `NULL`, c'est donc une absence de label et jamais un rejet. La base par défaut
est `.bestshot/dataset/bestshot.db`.

`DINOv2EmbeddingProvider` est en `eval()` avec tous les paramètres gelés et s'exécute sous
`torch.inference_mode()`. Les poids sont téléchargés explicitement avec
`bestshot models download embedding`; les inférences utilisent ensuite les fichiers locaux
uniquement.

## Configuration

```yaml
personal_pipeline:
  presampling:
    analysis_fps: 8
    bucket_seconds: 1.0
    keep_per_bucket: 2
    analysis_max_width: 640

embedding_model:
  repo_id: facebook/dinov2-small
  model_version: dinov2-vits14-1
  model_cache_dir: .bestshot/models/embedding/dinov2-vits14
  embedding_cache_dir: .bestshot/embeddings

dataset:
  database_path: .bestshot/dataset/bestshot.db
  preview_cache_dir: .bestshot/dataset/previews
```

## Tests

Les tests couvrent le filtre temporel, le classement local par netteté, le cache
d'embeddings, l'export des candidates et les mises à jour Qt depuis le thread UI.
