# Architecture V2 — présampling temporel

## Objectif

La V2 remplace la décision esthétique précoce par un premier passage local qui produit
une densité de candidates régulière et techniquement exploitable. Elle ne détermine pas
si une photo est « bonne » : elle choisit seulement, dans chaque courte fenêtre, les
frames les moins floues relativement à leurs voisines.

Le pipeline est entièrement local. Il ne transmet, ne persiste ni ne reconnaît aucune
image ou donnée biométrique. Les buffers de gris utilisés pour la comparaison sont
transitoires et libérés après chaque fenêtre.

## Chaîne de traitement

```text
vidéo
  │
  ├─ PyAVTemporalSamplingBackend : adaptateur d'infrastructure, décodage séquentiel
  │
  ├─ TemporalSampler : conserve environ 8 instants/s ; conversion ciblée en gris ≤ 640 px
  │
  ├─ CandidateGenerator : groupes fixes de 1 seconde
  │     │
  │     └─ SharpnessRanker : variance du Laplacien, classement local uniquement
  │
  └─ 2 PresampledCandidate / fenêtre non vide (timestamp et métadonnées source)
```

`infrastructure.temporal_sampling.PyAVTemporalSamplingBackend` ne convertit aucune frame
décodée en RGB ou Pillow. Après
le filtre temporel, et seulement pour une frame à analyser, il utilise PyAV pour la
réduire directement en niveaux de gris. Ainsi les frames entre deux instants
d'échantillonnage ne créent ni image Pillow, ni buffer RGB, ni aperçu persistant.

`TemporalSampler` maintient une cadence ciblée de `analysis_fps` à partir des timestamps
de présentation. Il compte toutefois toutes les frames effectivement décodées, y compris
les frames sans timestamp, afin que la commande puisse rapporter le travail réel.

## Fenêtres et netteté

Les fenêtres sont ancrées à l'origine temporelle de la vidéo : `[0, 1[`, `[1, 2[`, etc.
Chaque fenêtre non vide est classée avec la variance du Laplacien de son aperçu en gris.
`SharpnessRanker` retient les `keep_per_bucket` mesures les plus élevées et départage les
égalités par timestamp puis par index de frame. Les candidates sont ensuite remises dans
l'ordre temporel, ce qui rend la sortie déterministe.

La mesure de netteté est volontairement brute. Il n'existe ni normalisation, ni seuil,
ni agrégation entre fenêtres ou entre vidéos. Une fenêtre dont toutes les frames sont
floues conserve quand même ses deux meilleures frames disponibles. La V2 stabilise ainsi
la quantité de candidates sans prétendre évaluer leur valeur esthétique.

Le résultat `CandidateGenerationResult` ne conserve que les métadonnées légères des
`PresampledCandidate` (timestamp, index de frame, dimensions source, fenêtre et mesure
brute de netteté). Il ne contient pas de preview RGB. La mesure est conservée comme
métadonnée de provenance pour le dataset ; elle ne devient ni seuil, ni score global, ni
comparaison entre fenêtres ou vidéos.

## Composition et CLI

La logique V2 est dans `sampling/` :

```text
sampling/
  temporal_sampler.py     # port de décodage et filtre temporel
  sharpness_ranker.py     # variance du Laplacien et classement local
  candidate_generator.py  # fenêtres fixes et résultat de candidates
infrastructure/temporal_sampling.py # adaptateur PyAV et gris ciblé
services/presampling.py   # cas d'usage et formatage des indicateurs
cli.py                    # adaptateur Typer de la commande presample
```

La CLI n'orchestre pas de logique métier : `bestshot presample VIDEO` charge
`personal_pipeline.presampling`, inspecte les métadonnées locales avec ffprobe puis
appelle le cas d'usage. Elle affiche la durée, les frames vidéo décodées, les frames
réellement analysées, le nombre de candidates et les candidates par minute.

Cette commande ne crée pas de fichier et ne réalise ni détection de scènes, ni analyse
de visages, ni score esthétique ou composite. Elle se limite à une présélection temporelle
stable et techniquement exploitable.

## Embeddings visuels locaux

`bestshot embeddings VIDEO` réutilise les candidates V2 et les ingère dans le dataset
local. Il réalise d'abord le présampling, puis identifie les candidates absentes du cache.
Un second décodage PyAV reste séquentiel et ne convertit en RGB que ces frames manquantes
pour DINOv2 ; toutes les autres frames vidéo ne sont ni converties ni transmises au modèle.
Un passage ciblé supplémentaire peut écrire les aperçus réduits des seules candidates dans
le cache externe du dataset : il ne charge jamais la vidéo entière ni une frame 4K dans
SQLite.

```text
PresampledCandidate
  ├─ EmbeddingCache : cache hit → vecteur local réutilisé
  └─ PyAVCandidatePreviewReader → ImageEmbeddingProvider → vecteur L2 normalisé
```

`ImageEmbeddingProvider` est le port qui permet de remplacer le backbone sans modifier
le cache ou l'orchestration. Sa première implémentation,
`DINOv2EmbeddingProvider`, utilise DINOv2 ViT-S/14 (`facebook/dinov2-small`). Les poids
sont téléchargés uniquement après `bestshot models download embedding`, dans le cache
local configuré. Le chargement impose `local_files_only=True` et
`trust_remote_code=False` : l'inférence n'accède jamais au réseau et n'exécute aucun code
du dépôt du modèle.

Le backbone est passé en `eval()`, chacun de ses paramètres a
`requires_grad=False`, et l'inférence s'exécute sous `torch.inference_mode()`. CUDA est
sélectionnée lorsqu'elle est disponible, sinon le provider utilise le CPU. Aucun
entraînement ou gradient n'est donc possible dans ce composant.

`EmbeddingCache` ne stocke que le vecteur normalisé et ses métadonnées locales, jamais
les pixels. Sa clé inclut le chemin, la taille et la date de modification de la vidéo,
le timestamp, l'index de frame et la version du modèle. Une frame déjà analysée pour la
même version ne repasse donc pas dans DINOv2 lors d'un entraînement ultérieur du modèle
personnel. Modifiez `embedding_model.model_version` à chaque changement de poids ou de
backbone afin de créer un espace de cache distinct.

La commande affiche le device, le modèle, les embeddings nouvellement calculés, les cache
hits et le temps de traitement. Les vecteurs et aperçus réduits restent exclusivement sur
le disque local et sont référencés par SQLite ; aucun pixel ne sort de la machine.

## Dataset local de préférences

Le module `dataset/` conserve la collecte locale et `learning/` entraîne le head personnel.
Les structures binaires historiques restent lisibles, sans conversion automatique :

```text
VideoRecord    identité locale de la vidéo (chemin, hash, taille, date)
FrameRecord    candidate (vidéo, timestamp, index, preview, netteté, embedding, label)
FrameLabel     KEEP | REJECT | SKIP
TrainingModel  métadonnées réservées à un futur entraînement
```

`SQLiteDatasetRepository` applique ses migrations dans
`.bestshot/dataset/bestshot.db` par défaut. La table `frames` ne comporte aucune colonne
blob : `preview_reference` et `embedding_reference` pointent vers les caches locaux
externes. Il est donc impossible d'y stocker une frame 4K.

`FrameLabel.KEEP`, `REJECT` et `SKIP` sont historiques : ils ne sont plus le signal
d'entraînement par défaut. `SKIP` est représenté par `NULL` et n'est jamais interprété
comme un rejet. Une réinsertion de frame sans label conserve un label utilisateur existant.

Les commandes `bestshot dataset stats`, `bestshot dataset videos` et
`bestshot dataset reset-labels` créent ou migrent la base locale à la demande. Cette
dernière commande efface seulement les labels et remet les frames à `SKIP` ; elle ne
supprime jamais les candidates, caches ou vidéos.

## Préférences pairwise et ranking personnel

Le feedback V2 est relatif : l'utilisateur choisit la meilleure image parmi deux
candidates, ou indique une égalité ou une absence de jugement.

```text
PairwisePreference
  ├─ FIRST  : première frame affichée préférable à la seconde
  ├─ SECOND : seconde frame affichée préférable à la première
  ├─ EQUAL  : différence de score à rapprocher de zéro
  └─ SKIP   : aucune donnée d'entraînement (NULL dans SQLite)
```

La table `pairwise_preferences` stocke une paire d'identifiants en ordre croissant, avec
une contrainte d'unicité et des index par frame et choix. Le repository inverse le choix
lorsque l'UI soumet la paire dans l'autre sens : `A > B` et `B < A` mettent donc à jour la
même observation. Les anciens labels `KEEP`/`REJECT` ne sont jamais convertis en paires.

`learning.pair_generator` propose des paires proches dans le temps (`Nearby`), proches en
cosinus DINOv2 (`Similarity`) ou une alternance dédupliquée des deux (`Mixed`, défaut).
Les paires déjà enregistrées sont exclues, sauf revue explicitement demandée.

```text
candidates + caches DINOv2 locaux
  └─ pair_generator ──> UI PySide6 (thread SQLite séparé)
                           └─ pairwise_preferences
                                └─ RankingTrainer
                                     └─ LinearRankingModel + artefact versionné
```

`LinearRankingModel` est une unique couche linéaire. Sa perte RankNet traite `FIRST` et
`SECOND`; `EQUAL` ajoute une perte quadratique pondérée sur l'écart des scores. `SKIP` est
exclu avant tout split et toute optimisation. DINOv2 n'est pas instancié par le trainer et
ses poids frozen ne peuvent donc jamais entrer dans l'optimiseur.

La validation est groupée par vidéo. Une paire qui traverserait train et validation est
exclue de ce split plutôt que de créer une fuite. Les métriques JSON contiennent les
précisions `FIRST`, `SECOND`, `EQUAL` (avec `equality_margin`), la précision globale, les
compteurs train/validation et les paires croisées écartées.

Les artefacts sont locaux et immuables :

```text
.bestshot/models/personal/
  current.json
  model-0001/
    model.pt         # head linéaire uniquement, sans DINOv2
    metadata.json
    metrics.json
```

Un entraînement crée toujours un nouveau `model-XXXX` et met à jour seulement
`current.json`. `ranking-score` retourne un score relatif au modèle courant, jamais un
seuil de qualité V2. Il est prêt à être consommé par une sélection finale future sans
modifier la quantité stable produite par le présampling.

## Configuration

Les paramètres V2 sont validés dans `infrastructure.config` avant d'atteindre le
pipeline :

```yaml
personal_pipeline:
  presampling:
    analysis_fps: 8
    bucket_seconds: 1.0
    keep_per_bucket: 2
    analysis_max_width: 640

embedding_model:
  repo_id: facebook/dinov2-small
  revision: main
  model_version: dinov2-vits14-1
  model_cache_dir: .bestshot/models/embedding/dinov2-vits14
  embedding_cache_dir: .bestshot/embeddings

dataset:
  database_path: .bestshot/dataset/bestshot.db
  preview_cache_dir: .bestshot/dataset/previews

pair_generation:
  temporal_window_seconds: 5
  max_pairs_per_group: 10
  seed: 42

personal_ranking:
  model_type: linear
  equal_loss_weight: 0.5
  learning_rate: 0.001
  epochs: 100
  weight_decay: 0.0001
  validation_ratio: 0.20
  early_stopping_patience: 10
  equality_margin: 0.05
  seed: 42
```

Ils doivent tous être strictement positifs. Les valeurs par défaut donnent environ deux
candidates par seconde de vidéo, sauf lorsque la cadence ou la durée ne fournit pas assez
de frames dans une fenêtre.

## Tests

Les tests unitaires vérifient le filtre temporel, l'absence de conversion des frames
ignorées, le classement sans seuil absolu, la sélection par fenêtre, les compteurs du
rapport et l'adaptateur CLI. Ils vérifient également la normalisation, le gel de DINOv2,
le cache persistant, le décodage RGB ciblé et les compteurs d'embeddings. Ils couvrent
aussi la canonisation/déduplication des paires, les stratégies de proposition, le split
groupé par vidéo, les artefacts versionnés et l'ordre synthétique `A > B > C`. Les
intégrations éventuelles s'appuient uniquement sur des vidéos synthétiques locales et sont
marquées `integration`.
