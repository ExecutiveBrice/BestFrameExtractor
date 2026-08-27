# BestShotAI V2

BestShotAI V2 produit localement des candidates stables depuis une vidéo, puis calcule des
embeddings visuels DINOv2 pour les candidates. Les vidéos, les aperçus et les embeddings
restent sur la machine : aucune image n'est envoyée sur Internet.

## Installation

Pré-requis : Python 3.12 et `ffprobe` dans le `PATH`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,embedding]"
bestshot --help
```

## Utilisation

Les vidéos restent à leur emplacement d'origine. Définissez simplement leur chemin :

```bash
VIDEO="$HOME/Movies/vacances.mp4"
```

Le présampling décode séquentiellement la vidéo, analyse environ 8 frames/s et conserve
les deux frames les plus nettes de chaque fenêtre d'une seconde. La netteté sert
uniquement à classer les frames de leur propre fenêtre : elle n'est jamais un seuil de
qualité ni une comparaison entre vidéos.

```bash
bestshot presample "$VIDEO"
```

La commande affiche la durée, les frames décodées, les frames analysées, les candidates
générées et leur densité par minute. Elle ne crée aucun fichier.

## Embeddings DINOv2 locaux

Le provider initial est DINOv2 ViT-S/14 (`facebook/dinov2-small`). Son téléchargement est
explicite ; l'inférence suivante est limitée aux fichiers locaux et n'exécute pas de code
distant.

```bash
bestshot models download embedding
bestshot embeddings "$VIDEO"
```

DINOv2 est entièrement frozen (`eval`, aucun gradient) et utilise CUDA lorsqu'elle est
disponible, sinon le CPU. `embeddings` ne convertit en RGB pour DINOv2 que les candidates
absentes du cache. Les vecteurs L2 normalisés sont stockés dans `.bestshot/embeddings`, avec
une clé qui contient l'identité de la vidéo, la frame et la version du modèle. Les previews
réduits des seules candidates sont conservés hors SQLite dans `.bestshot/dataset/previews`
pour la revue locale ; aucune frame 4K ou image n'est envoyée sur Internet.

Les réglages sont dans [`config/default.yaml`](config/default.yaml) :
`personal_pipeline.presampling` pour les fenêtres temporelles et `embedding_model` pour
le backbone, ses poids et son cache.

## Dataset local de préférences et ranking personnel

Le dataset SQLite local est créé à la demande dans `.bestshot/dataset/bestshot.db`. Il
contient les vidéos, candidates, références aux previews/embeddings et préférences
pairwise. SQLite ne contient aucun blob image ni frame 4K.

Les anciens labels `KEEP`, `REJECT` et `SKIP` restent compatibles mais ne sont plus le
signal d'apprentissage par défaut. Le ranking personnel utilise `FIRST`, `SECOND`, `EQUAL`
et `SKIP` : ce dernier est une absence d'information et n'est jamais converti en rejet.

```bash
bestshot dataset stats
bestshot dataset videos
bestshot dataset reset-labels
```

`reset-labels` ne supprime ni les vidéos ni les candidates : il remet seulement tous les
labels à `SKIP`.

Après l'ingestion locale, générez puis comparez des paires. La fenêtre PySide6 est
optionnelle (`pip install -e ".[desktop]"), et les raccourcis sont `←`, `→`, `Espace`,
`Échap`.

```bash
bestshot preferences generate "$VIDEO"
bestshot preferences review "$VIDEO"
bestshot preferences stats
bestshot train-ranking
bestshot ranking-score ./photo.jpg
```

Seule une tête linéaire est entraînée ; DINOv2 reste frozen. Chaque entraînement écrit un
nouvel artefact dans `.bestshot/models/personal/model-XXXX/` et met à jour
`.bestshot/models/personal/current.json`, sans écraser un modèle précédent. Consultez le
[guide du ranking personnel](docs/PERSONAL_RANKING.md).

## Vérification

```bash
ruff check .
mypy src
pytest
```

Consultez [l'architecture V2](docs/ARCHITECTURE_V2.md) et le guide contributeur
[AGENTS.md](AGENTS.md).
