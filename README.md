# BestShotAI

BestShotAI sélectionne localement les meilleures images fixes d'une vidéo. Les vidéos,
aperçus et métadonnées restent sur la machine : aucune image n'est envoyée sur Internet.

## Installation

Pré-requis : Python 3.12, `ffmpeg` et `ffprobe` dans le `PATH`.

Sur macOS avec Homebrew :

```bash
brew install ffmpeg
```

Depuis la racine du dépôt :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
bestshot --help
```

## Où placer les vidéos

Vous pouvez laisser les vidéos où elles sont : dans `~/Movies`, sur un disque externe ou
dans un dossier de travail. BestShotAI ne les copie pas et ne les modifie pas. Passez
simplement leur chemin à chaque commande.

Les exports doivent être placés dans un dossier distinct, par exemple `./photos` ou
`~/Pictures/BestShotAI/vacances`. N'utilisez pas le dossier contenant vos originaux pour
éviter de mélanger sources et résultats.

```bash
VIDEO="$HOME/Movies/vacances.mp4"
```

## Démarrage rapide

Activez l'environnement virtuel (`source .venv/bin/activate`), puis lancez les commandes
suivantes :

```bash
bestshot info "$VIDEO"
bestshot scenes "$VIDEO"
bestshot candidates "$VIDEO"
bestshot analyse "$VIDEO" --technical-only
bestshot select "$VIDEO" --count 20
bestshot extract "$VIDEO" --count 30 --output ./photos
```

`info` affiche les métadonnées FFprobe. `scenes` détecte les changements de plan ; si
aucun changement n'est trouvé, la vidéo entière forme une scène. `candidates` montre le
nombre d'aperçus analysés par scène. `select` affiche les images retenues sans créer de
fichier. `extract` produit les images finales dans leur résolution native.

Le nombre demandé est un maximum : BestShotAI ne retient pas volontairement une image
sous le seuil de qualité afin de remplir un quota. La configuration limite aussi par
défaut chaque scène à trois images pour conserver de la diversité.

## Résultats d'export

```text
photos/
  vacances_0001.jpg
  vacances_0002.jpg
  manifest.json
```

`manifest.json` conserve, pour chaque export, le chemin de la source, le timestamp,
l'index de frame, la scène et le détail complet des scores. Pour exporter en PNG :

```bash
bestshot extract "$VIDEO" --count 30 --output ./photos-png --format png
```

La qualité JPEG, l'échantillonnage, les seuils de score et les limites de sélection sont
configurables dans [`config/default.yaml`](config/default.yaml).

Le dépôt prévu pour les aperçus de candidates se règle avec
`candidate_extraction.candidate_repository_dir` (par défaut `.bestshot/candidates`).
La commande `bestshot candidates "$VIDEO"` y crée un sous-dossier par vidéo avec les
aperçus JPEG réduits et un `manifest.json`. Les commandes d'analyse gardent leurs
candidates en flux et ne les exportent pas automatiquement.

## Visages et esthétique (optionnels)

La détection de visages MediaPipe ne fait aucune reconnaissance de personne. Son modèle
reste local ; tant que `models/face_landmarker.task` n'est pas présent, la sélection et
l'export continuent simplement avec le profil « sans visage ». Pour activer cette
analyse, placez un fichier MediaPipe Face Landmarker `.task` obtenu séparément dans ce
chemin (ou modifiez `face_scoring.model_path` dans la configuration).

Le score esthétique est également facultatif. L'adaptateur utilise le modèle local
[`rsinema/aesthetic-scorer`](https://huggingface.co/rsinema/aesthetic-scorer), basé sur
CLIP ViT-B/32 et entraîné sur PARA. BestShotAI charge uniquement son `state_dict` avec
`weights_only=True` et reconstruit localement l'architecture : aucun code distant du
dépôt n'est exécuté. Sans modèle, la commande continue avec une valeur neutre et indique
son état :

```bash
bestshot models
bestshot analyse "$VIDEO" --aesthetic
```

Pour l'activer, installez l'extra puis lancez le téléchargement explicite. Cette seule
commande récupère des poids de modèle ; elle ne transmet jamais vos vidéos ou images.

Si le dépôt de modèle demande une authentification Hugging Face, exportez votre jeton
dans le terminal. Le jeton n'est jamais enregistré par BestShotAI : la configuration ne
contient que le nom de cette variable (`HUGGINGFACE_TOKEN`).

```bash
export HUGGINGFACE_TOKEN="votre_jeton_huggingface"
pip install -e ".[dev,aesthetic]"
bestshot models download aesthetic
bestshot analyse "$VIDEO" --aesthetic
```

Les poids sont mis en cache dans `.bestshot/models/aesthetic`. CLIP utilise CUDA si elle
est disponible, sinon le CPU. Le score esthétique reste plafonné à 35 % du score global
par défaut. La section `aesthetic_model` de la configuration permet de modifier le dépôt,
le nom de fichier ou la plage de sortie (par défaut, `0` à `5`).
## Vérification locale

Pour vérifier l'installation :

```bash
ruff check .
mypy src
pytest
```

Validation effectuée le 24 août 2026 avec FFmpeg/ffprobe locaux et une vidéo synthétique
locale : inspection, scènes, candidates, analyse technique, analyse esthétique en mode
fallback, sélection et export JPEG ont tous été exécutés. L'export a produit 3 JPEG
nativement en 640 × 360 et un manifeste ; l'export PNG a aussi produit une image native
640 × 360. Les 52 tests automatisés passent.

Consultez également [l'architecture](docs/ARCHITECTURE.md) et le guide contributeur
[AGENTS.md](AGENTS.md).
