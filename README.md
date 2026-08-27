# BestShotAI V2

BestShotAI analyse localement les vidéos et sélectionne des candidates régulières : dans
chaque fenêtre temporelle, il conserve les frames les plus nettes relativement à leurs
voisines. Les vidéos, aperçus et embeddings DINOv2 restent sur la machine.

## Installation

Python 3.12, `ffprobe`, puis :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,embedding,desktop]"
```

## Utilisation

Téléchargez une fois le modèle local, puis utilisez l'interface pour choisir un dossier
de vidéos et lancer le traitement :

```bash
bestshot models download embedding
bestshot desktop
# ou : bestshot-gui
```

L'interface ne parcourt pas les sous-dossiers. Pour chaque vidéo, les candidates sont
exportées en JPEG pleine résolution dans `bestshot-candidates/`, à côté de la vidéo
source. Elles correspondent uniquement au présampling temporel V2 : aucune analyse de
scène, de visage ou de qualité esthétique n'est appliquée.

La ligne de commande reste disponible pour une vidéo :

```bash
bestshot presample ./video.mp4
bestshot embeddings ./video.mp4
```

`embeddings` calcule DINOv2 seulement pour les candidates non présentes dans le cache
local. DINOv2 reste frozen et l'inférence est limitée aux fichiers déjà téléchargés.

## Dataset local de préférences

Les vidéos analysées et leurs candidates sont indexées dans
`.bestshot/dataset/bestshot.db`. SQLite ne contient jamais les pixels des frames : les
aperçus et embeddings restent dans des caches locaux externes, référencés depuis la base.

Chaque candidate conserve la vidéo source et son hash, le timestamp, l'index de frame, la
netteté, ainsi que les références de preview et d'embedding. Son label est `KEEP`,
`REJECT` ou `SKIP`. `SKIP` est enregistré comme absence de label et ne représente jamais
un rejet.

```bash
bestshot dataset stats
bestshot dataset videos
bestshot dataset reset-labels
```

Le dataset prépare uniquement la collecte locale des préférences. Aucun modèle personnel
n'est entraîné à ce stade.

L'onglet **Apprentissage IA** ouvre un dossier `bestshot-candidates/` déjà créé par
l'analyse. Il affiche les aperçus locaux et permet de les marquer **Accepter**,
**Rejeter** ou **Passer**. Les écritures SQLite sont réalisées hors du thread de
l'interface ; aucun aperçu n'est envoyé hors de la machine.

L'onglet **Sélection IA** choisit un dossier de vidéos puis lance le traitement. Une tête
linéaire locale est entraînée sur les embeddings DINO déjà calculés et les seuls labels
`KEEP`/`REJECT` ; DINOv2 reste entièrement frozen. Les candidates prédites `KEEP` sont
exportées dans `bestshot-selection/`, à côté de chaque vidéo source.

## Vérification

```bash
ruff check .
mypy src
pytest
```

Consultez [l'architecture V2](docs/ARCHITECTURE_V2.md) pour les invariants du pipeline.
