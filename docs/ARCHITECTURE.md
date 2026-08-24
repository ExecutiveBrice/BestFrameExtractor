# Architecture de BestShotAI

## Principes

- Le traitement est entièrement local : aucune vidéo, image ou métadonnée n'est
  transmise à un service distant.
- Python 3.12 est la version cible.
- Les vidéos sont traitées comme des flux ; ne jamais charger une vidéo entière en
  mémoire.
- Le code est typé. Utiliser des `dataclass` pour les objets de données lorsque cela
  clarifie le modèle.
- Préférer le `logging` à `print`.
- Chaque nouveau composant doit être couvert par des tests automatisés.

## Organisation

```text
src/bestshot/
  cli.py             # Adaptateur CLI Typer : validation et délégation uniquement
  domain/            # Modèles, règles métier et interfaces indépendantes des E/S
  infrastructure/    # Adaptateurs FFmpeg/ffprobe, PyAV, fichiers et configuration
  services/          # Cas d'usage qui orchestrent le domaine et les adaptateurs
  video/             # Cas d'usage d'inspection vidéo et abstractions associées
```

`cli.py` ne contient aucune logique métier. Il appelle les cas d'usage définis dans
`services/`. Le domaine ne dépend pas de l'infrastructure. L'infrastructure porte les
détails liés aux bibliothèques externes : FFmpeg/ffprobe pour l'inspection et les
opérations vidéo, PyAV pour le décodage, PySceneDetect pour les scènes, et
OpenCV/Pillow pour l'analyse et l'export d'images.

## Ingestion vidéo

`domain.video_info.VideoInfo` est le modèle immuable des métadonnées nécessaires à
l'ingestion. `video.probe.FFprobeRunner` définit le port d'accès à ffprobe et
`video.probe.VideoProbe` transforme sa réponse JSON en modèle de domaine. Ainsi, la
logique d'interprétation des métadonnées ne dépend pas de `subprocess`.

`infrastructure.ffprobe.SubprocessFFprobeRunner` est l'adaptateur concret qui appelle
ffprobe. La commande `bestshot info VIDEO` reste un adaptateur mince : elle délègue
l'orchestration à `services.video_info` et n'appelle pas ffprobe directement.

## Détection de scènes

`domain.scene.Scene` représente une scène par son index (à partir de 1), son début,
sa fin et sa durée, en secondes. `video.scene_detector.SceneDetector` transforme les
limites temporelles en modèles du domaine. Son port `SceneDetectionBackend` permet de
tester ce comportement sans importer PySceneDetect.

`PySceneDetectBackend` utilise `AdaptiveDetector`, préférable lorsque les mouvements
de caméra sont fréquents. Il analyse le flux séquentiellement et ne produit ni image
extraite ni fichier intermédiaire. Les réglages `adaptive_threshold`,
`min_scene_len_frames`, `window_width` et `min_content_val` sont définis dans
`config/default.yaml`, puis validés par `infrastructure.config` avant usage. La
commande `bestshot scenes VIDEO` affiche simplement les scènes ainsi détectées.

## Génération des candidates

`domain.candidate_frame.CandidateFrame` préserve l'identifiant de scène, le timestamp,
le numéro de frame décodée, les dimensions source et un `PreviewImage` RGB. Cet aperçu
est un buffer mémoire redimensionné pour l'analyse : aucune image haute résolution
n'est exportée à cette étape.

`PyAVCandidateFrameBackend` décode une frame à la fois, dans l'ordre temporel, puis
la réduit à `analysis_max_width` avant de la céder au générateur. `CandidateExtractor`
échantillonne les scènes à la cadence `fps`. Les candidates sont exposées par un
itérateur et ne sont donc pas accumulées : la commande `bestshot candidates VIDEO`
ne conserve que les compteurs par scène. Les paramètres par défaut sont définis dans
la section `candidate_extraction` de `config/default.yaml` (3 images/s, largeur 960 px).

## Configuration

La configuration par défaut est `config/default.yaml`. Elle est lue par un adaptateur
d'infrastructure et convertie en objets typés avant d'atteindre le domaine.

## Tests

Les tests vivent dans `tests/` et doivent isoler le domaine des dépendances système.
Les tests d'intégration vidéo utiliseront des extraits courts et versionnés ou générés
localement. Ils portent explicitement le marqueur `integration` et peuvent être lancés
avec `pytest -m integration`.
