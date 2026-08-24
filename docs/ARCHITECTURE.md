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
commande `bestshot scenes VIDEO` affiche simplement les scènes ainsi détectées. Si
PySceneDetect ne détecte aucune coupe, l'adaptateur retourne la durée complète comme
une scène unique : les étapes suivantes restent donc utilisables pour toute vidéo.

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
`candidate_repository_dir` désigne le dépôt local des aperçus. La commande
`bestshot candidates VIDEO` y écrit chaque preview JPEG séquentiellement ainsi qu'un
manifeste, sans jamais enregistrer de frame native. Les analyses conservent des objets
transitoires en flux et ne déclenchent pas cette persistance.

## Score technique

`domain.technical_score.TechnicalScore` porte six qualités normalisées de `0` à `1`
(`1` est favorable) et leur moyenne pondérée : netteté, exposition, pixels brûlés,
pixels sous-exposés, contraste et faible flou de mouvement. Le module
`scoring.technical.TechnicalScorer` ne reçoit que `PreviewImage`, jamais l'image source
haute résolution. Il utilise la variance du Laplacien pour la netteté, la luminance
pour l'exposition et les pixels extrêmes, l'écart-type pour le contraste et une
heuristique d'anisotropie des gradients pour le flou de mouvement.

Tous les seuils et poids sont définis dans `technical_scoring` de `config/default.yaml`.
`bestshot analyse VIDEO --technical-only` traite les candidates en flux et affiche le
score technique moyen de chaque scène, sans conserver les aperçus après calcul.

## Score de visages

`scoring.face.MediaPipeFaceLandmarkerBackend` utilise le modèle local MediaPipe Face
Landmarker indiqué par `face_scoring.model_path`. Il ne réalise aucune reconnaissance
faciale : aucune identité, empreinte biométrique ou association entre images n'est
produite ni stockée. Pour chaque visage, `FaceAnalysis` contient seulement des mesures
de qualité (taille relative, yaw approximatif, ouverture des yeux, sourire, netteté et
coupe). `FaceScore` agrège le groupe en pénalisant la valeur la moins favorable pour
les yeux fermés, les visages tournés, flous ou coupés. L'absence de visage renvoie un
score global `None` et n'est pas un échec ni une mauvaise photo.

Les poids et tous les seuils sont configurables dans `face_scoring` de
`config/default.yaml`. Le modèle `.task` reste local. Si ce fichier est absent,
`create_face_scorer` fournit un score sans visage neutre afin que la sélection et
l'export restent possibles ; lorsqu'il est présent, le backend MediaPipe est utilisé.

## Score composite

`scoring.composite.CompositeScorer` combine les objets complets `TechnicalScore`,
`FaceScore` et les futurs scores esthétique et composition. Il sélectionne le profil
`people` dès qu'au moins un visage a un score, sinon `no_people`. Les valeurs et poids
de ces profils sont configurables dans `composite_scoring` de `config/default.yaml`.

`AestheticScoreProvider` est le port des modèles esthétiques. Son premier adaptateur,
`RsineAestheticScorer`, utilise localement `rsinema/aesthetic-scorer` : son checkpoint
est chargé comme un `state_dict` avec `torch.load(..., weights_only=True)`, puis le
backbone CLIP ViT-B/32 et la tête esthétique sont reconstruits dans notre code. Les poids
sont téléchargés explicitement et restent dans `.bestshot/models/aesthetic`; aucun code
distant du dépôt de modèle n'est exécuté. Lorsque ce modèle optionnel est absent ou
incompatible, `UnavailableAestheticScorer` retourne une valeur neutre explicitement
marquée et le pipeline reste fonctionnel. `CompositionScorer` demeure neutre jusqu'à son
implémentation. Le résultat `CompositeScore` inclut toujours les objets de score détaillés,
le profil retenu et une liste triée de `CompositeReason` (score, poids, contribution et
origine), jamais un simple flottant.

## Raffinement de candidates

`video.candidate_refiner.CandidateRefiner` prend les meilleures `RankedCandidate` de
chaque scène, limitées par `refinement.candidates_per_scene`, puis examine les frames
originales dans une fenêtre de ± `refinement.window_ms`. Le décodeur PyAV reste
séquentiel et le raffineur calcule au plus une fois les scores technique, visage et
composite pour chaque `frame_index`, même si plusieurs fenêtres se chevauchent.

La frame sélectionnée est rendue comme `RefinedCandidate` avec ses scores complets et
un aperçu réduit. Aucune frame haute résolution n'est exportée. `refinement.enabled`
permet de désactiver entièrement ce second passage.

## Dédoublonnage

`selection.deduplicate.Deduplicator` trie les `RankedCandidate` par score composite,
puis compare une candidate aux candidates déjà retenues seulement dans la fenêtre
`deduplication.temporal_window_ms`. La première implémentation
`PerceptualHashSimilarityScorer` emploie un hash perceptuel DCT sur les aperçus réduits
et conserve la mieux classée lorsqu'une similarité dépasse
`deduplication.similarity_threshold`.

Le résultat `DeduplicationResult` sépare les candidates retenues des `DuplicateCandidate`
avec la candidate gagnante, la similarité et l'écart temporel. `SimilarityScorer` est
un port : un score par embeddings peut le remplacer sans modifier la sélection.

## Sélection finale

`BestFrameSelector` reçoit les candidates classées, les scènes et le résultat de
dédoublonnage. Il applique `selection.minimum_score`, limite chaque scène par
`selection.max_per_scene`, puis sélectionne par tours entre les scènes pour préserver
la diversité temporelle. Les candidates sous le seuil ne servent jamais à remplir un
quota. `SelectionResult` contient les frames retenues, les doublons et chaque rejet.

## Export final

`FinalExporter` envoie chaque timestamp retenu à FFmpeg pour extraire directement une
frame de la vidéo originale, sans jamais agrandir les previews d'analyse. Il produit
des JPEG configurables ou des PNG, ainsi qu'un `manifest.json` contenant source,
timestamp, index de frame, scène et détails des scores.

## Configuration

La configuration par défaut est `config/default.yaml`. Elle est lue par un adaptateur
d'infrastructure et convertie en objets typés avant d'atteindre le domaine.

## Tests

Les tests vivent dans `tests/` et doivent isoler le domaine des dépendances système.
Les tests d'intégration vidéo utiliseront des extraits courts et versionnés ou générés
localement. Ils portent explicitement le marqueur `integration` et peuvent être lancés
avec `pytest -m integration`.
