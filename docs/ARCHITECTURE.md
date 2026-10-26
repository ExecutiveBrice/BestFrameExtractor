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
  desktop/           # Présentation PySide6 optionnelle (aucune logique de sélection)
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

`infrastructure.workflow_factory` assemble les adaptateurs locaux et le workflow de
sélection. La CLI et l'interface de bureau réutilisent cette même composition : elles ne
s'appellent pas l'une l'autre et ne dupliquent donc aucune règle métier.

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

## Traitement par lot

`services.batch` découvre les vidéos directement présentes dans un dossier et les traite
dans un ordre stable. La commande `bestshot batch DOSSIER --output PHOTOS` réutilise le
workflow de sélection puis exporte chaque vidéo dans son propre sous-dossier. Un échec
sur une vidéo est capturé dans le rapport final et n'interrompt pas le reste du lot.
`BatchExportRunner` est le cas d'usage réutilisable : il expose des événements de
progression indépendants de toute interface et conserve les chemins des exports dans son
résultat détaillé. L'événement `completed` d'une vidéo porte également ses chemins
d'images, ce qui permet une consultation immédiate avant la fin du lot.

Les événements de lot comprennent aussi les totaux cumulés de frames retenues et de
fichiers exportés. `BatchExportRunner.run` accepte un rappel d'arrêt coopératif : il est
vérifié avant chaque vidéo, après sa sélection, puis entre deux extractions. `FinalExporter`
conserve les images déjà extraites et écrit un manifeste partiel avant de signaler cet
arrêt. La CLI n'envoie pas ce rappel et conserve donc son comportement habituel.

## Application de bureau

`desktop.MainWindow` est une présentation PySide6 proposée par l'extra optionnel
`desktop`. Elle permet de choisir le dossier source et le dossier de destination, de
lancer un `PhotoPoolRunner` assemblé par `infrastructure.workflow_factory`, puis
d'afficher les photos exportées une par une. Elle utilise exclusivement le mode pool
avec CLIP et le format JPEG, sans sélecteur d'approche ni de format. Son seul réglage
d'analyse est la cadence de 1 à 10 photos par seconde, par pas entier de 1, initialisée
depuis `photo_pool.fps` et arrondie à l'entier le plus proche. Cette valeur est transmise
au runner sans réécrire le YAML. Les règles du pool déterminent le nombre d'exports,
sans quota ni dédoublonnage. Le traitement s'exécute dans `ProcessingWorker`,
déplacé dans un `QThread`, afin que les interactions de la fenêtre restent disponibles.
Le worker ne manipule aucun widget ; il ne transmet que les événements `BatchProgress`, le
résultat et les erreurs vers le thread de l'interface. À chaque événement de vidéo
terminée, la visionneuse reçoit et affiche immédiatement ses photos, sans attendre le
reste du lot. Elle ne conserve que les chemins des photos et charge un seul `QPixmap` à la
fois. Aucune vidéo, image ou métadonnée ne quitte la machine.

La fenêtre affiche les compteurs cumulés « retenues » et « exportées » à chaque événement
de sélection ou d'export. Son bouton Arrêter déclenche un `threading.Event` lu par le
worker ; l'opération en cours se termine proprement, mais aucune nouvelle frame ou vidéo
n'est démarrée ensuite.

`services.extraction_estimate` réutilise `find_videos` et `VideoProbe` pour cumuler les
durées disponibles, sans décodage. `VideoDurationSummary` estime le nombre de candidates
par durée totale × cadence, arrondi à l'entier supérieur. `desktop.EstimateWorker`
exécute cette inspection locale dans le `QThreadPool`, après une temporisation de saisie.
La fenêtre conserve les durées pour recalculer immédiatement lors d'un changement de
cadence et ignore les réponses d'un ancien dossier. L'estimation affichée précède le
filtrage qualité/CLIP ; elle signale les durées manquantes et les dossiers sans vidéo.

## Mode pool pour album

`domain.photo_pool` porte les paramètres, les mesures `PoolQuality` et `PoolContent`,
ainsi que `PhotoPoolPolicy`. La décision applique d'abord des exigences techniques
obligatoires, puis une similarité positive (personne OU paysage) et une marge par rapport
au contenu indésirable. Le résultat est accepté, à vérifier ou rejeté. Aucun score de
visage, orientation, esthétique, quota par scène ou dédoublonnage n'intervient.

`services.photo_pool.PhotoPoolRunner` orchestre ses ports de décodage, qualité, contenu,
écriture d'image et dépôt de rapports. Il ne conserve que des chemins et des compteurs ;
les décisions sont écrites séquentiellement. Il réutilise `BatchResult` et `BatchProgress`
pour la CLI et le worker Qt. L'arrêt coopératif est transmis jusqu'au décodeur et vérifié
entre les étapes coûteuses. Les erreurs vidéo sont isolées au sein du lot, avec manifeste
partiel pour les exports déjà produits.

`infrastructure.photo_pool` contient les adaptateurs PyAV, OpenCV, Pillow et fichiers.
PyAV décode séquentiellement et ne convertit en RGB que les frames échantillonnées.
Une seule image native transitoire circule entre les ports, sans accumulation de buffers.
OpenCV mesure la netteté sur des niveaux de gris **0 à 255**, à largeur d'analyse fixe,
avec un léger débruitage, un rapport aux gradients et un contrôle natif supplémentaire.
Pillow écrit les pixels natifs de cette même frame, sans réextraction à un timestamp
approximatif. Le dépôt crée des répertoires distincts pour chaque nouvel essai.

`infrastructure.clip_content.LocalClipContentScorer` charge le modèle officiel CLIP
image/texte et son processeur avec `local_files_only=True`, sans code distant. Le cadre
est inscrit entier dans un carré avec marges, pour éviter de perdre un sujet sur les
bords par recadrage central. Les embeddings texte sont calculés une fois ; seuls les
maximums de similarité par groupe et les descriptions gagnantes sont conservés dans
les rapports. Les embeddings d'images ne sont jamais stockés. Les similarités cosinus
ne constituent pas des probabilités calibrées. Les descriptions positives couvrent
explicitement les personnes de face, de profil et de dos. CUDA, MPS ou CPU sont choisis
selon leur disponibilité.

Seul `ClipContentModelManager.download`, invoqué par `bestshot models download clip`,
accède au réseau pour télécharger une liste de fichiers de modèle à une révision fixée.
Le mode pool échoue explicitement si le modèle manque ; il ne continue pas avec un
filtre sémantique absent. Les sections `photo_pool` et `clip_content` sont converties en
objets typés par l'adaptateur de configuration. Les règles de confidentialité restent
identiques : aucune vidéo, image ou métadonnée utilisateur n'est transmise.

## Configuration

La configuration par défaut est `config/default.yaml`. Elle est lue par un adaptateur
d'infrastructure et convertie en objets typés avant d'atteindre le domaine.

## Tests

Les tests vivent dans `tests/` et doivent isoler le domaine des dépendances système.
Les tests d'intégration vidéo utiliseront des extraits courts et versionnés ou générés
localement. Ils portent explicitement le marqueur `integration` et peuvent être lancés
avec `pytest -m integration`.
