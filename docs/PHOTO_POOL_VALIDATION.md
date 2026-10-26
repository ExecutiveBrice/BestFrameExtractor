# Essai local du mode pool — 26 octobre 2026

Les sept vidéos de `resources/` ont été parcourues intégralement, soit environ
20 minutes 55 secondes. L'essai utilise une candidate par seconde pour limiter le
volume d'export pendant le calibrage. La cadence proposée dans l'application reste
de trois candidates par seconde.

```bash
.venv/bin/bestshot pool resources --fps 1 --output .bestshot/photos-pool
```

| Vidéo | Photos dans le pool | À vérifier | Rejetées |
|---|---:|---:|---:|
| Film_famille_2018 | 176 | 20 | 13 |
| Film_famille_2019 | 161 | 45 | 2 |
| Film_famille_20192 | 134 | 73 | 26 |
| Film_famille_2020 | 156 | 15 | 37 |
| Film_famille_2021 | 105 | 15 | 34 |
| Film_famille_2022 | 162 | 38 | 30 |
| GoPro_video | 15 | 1 | 0 |
| **Total** | **909** | **207** | **142** |

Ces nombres décrivent les décisions du programme sur 1 258 candidates. Ils ne
constituent pas une mesure de précision : les vidéos ne disposent pas d'annotations
humaines indiquant, pour chaque frame, sa netteté et son admissibilité.

Les 142 rejets comprennent 134 échecs du filtre technique et 8 du filtre de contenu.
L'export a terminé sans erreur pour les sept vidéos. Toutes les dimensions des photos
exportées ont été vérifiées : 1 920 × 1 080 pour les montages et 3 840 × 2 160 pour
la GoPro. Les images restent similaires quand le plan ne change pas.

Un essai supplémentaire de la GoPro à la cadence par défaut a produit **45 photos 4K**
dans le pool et **3 à vérifier**, sur 48 candidates :

```bash
.venv/bin/bestshot pool resources/GoPro_video.MP4 --output .bestshot/photos-pool-3fps
```

## Vérifications

- 100 tests automatisés passent, dont un test vidéo généré localement pour vérifier
  timestamps, index de frame, dimensions natives, intervalle demandé et arrêt.
- `ruff check .`, `mypy src` et `git diff --check` passent.
- Les tests couvrent le rejet technique sans compensation par CLIP, l'absence de
  dédoublonnage/quota par scène, la séparation des cas incertains, les fichiers locaux,
  le téléchargement explicite à révision fixée, les erreurs et les exports partiels.
- Les planches de contrôle présentent les quatre nettetés retenues les plus faibles,
  puis huit instants répartis dans chaque vidéo. Les sept planches de photos retenues
  ont été inspectées, ainsi que des planches de rejets et de cas incertains.
- Une image réelle de la GoPro est acceptée dans son état initial ; ses variantes
  artificiellement floutées avec des rayons gaussiens de 4, 8 et 16 pixels sont rejetées.
- L'interface a été rendue hors écran et contrôlée visuellement dans le mode pool.

## Réglage et limites observées

Le seuil initial de netteté était trop permissif à taille native. Le réglage retenu
exige une variance de Laplacien d'au moins 70 à largeur 960, un rapport aux gradients
d'au moins 0,025 et une variance native d'au moins 20. Il utilise des niveaux de gris
sur l'échelle 0 à 255. Le décalage d'échelle du scorer technique historique a également
été corrigé et possède un test de régression avec les paramètres par défaut.

Les descriptions CLIP couvrent les personnes de face, de profil et de dos, ainsi que
des situations courantes : randonnée, piscine, camping, ski et kayak. Aucun score de
visage ou critère d'orientation ne limite l'admission. Les descriptions génériques de
flou ont été retirées du filtre sémantique après observation de nombreux faux doutes
sur des photos utiles ; la netteté est une condition technique indépendante.

Des photos utiles restent dans `a_verifier`, et certaines peuvent être rejetées. La
netteté globale ne garantit pas celle de chaque personne : un fond net peut masquer
un sujet en mouvement. Certains fondus de montage peuvent passer les deux filtres.
Les textes et bandes noires déjà incrustés dans les montages sont conservés. Les
labels « personnes » et « paysage » sont des hypothèses CLIP sur le cadre entier.

Les aperçus et rapports détaillés sont dans `.bestshot/photos-pool/`, et le second
essai GoPro dans `.bestshot/photos-pool-3fps/`. Ils restent dans le dépôt local ignoré
par Git ; les vidéos sources n'ont pas été modifiées. Aucune vidéo, image ou métadonnée
utilisateur n'a été envoyée à un service distant.
