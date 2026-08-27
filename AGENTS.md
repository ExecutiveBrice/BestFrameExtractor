# Guide contributeur

Le point d'entrée de la documentation technique est
[docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md). Les invariants du modèle personnel
sont détaillés dans [docs/PERSONAL_RANKING.md](docs/PERSONAL_RANKING.md).

- Respecter l'architecture domaine / infrastructure décrite dans ce document.
- Avant toute modification, consulter la configuration dans `config/default.yaml`.
- Ajouter ou adapter les tests dans `tests/` pour chaque nouveau composant.
- Ne pas modifier les règles de confidentialité locale du projet.
- Le présampling V2 ne doit pas inclure de détection de scènes, d'analyse de
  visages, de score esthétique ou de score composite.
- La netteté V2 sert uniquement à classer des frames d'une même fenêtre : aucun
  seuil absolu et aucune comparaison entre vidéos ne sont autorisés.
- DINOv2 reste entièrement frozen. Le modèle personnel n'entraîne que sa tête de ranking
  à partir d'embeddings locaux déjà calculés.
- Les anciens labels KEEP/REJECT/SKIP sont historiques : ne pas les convertir
  automatiquement en préférences pairwise, et ne jamais entraîner sur SKIP.
- Les paires persistées sont canoniques pour que A > B et B < A ne créent pas deux
  observations. Les accès SQLite de l'interface graphique restent hors du thread UI.
