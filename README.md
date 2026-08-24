# BestShotAI

BestShotAI analyse localement des vidéos familiales et en extrait automatiquement les
meilleures images fixes. Aucune vidéo ni image n'est envoyée sur Internet.

## État du projet

Le dépôt contient actuellement le socle du projet : organisation du code,
configuration et outillage de test. Aucun algorithme d'analyse vidéo n'est encore
implémenté.

## Pré-requis

- Python 3.12
- FFmpeg et ffprobe accessibles dans le `PATH`

## Installation de développement

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commandes

```bash
pytest
bestshot --help
```

Consultez [l'architecture](docs/ARCHITECTURE.md) pour les conventions et les
responsabilités des modules.
