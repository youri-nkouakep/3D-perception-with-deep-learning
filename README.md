# perception

Projet de vision par ordinateur.

## Installation

```bash
conda env create -f environment.yml
conda activate perception
pytest
```

## Structure

- `src/perception/` : code du package
- `tests/` : tests
- `notebooks/` : exploration
- `configs/` : fichiers de configuration
- `scripts/` : scripts d'entraînement / inférence
- `data/raw`, `data/processed` : données (non versionnées)
- `models/` : poids des modèles (non versionnés)
