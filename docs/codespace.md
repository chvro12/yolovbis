# GitHub Codespaces

## Démarrage

1. Ouvrez le dépôt sur GitHub → **Code** → **Codespaces** → **Create codespace on main**.
2. Le **devcontainer** installe les dépendances (`pip install -e ".[dev,ui,train_satellite]"`) et télécharge `yolov8s.pt` + le modèle VisDrone (Hugging Face).
3. Les checkpoints **versionnés dans Git** sont déjà présents :
   - `data/aerial_weights/dota_finetune_v1/run2/weights/best.pt` — détection véhicules (DOTA)
   - `data/runs/essai_cli_train/best.pt` — segmentation parking (APKLOT / essai)

## Commandes utiles

```bash
# Tests (réseau mocké)
pytest -q

# Diagnostic complet sur une adresse (YOLO véhicules DOTA + orthophoto IGN)
parking-capacity diagnose-address "2 Bd Industriel, 76270 Neufchâtel-en-Bray" \
  --dota-yolo \
  --cache-dir data/.cache_http \
  --out /tmp/diag

# Segmentation parking YOLO
parking-capacity test-segmentation-real \
  --weights data/runs/essai_cli_train/best.pt \
  --out /tmp/seg \
  "2 Bd Industriel, 76270 Neufchâtel-en-Bray"

# Interface web (écoute sur toutes les interfaces en Codespace)
parking-capacity ui --host 0.0.0.0 --port 7860
```

Ouvrez l’onglet **Ports** (7860) si le navigateur ne s’ouvre pas automatiquement.

## Poids non inclus dans Git

| Ressource | Taille | Obtention |
|-----------|--------|-----------|
| `datasets/` (DOTA) | ~4 Go | Entraînement local uniquement ; voir `scripts/prepare_dota_vehicles.py` |
| SegFormer UTEL | ~1 Go | Téléchargé au premier `run-address` / `diagnose-address` sans `--no-vision` |
| `yolov8s-obb.pt` | optionnel | Non requis pour l’inférence par défaut |

Relancer le script d’installation : `bash scripts/setup_codespace.sh`.

## Secrets optionnels

Copiez `.env.example` vers `.env` pour Mapillary, Roboflow, etc. (non requis pour le flux IGN + YOLO).
