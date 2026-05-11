# Entraînement des modèles (régression sur puces orthophoto)

## Prérequis

```bash
pip install -e ".[train]"
```

## Jeu de données

1. **Option A — labels tabulaires** : `catalog` → `harvest-labels` → `build-chips`  
2. **Option B — labels OSM dans une bbox** :  

```bash
parking-capacity harvest-real-dataset --out data/real_parking_dataset \
  --bbox "2.20,48.80,2.45,48.95" --country FR --cache-dir data/.cache_overpass
```

Puis (les puces sont déjà dans `--out`) :

```bash
parking-capacity train-model -d data/runs/parking_resnet18 \
  --chip-dir data/real_parking_dataset \
  --architecture resnet18 --epochs 40 --lr 1e-3 \
  --loss huber --target-transform log1p --split geo
```

## Architectures

| Nom | Usage |
|-----|--------|
| `tiny` | Tests synthétiques uniquement (moyenne RGB → MLP). |
| `resnet18` / `resnet50` | Backbones `torchvision`, tête régression 1 neurone. |
| `efficientnet_b0` | Alternative plus légère / efficace. |

## Pertes et cible

- `--loss mse` (défaut) ou `--loss huber` (moins sensible aux outliers).  
- `--target-transform log1p` : le modèle prédit `log1p(capacity)` ; métriques et inférence repassent en **places** via `expm1`.

## Split

- `--split random` : shuffle i.i.d. (léger risque de fuite spatiale).  
- `--split geo` : regroupement par cellule `(lat, lon)` arrondis puis répartition train/val des **cellules** (moins de fuite grossière).

## Artefacts produits (`--output-dir`)

- `model.pt`, `model_meta.json`, `summary.json`, `metrics_history.jsonl`  
- `metrics.csv`, `predictions_val.csv`, `dataset_manifest.csv`

## Savoir si le modèle est « bon »

- Regarder **MAE / RMSE / MAPE** sur `predictions_val.csv` (surtout buckets 0–10, 10–30, 30–100, 100+).  
- Comparer à une **baseline** tabulaire / surface (voir `baseline.py` et inférence adresse).  
- Si le ML est systématiquement pire que la baseline sur validation, **ne pas l’utiliser en production** : le message est aussi remonté côté inférence (`ml_vs_baseline_note`).
