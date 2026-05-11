# Entraînement des modèles satellite / aerial

Ce guide complète `docs/datasets.md` : formats, GPU recommandés, et **fine-tuning sur orthophotos IGN** (France).

## Dépendances optionnelles

```bash
pip install parking-capacity[train_satellite]
```

Inclut notamment `accelerate`, `datasets`, `ultralytics` (selon `pyproject.toml`).

## GPU

- **SegFormer / Mask2Former** : NVIDIA ≥ 16 Go VRAM recommandé pour entrées 512–1024 px ; sinon réduire `--batch-size` et `--img-size`.
- **YOLOv8** : 8–12 Go suffisent souvent pour `imgsz=640` et batch modeste.

## Formats produits par `datasets-prepare`

- **`parking_capacity_dataset/images`** — JPEG/PNG  
- **`masks`** — PNG binaire (parking)  
- **`labels`** — YOLO segmentation (polygones normalisés)  
- **`metadata.jsonl`** — ligne JSON par image (`split`, chemins relatifs)  
- **`coco_segmentation*.json`** — COCO instance segmentation  

## Commandes d’entraînement (CLI)

```bash
parking-capacity train-segformer --dataset apklot --epochs 50 --img-size 1024 --output-dir runs/sf

parking-capacity train-yolo-seg --dataset apklot --model yolov8m-seg.pt

parking-capacity train-mask2former --dataset apklot

parking-capacity train-vehicle-detector --dataset xview
```

Les chemins résolus via `data/datasets/dataset_registry.json` (`prepared_path`).

## Modules Python

- `python -m parking_capacity.training.train_segformer --dataset-root … --output-dir …`
- `python -m parking_capacity.training.train_yolov8_seg …`
- `python -m parking_capacity.training.train_vehicle_detector …` (nécessite `dataset.yaml` Ultralytics dans le jeu)

## Fine-tuning sur orthophotos IGN

1. Produire des puces WMS (voir `diagnose-address`, `harvest`, `make-training-run`).  
2. Annoter les masques parking (LabelMe, CVAT, etc.) dans le même répertoire que **`parking_capacity_dataset`**.  
3. Lancer `train-segformer` ou `train-yolo-seg` en pointant `--dataset-root` vers ce dossier (ou enregistrer un nouveau nom dans le registre).

Les domaines **urbains français** peuvent nécessiter un **fine-tune** depuis APKLOT / SpaceNet puis quelques centaines de puces IGN annotées.

## Inférence « deep satellite » produit

Le module `satellite_inference.py` enchaîne :

1. Segmentation parking (SegFormer chargé comme dans `vision_estimate`).  
2. Masques bâtiments / routes issus de la **fusion GIS** (`gis_fusion`).  
3. Surface utile et **capacité théorique** (m² / place).  
4. PNG de debug : `debug_segmentation_mask.png`, `debug_parking_polygon.png`, `debug_gis_segmentation_fusion.png`, etc.

## Benchmark

```bash
parking-capacity benchmark-satellite-modes --benchmark-dir ./bench_out --out ./summary.md
```

Agrège les `result.json` (MAE si capacité attendue, taux de refus, ratio de surestimation).

## Mask2Former

L’entrée CLI affiche la procédure recommandée : exporter **COCO** puis suivre les exemples officiels Hugging Face (`examples/pytorch/instance-segmentation`), car les hyperparamètres dépendent du backbone choisi.
