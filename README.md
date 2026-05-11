# parking-capacity

Outil en ligne de commande (France) : à partir d’une **adresse** (CSV), estime une **capacité de stationnement** en combinant :

1. **Géocodage** [API Adresse (BAN)](https://adresse.data.gouv.fr/outils/api-doc)  
2. **Parcelle cadastrale** [APICarto IGN – parcelle](https://apicarto.ign.fr/api/doc/cadastre)  
3. **Parkings OpenStreetMap** via [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) : somme des tags `capacity` sur les parkings qui intersectent la parcelle (priorité) ou un buffer autour du point  
4. **Orthophoto BD ORTHO** (WMS Géoplateforme [`data.geopf.fr/wms-r`](https://cartes.gouv.fr/aide/fr/guides-utilisateur/utiliser-les-services-de-la-geoplateforme/diffusion/wms-raster/)) : **géométrie parking** (OpenCV : Canny / Hough) en secours structurant ; **SegFormer** [UTEL-UIUC/SegFormer-large-parking](https://huggingface.co/UTEL-UIUC/SegFormer-large-parking) (poids `best_model.ckpt`, prétraitement compatible **nvidia/mit-b5**) n’est qu’**indice faible** sur le nombre de places tant qu’il n’est pas considéré comme modèle orthophoto *spécialisé* (`--visual-model-specialized`).  
5. **Optionnel** : régression ML sur puces, YOLO parking (`--yolo-weights` + `ultralytics`), backends futurs (`--visual-backend`).

## Installation

```bash
cd /chemin/vers/Yolo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

OpenCV (`opencv-python-headless`) est **inclus** dans les dépendances principales pour la géométrie parking.

Python **3.9+** recommandé (3.10+ si possible).

Les premiers lancements avec `--no-vision` désactivé téléchargent sur Hugging Face le backbone **nvidia/mit-b5** (préprocesseur + config) et le checkpoint **UTEL** (~ volumineux) ; le **GPU** est utilisé s’il est disponible.

Un inventaire des **sources de données France** (PAN, data.gouv, IGN, OSM, etc.) et les commandes `catalog` / `fetch-resource` sont décrits dans [`docs/sources-donnees-france.md`](docs/sources-donnees-france.md).

Documentation complémentaire : [`docs/architecture_audit.md`](docs/architecture_audit.md), [`docs/data_sources.md`](docs/data_sources.md), [`docs/model_training.md`](docs/model_training.md), [`docs/address_inference.md`](docs/address_inference.md), [`docs/limitations.md`](docs/limitations.md).

## Premier vrai modèle satellite parking (YOLOv8-seg + APKLOT)

Le produit distingue **segmentation de zones parking** (masques / polygones utiles) d’une **régression globale image → nombre de places**. Pour un premier modèle entraîné sur données réelles :

1. Installer les extras : `pip install -e ".[train_satellite]"` (Ultralytics, etc.).
2. Lancer le workflow automatisé (téléchargement APKLOT si besoin, préparation, layout YOLO, entraînement, exemples) :
   ```bash
   parking-capacity quickstart-apklot-yolo --dataset-subset small --run-name mon_run
   ```
   Sorties sous `data/runs/<run_name>/` : `train_metrics.json`, courbes, `sample_masks/`, `sample_overlays/`, `val_predictions/`, `best.pt`.
3. Jeu réduit Colab / test rapide : `--dataset-subset small` (sous-échantillon APKLOT).
4. Tester sur orthophoto IGN réelle + fusion GIS :
   ```bash
   parking-capacity test-segmentation-real \
     --weights data/runs/mon_run/best.pt \
     --out data/diagnostics/seg_test \
     "2 Bd Industriel, 76270 Neufchâtel-en-Bray"
   ```
5. Notebook Colab : [`notebooks/train_apklot_yolo.ipynb`](notebooks/train_apklot_yolo.ipynb).

Détails jeux de données : [`docs/datasets.md`](docs/datasets.md). Entraînement / GPU : [`docs/training_satellite_models.md`](docs/training_satellite_models.md).

Pourquoi **YOLOv8-seg** en premier : instances + masques vectorisables, bon compromis vitesse / qualité sur petits jeux ; pourquoi **APKLOT** : masques parking aériens alignés au problème ; la **segmentation** réduit toitures/voirie via post-traitement + **fusion BD TOPO / OSM** (`segmentation_gis_fusion.py`).

## Utilisation

### Interface web (une adresse)

```bash
pip install -e ".[ui]"
parking-capacity ui --port 7860
```

Ouvrez l’URL affichée (souvent `http://127.0.0.1:7860/`), saisissez l’adresse et cliquez sur **Analyser**. Cochez **Sans vision** pour aller plus vite (pas d’orthophoto ni SegFormer).

### Une adresse (sortie lisible ou JSON)

```bash
parking-capacity run-address "10 rue de la Santé, Paris" --radius-m 50
parking-capacity run-address "10 rue de la Santé, Paris" --format json
parking-capacity run-address "38 rue du Moulin à Vent, Paris" --radius-m 50 --source-priority hybrid \
  --cache-dir data/.cache_http --ml-checkpoint data/runs/lyon_small/model.pt
```

- **Défaut** : `--format pretty` (texte structuré : capacité, fourchette, méthode, confiance, sources, avertissements).  
- `--format json` / `json-pretty` : export machine.  
- `--force-ml` : utilise le checkpoint même si `model_meta.json` signale un jeu trop petit, synthétique ou un R² de validation négatif (sinon l’inférence ML est **ignorée** pour ne pas sur-vendre le modèle).  
- `--source-priority hybrid|aerial|osm` : ordre des sources (défaut **hybrid** : orthophoto toujours récupérée si possible, capacité OSM fiable prioritaire).  
- `--refresh-imagery` : ignore le cache WMS pour cette exécution.  
- `--cache-dir` : cache Overpass + puces orthophoto (PNG sous `wms_ortho/`).

### Workflow recommandé réel

1. **`diagnose-address`** sur quelques adresses cibles : export `chip.png`, `result.json`, `sources.json`, `debug_map.html`, `debug_overlay.png`, `warnings.txt` pour contrôle visuel.

   ```bash
   parking-capacity diagnose-address "38 rue du Moulin à Vent, Paris" --radius-m 50 --out data/diagnostics/moulin_vent --cache-dir data/.cache_http
   ```

   En CI ou sans réseau : ajouter `--mock` (HTTP simulé, pas de SegFormer).

2. **`make-training-run`** sur une petite bbox (ou un **preset** France : `paris_small`, `lyon_small`, `nantes_small`, `rennes_small`, `bordeaux_small`) pour enchaîner moisson OSM + puces + entraînement **resnet18** + évaluation + **`real_run_report.md`**.

   ```bash
   parking-capacity make-training-run --preset lyon_small --out data/runs/lyon_small --max-samples 300 --cache-dir data/.cache_http
   ```

3. **Lire `real_run_report.md`** : MAE, RMSE, R², comparaison baseline surface / ML, exemples de prédictions, avertissements si le jeu compte moins de 100 exemples ou si les métriques sont mauvaises.

4. **`run-address`** (ou `run` CSV) avec **`--ml-checkpoint`** seulement si le rapport conclut que le modèle est acceptable ; sinon rester sur **hybrid / OSM / surface / orthophoto** sans sur-vendre le ML. Sans `--force-ml`, un `model_meta.json` défavorable désactive l’inférence ML.

### Comment savoir si mon estimateur est fiable ?

Les **tests automatisés** ne remplacent pas une validation terrain. Workflow recommandé :

1. **`diagnose-address`** sur quelques cas représentatifs (puce + carte + `result.json`).  
2. **`benchmark-addresses`** sur un CSV d’adresses (`data/benchmark/addresses.csv` peut servir de modèle) avec éventuellement une colonne **`expected_capacity`** pour des métriques automatiques.  
3. Remplir **`manual_review.csv`** (généré dans le dossier benchmark) : ouvrir chaque `chip_path` / `overlay_path`, saisir **`human_count`**, éventuellement `accepted`.  
4. **`evaluate-manual-review`** sur ce CSV complété → `manual_eval_report.md` + `summary.json`.  
5. **`go-no-go-report`** en croisant le dossier benchmark, le `model.pt` et optionnellement le répertoire d’évaluation manuelle (`--manual-eval`).  
6. **Production** : n’utiliser le checkpoint comme source principale que si cette chaîne est concluante ; sinon privilégier OSM / surface / orthophoto et accepter des **fourchettes** ou **`low_confidence`**.

Exemples :

```bash
parking-capacity benchmark-addresses --input data/benchmark/addresses.csv --out data/benchmark/results --cache-dir data/.cache_http
parking-capacity evaluate-manual-review --input data/benchmark/results/manual_review.csv --out data/benchmark/manual_eval
parking-capacity go-no-go-report --benchmark data/benchmark/results --model data/runs/lyon_small/model.pt --out data/reports/go_no_go.md
```

Le fichier **`model_meta.json`** (à côté de `model.pt`) contient désormais `n_train_samples`, `n_val_samples`, `dataset_mode`, métriques de validation, `split_method`, `created_at`, et optionnellement `source_bbox` / `source_preset` pour tracer l’origine du jeu.

### Fichier CSV (lot)

```bash
parking-capacity run -i examples/addresses.sample.csv -o out/results.csv --radius-m 50 \
  --cache-dir data/.cache_http --source-priority hybrid
```

- Colonne d’adresse : `adresse` ou `address` (ou `--address-column`)  
- `--format jsonl` : une ligne JSON par adresse  
- `--no-vision` : uniquement BAN + parcelle + OSM (plus léger, pas de PyTorch sur l’image)  
- `--vision-compare` : lance aussi la vision lorsque OSM fournit déjà une capacité (pour comparer)  
- `--overpass-delay 1` : pause entre requêtes (courtoisie vis-à-vis des serveurs publics)  
- `--cache-dir data/.cache_http` : cache disque des réponses Overpass (POST)  
- `--no-aerial-first` : sans `capacity` OSM, privilégier surface / `parking_space` avant orthophoto  

### Catalogue des ressources (entraînement / labels)

```bash
parking-capacity catalog -o data/catalog_stationnement.csv
parking-capacity catalog -o data/catalog_pan.csv --no-datagouv
parking-capacity harvest-labels -c data/catalog_pan.csv -o data/harvested_labels.csv --max-files 30
parking-capacity build-chips --manifest data/harvested_labels.csv --out data/chip_dataset --max-rows 100
parking-capacity harvest-real-dataset --out data/real_parking_dataset --bbox "2.20,48.80,2.45,48.95" --country FR
parking-capacity fetch-resource "https://…" -o data/ressource.csv --max-mb 200
```

### Entraînement ML (régression capacité sur puces)

**Données réelles (objectif métier)** — enchaînement typique :

1. Moissonner les métadonnées : `parking-capacity catalog -o data/catalog.csv`  
2. Extraire labels tabulaires : `parking-capacity harvest-labels -c data/catalog.csv -o data/harvested_labels.csv` (CSV/GeoJSON issus du PAN / data.gouv)  
3. Télécharger les **orthophotos BD ORTHO** par point : `parking-capacity build-chips --manifest data/harvested_labels.csv --out data/chip_dataset`  
4. Entraîner : `parking-capacity train-model -d data/ml_real --chip-dir data/chip_dataset --architecture resnet18 --loss huber --target-transform log1p --split geo --epochs 30`  
5. Évaluer : `parking-capacity eval-model -c data/ml_real/model.pt --chip-dir data/chip_dataset --metrics-json data/ml_real/metrics_eval.json`

Alternative **labels OSM + orthophoto** dans une bbox :

```bash
parking-capacity harvest-real-dataset --out data/real_parking_dataset --bbox "2.20,48.80,2.45,48.95"
parking-capacity train-model -d data/runs/osm_fr --chip-dir data/real_parking_dataset --arch resnet18 --split geo
```

- **`train-model`** : `tiny` (tests synthétiques), `resnet18` / `resnet50` / `efficientnet_b0` avec `pip install -e ".[train]"`.  
- **`eval-model`** : métriques étendues ; option `--metrics-json`.

```bash
# Démo technique uniquement (pas des vraies aires de stationnement)
parking-capacity train-model -d data/ml_demo --synthetic-n 500 --epochs 15 --lr 0.01

# Production : puces WMS + labels issus du moissonnage
parking-capacity train-model -d data/ml_real --chip-dir data/chip_dataset --epochs 30
parking-capacity eval-model -c data/ml_real/model.pt --chip-dir data/chip_dataset
```

Sur **orthophotos réelles**, la qualité dépend des labels (`harvest-labels` / contrôle manuel) et du bruit ; prévoir `resnet18` et plus d’époques si besoin.

### Pourquoi la géométrie parking est préférable à une simple régression CNN globale

Un parking marqué présente une **structure répétitive** : places rectangulaires, **rangées parallèles**, **orientation dominante** des marquages, **espacement régulier**. Une régression qui mappe toute la puce vers un scalaire mélange bâtiments, végétation et bruit d’imagerie ; elle ne « voit » pas explicitement ces motifs. L’heuristique géométrique (contours, droites, clustering d’orientation, espacement) exploite directement ces régularités pour proposer un **ordre de grandeur de places visibles** et un score de confiance dédié, tout en laissant le pipeline dire **« je ne sais pas »** (`refuse_prediction`, capacité absente) lorsque la preuve visuelle ou OSM est insuffisante ou que la fourchette reste trop large (`range_quality_score`, `overall_reliability_score`).

**Commandes utiles** : `parking-capacity benchmark-vision "Adresse…" --out data/vision_bench/run1` (compare baseline surface, `geometry_only`, SegFormer, ML si `--ml-checkpoint`) ; `parking-capacity export-vision-dataset --input data/benchmark/results --out data/vision_dataset` (COCO minimal + `metadata.jsonl` pour pseudo-labels / entraînement YOLO ou Detectron).

## Limites et obligations

- **CGU / quotas** : respectez les conditions d’usage de la **BAN**, **APICarto**, **Overpass** (ne pas monopoliser les instances publiques ; augmenter `--overpass-delay` en batch) et de la [**Géoplateforme**](https://geoservices.ign.fr/). Les couches et paramètres WMS peuvent évoluer : en cas d’erreur `GetMap`, vérifiez la documentation IGN / cartes.gouv.fr.  
- **Précision** : les tags OSM sont **incomplets** ; la vision est entraînée hors contexte métier « clinique » et reste une **estimation** (parkings souterrains invisibles, ombres, etc.).  
- **Données** : le résultat doit être présenté avec les champs `primary_confidence`, `caveats` et `primary_source`.

## Tests

```bash
pytest
```

Les tests réseau sont **mockés** (pas d’appel Overpass/BAN réel en CI).
