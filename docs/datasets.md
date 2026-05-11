# Jeux de données satellite / aerial

Ce document décrit **où obtenir** APKLOT, DOTA, xView et SpaceNet, les **licences**, et comment les utiliser avec ce dépôt (`datasets-download`, `datasets-prepare`).

## Emplacement local

- Racine : `data/datasets/`
- Brut : `data/datasets/raw/<nom>/`
- Préparé : `data/datasets/prepared/<nom>/parking_capacity_dataset/`
- Registre : `data/datasets/dataset_registry.json`

## 1. APKLOT (parking — **mixed** satellite + caméra)

- **Source** : [langheran/APKLOT](https://github.com/langheran/APKLOT)  
- **Publication** : [MDPI Applied Sciences 2020](https://www.mdpi.com/2076-3417/10/15/5364) — le titre mentionne la perspective caméra ; le README du dépôt décrit **deux** modalités : vue **satellite** (captures **Google Maps API**, dossier typique **`1. Satellite`**) et vue **caméra** (**`2. Camera`**, LabelMe, perspectives terrain / mobile).  
- **Licence** : MIT sur le dépôt ; **images Maps** — respecter les [conditions Google](https://www.google.com/permissions/geoguidelines/).  
- **Git LFS** : les fichiers volumineux passent par **Git LFS**. Sans `git lfs pull`, le clone peut ne contenir que des **pointeurs LFS** ou majoritairement la partie **caméra** (chemins « Office Lens », « segmentation mobile », etc.). Pour l’**orthophoto / satellite**, vérifiez la présence du dossier **`1. Satellite`** après LFS.  
- **Préparation** :  
  ```bash
  parking-capacity datasets-download --dataset apklot
  cd data/datasets/raw/apklot && git lfs install && git lfs pull
  parking-capacity datasets-prepare --dataset apklot --apklot-view auto
  ```  
  - `--apklot-view auto` (défaut) : **satellite uniquement** (chemins contenant « Satellite »).  
  - `--apklot-view all` : satellite **+** caméra (ancien comportement tout-dépôt).  
  - `--apklot-view camera` : uniquement les chemins caméra.  
- **Inspection** : `parking-capacity inspect-dataset --dataset apklot` — compte approximatif satellite vs caméra, présence masques, **suitability_for_satellite_parking**.  
- **Orthophoto IGN** : ne présumez pas que APKLOT suffit ; préférez jeux **DOTA / xView / SpaceNet** pour l’aerial pur, ou une APKLOT correctement tirée de **`1. Satellite`**.

## 2. DOTA

- **Source** : [DOTA dataset](https://captain-whu.github.io/DOTA/)  
- **Licence** : **recherche académique uniquement** ; usage commercial interdit. Images Google Earth / GF-2 / JL-1 / CycloMedia — respecter chaque fournisseur.  
- **Téléchargement** : les liens sont sur **Google Drive / Baidu Pan** (changement fréquent). Définir si disponible :
  - `DOTA_TRAIN_ZIP_URL`, `DOTA_VAL_ZIP_URL` puis `datasets-download --dataset dota`.
  - Sinon : placer les `.zip` sous `data/datasets/raw/dota/` et extraire ; puis `datasets-prepare`.
- **Préparation** : boîtes orientées exportées en **YOLO OBB** (`yolo_obb/train|val|test`).

## 3. xView

- **Source** : [xView Dataset](https://xviewdataset.org/)  
- **Licence** : contrat d’inscription (Defense Digital Service / organisateurs) ; **pas de redistribution** des données.  
- **Téléchargement** : souvent **manuel** après inscription. Variable optionnelle :
  - `XVIEW_SYNC_CMD='aws s3 sync …'` si vous disposez d’un miroir autorisé.
  - ou `XVIEW_MANUAL_ROOT=/chemin/vers/données`.
- **Préparation** : indexation des fichiers (GeoTIFF / GeoJSON) dans `metadata.jsonl` ; conversions détaillées dépassent le périmètre de ce module (utiliser GDAL / pipelines Spark pour les très gros GeoJSON).

## 4. SpaceNet

- **Source** : [SpaceNet](https://spacenet.ai/) — données sur **AWS Registry Open Data**.  
- **Licence** : licence SpaceNet par défi ; lecture publique S3, attribution selon les pages du registre.  
- **Téléchargement** :
  ```bash
  aws s3 ls s3://spacenet-dataset/
  aws s3 sync s3://spacenet-dataset/<chemin_challenge>/ data/datasets/raw/spacenet/
  ```
  Ou définir `SPACENET_SYNC_CMD` pour automatiser.

## Multi-datasets

Le module `dataset_mix.py` fusionne plusieurs dossiers `parking_capacity_dataset` (symlinks par défaut). Utile pour **APKLOT + SpaceNet** (masques routes/bâtiments comme contexte supplémentaire exporté séparément).

## Commandes CLI

```bash
parking-capacity inspect-dataset --dataset apklot
parking-capacity benchmark-dataset-mosaics --out data/datasets/benchmark_mosaic.png

parking-capacity datasets-download --dataset apklot
parking-capacity datasets-prepare --dataset apklot

parking-capacity datasets-download --dataset dota   # si URLs définies
parking-capacity datasets-prepare --dataset dota

parking-capacity datasets-download --dataset spacenet   # si SPACENET_SYNC_CMD / aws
parking-capacity datasets-prepare --dataset xview
```

Pour les jeux **sans URL automatique**, le CLI affiche les instructions et écrit `DOWNLOAD_INSTRUCTIONS.txt` dans le dossier brut.
