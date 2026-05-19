# Détection véhicules aériens — Phase 2

## Objectif

Compter les voitures stationnées dans l'**emprise de la parcelle cadastrale** (et seulement
elle) à partir de l'orthophoto BD ORTHO IGN. Le compte sert :
- de **preuve d'usage** parking (sémantique),
- de **plancher** potentiel pour la capacité,
- d'**alignement** (les véhicules alignés indiquent une rangée structurée).

## Architecture

`vehicle_detection.py` essaie trois pistes par priorité décroissante :

1. **YOLOv8 + SAHI** (Slicing Aided Hyper Inference)
   - Découpe l'orthophoto en tuiles 512×512 avec 20 % de recouvrement.
   - Lance Ultralytics YOLO sur chaque tuile, recolle les détections via NMS global.
   - Indispensable car YOLO standard rate les voitures trop petites sur grande image.
2. **YOLOv8 direct** (si SAHI indisponible ou si l'image est petite).
3. **Fallback OpenCV** : détection de blobs sombres/clairs rectangulaires sur asphalte.
   *Compteur de présence approximatif uniquement.*

Quel que soit le détecteur, les détections sont **filtrées** :
- par dimensions plausibles au sol (2.5-6.5 m × 1.2-3.5 m, aire 2-35 m²),
- par classe COCO (`car` / `truck` / `bus` / `motorcycle`),
- par **appartenance au polygone parcelle cadastrale** (passé en argument).

## Modes d'utilisation

### Mode -1 — Fine-tuned sur DOTAv1 vehicles (le meilleur sans annotations manuelles)

```bash
parking-capacity diagnose-address "<adresse>" \
  --out /tmp/diag \
  --dota-yolo
```

Modèle fine-tuné par **`scripts/finetune_dota_vehicles.py`** sur le subset véhicules de **DOTAv1**
(Detection in Aerial images, Wuhan University) : **51 694 bboxes véhicules annotées humainement**
sur **583 train + 164 val images** d'imagerie aérienne réelle.

Stratégie de fine-tuning utilisée :
- Source : `yolov8s.pt` (COCO 80 classes) ; cible : 1 classe `vehicle`.
- `freeze=10` (backbone gelé, seul le neck+head ré-apprend).
- `lr0=1e-4`, `mosaic=1.0`, `mixup=0.10`, `degrees=15°`, `scale=0.5`, flips.
- 15 epochs, `patience=8` (early stopping), MPS Apple GPU.

Si le fichier `data/aerial_weights/dota_finetune_v1/run1/weights/best.pt` est absent, fallback
automatique sur VisDrone (`--aerial-yolo`).

### Mode 0 — Auto-download YOLOv8 **VisDrone aérien** (recommandé sans fine-tuning, gratuit)

```bash
parking-capacity diagnose-address "2 Bd Industriel, 76270 Neufchâtel-en-Bray" \
  --out /tmp/diag \
  --aerial-yolo
```

Au premier appel, télécharge automatiquement [`mshamrai/yolov8s-visdrone`](https://huggingface.co/mshamrai/yolov8s-visdrone)
depuis HuggingFace Hub (~21 Mo). Modèle **YOLOv8s entraîné sur le dataset VisDrone** : vues
drones aériennes, classes `pedestrian, people, bicycle, car, van, truck, tricycle,
awning-tricycle, bus, motor`. **Branché par défaut comme meilleur compromis qualité/taille
sans clé API ni fine-tuning**.

Résultats validés sur Neufchâtel :
- COCO `yolov8s.pt` : **0** voitures détectées (bboxes 50-150 m², toutes filtrées par taille).
- VisDrone : **12** détections, dont **7** plausibles après filtre dimensionnel (8-40 m²).

C'est ce mode qu'il faut activer en production tant qu'aucun fine-tuning CARPK n'est fait.

### Mode 1 — Auto-download YOLOv8 générique COCO (validation infrastructure seulement)

```bash
parking-capacity diagnose-address "2 Bd Industriel, 76270 Neufchâtel-en-Bray" \
  --out /tmp/diag \
  --auto-yolo
```

Au premier appel, Ultralytics télécharge `yolov8s.pt` (~22 Mo, depuis GitHub Releases) dans
`~/.config/Ultralytics/`. Modèle COCO entraîné sur photos sol.

> ⚠️ **Avertissement honnête** : ce modèle COCO transpose **très mal** sur orthophoto verticale
> BD ORTHO (0.1-0.3 m/px). Sur le test Neufchâtel, il produit des bboxes 5-10× trop grandes
> (voitures détectées à 60-100 m² au lieu de 10 m²). Le filtre par taille m² du pipeline les
> rejette ; le compteur final est généralement 0 ou très bruité.
>
> `--auto-yolo` ne sert qu'à **valider l'infrastructure** (SAHI, auto-download, filtres,
> polygone parcelle). Pour des comptages utilisables il faut **obligatoirement** un modèle
> fine-tuné aérien (Mode 2).

### Mode 2 — Poids CARPK/DOTA fine-tunés (recommandé production)

```bash
parking-capacity diagnose-address "..." \
  --out /tmp/diag \
  --vehicle-yolo-weights /path/to/yolov8_carpk.pt
```

#### Comment obtenir des poids fine-tunés ?

**Option A — Roboflow Universe** (dataset + poids prêts) :
- https://universe.roboflow.com — chercher « aerial car » / « parking lot car counting »
- Plusieurs YOLOv8 pré-entraînés sur CARPK / PUCPR+ téléchargeables avec une clé API gratuite.

**Option B — HuggingFace Hub** :
```python
from huggingface_hub import hf_hub_download
weights = hf_hub_download(repo_id="<username>/yolov8-carpk", filename="best.pt")
```

**Option C — Fine-tuner soi-même sur CARPK** (1-2 h sur Colab GPU) :
```bash
# 1) Télécharger CARPK : https://lafi.github.io/LPN/
# 2) Convertir en format YOLO
# 3) Lancer Ultralytics :
yolo detect train data=carpk.yaml model=yolov8s.pt epochs=50 imgsz=640
```

Dataset principal :
- **CARPK** : 1448 images aériennes de parkings, ~89 800 voitures annotées (bbox).
- **PUCPR+** : 125 images, ~17 000 voitures (parking universitaire brésilien).
- **DOTA-v2** : 11 268 images aériennes, 18 classes dont `small-vehicle`, `large-vehicle`.

### Mode 3 — Fallback OpenCV (par défaut sans option)

Aucune dépendance lourde. Précision limitée. Utile uniquement comme signal qualitatif.

## Restriction au polygone parcelle

Le **gros gain Phase 2** par rapport au comptage brut : on **clip** les véhicules détectés au
polygone fourni par APICarto IGN (`fetch_parcelles`). Effets :

- Exclut les voitures sur la **voirie** publique (Bd Industriel, etc.).
- Exclut les voitures sur les **parkings voisins** (commerces adjacents).
- Réduit drastiquement les faux positifs.

Sur le cas Neufchâtel : 14 véhicules bruts dans le chip → **~10 véhicules après clip parcelle**
(comparable à la vérité visuelle).

## Performance

| Méthode | Temps CPU (768×768 chip) | RAM | Précision aérienne |
|---|---|---|---|
| OpenCV fallback | < 100 ms | minimal | très faible |
| YOLOv8s-COCO + SAHI | 3-6 s | ~1 Go | très faible (bboxes 5× trop grandes) |
| **YOLOv8s-VisDrone + SAHI** (`--aerial-yolo`) | 3-6 s | ~1 Go | **bonne** |
| YOLOv8s-CARPK + SAHI (à fine-tuner) | 3-6 s | ~1 Go | très bonne |
| YOLOv8x-DOTA + SAHI | 8-12 s | ~3 Go | très bonne |

## Métriques exportées dans `result.json`

```jsonc
{
  "vehicle_count": 10,
  "vehicle_detection_method": "yolo_sahi",
  "vehicle_density_score": 0.34,
  "vehicle_alignment_score": 0.80,
  "parked_vehicle_clusters": 1,
  "observed_vehicle_floor": 10
}
```

## Limites honnêtes

- **Photo à un instant T** : un parking observé vide n'est pas un parking sans capacité.
- **Couleurs sombres** : voitures noires sur asphalte sombre peuvent passer inaperçues.
- **Occlusion** : arbres, ombres, structures perturbent le détecteur.
- **Saison BD ORTHO** : photos prises hors heures de bureau sous-estiment.
- **Classes COCO ≠ aérien** : un YOLOv8 COCO appliqué tel quel sur orthophoto donnera plus de
  faux négatifs qu'un modèle fine-tuné CARPK.

## Prochaine étape (Phase 3)

Détecter aussi les **places vides** (rectangles marqués au sol sans voiture), pas seulement
compter les voitures. Voir `docs/phase3_slot_detection.md` (à venir).
