"""Fine-tuning yolov8s-visdrone sur chips BD ORTHO françaises via self-pseudo-labeling.

Pipeline :
1. Récupérer ~30 chips BD ORTHO autour d'adresses françaises diverses (urbain + rural).
2. Faire prédire VisDrone (conf très bas) sur chaque chip → bboxes.
3. Filtrer par taille m² plausible + classes véhicules → labels conservés.
4. Sauver chips + labels YOLO (.txt) au format dataset.
5. Fine-tune yolov8s-visdrone sur ce dataset (15 epochs, MPS).
6. Validation sur Neufchâtel / Bouzonville / Vénissieux.

Limites honnêtes : self-pseudo-labeling ne peut pas inventer d'information manquante. Effet attendu :
- Réduit les faux positifs aberrants (bboxes 50+ m²).
- Spécialise sur résolution / couleurs BD ORTHO.
- Pas de gain en rappel sur véhicules manqués par le modèle source.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Tuple

import httpx
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("finetune")


VISDRONE_WEIGHTS = (
    "/Users/mac/Yolo/data/aerial_weights/models--mshamrai--yolov8s-visdrone/"
    "snapshots/996db9c3fbdf92ecdcf58411449a39232bfbcff6/best.pt"
)
DATASET_DIR = Path("/Users/mac/Yolo/data/finetune/french_aerial_v1")
WEIGHTS_OUT_DIR = Path("/Users/mac/Yolo/data/aerial_weights/finetuned_v1")

# ~30 adresses diverses : urbain dense, rural, périurbain, grand parking, petit commerce.
# Chacune produira 1 chip 768×768 à 160m × 160m de couverture.
SEED_ADDRESSES = [
    # Cas connus
    "2 Bd Industriel, 76270 Neufchâtel-en-Bray",
    "38 Rue du Moulin à Vent, Vénissieux",
    "2A Rue Saint-Hubert, 57320 Bouzonville",
    # Grands parkings (centres commerciaux probables)
    "Centre commercial Beaugrenelle, 75015 Paris",
    "1 Avenue de la Paix, 67000 Strasbourg",
    "Rue de la République, 69001 Lyon",
    # Petits sites
    "10 Place de la Mairie, 13100 Aix-en-Provence",
    "1 Rue de Rivoli, 75001 Paris",
    "5 Avenue Foch, 75116 Paris",
    "20 Boulevard Saint-Michel, 75006 Paris",
    # Périurbain
    "Hôpital, 33700 Mérignac",
    "Lycée, 33000 Bordeaux",
    "École primaire, 67000 Strasbourg",
    "Gare SNCF, 31000 Toulouse",
    "Mairie, 13100 Aix-en-Provence",
    # Industriel
    "Zone industrielle, 76200 Dieppe",
    "Z.A.C., 76600 Le Havre",
    "1 Rue de l'Industrie, 67000 Strasbourg",
    # Résidentiel
    "100 Rue de la République, 13002 Marseille",
    "50 Rue Pasteur, 31000 Toulouse",
    "10 Avenue Jean Jaurès, 38000 Grenoble",
    # Diversité régions
    "Place du Capitole, 31000 Toulouse",
    "Place Bellecour, 69002 Lyon",
    "Place Stanislas, 54000 Nancy",
    "Place de la Bourse, 33000 Bordeaux",
]


def fetch_orthophoto_chip(address: str, half_side_m: float = 80.0, chip_pixels: int = 768) -> Tuple[Image.Image, str, float, float]:
    """Géocode adresse + télécharge chip orthophoto BD ORTHO."""
    from parking_capacity.geocode import geocode_address
    from parking_capacity.imagery_wms import fetch_ortho_chip, DEFAULT_WMS_BASE, DEFAULT_WMS_LAYER

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        g = geocode_address(address, client=client)
        if g.score < 0.5:
            raise ValueError(f"BAN low score: {g.score}")
        chip = fetch_ortho_chip(
            g.lon, g.lat,
            half_side_m=half_side_m,
            width_px=chip_pixels, height_px=chip_pixels,
            wms_base=DEFAULT_WMS_BASE, layer=DEFAULT_WMS_LAYER,
            client=client,
            cache_dir=Path("/Users/mac/Yolo/data/finetune/cache"),
        )
        return chip.image, g.label, g.lon, g.lat


def detect_and_filter(chip: Image.Image, weights: str, m_per_px: float, conf_th: float = 0.20) -> List[Tuple[int, float, float, float, float]]:
    """Détecte avec VisDrone + filtre par classe + taille → bboxes YOLO normalisées.

    Retour : list de (class_id, x_center_norm, y_center_norm, w_norm, h_norm) avec class_id=0 (single 'vehicle').
    Filtres : car/van/truck/bus, dimensions véhicule plausibles.
    """
    from ultralytics import YOLO
    m = YOLO(weights)
    arr = np.asarray(chip.convert("RGB"))
    h_px, w_px = arr.shape[:2]

    r = m.predict(arr, conf=conf_th, verbose=False, imgsz=768)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    vehicle_classes = {"car", "van", "truck", "bus"}
    labels = []
    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i].cpu().numpy())
        cls_name = m.names[cls_id]
        if cls_name not in vehicle_classes:
            continue
        box = r.boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        # Filtre dimensions en m
        w_m = (x2 - x1) * m_per_px
        h_m = (y2 - y1) * m_per_px
        longe = max(w_m, h_m)
        shorte = min(w_m, h_m)
        if longe < 2.0 or longe > 8.5:
            continue
        if shorte < 1.0 or shorte > 4.0:
            continue
        if longe * shorte < 2.5 or longe * shorte > 30.0:
            continue
        # Conversion YOLO format (centre normalisé + taille normalisée)
        cx = ((x1 + x2) / 2) / w_px
        cy = ((y1 + y2) / 2) / h_px
        bw = (x2 - x1) / w_px
        bh = (y2 - y1) / h_px
        labels.append((0, cx, cy, bw, bh))  # single class "vehicle"
    return labels


def build_dataset() -> None:
    """Construit le dataset BD ORTHO + pseudo-labels."""
    log.info(f"Building dataset → {DATASET_DIR}")
    images_dir = DATASET_DIR / "images" / "train"
    labels_dir = DATASET_DIR / "labels" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    val_lbl_dir = DATASET_DIR / "labels" / "val"
    for d in (images_dir, labels_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    m_per_px = (80.0 * 2) / 768  # ~0.21 m/px

    n_train = 0
    n_val = 0
    n_labels_total = 0
    fails = []
    for i, addr in enumerate(SEED_ADDRESSES):
        try:
            chip, label, lon, lat = fetch_orthophoto_chip(addr)
            slug = f"{i:02d}_" + "".join(c for c in label[:30] if c.isalnum() or c in ("_", "-")).lower()
            labels = detect_and_filter(chip, VISDRONE_WEIGHTS, m_per_px)
            # 80/20 train/val split
            is_val = (i % 5 == 0)
            img_out = (val_img_dir if is_val else images_dir) / f"{slug}.png"
            lbl_out = (val_lbl_dir if is_val else labels_dir) / f"{slug}.txt"
            chip.save(img_out, format="PNG")
            with lbl_out.open("w") as f:
                for (cid, cx, cy, bw, bh) in labels:
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            if is_val:
                n_val += 1
            else:
                n_train += 1
            n_labels_total += len(labels)
            log.info(f"  [{i+1:02d}/{len(SEED_ADDRESSES)}] {addr[:40]:<40} → {len(labels):>3} véhicules {'(val)' if is_val else '(train)'}")
        except Exception as e:
            fails.append((addr, str(e)))
            log.info(f"  [{i+1:02d}/{len(SEED_ADDRESSES)}] {addr[:40]:<40} FAILED: {e}")

    log.info(f"\n  Train: {n_train} chips | Val: {n_val} chips | Total labels: {n_labels_total} | Failed: {len(fails)}")
    if fails:
        for addr, err in fails:
            log.info(f"    × {addr}: {err}")

    # YAML pour Ultralytics
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(f"""# French BD ORTHO aerial vehicle dataset (self-pseudo-labeled)
path: {DATASET_DIR}
train: images/train
val: images/val
names:
  0: vehicle
""")
    log.info(f"  YAML: {yaml_path}")


def fine_tune() -> str:
    """Fine-tune yolov8s-visdrone sur dataset français via MPS."""
    from ultralytics import YOLO

    WEIGHTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"\nFine-tuning yolov8s-visdrone on French aerial dataset (MPS, 15 epochs)...")
    m = YOLO(VISDRONE_WEIGHTS)
    results = m.train(
        data=str(DATASET_DIR / "dataset.yaml"),
        epochs=15,
        imgsz=768,
        batch=4,
        device="mps",
        project=str(WEIGHTS_OUT_DIR),
        name="run1",
        verbose=False,
        patience=10,
        save=True,
        save_period=-1,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    log.info(f"Best weights: {best}")
    return str(best)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("build", "all"):
        build_dataset()
    if cmd in ("train", "all"):
        fine_tune()
